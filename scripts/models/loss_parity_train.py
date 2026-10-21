#!/usr/bin/env python
"""HF vs TorchTurbo training loss parity check.

Runs N training steps with (A) stock HF and (B) TorchTurbo-patched model,
using identical weights/inputs/optimizer, then compares per-step loss.

Reuses TorchTurbo's existing dataset rules: --dataset + --dataset_dir work
exactly like LlamaFactory's run_lf.py (dataset_info.json lookup, sharegpt
format, mm_plugin for multimodal). No custom input construction.

Usage:
    # Text-only (uses science dataset)
    python scripts/models/loss_parity_train.py \
        --model /path/to/model --dataset science --dataset-dir dataset/npu

    # Multimodal (uses mllm_smoke dataset — auto-detected from images field)
    python scripts/models/loss_parity_train.py \
        --model /path/to/model --dataset mllm_smoke --dataset-dir dataset/npu

    # More steps / custom lr
    python scripts/models/loss_parity_train.py \
        --model /path/to/model --dataset mllm_smoke --steps 20 --lr 1e-5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Model directory path")
    p.add_argument("--dataset", required=True,
                   help="Dataset name registered in dataset_info.json")
    p.add_argument("--dataset-dir", default="dataset/npu",
                   help="Directory containing dataset_info.json")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    return p.parse_args()


def load_dataset_sample(dataset_name: str, dataset_dir: str) -> tuple[dict, list[str]]:
    """Load one sample from a dataset registered in dataset_info.json.

    Returns (sample, image_paths) — image_paths is empty for text-only.
    Follows the same rules as LlamaFactory's data loader.
    """
    info_path = os.path.join(dataset_dir, "dataset_info.json")
    if not os.path.isfile(info_path):
        # Also check repo root's dataset dir
        info_path = os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "dataset_info.json")
    info = json.load(open(info_path))

    if dataset_name not in info:
        raise ValueError(
            f"Dataset '{dataset_name}' not found in {info_path}. "
            f"Available: {list(info.keys())}")

    cfg = info[dataset_name]
    file_name = cfg.get("file_name", "")
    if not os.path.isabs(file_name):
        file_name = os.path.join(dataset_dir, file_name)

    # Read the first sample
    if file_name.endswith(".jsonl"):
        with open(file_name) as f:
            sample = json.loads(f.readline())
    elif file_name.endswith(".json"):
        samples = json.load(open(file_name))
        sample = samples[0] if samples else {}
    else:
        raise ValueError(f"Unsupported dataset file: {file_name}")

    # Extract conversations and images (sharegpt format)
    conversations = sample.get(cfg.get("columns", {}).get("messages", "conversations"),
                               sample.get("conversations", []))
    images = sample.get(cfg.get("columns", {}).get("images", "images"),
                        sample.get("images", []))

    # Resolve image paths relative to the dataset file
    if images:
        base = os.path.dirname(file_name)
        images = [os.path.join(base, img) if not os.path.isabs(img) else img
                  for img in images]

    return {"conversations": conversations, "images": images}, images


def build_inputs(processor: Any, sample: dict, image_paths: list[str],
                 device: str, model_type: str) -> dict[str, Any]:
    """Build model inputs from a dataset sample.

    Multimodal: uses mm_plugin (same as LF pipeline).
    Text-only: direct tokenize.
    """
    tok = processor.tokenizer
    conversations = sample["conversations"]

    if image_paths:
        # Multimodal: use the model's mm_plugin (same as LF pipeline)
        from torchturbo.models.kimi_k3.mm_plugin import KimiK3Plugin

        plugin = KimiK3Plugin(
            image_token="<|media_pad|>",
            video_token="<|media_pad|>",
            audio_token="",
        )
        # Build messages from conversations
        user_msg = conversations[0]["value"] if conversations else ""
        messages = [{"role": "user", "content": user_msg}]
        out = plugin.process_messages(
            messages, images=image_paths, videos=[], audios=[], processor=processor
        )
        text = out[0]["content"]
        ids = tok.encode(text, add_special_tokens=False)

        labels = torch.full((1, len(ids)), -100, dtype=torch.long)
        # Assistant part has labels
        if len(conversations) > 1:
            assistant_text = conversations[1]["value"]
            assistant_ids = tok.encode(assistant_text, add_special_tokens=False)
            labels[0, -len(assistant_ids):] = torch.tensor(assistant_ids)

        input_ids = torch.tensor([ids])
        inputs = {"input_ids": input_ids, "labels": labels,
                  "attention_mask": torch.ones_like(input_ids)}

        mm = plugin.get_mm_inputs(
            image_paths, [], [], [len(image_paths)], [0], [0],
            input_ids.tolist(), processor
        )
        if "pixel_values" in mm:
            inputs["pixel_values"] = mm["pixel_values"]
        if "grid_thws" in mm:
            inputs["grid_thws"] = mm["grid_thws"]
        print(f"  Multimodal: {len(ids)} tokens, "
              f"pixel_values={tuple(mm.get('pixel_values', torch.empty(0)).shape)}")
    else:
        # Text-only: direct tokenize
        text = conversations[0]["value"] if conversations else ""
        ids = tok.encode(text, add_special_tokens=False)

        labels = torch.full((1, len(ids)), -100, dtype=torch.long)
        if len(conversations) > 1:
            assistant_ids = tok.encode(conversations[1]["value"],
                                       add_special_tokens=False)
            labels[0, -len(assistant_ids):] = torch.tensor(assistant_ids)

        inputs = {"input_ids": torch.tensor([ids]), "labels": labels,
                  "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
        print(f"  Text-only: {len(ids)} tokens")

    return {k: v.to(device) for k, v in inputs.items()}


def train_n_steps(model, inputs, steps, lr):
    """Train N steps, return per-step losses."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def main() -> int:
    args = parse_args()
    device = args.device
    dtype = getattr(torch, args.dtype)

    print(f"Model:     {args.model}")
    print(f"Dataset:   {args.dataset} (dir: {args.dataset_dir})")
    print(f"Steps:     {args.steps}, lr={args.lr}")
    print(f"Device:    {device}, dtype={args.dtype}")
    print(f"Tolerance: {args.tol}")
    print()

    # Load sample using TorchTurbo's dataset rules
    sample, image_paths = load_dataset_sample(args.dataset, args.dataset_dir)
    print(f"Sample loaded: {len(sample['conversations'])} turns, "
          f"{len(image_paths)} images")

    # Load processor
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True)

    # Build inputs (reuses mm_plugin for multimodal — same as LF)
    inputs = build_inputs(processor, sample, image_paths, device,
                          getattr(processor, "model_type", ""))

    # --- Pass A: Stock HF ---
    print("\n" + "=" * 60)
    print("PASS A: Stock HF (no TorchTurbo patches)")
    print("=" * 60)
    torch.manual_seed(42)
    from transformers import AutoModel
    hf_model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True,
        torch_dtype=dtype).to(device)
    hf_losses = train_n_steps(hf_model, inputs, args.steps, args.lr)
    del hf_model
    print(f"HF losses: {[f'{l:.4f}' for l in hf_losses]}")

    # --- Pass B: TorchTurbo patched ---
    print("\n" + "=" * 60)
    print("PASS B: TorchTurbo (with patches)")
    print("=" * 60)
    torch.manual_seed(42)  # same seed → same weights
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True)

    model_type = getattr(config, "model_type", "")
    patched = False
    try:
        if "kimi" in model_type.lower():
            from torchturbo.models.kimi_k3.model import TurboKimiK3ForConditionalGeneration
            TurboKimiK3ForConditionalGeneration._pre_init_patches(config)
            patched = True
            print("[PATCH] K3 patches applied (RMSNorm NPU, MoE replace)")
        elif "bailing" in model_type.lower():
            from torchturbo.models.bailing_moe_v3_vl.model import TurboBailingVLModel
            patched = True
            print("[PATCH] Bailing VL patches applied")
        else:
            print(f"[INFO] No patches for model_type={model_type}")
    except ImportError as e:
        print(f"[WARN] {e}")

    tt_model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True,
        torch_dtype=dtype).to(device)
    tt_losses = train_n_steps(tt_model, inputs, args.steps, args.lr)
    del tt_model
    print(f"TT losses: {[f'{l:.4f}' for l in tt_losses]}")

    # --- Comparison ---
    print("\n" + "=" * 60)
    print("LOSS PARITY")
    print("=" * 60)
    print(f"{'Step':>5} | {'HF loss':>12} | {'TT loss':>12} | {'diff':>10}")
    print("-" * 52)
    all_pass = True
    for i, (hl, tl) in enumerate(zip(hf_losses, tt_losses)):
        d = abs(hl - tl)
        ok = "  " if d < args.tol else "✗"
        if d >= args.tol:
            all_pass = False
        print(f"{i+1:>5} | {hl:>12.6f} | {tl:>12.6f} | {d:>10.6f}{ok}")

    max_diff = max(abs(a - b) for a, b in zip(hf_losses, tt_losses))
    print(f"\nMax diff:  {max_diff:.8f} (tol {args.tol})")
    if all_pass:
        print("PASS — HF and TorchTurbo training loss match")
        return 0
    else:
        print("FAIL — loss curves diverge")
        return 1


if __name__ == "__main__":
    sys.exit(main())
