"""Labels handed to the model forward: SP must not expose full labels.

Multimodal models (kimi_k3) expand media placeholders inside forward and
need the labels there to return post-expansion supervision. But under
sequence parallelism the model's logits are shard-local (T/sp per rank)
while labels stay full-length, so a model that computes its own CE inside
forward (kimi_k25) dies with:

    ValueError: Expected input batch_size (58) to match target batch_size (117)

The engine's SP-aware loss (``_prepare_loss_labels``) is authoritative, so
labels must only be forwarded when sp_size == 1.
"""

import torch

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend


def test_model_forward_labels_passed_without_sequence_parallel():
    labels = torch.zeros(1, 8, dtype=torch.long)

    forwarded = FSDP2Backend._model_forward_labels(labels, sp_size=1)

    assert forwarded is labels


def test_model_forward_labels_omitted_under_sequence_parallel():
    labels = torch.zeros(1, 118, dtype=torch.long)

    forwarded = FSDP2Backend._model_forward_labels(labels, sp_size=2)

    assert forwarded is None


def test_model_forward_labels_omitted_for_larger_sp():
    labels = torch.zeros(1, 512, dtype=torch.long)

    assert FSDP2Backend._model_forward_labels(labels, sp_size=8) is None
