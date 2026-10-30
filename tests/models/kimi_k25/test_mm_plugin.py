"""Regression tests for kimi_k25 mm_plugin media handling.

Covers three behaviours:

1. sharegpt-style datasets hand the plugin image PATHS (str) while the
   model-dir vision processor's get_resize_config needs PIL images
   (media_input['image'].size) -> AttributeError: 'str' object has no
   attribute 'size' during dataset tokenization.
2. Kimi's processor declares no image_max_pixels, so the plugin must not
   inherit LLaMA-Factory's generic 768*768 clamp: the official pipeline
   bounds the token count via navit_resize (in_patch_limit /
   patch_limit_on_one_side / fixed_output_tokens) on the ORIGINAL image.
   A clamp would downscale large images and change the pad count.
3. Multiple images per sample: every placeholder gets its own N-pad block
   and every image its own resize config; a placeholder without a matching
   image must fail loudly instead of raising IndexError.
"""

import pytest
from PIL import Image

from torchturbo.models.kimi_k25.mm_plugin import KimiK25Plugin


class _StubVisualProcessor:
    def __init__(self):
        self.sizes = []

    def get_resize_config(self, media_input):
        assert media_input["type"] == "image"
        image = media_input["image"]
        assert isinstance(image, Image.Image), (
            f"expected PIL Image, got {type(image)!r}"
        )
        self.sizes.append(image.size)
        return {"num_tokens": 16}


class _StubProcessor:
    # BasePlugin._validate_input requires these attrs to exist (non-None).
    image_processor = object()
    video_processor = object()

    def __init__(self):
        self.media_processor = _StubVisualProcessor()


def _make_plugin():
    return KimiK25Plugin(
        image_token="<|media_pad|>",
        video_token="<|media_pad|>",
        audio_token=None,
    )


def test_process_messages_accepts_image_path(tmp_path):
    img_path = tmp_path / "img.png"
    Image.new("RGB", (64, 48), color=(255, 0, 0)).save(img_path)

    plugin = _make_plugin()
    messages = [{"role": "user", "content": "describe <image> please"}]

    out = plugin.process_messages(
        messages,
        images=[str(img_path)],
        videos=[],
        audios=[],
        processor=_StubProcessor(),
    )

    content = out[0]["content"]
    # N-pad protocol: exactly num_tokens pad tokens replace the placeholder.
    assert content.count("<|media_pad|>") == 16
    assert "<image>" not in content
    assert "describe" in content and "please" in content


def test_process_messages_accepts_pil_image_directly():
    pil = Image.new("RGB", (32, 32))

    plugin = _make_plugin()
    messages = [{"role": "user", "content": "describe <image>"}]

    out = plugin.process_messages(
        messages,
        images=[pil],
        videos=[],
        audios=[],
        processor=_StubProcessor(),
    )

    assert out[0]["content"].count("<|media_pad|>") == 16


def test_process_messages_expands_two_images(tmp_path):
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    Image.new("RGB", (32, 32)).save(first)
    Image.new("RGB", (64, 48)).save(second)

    processor = _StubProcessor()
    plugin = _make_plugin()

    out = plugin.process_messages(
        [{"role": "user", "content": "<image> versus <image>"}],
        images=[str(first), str(second)],
        videos=[],
        audios=[],
        processor=processor,
    )

    content = out[0]["content"]
    assert content.count("<|media_begin|>") == 2
    assert content.count("<|media_end|>") == 2
    assert content.count("<|media_pad|>") == 32  # 16 per image
    # Each image keeps its own resolution; no cross-talk between indices.
    assert processor.media_processor.sizes == [(32, 32), (64, 48)]


def test_process_messages_keeps_large_image_resolution(tmp_path):
    big = tmp_path / "big.png"
    Image.new("RGB", (1600, 1200)).save(big)

    processor = _StubProcessor()
    plugin = _make_plugin()

    plugin.process_messages(
        [{"role": "user", "content": "<image>"}],
        images=[str(big)],
        videos=[],
        audios=[],
        processor=processor,
    )

    # LLaMA-Factory's generic 768*768 clamp would shrink this to ~768x576.
    assert processor.media_processor.sizes == [(1600, 1200)]


def test_process_messages_rejects_placeholder_without_image():
    plugin = _make_plugin()

    with pytest.raises(ValueError, match="image placeholders"):
        plugin.process_messages(
            [{"role": "user", "content": "<image> and <image>"}],
            images=[Image.new("RGB", (32, 32))],
            videos=[],
            audios=[],
            processor=_StubProcessor(),
        )
