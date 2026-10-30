"""Regression tests for kimi_k25 mm_plugin path-image handling.

sharegpt-style datasets hand the plugin image PATHS (str), but the model-dir
vision processor's get_resize_config expects PIL images
(media_input['image'].size). get_mm_inputs already regularizes paths via
BasePlugin._regularize_images; process_messages must do the same or dataset
tokenization crashes with AttributeError: 'str' object has no attribute
'size'.
"""

from PIL import Image

from torchturbo.models.kimi_k25.mm_plugin import KimiK25Plugin


class _StubVisualProcessor:
    def get_resize_config(self, media_input):
        assert media_input["type"] == "image"
        # The contract under test: a PIL image, not the raw path string.
        assert isinstance(media_input["image"], Image.Image), (
            f"expected PIL Image, got {type(media_input['image'])!r}"
        )
        return {"num_tokens": 16}


class _StubProcessor:
    # BasePlugin._validate_input requires these attrs to exist (non-None).
    image_processor = object()
    video_processor = object()
    media_processor = _StubVisualProcessor()


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


def test_process_messages_accepts_pil_image_directly(tmp_path):
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
