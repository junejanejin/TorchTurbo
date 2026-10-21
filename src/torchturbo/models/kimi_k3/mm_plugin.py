"""LLamaFactory mm_plugin for Kimi-K3.

Handles image data: process_messages (expand placeholder into the official
media prompt) + get_mm_inputs (preprocess image into MoonViT inputs).
K3 media protocol: <|media_pad|> placeholder (one per image; the modeling
merge expands it into N feature slots), MoonViT-V2 vision tower.
"""
from copy import deepcopy
from io import BytesIO
from typing import TYPE_CHECKING, BinaryIO, Literal, Optional, TypedDict, Union
from typing_extensions import override

from llamafactory.extras.constants import IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER
from llamafactory.data.mm_plugin import BasePlugin
from llamafactory.extras.packages import (
    is_pillow_available,
    is_transformers_version_greater_than,
)

if is_pillow_available():
    from PIL import Image
    from PIL.Image import Image as ImageObject

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedTokenizer, ProcessorMixin

    class EncodedImage(TypedDict):
        path: Optional[str]
        bytes: Optional[bytes]

    ImageInput = Union[str, bytes, EncodedImage, BinaryIO, ImageObject]
    VideoInput = Union[str, BinaryIO, list[list[ImageInput]]]


class KimiK3Plugin(BasePlugin):
    """mm_plugin for Kimi-K3 VLM.

    K3 media protocol (matching the official processor / modeling):
    each image expands to
    ``<|media_begin|>image {W}x{H}<|media_content|>`` + N x ``<|media_pad|>``
    + ``<|media_end|>``, where N is the resolution-dependent token count
    (navit_resize). ``<|media_pad|>`` is the token the modeling-side merge
    counts (``config.media_placeholder_token_id`` = 163605 in the shipped
    tokenizer). Vision tower: MoonViT-V2; processor: kimi_k3_processor.
    """
    image_token: str = "<|media_pad|>"
    video_token: str = "<|media_pad|>"

    def _expand_image_placeholder(self, processor, image) -> str:
        """Official K3 media prompt for one image.

        Follows the official protocol: ONE ``<|media_pad|>`` per image — the
        modeling-side merge expands that single pad position into the N
        feature slots (``_token_occupation_table``), so emitting N pads here
        would over-expand. The resolution header comes from the official
        ``make_image_prompt``; without a loadable image fall back to a bare
        pad.
        """
        vp = getattr(processor, "image_processor", None)
        make = getattr(vp, "make_image_prompt", None)
        pil = self._load_pil(image)
        if make is not None and pil is not None:
            return make(pil.size[0], pil.size[1])
        return self.image_token

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list,
        processor: "ProcessorMixin",
        **kwargs,
    ):
        """Expand image placeholders in messages.

        Each image placeholder is replaced with the official K3 media prompt
        (resolution header + the real number of <|media_pad|> tokens), so
        the modeling-side merge (which counts media pads) sees exactly as
        many placeholder tokens as MoonViT will produce features.
        """
        images = images or []
        img_idx = 0
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        image = images[img_idx] if img_idx < len(images) else None
                        item["text"] = self._expand_image_placeholder(
                            processor, image
                        )
                        img_idx += 1
            elif isinstance(content, str) and IMAGE_PLACEHOLDER in content:
                parts = content.split(IMAGE_PLACEHOLDER)
                rebuilt = parts[0]
                for part in parts[1:]:
                    image = images[img_idx] if img_idx < len(images) else None
                    rebuilt += self._expand_image_placeholder(processor, image)
                    rebuilt += part
                    img_idx += 1
                message["content"] = rebuilt
        return messages

    @override
    def process_token_ids(
        self,
        input_ids: list[int],
        labels: Optional[list[int]],
        images: list,
        videos: list,
        audios: list,
        tokenizer: "PreTrainedTokenizer",
        processor: "ProcessorMixin",
        **kwargs,
    ) -> tuple[list[int], Optional[list[int]]]:
        """Keep token ids unchanged on the text-only path."""
        return input_ids, labels

    @staticmethod
    def _load_pil(image: "ImageInput") -> Optional["ImageObject"]:
        """Best-effort load of an ImageInput (path / bytes / file-like /
        PIL / K3 media dict) into a PIL image; None on failure."""
        if is_pillow_available() and isinstance(image, Image.Image):
            return image
        if isinstance(image, dict):
            inner = image.get("image")
            if is_pillow_available() and isinstance(inner, Image.Image):
                return inner
            image = inner
        try:
            if isinstance(image, bytes):
                return Image.open(BytesIO(image))
            if isinstance(image, str):
                return Image.open(image)
            if image is not None and hasattr(image, "read"):
                return Image.open(image)
        except Exception:
            return None
        return None

    def _get_image_token_count(self, processor, image=None, **kwargs) -> int:
        """Number of image tokens for one image.

        K3's token count depends on the image resolution (navit_resize
        via the vision processor's ``media_tokens_calculator`` /
        ``get_resize_config``). Returns 1 only as a fallback when the image
        is unavailable (e.g. LlamaFactory's dummy text-only batches) or the
        processor does not expose the calculator.
        """
        if image is None:
            return 1
        if is_pillow_available() and not isinstance(image, (Image.Image, dict)):
            # ImageInput may be a path / bytes / file-like object; load it
            # (BasePlugin._preprocess_image requires max/min pixel args and
            # is meant for already-loaded images).
            image = self._load_pil(image)
            if image is None:
                return 1
        vision_processor = getattr(processor, "image_processor", processor)
        calculator = getattr(vision_processor, "media_tokens_calculator", None)
        if calculator is None or image is None:
            return 1
        try:
            media = image if isinstance(image, dict) else {"type": "image", "image": image}
            return int(calculator(media))
        except Exception:
            # Malformed media should not break text preprocessing; keep the
            # legacy placeholder behavior.
            return 1

    @override
    def get_mm_inputs(
        self,
        images: list,
        videos: list,
        audios: list,
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: "ProcessorMixin",
        **kwargs,
    ):
        """Convert real images into K3 vision inputs (pixel_values/grid_thws).

        LlamaFactory passes images as path strings (real batches) or dummy
        PIL objects (text-only batches it pads to avoid FSDP hangs). The
        official kimi_k3 image processor consumes Moonshot's structured
        media dicts (``media['type']``). Gate on the K3 image placeholder
        token actually appearing in ``batch_ids`` — dummy batches never
        contain it — then load each image and run the official
        ``image_processor.preprocess`` to produce MoonViT inputs.
        """
        placeholder_id = getattr(processor, "media_placeholder_token_id", None)
        if placeholder_id is None:
            tok = getattr(processor, "tokenizer", None)
            if tok is not None:
                encoded = tok.encode(self.image_token, add_special_tokens=False)
                placeholder_id = encoded[0] if encoded else None
        has_real_image = any(
            placeholder_id is not None and placeholder_id in ids
            for ids in (batch_ids or [])
        )
        if not has_real_image:
            return {}

        medias = []
        for m in (images or []):
            pil = self._load_pil(m)
            if pil is not None:
                medias.append({"type": "image", "image": pil})
        if not medias:
            return {}

        vision_processor = getattr(processor, "image_processor", processor)
        feats = vision_processor.preprocess(medias, return_tensors="pt")
        mm_inputs = {}
        for key in ("pixel_values", "grid_thws"):
            if key in feats:
                mm_inputs[key] = feats[key]
        return mm_inputs
