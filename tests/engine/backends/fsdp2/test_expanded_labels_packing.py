"""Regression tests for multimodal label expansion vs packing boundaries.

Models that expand media placeholders inside forward (e.g. kimi_k3) return
post-expansion labels whose length no longer matches the packing boundaries
computed by the collator on the pre-expansion token layout. Reusing those
stale boundaries in the loss makes the boundary mask erase every supervised
token: CE sums to zero and training silently becomes a no-op.
"""

import torch

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend


def test_swap_expanded_labels_drops_stale_boundaries():
    labels = torch.full((1, 72), -100, dtype=torch.long)
    labels[0, 60:72] = 7
    expanded = torch.full((1, 337), -100, dtype=torch.long)
    expanded[0, 300:312] = 7
    stale_cu = torch.tensor([0, 72], dtype=torch.int32)

    swapped, cu = FSDP2Backend._swap_expanded_labels(labels, expanded, stale_cu)

    assert swapped is expanded
    assert cu is None, "packing boundaries must be dropped after expansion"


def test_swap_expanded_labels_keeps_text_path_untouched():
    labels = torch.zeros(1, 8, dtype=torch.long)
    sentinel_cu = torch.tensor([0, 8], dtype=torch.int32)

    swapped, cu = FSDP2Backend._swap_expanded_labels(labels, None, sentinel_cu)

    assert swapped is labels
    assert cu is sentinel_cu


def test_swap_expanded_labels_equal_length_keeps_boundaries():
    labels = torch.zeros(1, 8, dtype=torch.long)
    same_length = torch.zeros(1, 8, dtype=torch.long)
    sentinel_cu = torch.tensor([0, 8], dtype=torch.int32)

    swapped, cu = FSDP2Backend._swap_expanded_labels(
        labels, same_length, sentinel_cu
    )

    # Equal length means no expansion happened: engine-side labels and the
    # packing boundaries both stay untouched.
    assert swapped is labels
    assert cu is sentinel_cu


def test_without_boundaries_expanded_supervision_survives_shift():
    """The seam contract the fix relies on: collator marks sample starts as
    ignore_index, so shift-by-one needs no boundary metadata to stay safe."""
    seq_len = 337
    labels = torch.full((1, seq_len), -100, dtype=torch.long)
    labels[0, 300:312] = 7
    predictions = [torch.zeros(1, seq_len, 16)]

    shifted, normalizers = FSDP2Backend._prepare_loss_labels(
        predictions,
        labels,
        sp_size=1,
        sp_group=None,
        sp_rank=0,
        num_items_in_batch=None,
        cu_seq_lens=None,
    )

    # All 12 supervised tokens survive: the shift drops position 0 and the
    # supervised block sits at 300..311.
    assert (shifted[0] != -100).sum().item() == 12
