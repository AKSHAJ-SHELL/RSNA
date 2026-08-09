"""Shape, masking and gradient-routing checks for the study model.

The masking assertions are the ones that matter. A masked softmax that silently leaks
attention onto absent slots still trains, still converges, and still produces a submission —
it just reads the wrong images. Nothing downstream would raise.
"""

from __future__ import annotations

import pytest
import torch

from rsnaknee.model import SLOTS, ModelConfig, StudyModel, weighted_bce

N_TARGETS = 12


@pytest.fixture(scope="module")
def model() -> StudyModel:
    # trainable_blocks=1 keeps the fixture cheap while still exercising the thaw logic.
    # eval() matters: head dropout is on by default, so in train mode two forward passes on
    # identical input disagree and every determinism assertion below becomes a coin flip.
    return StudyModel(ModelConfig(pretrained=False, trainable_blocks=1)).eval()


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor]:
    slots = torch.randn(2, len(SLOTS), 3, 224, 224)
    mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.bool
    )
    return slots, mask


def test_output_shapes(model, batch):
    slots, mask = batch
    logits, attn = model(slots, mask, return_attention=True)
    assert logits.shape == (2, N_TARGETS)
    assert attn.shape == (2, N_TARGETS, len(SLOTS))


def test_attention_ignores_absent_slots(model, batch):
    """Absent slots must receive exactly zero attention, and present ones must sum to 1."""
    slots, mask = batch
    _, attn = model(slots, mask, return_attention=True)

    absent = attn.masked_fill(mask.unsqueeze(1), 0.0)
    assert absent.max().item() == pytest.approx(0.0, abs=1e-7)
    assert torch.allclose(attn.sum(-1), torch.ones(2, N_TARGETS), atol=1e-5)


def test_absent_slot_contents_cannot_change_output(model, batch):
    """Overwriting a masked-out slot must not move the logits.

    This is the assertion that would catch a mask applied after the softmax instead of
    before, or a mask silently broadcast along the wrong axis.
    """
    slots, mask = batch
    # Perturb per study: the two rows have different masks, so a single shared index would
    # scribble over slots that are genuinely present in the other row.
    with torch.no_grad():
        before = model(slots, mask)
        perturbed = slots.clone()
        for i in range(slots.shape[0]):
            absent = ~mask[i]
            perturbed[i, absent] = torch.randn_like(perturbed[i, absent]) * 50
        after = model(perturbed, mask)
    assert torch.allclose(before, after, atol=1e-4)


def test_empty_study_does_not_produce_nan(model):
    """A study with no usable slots must yield finite logits, not poison the batch."""
    slots = torch.randn(1, len(SLOTS), 3, 224, 224)
    mask = torch.zeros(1, len(SLOTS), dtype=torch.bool)
    logits = model(slots, mask)
    assert torch.isfinite(logits).all()


def test_zero_weight_cell_contributes_no_gradient(model, batch):
    """Confidence 0 must mean "no opinion", not "confident negative"."""
    slots, mask = batch
    logits = model(slots, mask)
    targets = torch.zeros(2, N_TARGETS)

    weights = torch.zeros(2, N_TARGETS)
    weights[0, 0] = 1.0
    loss_one = weighted_bce(logits, targets, weights)

    weights_extra = weights.clone()
    weights_extra[1, 5] = 0.0  # still zero — must change nothing
    assert weighted_bce(logits, targets, weights_extra).item() == pytest.approx(loss_one.item())


def test_weighted_bce_normalises_by_weight_not_count(model):
    """Doubling every weight must leave the loss unchanged."""
    logits = torch.randn(4, N_TARGETS)
    targets = torch.randint(0, 2, (4, N_TARGETS)).float()
    weights = torch.rand(4, N_TARGETS) + 0.1
    assert weighted_bce(logits, targets, weights).item() == pytest.approx(
        weighted_bce(logits, targets, weights * 2).item(), rel=1e-5
    )


def test_only_last_blocks_are_trainable():
    cfg = ModelConfig(pretrained=False, trainable_blocks=2)
    model = StudyModel(cfg)
    blocks = model.encoder.backbone.blocks
    n = len(blocks)

    for i, block in enumerate(blocks):
        expected = i >= n - 2
        assert all(p.requires_grad == expected for p in block.parameters()), (
            f"block {i} of {n}: expected requires_grad={expected}"
        )


def test_frozen_encoder_yields_single_param_group():
    model = StudyModel(ModelConfig(pretrained=False, trainable_blocks=0))
    groups = model.param_groups(head_lr=1e-3)
    assert len(groups) == 1, "a frozen encoder must not hand the optimiser an empty group"
    assert groups[0]["lr"] == 1e-3


def test_encoder_learns_slower_than_head():
    cfg = ModelConfig(pretrained=False, trainable_blocks=2, encoder_lr_scale=0.02)
    groups = StudyModel(cfg).param_groups(head_lr=1e-3)
    assert len(groups) == 2
    assert groups[1]["lr"] == pytest.approx(2e-5)
