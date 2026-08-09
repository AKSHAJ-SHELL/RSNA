"""Study-level model: slot encoder + per-class attention over series slots.

A study is not one image. It is several series acquired in one session, and each of the
twelve findings is read on particular sequences — an effusion on a fluid-sensitive sequence,
patellofemoral cartilage on an axial. So the study is presented to the encoder as a fixed
set of *slots*, one per (plane x fluid-sensitivity) combination, and the head is allowed to
choose which slots each diagnosis listens to.

The head design follows `pilkwang/rsna-knee-baseline-v1` §6 (see ATTRIBUTION.md), which is
the public floor here rather than our differentiator: per-diagnosis query, learned slot
identity, masked softmax over present slots.

Two things are deliberate and easy to get wrong:

Missing slots are masked out of the softmax, not zero-filled. Most studies do not have all
six — mean 5.5 series per study, and slots are not series. Feeding a zero vector for an
absent axial teaches the model that "absent" is a particular kind of image; renormalising
over what is present instead shifts that diagnosis onto sequences the study actually has.

The encoder is fine-tuned, not frozen, but only its last blocks and at a far lower learning
rate than the head. A self-supervised natural-image encoder has never seen the signal a torn
meniscus makes on proton density, so freezing it caps every downstream axis at once. Moving
all of it instead would destroy good early filters, since 4,407 weakly-labelled studies is
not enough supervision to improve generic edge detectors but is easily enough to damage them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsnaknee.reports import TARGETS

#: The six slots a study is decomposed into. Fluid-sensitive sequences (T2/PD/STIR) show
#: effusion, synovitis and marrow oedema; non-fluid-sensitive (T1/PD without fat-sat) show
#: anatomy and cartilage. Crossing that with the three planes covers what the protocol is
#: for, and keeps the slot count small enough that a per-class softmax over slots is a
#: 6-way choice rather than a diffuse average.
SLOTS: tuple[tuple[str, int], ...] = (
    ("Sagittal", 1),
    ("Sagittal", 0),
    ("Coronal", 1),
    ("Coronal", 0),
    ("Axial", 1),
    ("Axial", 0),
)


@dataclass
class ModelConfig:
    backbone: str = "vit_small_patch14_dinov2.lvd142m"
    pretrained: bool = True
    #: Encoder input size. DINOv2 ships at 518, which is far more than a slot image needs and
    #: costs quadratically: the pixel cache and the attention both grow with `image_size**2`.
    #: Must be divisible by the backbone patch size (14 for DINOv2), and this is the single
    #: most important knob for the efficiency track, so it is a first-class config field
    #: rather than something buried in a transform.
    image_size: int = 224
    #: Number of trailing transformer blocks left trainable. 0 freezes the encoder entirely,
    #: which is the cheap-ablation mode, not the mode we submit.
    trainable_blocks: int = 4
    proj_dim: int = 256
    attn_dim: int = 128
    dropout: float = 0.1
    n_slots: int = len(SLOTS)
    targets: list[str] = field(default_factory=lambda: list(TARGETS))
    #: Encoder learns this many times slower than the head. The head starts random and has
    #: everything to learn; the encoder starts from a good solution and needs only nudging.
    encoder_lr_scale: float = 0.02


class SlotEncoder(nn.Module):
    """Wraps a timm backbone and exposes one embedding per slot image."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            num_classes=0,
            img_size=cfg.image_size,
        )
        patch = getattr(self.backbone.patch_embed, "patch_size", (14, 14))[0]
        if cfg.image_size % patch:
            raise ValueError(
                f"image_size={cfg.image_size} is not divisible by the backbone patch size "
                f"{patch}; positional embeddings would not tile. Nearest valid sizes: "
                f"{cfg.image_size // patch * patch} or {(cfg.image_size // patch + 1) * patch}."
            )
        self.embed_dim: int = self.backbone.num_features
        self._freeze_all_but_last(cfg.trainable_blocks)

    def _freeze_all_but_last(self, k: int) -> None:
        """Freeze everything, then thaw the final `k` blocks and the norm that follows them.

        The final norm is thawed with the blocks on purpose: leaving a frozen norm on top of
        moving blocks fights the adaptation it is supposed to pass through.
        """
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        if k <= 0:
            return
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            raise AttributeError(f"{type(self.backbone).__name__} exposes no `.blocks` to thaw.")
        for block in blocks[-k:]:
            for p in block.parameters():
                p.requires_grad_(True)
        for name in ("norm", "fc_norm"):
            mod = getattr(self.backbone, name, None)
            if isinstance(mod, nn.Module):
                for p in mod.parameters():
                    p.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, S, 3, H, W) slot images -> (B, S, D) embeddings.

        Slots are folded into the batch so every slot in every study is one encoder pass;
        the alternative, looping over slots, serialises six forward passes for no gain.
        """
        b, s = x.shape[:2]
        flat = self.backbone(x.flatten(0, 1))
        return flat.view(b, s, -1)


class PerClassAttention(nn.Module):
    """Each diagnosis attends over the slots with its own query."""

    def __init__(self, cfg: ModelConfig, in_dim: int):
        super().__init__()
        n_targets = len(cfg.targets)
        self.project = nn.Sequential(
            nn.Linear(in_dim, cfg.proj_dim),
            nn.LayerNorm(cfg.proj_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.slot_identity = nn.Parameter(torch.zeros(cfg.n_slots, cfg.proj_dim))
        self.to_key = nn.Linear(cfg.proj_dim, cfg.attn_dim)
        self.queries = nn.Parameter(torch.empty(n_targets, cfg.attn_dim))
        self.classifier = nn.Parameter(torch.empty(n_targets, cfg.proj_dim))
        self.bias = nn.Parameter(torch.zeros(n_targets))
        self.attn_dim = cfg.attn_dim

        nn.init.normal_(self.queries, std=cfg.attn_dim**-0.5)
        nn.init.normal_(self.classifier, std=cfg.proj_dim**-0.5)
        nn.init.normal_(self.slot_identity, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, S, D), (B, S) bool -> logits (B, T) and attention (B, T, S).

        Attention is returned because it is the only window we have into *where* a diagnosis
        is being read from. With study-level labels there is nothing supervising it, so it is
        a diagnostic rather than a claim — but a head putting all of its ACL mass on an axial
        slot is telling us something is wrong upstream.
        """
        h = self.project(x) + self.slot_identity  # (B, S, P)
        keys = self.to_key(h)  # (B, S, A)
        scores = torch.einsum("ta,bsa->bts", self.queries, keys) / self.attn_dim**0.5

        # Masked softmax. -inf before the softmax rather than zeroing after, so the weights
        # renormalise over present slots instead of summing to less than one.
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        attn = scores.softmax(dim=-1)

        # A study with no slots at all would make the softmax all-NaN. Clamp to zero and let
        # the bias carry it, rather than poisoning the batch gradient.
        attn = torch.nan_to_num(attn, nan=0.0)

        context = torch.einsum("bts,bsp->btp", attn, h)
        logits = torch.einsum("btp,tp->bt", context, self.classifier) + self.bias
        return logits, attn


class StudyModel(nn.Module):
    """Slot images in, twelve logits out."""

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.encoder = SlotEncoder(self.cfg)
        self.head = PerClassAttention(self.cfg, self.encoder.embed_dim)

    def forward(
        self, slots: torch.Tensor, mask: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        logits, attn = self.head(self.encoder(slots), mask)
        return (logits, attn) if return_attention else logits

    def param_groups(self, head_lr: float) -> list[dict]:
        """Discriminative learning rates: encoder slow, head fast.

        Returns only parameters that require grad, so a frozen-encoder run does not hand the
        optimiser empty tensors to carry.
        """
        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        groups = [{"params": self.head.parameters(), "lr": head_lr}]
        if encoder_params:
            groups.append({"params": encoder_params, "lr": head_lr * self.cfg.encoder_lr_scale})
        return groups


def weighted_bce(
    logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """BCE where each (study, finding) cell carries its own confidence weight.

    A report that never mentions synovitis should pull on that head far less than one that
    names it. Passing `weight=0` for unmentioned cells is what makes "unmentioned" different
    from "negative" — without it, every silent report becomes a confident negative and the
    rarer findings drown.

    Normalised by total weight rather than cell count, so a batch that happens to contain
    many silent cells does not quietly shrink its own gradient.
    """
    loss = F.binary_cross_entropy_with_logits(logits, targets, weight=weights, reduction="sum")
    return loss / weights.sum().clamp_min(1.0)
