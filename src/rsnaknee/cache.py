"""The decoded-pixel cache and the dataset that reads it.

Reading dominates this pipeline. A study is several series of tens of slices — on the order
of 150 files — which is affordable once and ruinous once per epoch. So slots are decoded a
single time into a `uint8` memmap and every epoch after that is arithmetic.

Layout on disk, one directory per cache:

    pixels.npy     uint8 memmap, (n_studies, n_slots, slices, P, P)
    mask.npy       bool,  (n_studies, n_slots) — which slots the study actually has
    index.parquet  study order, so rows can be joined back to targets
    meta.json      the preprocessing this cache was built under

`meta.json` exists because nothing about a cache announces what it is. A model fitted at one
resolution, slice band or laterality convention loads cleanly against a cache built under
another, runs, and returns predictions computed from the wrong image. The metadata travels
with the pixels and is checked on open.

Memory: the array is memory-mapped, not read. Two processes opening the same cache share one
set of physical pages, which is what makes a second concurrent training run affordable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from rsnaknee.model import SLOTS

#: Slices stacked as encoder channels. Three is not a free parameter — it is what an
#: ImageNet/DINOv2 stem expects, and feeding it a 3-slice neighbourhood gives the encoder
#: local through-plane context for free.
GROUP = 3


@dataclass(frozen=True)
class CacheMeta:
    image_size: int
    slices: int
    n_slots: int
    n_studies: int
    laterality_normalised: bool
    #: Free-text description of the slice-selection rule, recorded so two caches built under
    #: different rules are distinguishable after the fact.
    slice_rule: str = "centred"

    def assert_compatible(self, image_size: int) -> None:
        if self.image_size != image_size:
            raise ValueError(
                f"Cache was built at {self.image_size}px but the model expects {image_size}px. "
                "Shapes would broadcast and the run would silently score the wrong pixels."
            )


def write_meta(root: Path, meta: CacheMeta) -> None:
    (root / "meta.json").write_text(json.dumps(asdict(meta), indent=2))


def read_meta(root: Path) -> CacheMeta:
    return CacheMeta(**json.loads((root / "meta.json").read_text()))


#: Studies per shard. At 224px x 8 slices x 6 slots this is a little under 1 GB per file.
#: Kaggle's kernel-output endpoint resets the connection partway through a single ~10 GB file,
#: and a failed transfer of one shard is retryable while a failed transfer of the whole cache
#: is three hours of decode thrown away.
SHARD_STUDIES = 441


class ShardedPixels:
    """Read-only view over pixel shards that indexes like one array.

    Each shard stays an independent memmap, so the OS pages in only what is touched and two
    processes opening the same cache still share physical pages. Concatenating them into one
    array would defeat both.
    """

    def __init__(self, paths: list[Path]):
        self.shards = [np.load(p, mmap_mode="r") for p in paths]
        self.counts = [len(s) for s in self.shards]
        self.offsets = np.cumsum([0] + self.counts)
        tail = self.shards[0].shape[1:]
        if any(s.shape[1:] != tail for s in self.shards):
            raise ValueError("Pixel shards disagree on shape beyond the first axis.")
        self.shape = (int(self.offsets[-1]), *tail)
        self.dtype = self.shards[0].dtype

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, i: int) -> np.ndarray:
        shard = int(np.searchsorted(self.offsets, i, side="right") - 1)
        return self.shards[shard][i - self.offsets[shard]]

    @property
    def nbytes(self) -> int:
        return sum(s.nbytes for s in self.shards)


def open_cache(root: str | Path):
    """Memory-map a cache directory. Never loads the pixels into the process.

    Accepts both layouts: a single `pixels.npy`, or `pixels_000.npy`, `pixels_001.npy`, ...
    """
    root = Path(root)
    meta = read_meta(root)

    single = root / "pixels.npy"
    shards = sorted(root.glob("pixels_*.npy"))
    if single.exists() and single.stat().st_size > 0:
        pixels = np.load(single, mmap_mode="r")
    elif shards:
        pixels = ShardedPixels(shards)
    else:
        raise FileNotFoundError(
            f"No pixel data in {root}. Expected pixels.npy or pixels_*.npy. "
            f"(A zero-byte pixels.npy usually means the download was reset partway.)"
        )

    mask = np.load(root / "mask.npy")
    index = pd.read_parquet(root / "index.parquet")["StudyInstanceUID"]
    expected = (meta.n_studies, meta.n_slots, meta.slices, meta.image_size, meta.image_size)
    if tuple(pixels.shape) != expected:
        raise ValueError(
            f"pixels is {tuple(pixels.shape)}, meta.json declares {expected}. "
            "A shard is missing or truncated — re-download rather than training on part of it."
        )
    return pixels, mask, pd.Index(index), meta


class StudyDataset(Dataset):
    """Serves one study as (slots, mask, targets, weights).

    Training draws a random group of `GROUP` consecutive slices per slot, which doubles as
    augmentation along the stack. Evaluation takes the centre group so the number is
    reproducible; averaging over all groups is an inference-time choice made in `predict`,
    not here, because it multiplies cost and the metric may not pay for it.
    """

    def __init__(
        self,
        pixels: np.memmap,
        mask: np.ndarray,
        index: pd.Index,
        targets: pd.DataFrame,
        weights: pd.DataFrame,
        rows: np.ndarray,
        train: bool,
        seed: int = 0,
    ):
        self.pixels = pixels
        self.mask = mask
        self.rows = rows
        self.train = train
        self.rng = np.random.default_rng(seed)

        # Align targets to cache row order once, so __getitem__ is pure indexing.
        uids = index[rows]
        self.targets = targets.reindex(uids).to_numpy(dtype=np.float32)
        self.weights = weights.reindex(uids).to_numpy(dtype=np.float32)
        if np.isnan(self.targets).any():
            raise ValueError("Targets contain NaN after alignment — a study has no extraction.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        stack = self.pixels[row]  # (n_slots, slices, P, P) uint8, still on disk
        n_slices = stack.shape[1]

        if self.train:
            start = int(self.rng.integers(0, max(1, n_slices - GROUP + 1)))
        else:
            start = max(0, (n_slices - GROUP) // 2)

        group = np.asarray(stack[:, start : start + GROUP], dtype=np.float32) / 255.0
        if group.shape[1] < GROUP:  # thin stack: repeat the last slice rather than zero-pad
            pad = np.repeat(group[:, -1:], GROUP - group.shape[1], axis=1)
            group = np.concatenate([group, pad], axis=1)

        # `.copy()` on the label rows: they are slices of a shared array, and handing torch a
        # non-writable view earns a warning now and undefined behaviour if anything downstream
        # ever writes in place.
        return (
            torch.from_numpy(group),
            torch.from_numpy(self.mask[row].copy()),
            torch.from_numpy(self.targets[i].copy()),
            torch.from_numpy(self.weights[i].copy()),
        )


def synthetic_cache(root: str | Path, n_studies: int = 64, image_size: int = 224, slices: int = 8):
    """Write a tiny random cache so the training loop can be exercised without real data.

    This exists to prove the loop runs, the shapes line up and the metric computes. It cannot
    tell us anything about accuracy — the labels are noise — and any number it produces should
    be treated as a smoke test result and nothing else.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    n_slots = len(SLOTS)
    pixels = np.lib.format.open_memmap(
        root / "pixels.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(n_studies, n_slots, slices, image_size, image_size),
    )
    pixels[:] = rng.integers(0, 256, pixels.shape, dtype=np.uint8)
    pixels.flush()

    mask = rng.random((n_studies, n_slots)) > 0.2
    mask[:, 0] = True  # every study has at least one slot
    np.save(root / "mask.npy", mask)

    uids = pd.Series([f"synthetic-{i:05d}" for i in range(n_studies)], name="StudyInstanceUID")
    uids.to_frame().to_parquet(root / "index.parquet")

    write_meta(
        root,
        CacheMeta(
            image_size=image_size,
            slices=slices,
            n_slots=n_slots,
            n_studies=n_studies,
            laterality_normalised=False,
            slice_rule="synthetic-random",
        ),
    )
    return root
