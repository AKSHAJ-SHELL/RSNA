"""Derive a smaller pixel cache from an existing one, without re-decoding.

Cache size goes as `slices x resolution²`, and on a 16 GB host the difference between fitting in
page cache and not is the difference between a GPU-bound run and an IO-bound one. Re-decoding on
Kaggle costs hours; resampling what we already have costs minutes.

    # 9.9 GB -> 5.6 GB
    .venv/bin/python scripts/downscale_cache.py \
        --src data/cache/r224s8 --dst data/cache/r168s8 --image-size 168

    # 9.9 GB -> 4.9 GB, keeping resolution
    .venv/bin/python scripts/downscale_cache.py \
        --src data/cache/r224s8 --dst data/cache/r224s4 --slices 4

Downscaling is lossy and one-way: 168px derived from a 224px cache is not identical to 168px
decoded from source, because it resamples an already-resampled image. It is close enough for the
slice-budget and resolution sweeps, and the final model should be trained from a cache decoded at
its own resolution.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from rsnaknee.cache import SHARD_STUDIES, open_cache, write_meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--image-size", type=int, default=0, help="0 keeps the source resolution")
    p.add_argument("--slices", type=int, default=0, help="0 keeps the source slice count")
    args = p.parse_args()

    import cv2

    pixels, mask, index, meta = open_cache(args.src)
    size = args.image_size or meta.image_size
    slices = args.slices or meta.slices

    if size > meta.image_size:
        raise SystemExit(
            f"Cannot upscale: source is {meta.image_size}px. Re-decode at {size}px instead — "
            "invented detail would look like real anatomy to the encoder."
        )
    if slices > meta.slices:
        raise SystemExit(f"Cannot add slices: source has {meta.slices}.")

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    n = meta.n_studies
    bounds = list(range(0, n, SHARD_STUDIES)) + [n]
    shards = [
        np.lib.format.open_memmap(
            dst / f"pixels_{k:03d}.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(bounds[k + 1] - bounds[k], meta.n_slots, slices, size, size),
        )
        for k in range(len(bounds) - 1)
    ]

    # Keep the centre band, matching how the source cache chose its slices — the joint is in the
    # middle and the outer slices are largely soft tissue.
    start = (meta.slices - slices) // 2

    for i in tqdm(range(n), desc=f"{meta.image_size}px/{meta.slices}sl -> {size}px/{slices}sl"):
        study = np.asarray(pixels[i][:, start : start + slices])
        if size != meta.image_size:
            study = np.stack(
                [
                    np.stack(
                        [cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA) for frame in slot]
                    )
                    for slot in study
                ]
            )
        k = min(i // SHARD_STUDIES, len(shards) - 1)
        shards[k][i - bounds[k]] = study

    for s in shards:
        s.flush()

    np.save(dst / "mask.npy", mask)
    index.to_series(name="StudyInstanceUID").reset_index(drop=True).to_frame().to_parquet(
        dst / "index.parquet"
    )
    write_meta(
        dst,
        replace(meta, image_size=size, slices=slices, slice_rule=f"{meta.slice_rule}->centred-{slices}"),
    )

    total = sum(s.nbytes for s in shards)
    print(f"wrote {dst}: {total / 1024**3:.2f} GB in {len(shards)} shards")


if __name__ == "__main__":
    main()
