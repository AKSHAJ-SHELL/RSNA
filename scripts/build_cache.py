"""Decode DICOM into a uint8 pixel cache.

Runs identically on a local subset and on Kaggle, where the competition data is already
mounted — the only difference is `--data-root` and `--limit`.

    # local, small, for validating preprocessing on real images
    .venv/bin/python scripts/build_cache.py --data-root data/raw --limit 100 \
        --out data/cache/r224s8 --image-size 224 --slices 8

    # Kaggle, everything
    python scripts/build_cache.py --data-root /kaggle/input/rsna-knee-abnormality-detection \
        --out /kaggle/working/cache/r224s8 --image-size 224 --slices 8

Laterality is decided once per *study*, not per series. The knee does not change between
sequences, and a per-series decision would let a noisy axial stack disagree with the sagittal
one and mirror half a study.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from rsnaknee.cache import CacheMeta, write_meta
from rsnaknee.dicom_io import load_series, read_header, study_side
from rsnaknee.model import SLOTS


def slot_of(plane: str, fluid: int) -> int | None:
    """Index into SLOTS for a (plane, fluid-sensitive) pair, or None if it is not one we keep."""
    for i, (slot_plane, slot_fluid) in enumerate(SLOTS):
        if slot_plane == plane and slot_fluid == int(fluid):
            return i
    return None


def series_dir(root: Path, split: str, study: str, series: str) -> Path:
    return root / f"{split}_series" / study / series


def build_one(job: tuple) -> tuple[str, np.ndarray, np.ndarray]:
    """Decode every slot of one study. Returns (uid, stack, mask)."""
    study, rows, root, split, n_slices, image_size = job
    n_slots = len(SLOTS)
    out = np.zeros((n_slots, n_slices, image_size, image_size), dtype=np.uint8)
    mask = np.zeros(n_slots, dtype=bool)

    # Group candidate series by slot, then decide laterality from every header in the study.
    by_slot: dict[int, list[tuple[str, list[Path]]]] = defaultdict(list)
    all_headers = []
    for row in rows:
        slot = slot_of(row["Anatomical_Plane"], row["Fluid_Sensitive"])
        if slot is None:
            continue
        directory = series_dir(root, split, study, row["SeriesInstanceUID"])
        paths = sorted(directory.glob("*.dcm"))
        if not paths:
            continue
        by_slot[slot].append((row["Anatomical_Plane"], paths))
        # Sample headers rather than reading all — laterality needs a median, not a census.
        all_headers.extend(h for h in (read_header(p) for p in paths[::8]) if h is not None)

    side = study_side(all_headers)

    for slot, candidates in by_slot.items():
        # More slices means better coverage of the joint; ties keep the first deterministically.
        plane, paths = max(candidates, key=lambda c: len(c[1]))
        stack = load_series(paths, plane=plane, side=side, n_slices=n_slices, image_size=image_size)
        if stack is not None:
            out[slot] = stack
            mask[slot] = True

    return study, out, mask


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--slices", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="0 = all studies")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    root = Path(args.data_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    series = pd.read_csv(root / f"{args.split}_series.csv")
    studies = sorted(series["StudyInstanceUID"].unique())

    # Keep only studies whose pixels are actually present — a local subset has the CSV for all
    # 4,407 but the DICOM for a handful, and silently emitting empty studies would poison
    # training with all-zero images that look like valid data.
    present = [s for s in studies if (root / f"{args.split}_series" / s).is_dir()]
    if args.limit:
        present = present[: args.limit]
    if not present:
        raise SystemExit(
            f"No study directories found under {root / (args.split + '_series')}. "
            "Download some DICOM first, or point --data-root at the Kaggle mount."
        )
    print(f"{len(present)} studies with pixels on disk (of {len(studies)} in the CSV)")

    grouped = {uid: rows.to_dict("records") for uid, rows in series.groupby("StudyInstanceUID")}
    jobs = [
        (uid, grouped[uid], root, args.split, args.slices, args.image_size) for uid in present
    ]

    pixels = np.lib.format.open_memmap(
        out / "pixels.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(len(present), len(SLOTS), args.slices, args.image_size, args.image_size),
    )
    masks = np.zeros((len(present), len(SLOTS)), dtype=bool)
    order = {uid: i for i, uid in enumerate(present)}

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for uid, stack, mask in tqdm(
            pool.map(build_one, jobs), total=len(jobs), desc="decoding"
        ):
            i = order[uid]
            pixels[i] = stack
            masks[i] = mask
    pixels.flush()

    np.save(out / "mask.npy", masks)
    pd.Series(present, name="StudyInstanceUID").to_frame().to_parquet(out / "index.parquet")
    write_meta(
        out,
        CacheMeta(
            image_size=args.image_size,
            slices=args.slices,
            n_slots=len(SLOTS),
            n_studies=len(present),
            laterality_normalised=True,
            slice_rule=f"centred-{args.slices}",
        ),
    )

    empty = int((~masks.any(axis=1)).sum())
    print(json.dumps({
        "studies": len(present),
        "gb": round(pixels.nbytes / 1024**3, 2),
        "mean_slots_present": round(float(masks.sum(axis=1).mean()), 2),
        "studies_with_no_slots": empty,
    }, indent=2))
    if empty:
        print(f"WARNING: {empty} studies decoded to nothing and will train on all-zero images.")


if __name__ == "__main__":
    main()
