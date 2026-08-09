"""Measure time and memory on MPS, and project full training runs.

Three questions this answers. Whether the encoder can be fine-tuned locally — if a run takes
many hours the work has to move to Kaggle. How large a batch fits before unified memory runs
out, given the pixel cache is resident at the same time. And where on the resolution curve
the efficiency submission should sit, since inference cost grows with the square of
resolution while cache size does too.

Timings are compute-only on synthetic tensors: no decode, no augmentation, no cache reads.
Treat them as the floor, not the forecast.

Run: .venv/bin/python scripts/bench_mps.py
"""

from __future__ import annotations

import os
import resource
import time

import torch

from rsnaknee.model import SLOTS, ModelConfig, StudyModel, weighted_bce

N_STUDIES = 4407
N_TARGETS = 12
HOLDOUT = 0.2  # baseline holds out one fifth; the rest is what an epoch actually walks


def rss_gb() -> float:
    """Peak resident set size. macOS reports ru_maxrss in bytes, unlike Linux."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3


def bench(image_size: int, batch: int, trainable_blocks: int, steps: int = 6) -> dict | None:
    device = torch.device("mps")
    torch.mps.empty_cache()

    try:
        model = StudyModel(
            ModelConfig(pretrained=False, image_size=image_size, trainable_blocks=trainable_blocks)
        ).to(device)
        opt = torch.optim.AdamW(model.param_groups(head_lr=1e-3))

        slots = torch.randn(batch, len(SLOTS), 3, image_size, image_size, device=device)
        mask = torch.ones(batch, len(SLOTS), dtype=torch.bool, device=device)
        targets = torch.randint(0, 2, (batch, N_TARGETS), device=device).float()
        weights = torch.rand(batch, N_TARGETS, device=device)

        def step():
            opt.zero_grad(set_to_none=True)
            weighted_bce(model(slots, mask), targets, weights).backward()
            opt.step()

        step()
        torch.mps.synchronize()

        start = time.perf_counter()
        for _ in range(steps):
            step()
        torch.mps.synchronize()
        train_s = (time.perf_counter() - start) / steps

        peak_gb = torch.mps.driver_allocated_memory() / 1024**3

        model.eval()
        with torch.no_grad():
            torch.mps.synchronize()
            t0 = time.perf_counter()
            for _ in range(steps):
                model(slots, mask)
            torch.mps.synchronize()
            infer_s = (time.perf_counter() - t0) / steps

        train_epoch_min = train_s / batch * N_STUDIES * (1 - HOLDOUT) / 60
        return {
            "res": image_size,
            "batch": batch,
            "blocks": trainable_blocks,
            "studies_per_s": batch / train_s,
            "epoch_min": train_epoch_min,
            "infer_ms": infer_s / batch * 1000,
            "peak_gb": peak_gb,
        }
    except RuntimeError as exc:
        print(f"  res={image_size} batch={batch} blocks={trainable_blocks}: FAILED — {exc}")
        return None
    finally:
        torch.mps.empty_cache()


def cache_gb(image_size: int, slices: int) -> float:
    return N_STUDIES * len(SLOTS) * slices * image_size**2 / 1024**3


def main() -> None:
    total_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    print(f"MPS | {total_ram:.0f} GB unified memory | {N_STUDIES} studies | {len(SLOTS)} slots")
    print(f"epoch = {int(N_STUDIES * (1 - HOLDOUT))} studies (80% train split)\n")

    print("=== batch-size scaling @ 224px, 4 trainable blocks ===")
    print(f"{'batch':>6}{'studies/s':>11}{'epoch(min)':>12}{'peak GB':>10}")
    for b in (2, 4, 8, 16, 32):
        r = bench(224, b, 4)
        if r:
            print(
                f"{r['batch']:>6}{r['studies_per_s']:>11.1f}"
                f"{r['epoch_min']:>12.1f}{r['peak_gb']:>10.1f}"
            )

    print("\n=== resolution x adaptation @ batch 8 ===")
    print(f"{'res':>6}{'blocks':>8}{'epoch(min)':>12}{'infer(ms)':>11}{'peak GB':>10}")
    rows = []
    for res in (168, 224, 280, 336):
        for blocks in (0, 4, 12):
            r = bench(res, 8, blocks)
            if r:
                rows.append(r)
                print(
                    f"{r['res']:>6}{r['blocks']:>8}{r['epoch_min']:>12.1f}"
                    f"{r['infer_ms']:>11.1f}{r['peak_gb']:>10.1f}"
                )

    print("\n=== projected wall-clock for a full run (batch 8, 4 blocks) ===")
    print(f"{'res':>6}{'1 fold x30ep':>15}{'5 folds x30ep':>16}{'cache@16sl':>12}")
    for res in (168, 224, 280, 336):
        m = next((r["epoch_min"] for r in rows if r["res"] == res and r["blocks"] == 4), None)
        if m:
            print(
                f"{res:>6}{m * 30 / 60:>14.1f}h{m * 30 * 5 / 60:>15.1f}h"
                f"{cache_gb(res, 16):>11.1f}G"
            )

    print(f"\npeak process RSS during benchmark: {rss_gb():.1f} GB")


if __name__ == "__main__":
    main()
