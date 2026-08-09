"""Training entry point.

    .venv/bin/python -m rsnaknee.train --smoke
    .venv/bin/python -m rsnaknee.train --cache data/cache/r224s8 \
        --targets data/targets/v1.parquet --folds data/folds.csv --fold 0

Every run appends one row to `experiments/runs.csv`, including the inference time. That column
is populated from the first run rather than retrofitted, because the efficiency track scores
it and reconstructing timings for old runs in October is exactly the avoidable pain we are
trying to design out.

Model selection uses the mean of twelve per-label AUCs, matching the competition metric, and
the per-label spread is logged alongside it: a label sitting at chance costs (M-0.5)/12 of the
final score no matter how good the other eleven are, and a single mean hides that completely.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from rsnaknee.cache import StudyDataset, open_cache, synthetic_cache
from rsnaknee.model import ModelConfig, StudyModel, weighted_bce
from rsnaknee.reports import TARGETS

RUN_LOG = Path("experiments/runs.csv")


#: Graded targets are binarised at their midpoint purely so a held-out AUC can be computed.
#: The grades are 0.0 / 0.35 / 0.65 / 0.90, so this puts "trace" and "mild" on the negative
#: side — which matches how the annotation was made: the annotator marked only findings they
#: judged significant, while the reporting radiologist mentions everything they see.
#:
#: This threshold decides which epoch and configuration are kept. It never touches a submitted
#: score, and it never touches training: the model is fitted against the *graded* values,
#: because under a rank metric the ordering between trace and marked is free information.
EVAL_THRESHOLD = 0.5


def per_label_auc(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """AUC per target, NaN where a label has only one class present.

    A single-class column is not a score of 0.5 — it is an absence of information, and
    averaging it in as 0.5 would quietly drag the mean toward chance on exactly the rare
    labels we most need to watch.
    """
    binary = (y_true >= EVAL_THRESHOLD).astype(int)
    out = {}
    for i, name in enumerate(TARGETS):
        col = binary[:, i]
        out[name] = float(roc_auc_score(col, y_score[:, i])) if len(np.unique(col)) > 1 else np.nan
    return out


def synchronize(device: torch.device) -> None:
    """Block until queued GPU work finishes.

    Both CUDA and MPS dispatch asynchronously, so a timer that does not synchronise measures how
    fast Python queued the work, not how long it took. That would make the efficiency-track
    column optimistic by a wide and silently varying margin.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def evaluate(model, loader, device, amp: bool = False) -> tuple[dict[str, float], float]:
    model.eval()
    scores, truths = [], []
    n = 0
    synchronize(device)  # do not bill the previous epoch's tail to this measurement
    start = time.perf_counter()
    for slots, mask, targets, _ in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(slots.to(device), mask.to(device))
        scores.append(torch.sigmoid(logits.float()).cpu().numpy())
        truths.append(targets.numpy())
        n += len(targets)
    synchronize(device)
    ms_per_study = (time.perf_counter() - start) / max(n, 1) * 1000
    return per_label_auc(np.concatenate(truths), np.concatenate(scores)), ms_per_study


def train_one_fold(args) -> dict:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    if args.smoke:
        cache_root = synthetic_cache(
            Path(args.workdir) / "synthetic", n_studies=64, image_size=args.image_size, slices=8
        )
    else:
        cache_root = Path(args.cache)

    pixels, mask, index, meta = open_cache(cache_root)
    meta.assert_compatible(args.image_size)

    if args.smoke:
        rng = np.random.default_rng(0)
        targets = pd.DataFrame(
            rng.integers(0, 2, (len(index), len(TARGETS))).astype(float),
            index=index,
            columns=TARGETS,
        )
        weights = pd.DataFrame(1.0, index=index, columns=TARGETS)
        fold = pd.Series(rng.integers(0, 5, len(index)), index=index)
    else:
        extracted = pd.read_parquet(args.targets)
        targets = extracted[TARGETS]
        weight_cols = [f"{t}__confidence" for t in TARGETS]
        if all(c in extracted.columns for c in weight_cols):
            weights = extracted[weight_cols].set_axis(TARGETS, axis=1)
        else:
            # No confidence column: fall back to uniform, but say so. Silently treating an
            # unmentioned finding as a confident negative is the failure this pipeline is
            # built to avoid, so it should never happen by accident.
            print("WARNING: no __confidence columns found; weighting all cells equally.")
            weights = pd.DataFrame(1.0, index=extracted.index, columns=TARGETS)
        fold = pd.read_csv(args.folds, index_col="StudyInstanceUID")["fold"].reindex(index)

    is_val = (fold.to_numpy() == args.fold)
    train_rows = np.flatnonzero(~is_val)
    val_rows = np.flatnonzero(is_val)
    print(f"cache={cache_root} studies={len(index)} train={len(train_rows)} val={len(val_rows)}")

    common = dict(pixels=pixels, mask=mask, index=index, targets=targets, weights=weights)
    train_ds = StudyDataset(**common, rows=train_rows, train=True, seed=args.seed)
    val_ds = StudyDataset(**common, rows=val_rows, train=False)

    loader_kwargs = dict(batch_size=args.batch, num_workers=args.num_workers)
    if device.type == "cuda":
        loader_kwargs["pin_memory"] = True  # enables the async host->device copy
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    cfg = ModelConfig(image_size=args.image_size, trainable_blocks=args.trainable_blocks)
    model = StudyModel(cfg).to(device)
    opt = torch.optim.AdamW(model.param_groups(args.lr), weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=[g["lr"] for g in opt.param_groups],
        total_steps=args.epochs * max(1, len(train_loader)),
    )

    best = {"macro_auc": -1.0}
    out_dir = Path(args.workdir) / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_start = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        running, seen = 0.0, 0
        for slots, msk, tgt, wt in train_loader:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
                logits = model(slots.to(device), msk.to(device))
            # Loss in fp32: bf16 has ~3 decimal digits of mantissa, and the weighted sum here is
            # divided by a total weight that can be large, which is exactly where it loses them.
            loss = weighted_bce(logits.float(), tgt.to(device), wt.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()
            running += loss.item() * len(tgt)
            seen += len(tgt)

        aucs, ms = evaluate(model, val_loader, device, amp=args.amp)
        macro = float(np.nanmean(list(aucs.values())))
        chance = [k for k, v in aucs.items() if not np.isnan(v) and v < 0.55]
        print(
            f"epoch {epoch + 1:>3}/{args.epochs}  loss {running / max(seen, 1):.4f}  "
            f"macroAUC {macro:.4f}  {ms:.1f} ms/study"
            + (f"  at-chance: {','.join(chance)}" if chance else "")
        )

        if macro > best["macro_auc"]:
            best = {"macro_auc": macro, "epoch": epoch + 1, "ms_per_study": ms, **aucs}
            torch.save(
                {"state_dict": model.state_dict(), "cfg": asdict(cfg), "meta": asdict(meta)},
                out_dir / "best.pt",
            )

    best["train_min"] = (time.perf_counter() - train_start) / 60
    (out_dir / "best.json").write_text(json.dumps(best, indent=2))
    return best


def log_run(args, best: dict) -> None:
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "smoke": args.smoke,
        "fold": args.fold,
        "image_size": args.image_size,
        "trainable_blocks": args.trainable_blocks,
        "batch": args.batch,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "macro_auc": best["macro_auc"],
        "best_epoch": best.get("epoch"),
        "train_min": round(best.get("train_min", float("nan")), 2),
        "infer_ms_per_study": round(best.get("ms_per_study", float("nan")), 2),
        "notes": args.notes,
        **{f"auc__{t}": best.get(t) for t in TARGETS},
    }
    frame = pd.DataFrame([row])
    frame.to_csv(RUN_LOG, mode="a", header=not RUN_LOG.exists(), index=False)
    print(f"logged -> {RUN_LOG}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true", help="run on a synthetic cache; proves the loop only")
    p.add_argument("--cache", type=str, help="path to a pixel cache directory")
    p.add_argument("--targets", type=str, help="parquet of extracted targets + __confidence columns")
    p.add_argument("--folds", type=str, help="CSV from rsnaknee.splits")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--trainable-blocks", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        help="defaults to the best available accelerator",
    )
    p.add_argument(
        "--amp",
        action="store_true",
        help="bfloat16 autocast. Roughly halves step time on CUDA; leave off on MPS, where "
             "bf16 support is uneven and the win is small.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="0 is right for a memmap cache: workers each fault in their own pages and "
             "multiply resident memory without hiding any decode work.",
    )
    p.add_argument("--workdir", type=str, default="runs")
    p.add_argument("--notes", type=str, default="")
    args = p.parse_args()

    if not args.smoke and not (args.cache and args.targets and args.folds):
        p.error("--cache, --targets and --folds are required unless --smoke is given.")

    log_run(args, train_one_fold(args))


if __name__ == "__main__":
    main()
