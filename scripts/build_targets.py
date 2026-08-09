"""Extract targets from reports, measure the extractor, and write the fold split.

    .venv/bin/python scripts/build_targets.py

Writes:
    data/targets/v1.parquet   scores + __confidence per finding
    data/folds.csv            report-hash-grouped, multi-label stratified

Prints the two diagnostics that matter. Coverage by language needs no ground truth and runs
on all 4,407 studies, so it is the number that actually resolves differences between
extractors. Agreement against the 58 annotated studies is the only signal read from images
rather than text, and is reported with its confidence interval precisely because it is too
small to arbitrate anything.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rsnaknee.extract import extract_frame
from rsnaknee.reports import COMMONLY_REPORTED, TARGETS, annotation_agreement, coverage_rate, detect_languages
from rsnaknee.splits import check_no_group_leak, make_folds, save_folds

TRAIN_CSV = Path("data/raw/train.csv")
LANG_CACHE = Path("data/raw/_lang.csv")
TARGETS_OUT = Path("data/targets/v1.parquet")
FOLDS_OUT = Path("data/folds.csv")


def main() -> None:
    train = pd.read_csv(TRAIN_CSV).set_index("StudyInstanceUID")
    print(f"loaded {len(train)} studies")

    if LANG_CACHE.exists():
        langs = pd.read_csv(LANG_CACHE, index_col="StudyInstanceUID")["lang"].reindex(train.index)
    else:
        langs = detect_languages(train["Report"])
        langs.rename("lang").to_frame().to_csv(LANG_CACHE, index_label="StudyInstanceUID")

    extracted = extract_frame(train["Report"])
    TARGETS_OUT.parent.mkdir(parents=True, exist_ok=True)
    extracted.to_parquet(TARGETS_OUT)
    print(f"wrote {TARGETS_OUT}")

    conf = extracted[[f"{t}__confidence" for t in TARGETS]].set_axis(TARGETS, axis=1)

    print("\n=== positive rate (score > 0.5, among mentioned) ===")
    for t in TARGETS:
        mentioned = conf[t] > 0
        rate = (extracted.loc[mentioned, t] > 0.5).mean() if mentioned.any() else float("nan")
        print(f"  {t:<18} mentioned {mentioned.mean():>5.1%}   positive-if-mentioned {rate:>5.1%}")

    print("\n=== coverage by language (fraction of reports where the finding was mentioned) ===")
    cov = coverage_rate(conf, langs)
    print(cov[COMMONLY_REPORTED + ["n"]].round(3).to_string())

    print("\n  ^ these four are findings a knee report almost always comments on.")
    print("    Low coverage here is extractor blindness, not genuine silence.")

    annotated = train[train[TARGETS].notna().all(axis=1)]
    print(f"\n=== agreement vs the {len(annotated)} annotated studies ===")
    agree = annotation_agreement(extracted[TARGETS], annotated[TARGETS])
    print(agree.round(3).to_string())
    print(f"\n  mean agreement {agree.agreement.mean():.3f}")
    print(f"  mean 95% CI width {agree.ci_width.mean():.3f}  <- why this cannot select models")

    folds = make_folds(train.assign(**{t: extracted[t] for t in TARGETS}), n_folds=5, seed=0)
    check_no_group_leak(train, folds)
    save_folds(folds, FOLDS_OUT)
    print(f"\nwrote {FOLDS_OUT}: {folds.value_counts().sort_index().to_dict()}")
    print("no duplicate-report group spans folds")


if __name__ == "__main__":
    main()
