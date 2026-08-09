"""Fold assignment.

One leak dominates here and it is not the usual one. Some reports are byte-identical across
studies — a template read for an unremarkable knee — and every study sharing a report derives
the same target vector. Splitting such a group across the divide scores the model on a target
whose source it trained on. 183 of 4,407 studies sit in a duplicate group, the largest
covering 37 studies, so the effect is small but entirely avoidable.

Grouping therefore happens on the *report text*, not the study id. Stratification then runs
over the group, not the row, because assigning a 37-study group is one decision rather than
thirty-seven.

Folds are written to disk and read back rather than recomputed. A split that silently changes
between experiments makes every comparison in the run log meaningless, and nothing about a
recomputed split announces that it moved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rsnaknee.reports import TARGETS, report_hash


def make_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 0,
    text_col: str = "Report",
    targets: list[str] | None = None,
) -> pd.Series:
    """Assign each study a fold, keeping duplicate-report groups whole.

    Stratification is multi-label and iterative: with twelve correlated targets, striping on
    any single one leaves the others free to drift, and the rare findings are exactly the ones
    that drift worst. Group label vectors are averaged over their members, so a group of
    template normals presents as the negative block it is.

    Args:
        df: one row per study, indexed by StudyInstanceUID.
        n_folds: number of folds.
        seed: shuffles group order before assignment.
        text_col: column holding the report text.
        targets: label columns to stratify on. Missing values count as negative *for
            stratification only* — this decides which fold a study lands in, never what it is
            trained against.

    Returns:
        Integer fold per study, aligned to `df.index`.
    """
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

    targets = targets or TARGETS
    groups = df[text_col].fillna("").map(report_hash)

    label_frame = df[targets].fillna(0.0)
    label_frame = (label_frame > 0.5).astype(int)
    per_group = label_frame.groupby(groups.values).mean()
    per_group = (per_group > 0.5).astype(int)

    # Drop label columns with no positives at group level; the stratifier cannot balance a
    # constant column and silently degrades if handed one.
    usable = per_group.loc[:, per_group.sum() > 0]
    if usable.empty:
        raise ValueError(
            "No target column has a positive at group level — cannot stratify. "
            "Check that targets were extracted before splitting."
        )

    order = np.random.default_rng(seed).permutation(len(per_group))
    shuffled = per_group.iloc[order]

    splitter = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    group_fold = pd.Series(-1, index=shuffled.index, dtype=int)
    for fold, (_, held) in enumerate(splitter.split(shuffled.values, usable.iloc[order].values)):
        group_fold.iloc[held] = fold

    if (group_fold < 0).any():
        raise RuntimeError("Some groups were never assigned a fold.")

    return groups.map(group_fold).rename("fold")


def save_folds(folds: pd.Series, path: str | Path) -> Path:
    """Persist folds so every later run reads the same split."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    folds.rename("fold").to_frame().to_csv(path, index_label="StudyInstanceUID")
    return path


def load_folds(path: str | Path) -> pd.Series:
    return pd.read_csv(path, index_col="StudyInstanceUID")["fold"]


def check_no_group_leak(df: pd.DataFrame, folds: pd.Series, text_col: str = "Report") -> None:
    """Raise if any duplicate-report group spans more than one fold.

    Cheap to run and the only thing standing between us and a validation number that looks
    better than the model is.
    """
    groups = df[text_col].fillna("").map(report_hash)
    spread = folds.groupby(groups.values).nunique()
    offenders = spread[spread > 1]
    if len(offenders):
        raise AssertionError(
            f"{len(offenders)} report group(s) span multiple folds; "
            f"worst spans {int(offenders.max())} folds."
        )
