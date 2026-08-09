"""Report handling: identity, language, and the instruments that judge an extractor.

The competition gives 4,407 reports and 58 image-read annotations. Extraction quality is
therefore the ceiling on supervision quality, and we need to compare extractors *without*
leaning on 58 studies, where the AUC standard error (~0.07) swamps every difference we care
about.

Two instruments here, in increasing order of how much we should trust them:

`coverage_rate`   needs no ground truth at all. A lexicon's failure mode is silence, not
                  error, so "did anything match?" is measurable on all 4,407 and, broken out
                  by language, says *where* an extractor is blind. This is the instrument
                  that scales.

`annotation_agreement`  compares against the 58 real labels. This is the only signal read
                  from images rather than text, and it is also far too small to arbitrate.
                  It reports a Wilson interval so the noise is impossible to ignore.

The idea of measuring silence is from `pilkwang/rsna-knee-baseline-v1` §2 (see ATTRIBUTION.md).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

#: Findings a knee MRI report almost always comments on. Silence on these is far more
#: likely to be extractor blindness than genuine omission, so coverage here is the
#: sharpest diagnostic we have.
COMMONLY_REPORTED = ["ACL", "Medial Meniscus", "Lateral Meniscus", "Effusion"]


@dataclass(frozen=True)
class Extraction:
    """One (report, finding) reading.

    score
        Graded severity in [0, 1]. 0.0 means explicitly negated, not "unknown" — the
        distinction matters because the metric reads order, so a trace effusion must land
        between a negated one and a marked one rather than collapsing onto either.
    confidence
        0.0 when nothing in the report bore on this finding. Becomes the sample weight, so
        an unmentioned finding pulls weakly on its head instead of asserting a negative.
    """

    score: float
    confidence: float

    @property
    def mentioned(self) -> bool:
        return self.confidence > 0.0


def report_hash(text: str) -> str:
    """Stable identity for a report's *text*, used to keep duplicates in one fold.

    Some reports are byte-identical across studies — a template read for an unremarkable
    knee. 183 of 4,407 studies sit in such a group here, the largest covering 37 studies.
    Every study in a group derives the same target vector, so splitting a group across the
    holdout scores the model on a target whose source it trained on.

    Whitespace is collapsed before hashing: a report differing only in line wrapping is the
    same read, and treating it as distinct would reintroduce the leak this prevents.
    """
    return hashlib.sha256(" ".join(str(text).split()).encode("utf-8")).hexdigest()


def add_report_groups(df: pd.DataFrame, text_col: str = "Report") -> pd.DataFrame:
    """Attach a `report_group` column for group-aware splitting."""
    out = df.copy()
    out["report_group"] = out[text_col].fillna("").map(report_hash)
    return out


def detect_languages(texts: pd.Series) -> pd.Series:
    """Best-effort language label per report.

    Used only to *stratify diagnostics*, never to route extraction. Routing would mean
    committing to a guess before reading any evidence, and a misrouted report loses every
    finding at once rather than degrading gracefully.
    """
    from lingua import LanguageDetectorBuilder

    detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
    return texts.fillna("").map(lambda s: (lambda d: d.name if d else "UNKNOWN")(detector.detect_language_of(s)))


def coverage_rate(
    extractions: pd.DataFrame,
    languages: pd.Series,
    targets: list[str] | None = None,
) -> pd.DataFrame:
    """Fraction of reports where the extractor said *anything* about each finding.

    Needs no annotations, so it runs on all 4,407 studies. Low coverage on a
    `COMMONLY_REPORTED` finding in a given language is near-proof of extractor blindness,
    because those findings are exactly the ones a knee report does not skip.

    Args:
        extractions: confidence per finding, indexed like `languages`. Columns are targets.
        languages: language label per report.

    Returns:
        Rows = language (plus an `ALL` row), columns = targets, values = coverage in [0, 1].
        An `n` column carries the report count so thin languages are visibly thin.
    """
    targets = targets or TARGETS
    mentioned = extractions[targets] > 0.0
    by_lang = mentioned.groupby(languages.values).mean()
    by_lang["n"] = languages.value_counts()
    overall = mentioned.mean().to_frame().T
    overall.index = ["ALL"]
    overall["n"] = len(mentioned)
    return pd.concat([overall, by_lang.sort_values("n", ascending=False)])


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — honest at the small n and extreme rates we have here.

    A normal-approximation interval would produce bounds outside [0, 1] on a finding with
    9 positives out of 58, which is exactly the regime this competition puts us in.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    halfwidth = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, centre - halfwidth), min(1.0, centre + halfwidth))


def annotation_agreement(
    extracted_scores: pd.DataFrame,
    annotations: pd.DataFrame,
    targets: list[str] | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Agreement between extracted labels and the 58 image-read annotations.

    Reported, never allowed to arbitrate. With 58 studies and as few as 9 positives on MCL,
    the confidence interval on any single cell is wide enough to contain most of the
    differences between two serious extractors. The interval is returned alongside the point
    estimate specifically so it cannot be quietly ignored.

    Only rows present in both frames are compared, so callers should pass the *holdout*
    slice: the annotated studies stay in training at elevated weight, and scoring a model on
    upweighted training rows measures memorisation and reports it as skill.
    """
    targets = targets or TARGETS
    shared = extracted_scores.index.intersection(annotations.index)
    if len(shared) == 0:
        raise ValueError("No overlapping studies between extractions and annotations.")

    rows = []
    for target in targets:
        truth = annotations.loc[shared, target]
        pred = (extracted_scores.loc[shared, target] >= threshold).astype(float)
        valid = truth.notna()
        n = int(valid.sum())
        agree = int((pred[valid] == truth[valid]).sum())
        lo, hi = _wilson(agree, n)
        rows.append(
            {
                "target": target,
                "n": n,
                "n_pos": int(truth[valid].sum()),
                "agreement": agree / n if n else float("nan"),
                "ci_low": lo,
                "ci_high": hi,
                "ci_width": hi - lo,
            }
        )
    return pd.DataFrame(rows).set_index("target")


def extractor_disagreement(
    a: pd.DataFrame,
    b: pd.DataFrame,
    targets: list[str] | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Where two extractors disagree, on all 4,407 studies.

    Disagreement is itself usable supervision: a study both readers call positive is a
    firmer target than one they split on. Feeding the split cases at reduced weight is
    strictly better than picking a winner per study, and unlike `annotation_agreement` this
    is measured at a sample size that can actually resolve differences.
    """
    targets = targets or TARGETS
    shared = a.index.intersection(b.index)
    pa = (a.loc[shared, targets] >= threshold).astype(int)
    pb = (b.loc[shared, targets] >= threshold).astype(int)
    both = (pa & pb).sum()
    either = (pa | pb).sum()
    return pd.DataFrame(
        {
            "n": len(shared),
            "a_pos": pa.sum(),
            "b_pos": pb.sum(),
            "both_pos": both,
            "disagree": (pa != pb).sum(),
            "disagree_rate": (pa != pb).mean(),
            "jaccard": (both / either.replace(0, np.nan)),
        }
    )
