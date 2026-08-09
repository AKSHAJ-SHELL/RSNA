"""Multilingual report -> twelve graded findings.

This is the primary supervision. 4,407 studies carry a report; 58 carry an image-read label.
Whatever this module fails to read is simply absent from training, so its blind spots become
the model's blind spots — which is why `reports.coverage_rate` measures silence rather than
accuracy, and why that measurement runs on every study instead of the 58.

Three decisions shape the design.

**No language routing.** Every cue lexicon carries all languages at once and each clause is
tested against the union. Routing means committing to a guess before reading any evidence,
and a misrouted report loses all twelve findings rather than degrading gracefully. Greek and
Cyrillic cues cannot collide with Latin-script ones, and the Latin-script anatomical
vocabularies here are close enough ("meniscus"/"menisco"/"menisk"/"menisküs") that a shared
stem is usually right.

**Grade, never binarise.** The reporting radiologist and the competition annotator do not
share a threshold — a report saying *small joint effusion* may sit against a negative
annotation. A rule of the form `term present => positive` is wrong by construction. Since the
metric reads only order, grading a mention costs nothing and strictly adds information.

**Silence is not negation.** A report that never mentions synovitis gets confidence 0 on that
finding, not a zero label. The confidence becomes the sample weight in `weighted_bce`, so an
unmentioned finding pulls weakly instead of asserting a negative — which matters enormously
for the rare findings, where spurious negatives would drown the few real positives.

Scope is honest about its limits: this is a lexicon, so its failure mode is silence, and
coverage by language is the number that says where it needs help. See ATTRIBUTION.md — the
approach follows `pilkwang/rsna-knee-baseline-v1` §2.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd
import regex as re

from rsnaknee.reports import TARGETS

# --------------------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------------------

#: Folded before casefolding. Turkish dotted/dotless i is the important one: without this,
#: "İZLENMEZ" (is not observed) and "izlenmez" diverge, and Turkish is 12% of the corpus.
_PRE = str.maketrans(
    {
        "ı": "i", "İ": "i", "I": "i",
        "ß": "ss", "đ": "d", "Đ": "d",
        "ø": "o", "Ø": "o", "æ": "ae", "Æ": "ae",
        "ć": "c", "č": "c", "ž": "z", "š": "s",
    }
)


def normalize(text: str) -> str:
    """Fold case, accents and separators while keeping Greek and Cyrillic letters.

    NFKD splits Latin accents into base + combining mark, which `Mn` filtering then drops.
    Greek and Cyrillic survive because their letters are not decomposable this way — stripping
    them would erase 12% of the corpus.
    """
    text = str(text).translate(_PRE).casefold()
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", stripped)


def clauses(text: str) -> list[str]:
    """Split into scope units for negation.

    Negation scope in radiology reports is the clause, not the sentence: "no meniscal tear,
    moderate effusion" contains one negated and one asserted finding, and treating the whole
    sentence as negated would lose the effusion.
    """
    return [c.strip() for c in re.split(r"[.;:\n•·]|(?<=\w),(?=\s)", text) if c.strip()]


def _rx(*patterns: str) -> re.Pattern:
    return re.compile("|".join(patterns), re.IGNORECASE)


def _near(side: str, part: str, gap: int = 4) -> str:
    """Match a side word and an anatomy word within `gap` words, in either order.

    Strict adjacency loses a whole class of real mentions. Croatian writes "Medijalni i
    lateralni menisk" — one noun serving two adjectives — and Spanish writes "menisco medial"
    while German writes "mediale Meniskus". Requiring the two tokens to touch drops the first
    entirely and forces a separate pattern per language for the rest.

    The gap is deliberately short. Widen it and "medial" from one clause starts binding to
    "meniscus" in the next, which turns a coverage gain into a precision loss on exactly the
    four sided targets that motivated laterality handling in the first place.
    """
    return rf"{side}(?:\W+\w+){{0,{gap}}}\W+{part}|{part}(?:\W+\w+){{0,{gap}}}\W+{side}"


# --------------------------------------------------------------------------------------
# Modifiers — negation, hedging, severity
# --------------------------------------------------------------------------------------

NEGATION = _rx(
    # en
    r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\bnegative for\b", r"\babsen\w*",
    r"\bunremarkable\b", r"\bintact\b", r"\bnormal\b", r"\bfree of\b", r"\bpreserved\b",
    # es
    r"\bsin\b", r"\bno hay\b", r"\bausen\w*", r"\bnormale?s?\b", r"\bconservad\w*", r"\bintegr\w*",
    # fr
    r"\bpas de\b", r"\bsans\b", r"\baucune?\b", r"\bnormale?\b",
    # nl / de
    r"\bgeen\b", r"\bzonder\b", r"\bniet\b", r"\bkein\w*", r"\bohne\b", r"\bunauffallig\w*",
    r"\bregelrecht\b", r"\bnormale?r?s?\b",
    # tr
    r"\bizlenme\w*", r"\bsaptanma\w*", r"\bgozlenme\w*", r"\byoktur\b", r"\byok\b",
    r"\bnormaldir\b", r"\bdogald\w*", r"\bmevcut degil\b",
    # hr / bs / sr
    r"\bbez\b", r"\bnema\b", r"\bnije\b", r"\buredan\b", r"\burednog\b", r"\bocuvan\w*",
    # el
    r"\bδεν\b", r"\bχωρις\b", r"\bαπουσια\b", r"\bφυσιολογικ\w*", r"\bακεραι\w*",
    # bg / ru
    r"\bбез\b", r"\bне\b", r"\bняма\b", r"\bнорм\w*", r"\bзапазен\w*",
)

HEDGE = _rx(
    r"\bpossibl\w*", r"\bprobabl\w*", r"\bsuspicio\w*", r"\bsuspect\w*", r"\bcannot exclude\b",
    r"\bquestionable\b", r"\bmay represent\b", r"\bequivocal\b", r"\blikely\b",
    r"\bposible\b", r"\bprobable\b", r"\bsospech\w*", r"\bdudos\w*", r"\bno se puede excluir\b",
    r"\bmogelijk\b", r"\bverdacht\w*", r"\bmoglich\w*", r"\bfraglich\w*",
    r"\bsuphel\w*", r"\bolasi\b", r"\bkuskul\w*", r"\bdusundur\w*",
    r"\bmoguc\w*", r"\bsumnj\w*", r"\bvjerojatn\w*",
    r"\bπιθαν\w*", r"\bυποψια\b", r"\bενδεχομεν\w*",
    r"\bвъзможн\w*", r"\bвероятн\w*", r"\bсъмнени\w*",
)

SEVERE = _rx(
    r"\bsevere\w*", r"\bmarked\w*", r"\blarge\b", r"\bgross\w*", r"\badvanced\b", r"\bcomplete\b",
    r"\bfull[- ]thickness\b", r"\bgrade i{3}\b", r"\bgrade 3\b", r"\bextensive\b", r"\bmassive\b",
    r"\bsevera\w*", r"\bgrave\b", r"\bavanzad\w*", r"\bmarcad\w*", r"\bimportante\b", r"\bgran\b",
    r"\bausgepragt\w*", r"\bschwer\w*", r"\bhochgradig\w*", r"\bgroße?r?\b", r"\buitgebreid\w*",
    r"\bileri\b", r"\bbelirgin\b", r"\byaygin\b", r"\bbuyuk\b", r"\btam kat\b",
    r"\bizrazit\w*", r"\bteska\b", r"\bopsezn\w*", r"\bvelik\w*",
    r"\bσοβαρ\w*", r"\bεκτεταμεν\w*", r"\bμεγαλ\w*", r"\bπληρη\w*",
    r"\bизразен\w*", r"\bтежк\w*", r"\bголям\w*", r"\bобширн\w*",
)

MILD = _rx(
    r"\btrace\b", r"\bminimal\w*", r"\bmild\w*", r"\bsmall\b", r"\bslight\w*", r"\btiny\b",
    r"\bgrade i\b", r"\bgrade 1\b", r"\blow[- ]grade\b", r"\bearly\b", r"\bsubtle\b",
    r"\bleve\b", r"\bminim\w*", r"\bpequen\w*", r"\bligera\w*", r"\bdiscret\w*", r"\binicial\b",
    r"\bgering\w*", r"\bleicht\w*", r"\bmaßig\w*", r"\bklein\w*", r"\bdiscrete\b", r"\blicht\w*",
    r"\bhafif\b", r"\bminimal\b", r"\bsilik\b", r"\bkucuk\b", r"\baz\b",
    r"\bblag\w*", r"\bdiskretn\w*", r"\bmanj\w*",
    r"\bηπι\w*", r"\bελαφρ\w*", r"\bμικρ\w*", r"\bελαχιστ\w*",
    r"\bлек\w*", r"\bминимал\w*", r"\bмалк\w*", r"\bнеznачител\w*",
)

# --------------------------------------------------------------------------------------
# Finding cues
# --------------------------------------------------------------------------------------

#: Anatomical stems are deliberately loose — medical vocabulary is highly cognate across
#: these languages, so a shared stem usually matches everywhere and costs a rare false hit.
CUES: dict[str, re.Pattern] = {
    "ACL": _rx(
        r"\bacl\b", r"\blca\b", r"\bvkb\b", r"\bpkl\b",
        r"anterior cruciate", r"cruciate ligament",
        # Spanish/Portuguese routinely write the pair — "ligamentos cruzados y colaterales" —
        # with no "anterior" anywhere, so requiring the qualifier loses most of the corpus.
        r"cruzad\w* anterior", r"ligament\w* cruzad\w*",
        r"croise anterieur", r"ligament\w* croise\w*",
        r"vordere[sn]? kreuzband", r"\bkreuzband\w*", r"voorste kruisband", r"\bkruisband\w*",
        r"on capraz", r"capraz bag",
        # Croatian: "križni ligament", not "ukriženi" — normalisation folds ž to z.
        r"krizn\w*\W+(?:\w+\W+){0,2}?ligament\w*", r"prednj\w*\W+(?:\w+\W+){0,2}?krizn\w*",
        r"προσθι\w* χιαστ\w*", r"χιαστ\w* συνδεσμ\w*",
        r"предн\w* кръстн\w*", r"кръстн\w* връзк\w*",
    ),
    "MCL": _rx(
        r"\bmcl\b", r"medial collateral", r"colateral (?:medial|interno)", r"collateral (?:medial|tibial)",
        r"innenband", r"mediale[sn]? kollateralband", r"mediale collaterale",
        r"medial kollateral", r"ic collateral", r"medijaln\w* kolateraln\w*",
        r"εσω πλαγι\w*", r"вътрешн\w* колатерал\w*", r"lcm\b",
    ),
    # `menisk\w*` covers meniscus / menisco / menisküs / menisk / meniskus in one stem; the
    # side word is matched by proximity so coordinated phrases survive.
    "Medial Meniscus": _rx(
        _near(r"(?:medial\w*|medijaln\w*|intern\w*|inner|binnen|unutarnj\w*|medyal\w*)", r"menisk?\w*"),
        r"innenmeniskus", r"binnenmeniscus", r"\bic menisc\w*",
        _near(r"εσω", r"μηνισκ\w*"), _near(r"(?:вътреш\w*|медиал\w*)", r"менискус\w*"),
    ),
    "Lateral Meniscus": _rx(
        _near(r"(?:lateral\w*|lateraln\w*|extern\w*|aussen|buiten|vanjsk\w*)", r"menisk?\w*"),
        r"aussenmeniskus", r"buitenmeniscus",
        _near(r"εξω", r"μηνισκ\w*"), _near(r"(?:външ\w*|латерал\w*)", r"менискус\w*"),
    ),
    # Degenerative vocabulary shares stems across these languages (arthros/artros/artroz/
    # gonartroz/αρθρωσ/артроз), so one stem group plus a proximate side word replaces a
    # per-language pattern list. The gap is wider here: reports say "gonartroza, izraženija u
    # medijalnom kompartmentu" with the side several words from the diagnosis.
    "Medial OA": _rx(
        _near(
            r"(?:medial\w*|medijaln\w*|intern\w*|inner|εσω|медиал\w*|вътреш\w*)",
            r"(?:osteoarthr\w*|arthros\w*|artros\w*|artroz\w*|gonarthros\w*|gonartroz\w*"
            r"|chondromalac\w*|kondromalaz\w*|hondromalac\w*|degenerat\w*|femorotibial\w*"
            r"|femorotibijaln\w*|αρθρωσ\w*|αρθριτ\w*|артроз\w*)",
            gap=6,
        ),
        r"innere? gonarthrose", r"\bic femorotibial",
    ),
    "Lateral OA": _rx(
        _near(
            r"(?:lateral\w*|lateraln\w*|extern\w*|aussen|εξω|латерал\w*|външ\w*)",
            r"(?:osteoarthr\w*|arthros\w*|artros\w*|artroz\w*|gonarthros\w*|gonartroz\w*"
            r"|chondromalac\w*|kondromalaz\w*|hondromalac\w*|degenerat\w*|femorotibial\w*"
            r"|femorotibijaln\w*|αρθρωσ\w*|αρθριτ\w*|артроз\w*)",
            gap=6,
        ),
        r"außere? gonarthrose",
    ),
    "PF OA": _rx(
        r"patellofemoral\w*", r"patelo?femoral\w*", r"retropatellar\w*", r"femoropatelar\w*",
        r"femoropatellar\w*", r"chondromalac\w* patell\w*", r"kondromalazi\w* patella",
        r"patella (?:arthrose|artroz)\w*", r"επιγονατιδομηριαι\w*", r"пателофеморал\w*",
        r"\bpfj\b", r"patellofemoraln\w*",
    ),
    "Effusion": _rx(
        r"\beffusion\w*", r"joint fluid", r"\bderrame\w*", r"epanchement", r"\bhydrops\b",
        r"gelenkerguss", r"\berguss\w*", r"gewrichtsvocht", r"\bvocht\w*",
        # Turkish writes "mayii artışı" or "sıvı miktarı arttı" — the fluid noun stands alone
        # and "eklem sıvısı" as a fixed phrase misses nearly all of it.
        r"\befuzyon\w*", r"\bmayi\w*", r"\bsivi\w*",
        r"\bizljev\w*", r"\bizliv\w*", r"\btekucin\w*", r"zglobn\w* tekucin\w*",
        r"αρθρικ\w* υγρ\w*", r"\bσυλλογη\w*", r"\bυδραρθρ\w*",
        r"\bизлив\w*", r"ставн\w* течност", r"\bтечност\b",
    ),
    "Synovitis": _rx(
        r"synovit\w*", r"sinovit\w*", r"synovial (?:thicken|proliferat|hypertroph)\w*",
        r"synovialit\w*", r"synovialis\w* (?:verdickung|proliferation)", r"sinovijalit\w*",
        r"υμενιτ\w*", r"синовит\w*", r"pannus",
    ),
    "Baker's": _rx(
        r"baker\w*", r"popliteal cyst", r"quiste de baker", r"kyste de baker", r"bakerzyste",
        r"bakerse cyste", r"poplite\w* (?:cyst|kist|zyste)\w*", r"baker kist\w*",
        r"bakerova cista", r"κυστη baker", r"киста на бейкър", r"поплитеал\w* киста",
    ),
    "Contusion": _rx(
        r"contusi\w*", r"kontuzi\w*", r"kontuzyon\w*", r"\bbruise\w*",
        r"knochenmark[- ]?odem", r"beenmergoedeem", r"botcontusie",
        # Bone + oedema in either order covers "bone marrow oedema", "edema óseo",
        # "koštani edem", "kemik iliği ödemi" without a pattern per language.
        _near(
            r"(?:bone|marrow|osse\w*|oseo|osea|medula|knochen|been?merg|kemik|ilig\w*"
            r"|kostan\w*|kosti|οστ\w*|μυελ\w*|кост\w*|мозъчен)",
            r"(?:oedema|edema|odem\w*|oedeem|odema|edem\w*|οιδημα\w*|оток\w*)",
            gap=3,
        ),
    ),
    "Fracture": _rx(
        r"fractur\w*", r"fractura", r"fracture", r"\bfraktur\w*", r"\bbreuk\b", r"kirik",
        r"\bfraktura\b", r"\bprijelom\w*", r"καταγμα\w*", r"фрактура", r"счупван\w*",
        r"avulsion", r"avulsi\w*",
    ),
}

#: Grades. Values are ordinal, not calibrated probabilities — the metric reads order only, so
#: what matters is that trace < unqualified < marked and that a negation lands below all three.
SCORE_NEGATED = 0.0
SCORE_MILD = 0.35
SCORE_PLAIN = 0.65
SCORE_SEVERE = 0.90

#: Confidence, which becomes the sample weight. A hedged mention is real evidence but should
#: not pull as hard as an unhedged one.
CONF_NONE = 0.0
CONF_HEDGED = 0.4
CONF_STATED = 1.0


@dataclass(frozen=True)
class Reading:
    score: float
    confidence: float


def read_finding(text_clauses: list[str], cue: re.Pattern) -> Reading:
    """Decide one finding from the clauses that mention it.

    When clauses disagree — a negation in the technique section, an assertion in the
    impression — the strongest *asserted* reading wins. Radiology reports restate their
    positives in the impression, so a later assertion is a confirmation rather than a
    contradiction, and taking the maximum matches how the report is meant to be read.
    """
    best = Reading(SCORE_NEGATED, CONF_NONE)
    for clause in text_clauses:
        if not cue.search(clause):
            continue

        negated = bool(NEGATION.search(clause))
        hedged = bool(HEDGE.search(clause))

        if negated and not hedged:
            score, confidence = SCORE_NEGATED, CONF_STATED
        else:
            if SEVERE.search(clause):
                score = SCORE_SEVERE
            elif MILD.search(clause):
                score = SCORE_MILD
            else:
                score = SCORE_PLAIN
            confidence = CONF_HEDGED if hedged else CONF_STATED

        # Prefer the reading that asserts most strongly; among equals prefer the confident one.
        if (score, confidence) > (best.score, best.confidence):
            best = Reading(score, confidence)
        elif best.confidence == CONF_NONE:
            best = Reading(score, confidence)
    return best


def extract_report(text: str) -> dict[str, Reading]:
    parts = clauses(normalize(text))
    return {target: read_finding(parts, CUES[target]) for target in TARGETS}


def extract_frame(reports: pd.Series, silence_weight: float = 0.25) -> pd.DataFrame:
    """Extract every report into a frame of scores plus `__confidence` columns.

    The confidence columns are what `train.py` reads as sample weights; emitting them in the
    same file keeps a target set and its weighting from ever drifting apart.

    **Silence as weak negative evidence.** Leaving every unmentioned cell at weight 0 discards
    54% of the label matrix. But silence is not uniformly uninformative: reports vary enormously
    in how much ground they cover — English reports here mention a median of 7 findings in 1,317
    characters, Spanish 3 in 753 — and a report that discusses seven findings without mentioning
    a Baker's cyst is real, if weak, evidence that there isn't one. A terse positives-only report
    says nothing by the same omission.

    So an unmentioned finding is scored 0 and weighted `silence_weight x thoroughness`, where
    thoroughness is the fraction of the twelve that report *did* mention. Thorough reports lend
    their silence weight; terse ones stay near zero and are effectively still ignored.

    This is a genuine bet, not a free lunch: if the reasoning is wrong it injects false negatives
    on exactly the rare findings that can least afford them. `silence_weight` is the knob, it is
    deliberately well below the 1.0 of a stated finding, and 0.0 restores the old behaviour for
    a clean A/B.
    """
    rows = [extract_report(text) for text in reports.fillna("")]
    scores = pd.DataFrame(
        [{t: r[t].score for t in TARGETS} for r in rows], index=reports.index, columns=TARGETS
    )
    confidence = pd.DataFrame(
        [{t: r[t].confidence for t in TARGETS} for r in rows], index=reports.index
    )

    if silence_weight > 0:
        mentioned = confidence > 0
        thoroughness = mentioned.sum(axis=1) / len(TARGETS)
        implied = pd.DataFrame(
            np.outer(silence_weight * thoroughness, np.ones(len(TARGETS))),
            index=reports.index,
            columns=TARGETS,
        )
        confidence = confidence.where(mentioned, implied)

    confidence.columns = [f"{t}__confidence" for t in TARGETS]
    return pd.concat([scores, confidence], axis=1)
