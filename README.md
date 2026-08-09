# RSNA Knee Abnormality Detection

Competition: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
Deadline **2026-10-22** · $77,000 · Research category · rules already accepted on this account.

---

## ⚠️ The single most important fact about this competition

**Only 58 of 4,407 training studies have ground-truth labels. All 4,407 have radiology reports.**

```
train.csv: 4,407 rows × 14 cols  (StudyInstanceUID, Report, + 12 targets)
  Report non-null ............ 4,407  (100%)
  All 12 targets non-null .....   58  (1.3%)
  Any target non-null .........   58  (1.3%)
```

This is a **weakly-supervised** competition. The reports *are* the training signal. Report→label
extraction is not an auxiliary trick — it is the primary supervision, and it must work across
**12 languages**.

Verified 2026-08-08 by direct inspection. No hidden label file exists: probing
`train_labels.csv`, `validation.csv`, `train_reports.csv`, `labels.csv`, `train_metadata.csv`
all return absent. The five CSVs below are the entire tabular dataset.

## Data files

| File | Size | Shape | Notes |
|---|---|---|---|
| `train.csv` | 5.4 MB | 4,407 × 14 | UID, Report (multilingual free text), 12 targets — 98.7% NaN |
| `train_series.csv` | 3.3 MB | 24,371 × 5 | UID, SeriesUID, `Fluid_Sensitive`, `Fat_Suppression`, `Anatomical_Plane` |
| `test.csv` | 212 B | 3 × 1 | Public placeholder; real test is hidden |
| `test_series.csv` | 2.2 KB | 15 × 5 | Same schema as train_series |
| `sample_submission.csv` | 470 B | 3 × 13 | UID + 12 probability columns, all 0.5 |

**No `Report` column in the test schema** — inference is image-only. Reports are a training-time
signal only; they cannot be read at test time.

## The 12 targets

`ACL` · `MCL` · `Medial Meniscus` · `Lateral Meniscus` · `Medial OA` · `Lateral OA` · `PF OA` ·
`Effusion` · `Synovitis` · `Baker's` · `Contusion` · `Fracture`

"Medial/Lateral" = knee **compartment**, not left/right body side. Which image side is medial
depends on whether the knee is left or right — so left/right canonicalization is still what
protects these four targets.

### Positives among the 58 labelled studies

| Class | Pos/58 | | Class | Pos/58 |
|---|---|---|---|---|
| Effusion | 35 | | ACL | 24 |
| Synovitis | 27 | | Lateral Meniscus | 23 |
| Medial Meniscus | 26 | | PF OA | 21 |
| Contusion | 19 | | Fracture | 18 |
| Medial OA | 15 | | Baker's | 12 |
| Lateral OA | 11 | | MCL | 9 |

These rates are **very high** (Effusion 60%, ACL 41%) — far above general-population prevalence.
If the hidden test set is annotated the same way, it is enriched for abnormality, and
report-derived label prevalence on the 4,407 will not match it. Watch for that shift.

**58 studies cannot select models.** AUC standard error at n=58 is roughly ±0.07. Use them as a
smoke test and a directional anchor, never as a leaderboard.

## Series structure

24,371 series over 4,407 studies — **mean 5.5 series/study** (min 3, median 5, max 14).
Planes: Sagittal 9,864 · Coronal 8,609 · Axial 5,898.
`Fluid_Sensitive` and `Fat_Suppression` are binary flags — free series-type embeddings for fusion.

## Reports

Mean 1,098 chars (min 52, max 4,743). Confirmed languages in samples: Spanish, Dutch, Greek,
Turkish. Reports frequently state laterality in-text (`SAĞ DİZ` = right knee,
`ΔΕΞΙΟΥ ΓΟΝΑΤΟΣ` = right knee) — **use this to validate DICOM left/right canonicalization for free.**

## Public leaderboard context (2026-08-08, day 4, 548 teams)

Top 0.936 · `pilkwang` (baseline author) 0.891 · the "0.809" figure from the original build plan is
stale. Reproduce whatever the current notebook version scores; the number is an oracle, not a target.

## Environment

Apple M5 Pro · 64 GB RAM · 18 cores · torch 2.13.0 with MPS · Python 3.12.12 (pyenv) · 303 GB free.

**Raw DICOM cannot be stored locally.** 24,371 series × ~25 slices × ~1.4 MB ≈ **700–850 GB**
against 303 GB free. Embedding extraction therefore runs on Kaggle (data already mounted); only the
sub-2 GB embedding cache comes down for local head training.

## Metric (confirmed)

Unweighted mean of twelve per-label ROC AUCs: `Score = (1/12) · Σ AUC_i`.

Three consequences, all load-bearing:
- **Only order matters.** AUC is invariant to any monotone transform, so calibration and
  thresholds are worthless — and ensembles must combine by **rank**, not by averaging
  probabilities (averaging probabilities lets the most confident member dominate).
- **Every label costs the same.** A label left at chance forfeits `(M − 0.5)/12` ≈ **0.029** at
  M = 0.85, no matter how good the other eleven are. **Rare findings deserve more attention than
  common ones**, which inverts the usual instinct.
- **Prevalence drift is survivable**; thresholds are not.

## Raw dataset size (measured)

All 15 test series enumerated from the file manifest: **557 files, mean 37.1 slices/series**
(median 30, one 160-slice outlier), **mean 1.03 MB/file**.

Extrapolating to 24,371 train series: **≈ 750–930 GB**. Against 303 GB free, local storage of raw
DICOM is impossible. Confirmed: decode happens on Kaggle.

The full manifest is exactly: `sample_submission.csv`, `test.csv`, `test_series.csv`, `train.csv`,
`train_series.csv`, `test_series/`, `train_series/`. Nothing else — the 58-label finding is
confirmed against the manifest, not merely inferred from probing.

## What the public baseline already does

`pilkwang/rsna-knee-baseline-v1` (169 votes, 143k chars) is unusually well-reasoned and **already
implements both of our intended differentiators**:

- **§6 per-class attention pooling** — each of the 12 diagnoses gets its own query `q_o` attending
  over slot embeddings. This is our "per-class attention" idea, already shipped.
- **§6 cross-series fusion** — learned slot-identity embeddings `e_s` plus a **masked softmax** that
  renormalises over present series. This is our "cross-series fusion with masking" idea, shipped.

It also contains several things our plan did **not** have:

| Technique | Why it matters |
|---|---|
| **Report-hash grouping in splits** | Some reports are byte-identical across studies (template normals). Grouping by *study* is insufficient — must group by report hash or the holdout is leaked. |
| **Laterality from patient coordinates** | The `Laterality` DICOM tag is Type 2C and **absent on ~half of studies, by whole vendor**. Sign of the image-centre x in the patient coordinate system recovers it. Studies near the midline are left unresolved rather than guessed. |
| **Sagittal laterality ≠ flip** | Coronal/axial: mirror the last axis. **Sagittal: reverse slice order** — mirroring a sagittal slice does nothing. |
| **Graded, not binary, targets** | "small effusion" vs "marked effusion" — grading the mention is free upside because only order is read. Radiologist and annotator do not share a threshold. |
| **Confidence as sample weight** | A report that never mentions synovitis pulls weakly on that head instead of asserting a negative. |
| **Annotated 58 kept in training at elevated weight** | They are the only labels read from images. The annotation *check* is therefore restricted to the holdout — scoring on upweighted training rows measures memorisation. |
| **Fine-tuned encoder, not frozen** | A frozen natural-image encoder is bounded by a vocabulary that never saw a torn meniscus on PD. Last blocks only, encoder LR ≪ head LR. |
| **Cache decoded uint8 pixels, not embeddings** | `bytes = N_study × N_slot × S × P²` — grows **quadratically in resolution**, linearly in slices. Coverage is the cheap axis. |

## Resource usage and train times (M5 Pro, 64 GB, MPS)

`scripts/bench_mps.py` — DINOv2 ViT-S/14, 6 slots/study, epoch = 3,525 studies (80% split).
**Compute only**: synthetic tensors, no decode, no augmentation, no cache reads. This is the
floor, not the forecast.

### Batch-size scaling @ 224px, 4 trainable blocks

| Batch | studies/s | Epoch (min) | Peak GB |
|---|---|---|---|
| 2 | 32.1 | 1.8 | 1.2 |
| **8** | **31.8** | **1.8** | **3.2** |
| 16 | 31.3 | 1.9 | 4.2 |
| 32 | 29.4 | 2.0 | 8.2 |

Throughput is flat from 2→16 and *degrades* at 32 — the encoder already sees `batch × 6`
images per step, so the GPU is saturated at batch 2. **Batch 8 is the pick**: same speed,
still only 3.2 GB.

### Resolution × adaptation depth @ batch 8

| res | blocks | Epoch (min) | Infer (ms/study) | Peak GB |
|---|---|---|---|---|
| 168 | 0 | 0.5 | 8.6 | 1.2 |
| 168 | 4 | 1.0 | 8.9 | 2.1 |
| 168 | 12 | 2.0 | 9.3 | 5.2 |
| 224 | 0 | 0.9 | 17.6 | 1.1 |
| **224** | **4** | **1.9** | **17.7** | **3.2** |
| 224 | 12 | 3.9 | 17.6 | 7.2 |
| 280 | 4 | 3.3 | 29.7 | 5.2 |
| 336 | 4 | 5.5 | 46.1 | 7.2 |
| 336 | 12 | 11.2 | 47.0 | 16.2 |

Epoch time is **linear in trainable blocks** (0→4→12 doubles, then doubles again) and
**quadratic in resolution**. Inference time is *independent* of trainable blocks — no backward
pass — so adaptation depth is free at submission time and only costs training wall-clock.

### Full-run wall-clock (batch 8, 4 blocks, 30 epochs)

| res | 1 fold | 5 folds | Cache @16 sl | Cache + model |
|---|---|---|---|---|
| 168 | 0.5 h | 2.4 h | 11.1 GB | 13 GB |
| **224** | **1.0 h** | **4.8 h** | **19.8 GB** | **23 GB** |
| 280 | 1.7 h | 8.4 h | 30.9 GB | 36 GB |
| 336 | 2.7 h | 13.6 h | 44.5 GB | 52 GB |

Pixel cache `= N_study × N_slot × S × P²` (uint8).

**Memory is not the binding constraint — wall-clock is.** Peak MPS is 3.2 GB at the working
config and peak process RSS was 0.6 GB; the 19.8 GB cache dominates, and even 336px fits in
64 GB. What actually limits us is that 5 folds at 336px is a 13.6-hour overnight run, against
4.8 h at 224.

**Efficiency-track read:** inference is 8.6 ms/study at 168 vs 46.1 at 336 — a **5.4× spread**
for 4× the pixels. Resolution is the dominant lever, confirming the plan's ranking.

## Running on Windows + CUDA

### Setup

```powershell
git clone https://github.com/AKSHAJ-SHELL/RSNA.git
cd RSNA

winget install --id=astral-sh.uv -e     # or: pip install uv
uv venv --python 3.12
```

**Install torch from the CUDA index, before anything else.** On Windows the default PyPI wheel
is **CPU-only**, so `uv pip install -e .` alone yields a torch that reports
`cuda.is_available() == False` and trains at CPU speed with no error message.

```powershell
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -e .
```

Match the index to your driver: `cu121`, `cu124`, or `cu128`. Check with `nvidia-smi`.

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.venv\Scripts\python.exe -m pytest -q
```

### Train

```powershell
.venv\Scripts\python.exe -m rsnaknee.train `
  --cache data\cache\r224s8 `
  --targets data\targets\v1.parquet `
  --folds data\folds.csv `
  --fold 0 --epochs 30 --batch 32 --image-size 224 `
  --device cuda --amp
```

`--device` already defaults to CUDA when it is available; it is spelled out above because a
silent fallback to CPU is the failure mode worth being loud about.

Notes specific to this machine and OS:

- **`--amp`** enables bfloat16 autocast, which roughly halves step time on Ampere or newer. The
  loss is computed in fp32 regardless — bf16 carries about three decimal digits, and the
  weighted sum here divides by a total weight large enough to lose them. Skip `--amp` on GTX
  10-series or older, which has no usable bf16.
- **`--batch 32`** rather than the Mac's 8. The encoder sees `batch x 6` images per step, so 32
  means 192 images in flight; drop to 16 if you hit OOM. On the M5 throughput was flat from 2 to
  16 because the GPU saturated almost immediately — a discrete CUDA card will not.
- **`--num-workers` stays 0.** The cache is a memmap; workers each fault in their own pages and
  multiply resident memory without hiding any decode work. Windows uses spawn rather than fork,
  which makes workers more expensive here, not less.
- **Paths use backslashes** and PowerShell continues lines with a backtick, not `\`.

### Generating targets and folds on Windows

`data/targets/v1.parquet` and `data/folds.csv` are gitignored, so regenerate them after cloning
— it takes under a minute and needs only `train.csv`:

```powershell
kaggle competitions download -c rsna-knee-abnormality-detection -f train.csv -p data\raw
kaggle competitions download -c rsna-knee-abnormality-detection -f train_series.csv -p data\raw
.venv\Scripts\python.exe scripts\build_targets.py
```

## Open items

- [x] Exact evaluation metric — mean of 12 per-label ROC AUCs.
- [x] Raw dataset size — ~750–930 GB.
- [ ] Exact efficiency-track scoring formula — lives at
      `/competitions/rsna-knee-abnormality-detection/overview/efficiency-prize-evaluation`.
      Kaggle overview pages are JS-rendered and not machine-fetchable; **needs a human paste.**
- [ ] Prize age eligibility — Rules tab (same constraint).
- [ ] Verify notebook licenses — `kernel-metadata.json` returns `licenseName: None` for both pulled
      notebooks, so the license is only visible on the web page. Not yet confirmed for any.
