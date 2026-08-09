# Attribution

Every public notebook, dataset, and model weight this project uses. Updated **at the moment of each
borrow**, never retroactively. Competition winners must open-source; reconstructing provenance in
the final week is miserable.

**Licenses below are UNVERIFIED placeholders** except where marked ✅. Verify each notebook's license
on its Kaggle page before using any of its code, and mark it here.

## Public Kaggle notebooks

| Notebook | Author | License | Used for | Verified |
|---|---|---|---|---|
| [`pilkwang/rsna-knee-baseline-v1`](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1) | Pilkwang Kim | Apache 2.0 (claimed) | Phase 0 correctness oracle; DICOM I/O, series ID, laterality normalization, mm resampling, slice ordering | ☐ |
| [`ryanholbrook/rsna-knee-abnormalities-efficiency-lb`](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb) | Ryan Holbrook (competition host) | TBD | Efficiency-track scoring formula — reference only | ☐ |
| [`wguesdon/rsna-knee-dinov2-at-meniscus-resolution`](https://www.kaggle.com/code/wguesdon/rsna-knee-dinov2-at-meniscus-resolution) | Will | TBD | Input-resolution tradeoff; informs slice/resolution budget | ☐ |
| [`prvsiyan/rsna-knee-read-the-report-then-the-knee`](https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee) | prvsiyan | TBD | Report-text → label extraction (now the primary training signal) | ☐ |
| [`avikdas567/multimodal-multi-plane-2-5d-cnn-knee-mri-detection`](https://www.kaggle.com/code/avikdas567/multimodal-multi-plane-2-5d-cnn-knee-mri-detection) | Avik Das | TBD | 2.5D multi-plane encoder for the ensemble | ☐ |
| [`romanrozen/rsna-knee-data-structure-eda-baseline`](https://www.kaggle.com/code/romanrozen/rsna-knee-data-structure-eda-baseline) | Roman Rozen | TBD | EDA reference | ☐ |

## Datasets

| Dataset | Source | License | Used for | Verified |
|---|---|---|---|---|
| RSNA Knee Abnormality Detection | [Kaggle competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) | Competition rules | Training/eval data | ✅ |

## Model weights

| Weights | Source | License | Used for | Verified |
|---|---|---|---|---|
| DINOv2 ViT-S/14 | Meta / `timm` | Apache 2.0 | Slice encoder | ☐ |
| DINOv2 ViT-B/14 | Meta / `timm` | Apache 2.0 | Ensemble encoder | ☐ |
| RadImageNet | RadImageNet | Non-commercial — **check competition compatibility** | Ensemble encoder (candidate) | ☐ |

## Our own contributions

Per-class gated-attention MIL head, cross-series fusion with series-type embeddings, multilingual
report→label extraction schema, slice-budget efficiency analysis.
