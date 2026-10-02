# AgricultureAi

Hierarchical crop classification on the BreizhCrops Sentinel-2 time-series dataset.
The production stack is a **4-tier DMOE pipeline** (Dynamic Mixture of Experts) with a
**late-fusion XGBoost meta-classifier** on cached branch outputs.

## Problem

Classify **9 BreizhCrops classes** on held-out region `frh04`:

| Raw id | Class |
| --- | --- |
| 0–1 | Barley, Wheat |
| 2–3 | Rapeseed, Corn |
| 4–6 | Sunflower, Orchards, Nuts |
| 7–8 | Permanent / Temporary meadows |

**Train regions:** `frh01`, `frh02`, `frh03`  
**Eval region:** `frh04` (122,708 parcels at L2A)  
**Spectral level:** `L2A` (10 channels, cloud-filtered; max sequence length 45)  
Previous L1C (13 channels) checkpoints are backed up in `artifacts/L1C_backup/`.

The main challenge is **orchards** (552 eval parcels): MSResNet baseline reaches ~80.8% global
accuracy but only ~16% orchard recall. The meta-classifier reaches 37–57% orchard recall at
80.7–81.9% global accuracy, depending on the operating point (see Results).

## Architecture

```
x (B,T,C) → L1 Transformer router (4 super-classes)
  ├─ Rapeseed / Corn → global {2, 3}
  ├─ Cereals → L2a LTAE → {barley, wheat}
  └─ Grasslands → L2b GL-TAE anomaly gate
        ├─ Grass → L2c MSResNet → {permanent, temporary meadows}
        └─ Anomaly → L3 ProtoNet (Mahalanobis) → {sunflower, orchards, nuts} + meadow rejection
```

The meta-classifier replaces the hand-crafted probability cascade for final predictions:
compact 39-d features (L3 distances, cereal features, MSResNet logits, Transformer block,
8 parcel-metadata features), an XGBoost + CatBoost blend, and decode-time orchard / meadow
log-biases that pick the operating point.

L2a, L3 and the pretrained MSResNet / Transformer were trained on the stock BreizhCrops
transform (every parcel resampled to 45 dates, never zero-padded); their cached outputs are
computed from 8 seeded resampled views (`patch_l2a_features.py`, `patch_l3_distances.py`,
`patch_global_logits.py`), not from zero-padded series.

Implementation: `src/agricultureai/pipeline.py` (`DMOEPipeline`).

## Production workflow

### 1. Train experts

```bash
python scripts/train_transformer_breizhcrops.py
python scripts/train_expert_cereals.py
python scripts/train_l2b_anomaly_gate.py
python scripts/train_l2c_meadow_expert.py
python scripts/train_l3_class_balanced_head.py \
  --epochs 15 --batch-size 256 --max-samples 10000 --orchard-multiplier 3.0
```

Checkpoints land in `artifacts/checkpoints/<run_name>/best.pt`.

### 2. Build meta-classifier caches

```bash
python scripts/sweep_dmoe_parameters.py --extract-only
python scripts/append_msresnet_logits.py
python scripts/append_transformer_logits.py
python scripts/patch_l3_distances.py \
  --mode linear --linear-weights artifacts/checkpoints/level3_linear_head/best.pt
python scripts/append_parcel_metadata.py \
  artifacts/cached_logits_train.pt artifacts/cached_logits_eval.pt \
  --with-area --with-doy
```

Caches: `artifacts/cached_logits_train.pt`, `artifacts/cached_logits_eval.pt`.

### 3. Train and evaluate meta-classifier

```bash
python scripts/train_meta_classifier.py --orchard-multiplier 25
python scripts/compare_meta_vs_msresnet.py --orchard-multiplier 25
python scripts/sweep_meta_m_area.py
```

### 4. Evaluate DMOE cascade (without meta)

```bash
python scripts/evaluate_dmoe_pipeline.py \
  --l1-weights  artifacts/checkpoints/transformer_l1_4class/best.pt \
  --l2a-weights artifacts/checkpoints/expert_cereals/best.pt \
  --l2b-weights artifacts/checkpoints/l2b_anomaly_gate/best.pt \
  --l2c-weights artifacts/checkpoints/l2c_meadow_expert/best.pt \
  --l3-weights  artifacts/checkpoints/level3_linear_head/best.pt
```

### Baselines

```bash
python scripts/evaluate_pretrained_models.py --full-region
python main.py   # quick Transformer baseline on frh04
```

## Scripts

| Script | Purpose |
| --- | --- |
| `train_transformer_breizhcrops.py` | L1 4-class router |
| `train_expert_cereals.py` | L2a barley/wheat (LTAE) |
| `train_l2b_anomaly_gate.py` | L2b grass vs anomaly (GL-TAE) |
| `train_l2c_meadow_expert.py` | L2c meadow split (MSResNet) |
| `train_l3_class_balanced_head.py` | L3 linear head (production) |
| `sweep_dmoe_parameters.py` | Extract DMOE logits to cache |
| `append_msresnet_logits.py` | Add MSResNet features to cache |
| `append_transformer_logits.py` | Add Transformer features to cache |
| `append_parcel_metadata.py` | Add 6-d parcel metadata to cache |
| `patch_l3_distances.py` | Refresh L3 distances in cache |
| `train_meta_classifier.py` | XGBoost late fusion |
| `sweep_meta_m_area.py` | Sweep orchard multiplier (production tuning) |
| `compare_meta_vs_msresnet.py` | Side-by-side per-class metrics |
| `evaluate_dmoe_pipeline.py` | Full 4-tier pipeline eval |
| `evaluate_pretrained_models.py` | MSResNet / Transformer baselines |
| `count_class_distribution.py` | Label counts per region |
| `generate_full_presentation.py` | Build PowerPoint summary |

## Results (frh04, held out)

Decode settings (blend, orchard multiplier `m`, orchard / meadow biases) are chosen out of
sample: frh04 is cut into 10 km spatial blocks grouped into 5 folds, and each fold is scored
with the settings a fixed rule picks on the other four (`scripts/evaluate_meta_protocol.py`).
Meta rows are means over 3 training seeds; brackets are 95% spatial block-bootstrap
intervals (seed 42).

| Model | Global acc | Orchard recall | Orchard precision | Wheat recall |
| --- | --- | --- | --- | --- |
| MSResNet baseline (L1C, stock transform) * | 80.77% | 15.73% | – | 97.58% |
| DMOE cascade only (no meta-classifier) | 79.65% | 21.74% | 5.73% | 97.37% |
| Previous best: XGB @ m=25, 64-d (L1C, chosen on frh04) * | 80.76% | 35.80% | – | 97.57% |
| **Meta, accuracy-first** | **81.87%** [81.10, 82.77] | **36.9%** [31.7, 42.8] | 14.5% | 98.01% |
| **Meta, orchard-first** | **80.72%** [79.83, 81.60] | **56.9%** [53.1, 62.8] | 7.6% | 97.99% |
| Meta, closest to stretch | 80.47% [79.63, 81.46] | 60.0% [55.5, 65.1] | 7.0% | 98.00% |

\* L1C evaluation set (122,614 parcels, 553 orchards); not paired with the L2A rows.
Rules: accuracy-first = max accuracy with orchard recall ≥ 35%; orchard-first = max orchard
recall with accuracy ≥ 80.69%; closest to stretch = smallest shortfall to (81%, 60%).
Production targets (accuracy > 80.69%, orchard recall > 35%) are met; the stretch target
(≥ 81% and ≥ 60%) is not. The DMOE cascade row uses `evaluate_dmoe_pipeline.py` defaults.

What moved the numbers (paired block bootstrap, out-of-sample selection, seeds 42 / 1 / 2):

- Re-extracting the MSResNet features from resampled views instead of zero-padded series:
  +0.50 / +0.42 / +0.36 accuracy (accuracy-first) and +5.3 / +5.1 / +3.6 orchard recall
  (orchard-first); every 95% interval excludes zero.
- The same fix for L2a / L3: +8.5 points of cereal accuracy in the cascade, but within seed
  noise once the meta-classifier is on top. For the Transformer features: no measurable effect.
- Choosing decode settings on frh04 itself inflated accuracy by at most 0.15 points.

### Feature-layout ablation

`scripts/ablate_meta_layouts.py`: XGBoost only, `m = 1`, no decode bias, 3 seeds (mean ± sd);
deltas are paired against the compact layout.

| Layout | Features | Accuracy | Orchard recall | Orchard precision | Δ acc | Δ orchard |
| --- | --- | --- | --- | --- | --- | --- |
| full | 66 | 81.89 ± 0.03 | 9.2 ± 0.2 | 18.1 | −0.09 | −21.4 |
| minus L2b + L2c | 60 | 82.02 ± 0.01 | 30.7 ± 0.7 | 18.9 | +0.04 | +0.1 |
| **compact** | **39** | **81.98 ± 0.05** | **30.6 ± 1.1** | **18.9** | – | – |
| compact − L3 distances | 34 | 81.96 ± 0.02 | 28.6 ± 1.8 | 17.2 | −0.01 | −2.1 |
| compact − cereal | 32 | 81.73 ± 0.03 | 29.4 ± 0.4 | 17.7 | −0.25 | −1.2 |
| compact − MSResNet | 30 | 81.35 ± 0.02 | 29.1 ± 0.9 | 13.6 | −0.63 | −1.6 |
| compact − Transformer | 29 | 81.85 ± 0.05 | 30.4 ± 0.8 | 16.4 | −0.13 | −0.2 |
| compact − metadata | 31 | 81.59 ± 0.01 | 34.9 ± 0.3 | 11.9 | −0.39 | +4.3 |

The L2b / L2c features collapse orchard recall (full: 9.2%); the compact layout matches the
60-d one with 39 features. MSResNet logits are the most valuable block; parcel metadata
trades 4.3 points of orchard recall for 0.39 points of accuracy.

### Reproducing

```bash
python scripts/patch_l2a_features.py artifacts/cached_logits_train.pt artifacts/cached_logits_eval.pt
python scripts/patch_l3_distances.py --l3-weights artifacts/checkpoints/level3_protonet_l2a_episodic/best.pt \
  artifacts/cached_logits_train.pt artifacts/cached_logits_eval.pt
python scripts/patch_global_logits.py --models msresnet artifacts/cached_logits_train.pt artifacts/cached_logits_eval.pt
python scripts/sweep_compact_dual_bias.py --seed 42 --proba-dir artifacts/pruned_proba_cache_main  # repeat per seed
python scripts/evaluate_meta_protocol.py --arm main=artifacts/pruned_proba_cache_main
python scripts/ablate_meta_layouts.py --seeds 42,1,2
```

Each `patch_*` script rewrites a cache in place only after a verified backup, or writes a
copy with `--output-suffix`.

## Installation

```bash
uv sync
# or: python -m pip install -e .
```

Optional dev / tracking extras:

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[tracking]"
```

On Windows, if not installed editable:

```powershell
$env:PYTHONPATH = "src"
python scripts/train_meta_classifier.py
```

## Project layout

```text
.
├── main.py
├── scripts/           # training and evaluation entry points
├── src/agricultureai/
│   ├── data/          # loaders, label maps, episodic L3 sampling
│   ├── evaluation/    # shared eval helpers
│   ├── meta_features.py  # 64-d feature layout + XGB helpers
│   ├── models/        # Transformer, LTAE, GL-TAE, MSResNet, ProtoNet
│   ├── pipeline.py    # DMOEPipeline
│   └── training/      # trainer, checkpoints, losses
├── tests/
└── artifacts/         # checkpoints, caches, logs (gitignored)
```

## Tests

```bash
pytest
```

GitHub Actions runs the same suite on push/PR (`.github/workflows/python-app.yml`).

## Notes

- Generated checkpoints, caches, Excel reports, and datasets are gitignored.
- Checkpoints are self-describing (`architecture_config` + `training_metadata`).
- W&B logging: `--wandb --wandb-project dmoe-thesis` on training scripts.
