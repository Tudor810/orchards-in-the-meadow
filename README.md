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
**Eval region:** `frh04` (122,614 parcels)  
**Spectral level:** `L1C` (13 channels, max sequence length 45)

The main challenge is **orchards** (553 eval samples): MSResNet baseline reaches ~80.8% global
accuracy but only ~16% orchard recall. The meta-classifier raises orchard recall to ~36% at
similar global accuracy.

## Architecture

```
x (B,T,C) → L1 Transformer router (4 super-classes)
  ├─ Rapeseed / Corn → global {2, 3}
  ├─ Cereals → L2a LTAE → {barley, wheat}
  └─ Grasslands → L2b GL-TAE anomaly gate
        ├─ Grass → L2c MSResNet → {permanent, temporary meadows}
        └─ Anomaly → L3 linear head → {sunflower, orchards, nuts} + meadow rejection
```

Meta-classifier (64-dim: 58 neural + 6 parcel metadata) replaces the hand-crafted probability
cascade for final predictions.

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

## Results (frh04, production config)

| Model | Global acc | Orchard recall | Wheat recall |
| --- | --- | --- | --- |
| MSResNet baseline | 80.77% | 15.73% | 97.58% |
| DMOE cascade only | ~75–78% | ~34% | ~81% |
| **Meta @ m=25 (64-d)** | **80.76%** | **35.80%** | **97.57%** |

Targets in code: accuracy > 80.69%, orchard recall > 35%.

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
