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
**Spectral level:** `L2A` (10 channels; max sequence length 45). The series are not
cloud-free: the per-date cloud fraction `CLD` is above zero on 60–68% of observations, and
those dates are noisy.  
Previous L1C (13 channels) checkpoints are backed up in `artifacts/L1C_backup/`.

The main challenge is **orchards** (552 eval parcels): an MSResNet trained on these L2A series
reaches 82.5% global accuracy but only 13–17% orchard recall (the pretrained L1C MSResNet:
80.8% / 16%). The meta-classifier reaches 50–77% orchard recall at 80.8–82.6% global
accuracy, depending on the operating point (see Results).

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
8 parcel-metadata features) plus 15 phenology features from clear dates
(`compute_phenology_features.py`) and the 9 logits of an MSResNet trained on cloud-screened
L2A dates (`train_series_classifier.py`), an XGBoost + CatBoost blend, and decode-time
orchard / meadow log-biases that pick the operating point.

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
| `compute_phenology_features.py` | 15 orchard-vs-meadow phenology features |
| `build_series_store.py` | Pack L2A series + per-date clear-sky flags per region |
| `train_series_classifier.py` | 9-class MSResNet on all or cloud-screened L2A dates; TTA logit export |
| `train_meta_variants.py` | Meta-classifier variants with extra feature blocks (`--extra`) |
| `evaluate_meta_protocol.py` | Out-of-sample decode selection, block bootstrap, paired arms |
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
Rows marked † are means over 3 training seeds (network and meta-classifier seed 42, 1, 2);
brackets are 95% spatial block-bootstrap intervals (seed 42).

| Model | Global acc | Orchard recall | Orchard precision | Wheat recall |
| --- | --- | --- | --- | --- |
| MSResNet baseline (L1C, pretrained, stock transform) * | 80.77% | 15.73% | – | 97.58% |
| Previous best: XGB @ m=25, 64-d (L1C, chosen on frh04) * | 80.76% | 35.80% | – | 97.57% |
| DMOE cascade only (no meta-classifier) | 79.65% | 21.74% | 5.73% | 97.37% |
| MSResNet trained on L2A, clear dates (argmax) † | 82.47% | 17.2% | 54.9% | 97.81% |
| Meta + phenology, accuracy-first † | 82.01% | 45.5% | 15.3% | 98.07% |
| Meta + phenology, orchard-first † | 80.74% | 66.1% | 8.0% | 98.02% |
| Meta + phenology, closest to stretch † | 81.23% | 61.0% | 9.7% | 98.03% |
| **Meta + phenology + L2A network, accuracy-first** † | **82.63%** [81.85, 83.45] | **49.8%** [43.9, 55.3] | 20.6% | 98.24% |
| **… orchard-first** † | **80.79%** [80.06, 81.83] | **77.4%** [73.1, 82.0] | 7.7% | 98.22% |
| **… orchard-first, accuracy ≥ 81%** † | **81.05%** [80.22, 81.95] | **75.2%** [70.6, 80.4] | 8.5% | 98.23% |
| **… closest to stretch** † | **82.12%** [81.22, 82.91] | **60.1%** [54.3, 64.5] | 13.5% | 98.23% |

\* L1C evaluation set (122,614 parcels, 553 orchards); not paired with the L2A rows.
Rules: accuracy-first = max accuracy with orchard recall ≥ 35%; orchard-first = max orchard
recall with accuracy ≥ 80.69% (or ≥ 81%); closest to stretch = smallest shortfall to
(81%, 60%). Production targets (accuracy > 80.69%, orchard recall > 35%) are met. The stretch
target (≥ 81% and ≥ 60%) is met by the "accuracy ≥ 81%" row on every seed (81.09 / 75.5,
81.02 / 74.6, 81.03 / 75.4): orchard recall clears 60% with its whole interval, while
accuracy sits at the floor by construction (seed 42 interval 80.2–82.0). The DMOE cascade
row uses `evaluate_dmoe_pipeline.py` defaults. Full logs:
`artifacts/meta_protocol_cloud_3seeds.log`, `artifacts/max_orchard_at81_3seeds.log`.

What moved the numbers (paired block bootstrap, out-of-sample selection, seeds 42 / 1 / 2):

- The L2A network block (an MSResNet trained from scratch on frh01–frh03 L2A series,
  cloud-screened dates, 9 TTA logits added to the meta-classifier): +0.68 / +0.67 / +0.52
  accuracy and +3.4 / +2.7 / +6.7 orchard recall (accuracy-first; orchard precision
  15.3 → 20.6%), +11.2 / +10.0 / +12.5 orchard recall (orchard-first), +0.91 / +0.88 / +0.86
  accuracy at similar orchard recall (−2.4 / +0.2 / −0.4, intervals span zero; closest to
  stretch), and orchard average precision 26.5 → 34.5. Every accuracy interval at
  accuracy-first and closest-to-stretch, and every orchard-recall interval at orchard-first,
  excludes zero. Training on all dates instead gives nearly the same gain (below).
- Against that L2A network on its own (same seed, `artifacts/meta_vs_l2a_network_3seeds.log`):
  the accuracy-first meta-classifier matches its accuracy (+0.01 / +0.25 / +0.23, intervals
  span zero) and adds +33.3 / +30.6 / +33.9 points of orchard recall; closest to stretch
  gives up 0.22–0.59 accuracy for +42 to +44.
- Phenology features (below): +9.1 / +8.5 / +8.3 orchard recall (accuracy-first),
  +8.5 / +10.3 / +8.9 (orchard-first) and +0.62 / +0.92 / +0.76 accuracy (closest to
  stretch); orchard average precision 22 → 26.5. Every orchard-recall interval excludes zero.
- Re-extracting the MSResNet features from resampled views instead of zero-padded series:
  +0.50 / +0.42 / +0.36 accuracy (accuracy-first) and +5.3 / +5.1 / +3.6 orchard recall
  (orchard-first); every 95% interval excludes zero.
- The same fix for L2a / L3: +8.5 points of cereal accuracy in the cascade, but within seed
  noise once the meta-classifier is on top. For the Transformer features: no measurable effect.
- Choosing decode settings on frh04 itself inflated accuracy by at most 0.15 points.
- Stacking leakage (the experts and pretrained models saw frh01–frh03, so the meta-classifier's
  training features are in-sample): training it instead on frh04's own spatial folds, where
  every expert feature is out of sample (`train_meta_variants.py --train-on frh04-oof`),
  changed nothing measurable (seed 42: −0.15 to +0.04 accuracy, all intervals span zero),
  despite 5× less training data. With phenology features as well, it matches the
  frh01–frh03-trained model at the orchard-first and stretch points; at the accuracy-first
  point it lands on a higher-precision trade-off (82.21% accuracy, 37.0% orchard recall,
  22.3% orchard precision; seed 42). The L2A network block is in-sample on frh01–frh03 in
  the same way and has not been re-tested this way.

### Phenology features

96% of false orchards are meadows, so `scripts/compute_phenology_features.py` adds 15
per-parcel features aimed at that confusion, computed from clear dates only (`CLD == 0`):
seasonal NDVI means and winter minimum, NDVI spread, summer-minus-winter contrast,
mowing-like NDVI drops in April–September, spring / summer red-edge (NDRE) and moisture
(NDMI) means, the number of clear dates, and parcel compactness (4π·area / perimeter²).
On frh04 the strongest single features separate orchards from meadows like this
(separability = max(AUROC, 1 − AUROC)):

| Feature | Separability | Direction |
| --- | --- | --- |
| spring NDMI | 0.74 | meadows wetter |
| spring NDVI | 0.72 | meadows greener |
| spring NDRE | 0.71 | meadows higher |
| largest NDVI drop (Apr–Sep) | 0.70 | meadows (cuts / grazing) |
| summer − winter NDVI | 0.69 | orchards gain into summer |

Grass greens up early and is cut; deciduous trees peak later and keep their summer canopy.

### L2A network and cloud screening

`CLD` is a per-date cloud fraction, above zero on 60–68% of dates.
`agricultureai.data.cloud_mask.clear_observation_mask` keeps a date when no band is missing,
`CLD ≤ 0.1`, `EDG ≤ 0.5`, blue reflectance (B2) < 0.12, and it is not an isolated NDVI dip
(more than 0.15 below both neighbours within 20 days: a missed cloud or shadow; a cut that
does not recover at once is kept). That leaves 32–40% of dates per region (median 11–14
per parcel); a parcel with no clear date keeps all of them. `build_series_store.py` packs
the series and flags; `train_series_classifier.py` trains the same 9-class MSResNet on all
dates or on clear dates only (identical seed, schedule and 10 epochs; frh04 is never used to
pick an epoch; BatchNorm statistics are recomputed after the last epoch, without which
eval-mode accuracy lagged by ~20 points), and exports logits averaged over 8 resampled views.

Clear dates minus all dates, per network seed (42 / 1 / 2), paired block bootstrap:

| Comparison | Accuracy | Orchard recall | Notes |
| --- | --- | --- | --- |
| Network alone (argmax) | +0.16 / −0.11 / −0.09 | **+4.0 / +5.4 / +3.1** | orchard intervals [+1.6, +6.6], [+2.6, +8.4], [0.0, +6.3]; wheat −0.3 to −0.5 |
| Meta, orchard AP (no selection) | – | AP +1.1 / +1.3 / +1.0 | each interval includes zero (P(>0) = 0.90 / 0.97 / 0.89) |
| Meta, accuracy-first | +0.03 / +0.02 / −0.13 | +1.6 / +0.5 / +3.1 | intervals span zero |
| Meta, orchard-first | +0.23 / 0.00 / −0.03 | +0.7 / +0.4 / 0.0 | intervals span zero (seed 42 accuracy aside) |
| Meta, orchard-first at ≥ 81% | – | +1.6 / +1.1 / −0.2 | intervals span zero |
| Meta, closest to stretch | −0.11 / +0.03 / −0.05 | −1.3 / +0.7 / −1.1 | intervals span zero |

Cloud screening makes the network itself a better orchard detector on every seed, at the
same accuracy, and the fused model ranks orchards slightly better (AP +1 on every seed).
At the selected operating points the two variants are indistinguishable: almost all of the
meta-classifier gain comes from adding an L2A-trained network at all (all-dates variant:
82.66 / 48.1 accuracy-first, 80.73 / 77.0 orchard-first, 82.16 / 60.7 closest to stretch).

**L1C control: the input, not a second network.** The same MSResNet, trainer, schedule and
seeds on the same 10 bands at L1C (top-of-atmosphere; `build_series_store.py --level L1C`,
all dates, logits aligned to the cache rows with `--align-cache`, NaN for the 151 eval
parcels without an L1C series) gives the meta-classifier almost nothing over phenology alone
(3 seeds, `artifacts/meta_protocol_l1c_control.log`, `artifacts/l1c_control_eval.log`):

| Extra block | Acc-first acc / orchard | Orchard-first orchard | Acc ≥ 81% orchard | Closest-stretch acc | Orchard AP |
| --- | --- | --- | --- | --- | --- |
| none (phenology only) | 82.01 / 45.5 | 66.1 | 63.7 | 81.23 | 26.5 |
| L1C network | 82.12 / 43.5 | 67.0 | 64.5 | 81.30 | 26.8 |
| L2A network (all dates) | 82.66 / 48.1 | 77.0 | 74.3 | 82.16 | 33.3 |

L2A minus L1C, paired per seed: +0.49 / +0.55 / +0.58 accuracy (accuracy-first), +10.7 /
+9.1 / +10.3 orchard recall (orchard-first) and +0.87 / +0.82 / +0.89 accuracy (closest to
stretch); every one of these intervals excludes zero. On its own the L1C network is also
weaker (−0.66 to −1.11 accuracy, −4.4 to −9.1 orchard recall against the L2A network, all
intervals excluding zero). So the gain comes from atmospherically corrected input, not from
ensembling a second network; L1C here also has 2.4× more (cloudier) dates per parcel, which
is part of what the two levels differ in as distributed.

Note: the raw L1C HDF5 columns are in CSV order (B1, B10, B11, B12, B2 … B8A, B9, QA…, doa),
not the `SELECTED_BANDS` order the stock BreizhCrops transform indexes with, so the
pretrained L1C models see their 13 channels permuted (consistently in training and test).
`build_series_store.py` uses the verified layout.

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
python scripts/compute_phenology_features.py artifacts/cached_logits_train.pt --output artifacts/phenology_train.npz
python scripts/compute_phenology_features.py artifacts/cached_logits_eval.pt --output artifacts/phenology_eval.npz
python scripts/train_meta_variants.py --seed 42 --proba-dir artifacts/pruned_proba_cache_pheno \
  --extra pheno=artifacts/phenology_train.npz,artifacts/phenology_eval.npz  # repeat per seed
python scripts/evaluate_meta_protocol.py --arm pheno=artifacts/pruned_proba_cache_pheno
python scripts/ablate_meta_layouts.py --seeds 42,1,2

# L2A-trained MSResNet block (cloud-screened dates)
for r in frh01 frh02 frh03 frh04; do
  python scripts/build_series_store.py $r --output artifacts/series_store/$r.npz
done
CK=artifacts/checkpoints/msresnet_l2a_clear
python scripts/train_series_classifier.py --dates clear --seed 42 --out-dir $CK  # resumes; rerun until done
python scripts/train_series_classifier.py --dates clear --seed 42 --out-dir $CK --export train
python scripts/train_series_classifier.py --dates clear --seed 42 --out-dir $CK --export eval
python scripts/train_meta_variants.py --seed 42 --proba-dir artifacts/pruned_proba_cache_phenomsclear \
  --extra pheno=artifacts/phenology_train.npz,artifacts/phenology_eval.npz \
  --extra msclear=$CK/logits_train.npz,$CK/logits_eval.npz
python scripts/evaluate_meta_protocol.py --pair phenomsclear:pheno \
  --arm pheno=artifacts/pruned_proba_cache_pheno \
  --arm phenomsclear=artifacts/pruned_proba_cache_phenomsclear
```

Replicates pair network seed *s* with meta-classifier seed *s* (`--seed 1`, `--seed 2`,
`--out-dir ${CK}_seed1`, `--proba-dir ..._seed1`); `--dates all` gives the all-dates control.
The L1C control: `build_series_store.py <region> --level L1C --output artifacts/series_store_l1c/<region>.npz`,
then `train_series_classifier.py --dates all --store-dir artifacts/series_store_l1c` and export
with `--align-cache artifacts/cached_logits_{train,eval}.pt`.

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
│   ├── data/          # loaders, label maps, episodic L3 sampling, cloud mask
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
