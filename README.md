# Orchards in the Meadow

Code for the paper *Orchards in the Meadow: Seasonal Evidence and a Two-Stage Orchard Verifier for Rare-Class Crop Mapping from Sentinel-2 Time Series* (Bodea, 2026); see [How to cite](#how-to-cite).

Rare-class (orchard) crop mapping on the BreizhCrops Sentinel-2 time-series dataset.
The final system (design frozen on 2026-10-09, see `FROZEN_DESIGN.md`) is a **late-fusion
meta-classifier** (XGBoost + CatBoost) over phenology features, parcel metadata, pretrained
L1C networks and an MSResNet trained on L2A, followed by a **two-stage orchard verifier**.
The **4-tier DMOE expert hierarchy** the project started from is kept as code and as an
ablation; its outputs are not part of the final design. A pre-registered four-region
rotation (see Results) confirms the phenology and L2A-network gains in every region; the
verifier helps in three of the four regions and is neutral in frh03.

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
80.8% / 16%). The final meta-classifier reaches 48.7–77.8% orchard recall at 80.8–82.7%
global accuracy, depending on the operating point; with the verifier, 45.5% of predicted
orchards are correct at 82.90% accuracy (see Results). Rotating the test region over all four
regions gives 80.1–83.5% accuracy at 51–67% orchard recall (accuracy-first).

## Architecture

Final system:

```
features (51)
  ├─ 8 parcel-metadata features
  ├─ 15 phenology features from clear dates
  ├─ pretrained BreizhCrops MSResNet + Transformer on L1C (19, 8-view TTA)
  └─ MSResNet trained on L2A clear dates (9 logits, 8-view TTA, mean of 3 seeds)
→ meta-classifier: XGBoost + CatBoost, orchard weight m, decode-time log-biases
→ orchard verifier: parcels above a candidate threshold (keeps 90% of out-of-fold training
  orchards) re-scored by a binary XGBoost with monthly clear-date means, parcel geometry and
  neighbour context; orchard above a threshold chosen out of sample, else the meta label
```

Four-tier expert cascade (the starting point; tested as a feature block and dropped):

```
x (B,T,C) → L1 Transformer router (4 super-classes)
  ├─ Rapeseed / Corn → global {2, 3}
  ├─ Cereals → L2a LTAE → {barley, wheat}
  └─ Grasslands → L2b GL-TAE anomaly gate
        ├─ Grass → L2c MSResNet → {permanent, temporary meadows}
        └─ Anomaly → L3 ProtoNet (Mahalanobis) → {sunflower, orchards, nuts} + meadow rejection
```

The meta-classifier replaced the hand-crafted probability cascade. It started on the compact
39-d layout (L3 distances, cereal features, MSResNet logits, Transformer block, 8
parcel-metadata features); the final design drops the expert blocks (L3 distances, cereal
features: `train_meta_variants.py --drop-blocks l3_raw,cereal`) and adds 15 phenology
features from clear dates (`compute_phenology_features.py`) and the 9 logits of an MSResNet
trained on cloud-screened L2A dates (`train_series_classifier.py`), averaged over three
network seeds. An XGBoost + CatBoost blend and decode-time orchard / meadow log-biases pick
the operating point; `orchard_verifier.py` adds the second stage.

L2a, L3 and the pretrained MSResNet / Transformer were trained on the stock BreizhCrops
transform (every parcel resampled to 45 dates, never zero-padded); their cached outputs are
computed from 8 seeded resampled views (`patch_l2a_features.py`, `patch_l3_distances.py`,
`patch_global_logits.py`), not from zero-padded series.

The expert cascade is `src/agricultureai/pipeline.py` (`DMOEPipeline`).

## Expert-cascade workflow (starting point)

### 1. Train experts

```bash
python scripts/cascade/train_transformer_breizhcrops.py
python scripts/cascade/train_expert_cereals.py
python scripts/cascade/train_l2b_anomaly_gate.py
python scripts/cascade/train_l2c_meadow_expert.py
python scripts/cascade/train_l3_protonet.py --level L2A --checkpoint-dir artifacts/checkpoints/level3_protonet
```

Checkpoints land in `artifacts/checkpoints/<run_name>/best.pt`.

### 2. Build meta-classifier caches

```bash
python scripts/cascade/build_meta_caches.py
```

`build_meta_caches.py` extracts the branch outputs (`sweep_dmoe_parameters.py --extract-only`),
aligns the pretrained L1C MSResNet / Transformer logits by parcel id
(`append_l1c_cached_global_logits.py`; the pretrained models need 13-band L1C input), patches
the L3 Mahalanobis distances and appends the 8-d parcel metadata. Then refresh the 8-view
features as in Reproducing below (`patch_l2a_features.py`, `patch_l3_distances.py`,
`patch_global_logits.py`). Caches: `artifacts/cached_logits_train.pt`,
`artifacts/cached_logits_eval.pt`.

### 3. Evaluate DMOE cascade (without meta)

```bash
python scripts/cascade/evaluate_dmoe_pipeline.py \
  --l1-weights  artifacts/checkpoints/transformer_l1_4class/best.pt \
  --l2a-weights artifacts/checkpoints/expert_cereals/best.pt \
  --l2b-weights artifacts/checkpoints/l2b_anomaly_gate/best.pt \
  --l2c-weights artifacts/checkpoints/l2c_meadow_expert/best.pt \
  --l3-weights  artifacts/checkpoints/level3_protonet/best.pt
```

### Baselines

```bash
python scripts/cascade/evaluate_pretrained_models.py --full-region
```

## Scripts

Scripts run from the repository root (`python scripts/<stage>/<script>.py`); code shared between
scripts lives in `src/agricultureai`.

**`scripts/data/`**: per-parcel inputs

| Script | Purpose |
| --- | --- |
| `build_series_store.py` | Pack L2A (or L1C) series + per-date clear-sky flags per region |
| `compute_phenology_features.py` | 15 orchard-vs-meadow phenology features from clear dates |

**`scripts/cascade/`**: four-tier expert cascade, branch-output caches, pretrained baselines

| Script | Purpose |
| --- | --- |
| `train_transformer_breizhcrops.py` | L1 4-class router |
| `train_expert_cereals.py` | L2a barley / wheat (LTAE) |
| `train_l2b_anomaly_gate.py` | L2b grass vs anomaly (GL-TAE) |
| `train_l2c_meadow_expert.py` | L2c permanent / temporary meadows (MSResNet) |
| `train_l3_protonet.py` | L3 rare-class ProtoNet (Mahalanobis) |
| `evaluate_dmoe_pipeline.py` | Full cascade on frh04 |
| `sweep_dmoe_parameters.py` | Extract branch outputs to a cache; cascade decode sweep |
| `build_meta_caches.py` | Build the train / eval caches from the expert checkpoints |
| `append_msresnet_logits.py`, `append_transformer_logits.py` | Pretrained L1C network logits into an L1C cache |
| `append_l1c_cached_global_logits.py` | Align those logits into the L2A caches by parcel id |
| `append_parcel_metadata.py` | 8 parcel-metadata features |
| `patch_global_logits.py`, `patch_l2a_features.py`, `patch_l3_distances.py` | Recompute pretrained-network / L2a / L3 outputs from 8 resampled views |
| `evaluate_pretrained_models.py` | Pretrained MSResNet / Transformer baselines |

**`scripts/networks/`**: series networks trained on L2A

| Script | Purpose |
| --- | --- |
| `train_series_classifier.py` | 9-class MSResNet on all or cloud-screened dates (optional class-weighted loss); TTA logit export |
| `network_as_arm.py` | A network's frh04 logits as a protocol arm (same decision dial as the meta-classifier) |

**`scripts/meta/`**: meta-classifier, orchard verifier, evaluation

| Script | Purpose |
| --- | --- |
| `train_meta_variants.py` | Meta-classifier with extra feature blocks / dropped blocks / frh04 cross-fitting |
| `evaluate_meta_protocol.py` | Out-of-sample decode selection, block bootstrap, paired arms |
| `orchard_verifier.py` | Two-stage orchard verifier (`oof` / `features` / `verify`) |
| `compute_model_metrics.py` | Confusion matrices, micro / macro P, R, F1, orchard error shares |
| `paired_ap_ci.py` | Paired block-bootstrap interval for the orchard-AP difference of two arms |
| `recall_at_accuracy_floor.py` | Cross-fitted orchard recall at an accuracy floor (default 81%) and orchard AP, paired |
| `ablate_meta_layouts.py` | Feature-layout ablation |

**`scripts/rotation/`**: pre-registered region rotation

| Script | Purpose |
| --- | --- |
| `region_rotation.py` | Four-region rotation of the frozen design (`prep` / `meta` / `oof` / `verify`) |
| `diagnose_rotation_verifier.py` | Post-hoc verifier variants (candidate share, feature groups) |

## Results (frh04, held out)

Decode settings (blend, orchard multiplier `m`, orchard / meadow biases) are chosen out of
sample: frh04 is cut into 10 km spatial blocks grouped into 5 folds, and each fold is scored
with the settings a fixed rule picks on the other four (`scripts/meta/evaluate_meta_protocol.py`).
Rows marked † are means over 3 training seeds (network and meta-classifier seed 42, 1, 2);
brackets are 95% spatial block-bootstrap intervals (seed 42). The logs cited below are in
[`logs/`](logs/README.md), whose index maps each one to the tables of the paper; the scripts write
them to `artifacts/`.

| Model | Global acc | Orchard recall | Orchard precision | Wheat recall |
| --- | --- | --- | --- | --- |
| MSResNet baseline (L1C, pretrained, stock transform) * | 80.77% | 15.73% | – | 97.58% |
| DMOE cascade only (no meta-classifier) | 79.65% | 21.74% | 5.73% | 97.37% |
| MSResNet trained on L2A, clear dates (argmax) † | 82.47% | 17.2% | 54.9% | 97.81% |
| Meta + phenology, accuracy-first † | 82.01% | 45.5% | 15.3% | 98.07% |
| Meta + phenology, orchard-first † | 80.74% | 66.1% | 8.0% | 98.02% |
| Meta + phenology, closest to stretch † | 81.23% | 61.0% | 9.7% | 98.03% |
| Meta + phenology + L2A network (with experts, one network), accuracy-first † | 82.63% [81.85, 83.45] | 49.8% [43.9, 55.3] | 20.6% | 98.24% |
| … orchard-first † | 80.79% [80.06, 81.83] | 77.4% [73.1, 82.0] | 7.7% | 98.22% |
| … orchard-first, accuracy ≥ 81% † | 81.05% [80.22, 81.95] | 75.2% [70.6, 80.4] | 8.5% | 98.23% |
| … closest to stretch † | 82.12% [81.22, 82.91] | 60.1% [54.3, 64.5] | 13.5% | 98.23% |
| **Final design (no experts, 3-seed L2A network), accuracy-first** † | **82.67%** [81.80, 83.46] | **48.7%** [43.0, 54.1] | 21.7% | 98.26% |
| **… orchard-first** † | **80.78%** [79.84, 81.64] | **77.8%** [73.9, 83.3] | 7.8% | 98.20% |
| **… closest to stretch** † | **82.18%** [81.33, 83.04] | **60.4%** [54.1, 64.1] | 13.6% | 98.24% |
| **… + orchard verifier, accuracy-first** † ‡ | **82.90%** | **36.4%** | **45.5%** | 98.26% |

\* L1C evaluation set (122,614 parcels, 553 orchards); not paired with the L2A rows.
Rules: accuracy-first = max accuracy with orchard recall ≥ 35%; orchard-first = max orchard
recall with accuracy ≥ 80.69% (or ≥ 81%); closest to stretch = smallest shortfall to
(81%, 60%). Production targets (accuracy > 80.69%, orchard recall > 35%) are met. The stretch
target (≥ 81% and ≥ 60%) is met by the "accuracy ≥ 81%" row on every seed (81.09 / 75.5,
81.02 / 74.6, 81.03 / 75.4): orchard recall clears 60% with its whole interval, while
accuracy sits at the floor by construction (seed 42 interval 80.2–82.0). The DMOE cascade
row uses `evaluate_dmoe_pipeline.py` defaults with 8-view inputs (`logs/run_cascade_eval.log`;
decoding the cached outputs instead, as `compute_model_metrics.py --models cascade` does, gives
79.66% / 21.2%, `logs/model_metrics_cascade_frh04.log`). Full logs:
`logs/meta_protocol_cloud_3seeds.log`, `logs/max_orchard_at81_3seeds.log`; final
design: `logs/meta_protocol_final.log`. ‡ Verifier decode: orchard where the verifier
score passes a threshold chosen out of sample, else the meta argmax without orchards
(`logs/orchard_verifier_final/verify_seed*.log`). Paired confusion matrices and micro /
macro precision, recall and F1 for the baseline, the meta-classifier and the final system:
`scripts/meta/compute_model_metrics.py` (`logs/model_metrics_frh04.log`).

What moved the numbers (paired block bootstrap, out-of-sample selection, seeds 42 / 1 / 2):

- Orchard verifier (`orchard_verifier.py`, on the final meta-classifier): orchard AP
  34.9 → 39.1 (+4.29 / +3.83 / +4.34, every interval excludes zero); at the accuracy-first
  point precision 34.8 → 45.5% at the same 36% recall and 82.83 → 82.90% accuracy; at the 81%
  accuracy floor little change (recall 74.5 → 76.8%). Single feature groups add at most 1.2 AP
  points (monthly means); only all of them together give the gain. In the region rotation
  (below) the gain holds in frh01, frh02 and frh04 but not in frh03.
- Expert blocks (L3 distances, cereal features): dropping them moves accuracy by at most 0.15
  points and orchard AP 34.5 → 34.0; accuracy-first orchard recall −2.4 / 0.0 / −3.4
  (`logs/meta_protocol_noexperts.log`). Left out of the final design.
- Averaging three L2A network seeds (no-experts base): orchard AP 34.0 → 35.3, accuracy-first
  82.61 → 82.67% and 47.9 → 48.7% orchard recall (`logs/meta_protocol_final.log`).
- The pretrained L1C networks are still needed: without them accuracy-first accuracy falls
  82.61 → 81.94% and orchard AP 34.0 → 31.4 (`logs/meta_protocol_checks245.log`).
- Vegetation indices as extra L2A-network inputs (`train_series_classifier.py --indices`): no
  gain (network 82.0–82.2% vs 82.5%; meta accuracy 82.40 vs 82.61%, orchard AP 33.3 vs 34.0;
  `logs/meta_protocol_indices.log`).
- End-to-end alternatives, given the same decision dial (biases and out-of-sample selection;
  `scripts/networks/network_as_arm.py`, `logs/meta_protocol_e2e.log`, `logs/e2e/paired_ap_ci.log`):
  the L2A network alone reaches 82.24% / 37.0% orchard recall at accuracy-first (argmax: 17.2%);
  trained with a class-weighted loss (`train_series_classifier.py --class-weights orchard` ×10,
  or `sqrt_inv` capped at 10) 81.69% / 45.4% and 82.01% / 40.6%. Weighting shifts the operating
  point but not the ranking: orchard AP 25.2–27.7 vs 26.1 unweighted (every interval spans zero),
  against 35.3 for the final design (+7.5 to +10.2, every interval excludes zero). The final
  design leads at every operating point: +0.6 to +1.2 accuracy (accuracy-first), +14.5 to +18.5
  orchard recall (orchard-first: 77.8% vs at most 62%), +1.1 to +1.7 accuracy (closest to
  stretch). Subtracting the log class weights from the network's logits changes none of this.

- The L2A network block (an MSResNet trained from scratch on frh01–frh03 L2A series,
  cloud-screened dates, 9 TTA logits added to the meta-classifier): +0.68 / +0.67 / +0.52
  accuracy and +3.4 / +2.7 / +6.7 orchard recall (accuracy-first; orchard precision
  15.3 → 20.6%), +11.2 / +10.0 / +12.5 orchard recall (orchard-first), +0.91 / +0.88 / +0.86
  accuracy at similar orchard recall (−2.4 / +0.2 / −0.4, intervals span zero; closest to
  stretch), and orchard average precision 26.5 → 34.5. Every accuracy interval at
  accuracy-first and closest-to-stretch, and every orchard-recall interval at orchard-first,
  excludes zero. Training on all dates instead gives nearly the same gain (below).
- Against that L2A network on its own (same seed, `logs/meta_vs_l2a_network_3seeds.log`):
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
  22.3% orchard precision; seed 42). Re-tested for the final design (L2A network block
  included, 3 seeds; `logs/meta_protocol_leakage_final.log`): accuracy changes by
  −0.08 to +0.12 points at all three operating points and orchard AP by −0.7 to −1.0
  (35.3 → 34.5; `scripts/meta/paired_ap_ci.py`), every interval spanning zero. At the default
  decoding the out-of-sample model is only more conservative on orchards (accuracy-first:
  37.5% recall at 30.7% precision instead of 48.7% at 21.7%; orchard F1 +3.0 to +4.5).
  In-sample training features do not inflate the reported results.

### Region rotation (pre-registered)

Every design choice above was made on frh04, so the design was frozen and four checks
written down (`FROZEN_DESIGN.md`) before rotating the test region over all four regions. The
rotation uses a rotation-safe variant: the pretrained BreizhCrops networks saw frh01–frh03, so
they are replaced by our own L1C MSResNet trained per fold, and one L2A network seed is used
per fold. Arms: B = metadata + phenology + L1C net + L2A net; A = B without the L2A net;
D = B without phenology; C = B + verifier. Means of meta seeds 42 / 1 / 2:

| Test region | Parcels / orchards | Accuracy-first acc / orch. recall / precision (B) | Orchard-first acc / recall (B) | Orchard AP B / A / D | Verifier AP (C − B baseline) |
| --- | --- | --- | --- | --- | --- |
| frh01 | 178,632 / 937 | 83.52% / 51.5% / 22.2% | 80.83% / 80.7% | 40.0 / 26.9 / 36.4 | 39.0 → 42.5 |
| frh02 | 140,782 / 348 | 82.72% / 52.6% / 9.2% | 80.93% / 71.3% | 29.4 / 24.2 / 26.1 | 29.3 → 33.8 |
| frh03 | 166,367 / 1,217 | 80.13% / 67.2% / 18.3% | floor not reached | 43.4 / 28.8 / 37.7 | 42.8 → 42.4 |
| frh04 | 122,708 / 552 | 82.40% / 50.9% / 17.8% | 80.77% / 75.2% | 32.5 / 21.3 / 28.7 | 31.9 → 36.2 |

1. **L2A network (B − A): holds in all four regions.** Orchard AP +5.0 to +14.9, orchard-first
   recall +5.7 to +19.9; all 12 intervals of each exclude zero.
2. **Phenology (B − D): holds in all four regions.** Orchard recall rises at both operating
   points for every region and seed (+0.6 to +7.1; 21 of 24 intervals exclude zero); orchard
   AP +3.0 to +5.9, all intervals exclude zero.
3. **Verifier (C − B): holds in three regions, fails in frh03.** Orchard AP +3.0 to +4.8 in
   frh01, frh02 and frh04 (all 9 intervals exclude zero; accuracy-first precision 44 → 50%,
   20 → 32%, 28 → 42%). In frh03: −1.1 / −0.8 / +0.7, every interval spans zero, and
   accuracy-first precision falls 52 → 47%. frh03 has the most orchards (0.73% of parcels)
   and the highest meta AP before the verifier, partly because AP rises with prevalence.
   Post-hoc (`scripts/rotation/diagnose_rotation_verifier.py`, `logs/rotation/diagnosis.log`;
   the frozen design is unchanged): frh03 holds 40% of all orchards, so its fold trains on
   the fewest (1,837 vs 2,117–2,706) and gets the loosest candidate threshold (τ 0.0055 vs
   0.009–0.019; 18% of parcels become candidates instead of 8–10%). A 9% candidate share in
   every fold removes the frh03 loss (AP −0.3 to +0.3) and keeps the gain elsewhere (+3.0 to
   +4.4), but creates no gain in frh03; dropping the neighbour or monthly features does not
   help either. The threshold is part of the problem, not all of it.
4. **Absolute level:** 80.1–83.5% accuracy at 51–67% orchard recall (accuracy-first).
   Orchard precision is lowest where orchards are rarest (9% in frh02). frh03 never reaches
   the 80.69% accuracy floor, so all three selection rules pick the default decode there.

The variant on frh04 sits below the full design (82.40% / 50.9%, AP 32.5 vs 82.67% / 48.7%,
AP 35.3), as expected without the pretrained networks and with one L2A seed. Logs:
`logs/rotation/<region>/protocol_{B,ablation}.log` and `verify*.json`; AP intervals for
checks 1–2 (`scripts/meta/paired_ap_ci.py`): `logs/rotation/rotation_ap_ci.log`.

### Phenology features

96% of false orchards are meadows and 98% of missed orchards are called meadows (meta-classifier,
`compute_model_metrics.py`: 96.4% / 97.5%), so `scripts/data/compute_phenology_features.py` adds 15
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
(3 seeds, `logs/meta_protocol_l1c_control.log`, `logs/l1c_control_eval.log`):

| Extra block | Acc-first acc / orchard | Orchard-first orchard | Acc ≥ 81% orchard | Closest-stretch acc | Orchard AP |
| --- | --- | --- | --- | --- | --- |
| none (phenology only) | 82.01 / 45.5 | 66.1 | 63.7 | 81.23 | 26.5 |
| L1C network | 82.12 / 43.5 | 67.0 | 64.5 | 81.30 | 26.8 |
| L2A network (all dates) | 82.66 / 48.1 | 77.0 | 74.3 | 82.16 | 33.3 |

These rows are the earlier configuration (expert blocks included, one network per seed). The paper's Table 2
reruns the comparison on the final design, each network block averaging three seeds
(`logs/meta_protocol_table2_final.log`, `logs/table2_final_at81.log`, `logs/table2_final_ap_ci.log`): orchard AP
25.7 with no network, 26.3 with the L1C network, 34.3 / 35.3 with the L2A network on all / clear dates; orchard
recall at accuracy ≥ 81% 60.7, 61.4, 75.4 and 75.2.

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

`scripts/meta/ablate_meta_layouts.py` (`logs/ablate_meta_layouts_msfix.log`): XGBoost only, `m = 1`,
no decode bias, 3 seeds (mean ± sd);
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
python scripts/cascade/patch_l2a_features.py artifacts/cached_logits_train.pt artifacts/cached_logits_eval.pt
python scripts/cascade/patch_l3_distances.py --l3-weights artifacts/checkpoints/level3_protonet_l2a_episodic/best.pt \
  artifacts/cached_logits_train.pt artifacts/cached_logits_eval.pt
python scripts/cascade/patch_global_logits.py --models msresnet artifacts/cached_logits_train.pt artifacts/cached_logits_eval.pt
python scripts/data/compute_phenology_features.py artifacts/cached_logits_train.pt --output artifacts/phenology_train.npz
python scripts/data/compute_phenology_features.py artifacts/cached_logits_eval.pt --output artifacts/phenology_eval.npz
python scripts/meta/train_meta_variants.py --seed 42 --proba-dir artifacts/pruned_proba_cache_pheno \
  --extra pheno=artifacts/phenology_train.npz,artifacts/phenology_eval.npz  # repeat per seed
python scripts/meta/evaluate_meta_protocol.py --arm pheno=artifacts/pruned_proba_cache_pheno
python scripts/meta/ablate_meta_layouts.py --seeds 42,1,2

# L2A-trained MSResNet block (cloud-screened dates)
for r in frh01 frh02 frh03 frh04; do
  python scripts/data/build_series_store.py $r --output artifacts/series_store/$r.npz
done
CK=artifacts/checkpoints/msresnet_l2a_clear
python scripts/networks/train_series_classifier.py --dates clear --seed 42 --out-dir $CK  # resumes; rerun until done
python scripts/networks/train_series_classifier.py --dates clear --seed 42 --out-dir $CK --export train
python scripts/networks/train_series_classifier.py --dates clear --seed 42 --out-dir $CK --export eval
python scripts/meta/train_meta_variants.py --seed 42 --proba-dir artifacts/pruned_proba_cache_phenomsclear \
  --extra pheno=artifacts/phenology_train.npz,artifacts/phenology_eval.npz \
  --extra msclear=$CK/logits_train.npz,$CK/logits_eval.npz
python scripts/meta/evaluate_meta_protocol.py --pair phenomsclear:pheno \
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

Final design and orchard verifier (`artifacts/checkpoints/msresnet_l2a_clear_ens3/logits_{train,eval}.npz`
is the element-wise mean of the three seeds' exported logits, same `parcel_ids` / `labels` /
`columns`):

```bash
ENS=artifacts/checkpoints/msresnet_l2a_clear_ens3
V=artifacts/orchard_verifier_final
for s in 42 1 2; do
  sfx=$([ $s = 42 ] || echo _seed$s)
  for m in 1 2 3; do
    python scripts/meta/train_meta_variants.py --seed $s --multipliers $m --drop-blocks l3_raw,cereal \
      --proba-dir artifacts/pruned_proba_cache_noexpens$sfx \
      --extra pheno=artifacts/phenology_train.npz,artifacts/phenology_eval.npz \
      --extra msens=$ENS/logits_train.npz,$ENS/logits_eval.npz
  done
  for r in frh01 frh02 frh03; do
    python scripts/meta/orchard_verifier.py oof --region $r --seed $s --work-dir $V \
      --base-tag noexpens --drop-blocks l3_raw,cereal --net-dir $ENS
  done
  python scripts/meta/orchard_verifier.py features --seed $s --work-dir $V --base-tag noexpens
  python scripts/meta/orchard_verifier.py verify --seed $s --work-dir $V --base-tag noexpens
done
A=artifacts/pruned_proba_cache_noexpens
python scripts/meta/evaluate_meta_protocol.py --arm final=$A,${A}_seed1,${A}_seed2
python scripts/meta/compute_model_metrics.py   # confusion matrices, micro / macro P, R, F1
# stacking-leakage test: same arm, meta-classifier cross-fitted on frh04's own folds
for s in 42 1 2; do
  sfx=$([ $s = 42 ] || echo _seed$s)
  python scripts/meta/train_meta_variants.py --train-on frh04-oof --seed $s --drop-blocks l3_raw,cereal \
    --proba-dir artifacts/pruned_proba_cache_noexpensoof$sfx \
    --extra pheno=artifacts/phenology_train.npz,artifacts/phenology_eval.npz \
    --extra msens=$ENS/logits_train.npz,$ENS/logits_eval.npz
done
O=artifacts/pruned_proba_cache_noexpensoof
python scripts/meta/evaluate_meta_protocol.py --arm final=$A,${A}_seed1,${A}_seed2 \
  --arm finaloof=$O,${O}_seed1,${O}_seed2 --pair finaloof:final \
  --output-log artifacts/meta_protocol_leakage_final.log --output-json artifacts/meta_protocol_leakage_final.json
python scripts/meta/paired_ap_ci.py $O $A
# Table 2 on the final configuration: the network block removed or replaced by 3-seed averages
# (msresnet_{l1c,l2a}_all_ens3 = element-wise mean of the three seeds' logits, like msresnet_l2a_clear_ens3)
for s in 42 1 2; do
  sfx=$([ $s = 42 ] || echo _seed$s)
  for m in 1 2 3; do
    for net in none l1c_all l2a_all; do
      case $net in
        none) arm=noexpnone; extra=() ;;
        l1c_all) arm=noexpl1cens; E=artifacts/checkpoints/msresnet_l1c_all_ens3; extra=(--extra msl1cens=$E/logits_train.npz,$E/logits_eval.npz) ;;
        l2a_all) arm=noexpallens; E=artifacts/checkpoints/msresnet_l2a_all_ens3; extra=(--extra msallens=$E/logits_train.npz,$E/logits_eval.npz) ;;
      esac
      python scripts/meta/train_meta_variants.py --seed $s --multipliers $m --drop-blocks l3_raw,cereal \n        --proba-dir artifacts/pruned_proba_cache_$arm$sfx \n        --extra pheno=artifacts/phenology_train.npz,artifacts/phenology_eval.npz "${extra[@]}"
    done
  done
done
P=artifacts/pruned_proba_cache_
ARMS="--arm final=${P}noexpens,${P}noexpens_seed1,${P}noexpens_seed2 --arm none=${P}noexpnone,${P}noexpnone_seed1,${P}noexpnone_seed2
  --arm l1cnet=${P}noexpl1cens,${P}noexpl1cens_seed1,${P}noexpl1cens_seed2 --arm l2aall=${P}noexpallens,${P}noexpallens_seed1,${P}noexpallens_seed2"
PAIRS="--pair final:none --pair final:l1cnet --pair final:l2aall --pair l2aall:l1cnet --pair l1cnet:none"
python scripts/meta/evaluate_meta_protocol.py $ARMS $PAIRS \n  --output-log artifacts/meta_protocol_table2_final.log --output-json artifacts/meta_protocol_table2_final.json
python scripts/meta/recall_at_accuracy_floor.py $ARMS $PAIRS --output-log artifacts/table2_final_at81.log
python scripts/meta/paired_ap_ci.py ${P}noexpens ${P}noexpnone   # likewise for the other pairs
# end-to-end baselines with the same decision dial (unweighted, orchards x10, capped sqrt-inverse)
for s in 42 1 2; do
  sfx=$([ $s = 42 ] || echo _seed$s)
  python scripts/networks/network_as_arm.py artifacts/checkpoints/msresnet_l2a_clear$sfx/logits_eval.npz artifacts/pruned_proba_cache_l2anetdial$sfx
  for w in "orchw10 --class-weights orchard --orchard-weight 10" "sqrtinv --class-weights sqrt_inv --weight-cap 10"; do
    set -- $w; n=$1; shift; CKW=artifacts/checkpoints/msresnet_l2a_clear_$n$sfx
    python scripts/networks/train_series_classifier.py --dates clear --seed $s --out-dir $CKW "$@" --time-budget-min 600
    python scripts/networks/train_series_classifier.py --dates clear --seed $s --out-dir $CKW "$@" --export eval --align-cache artifacts/cached_logits_eval.pt
    python scripts/networks/network_as_arm.py $CKW/logits_eval.npz artifacts/pruned_proba_cache_${n}dial$sfx
  done
done
# then evaluate_meta_protocol.py with --arm final=... --arm l2anetdial=... --arm orchw10dial=... --arm sqrtinvdial=...
# --pair final:<arm>, and scripts/meta/paired_ap_ci.py artifacts/pruned_proba_cache_noexpens artifacts/pruned_proba_cache_<arm>
```

Region rotation (design and checks pre-registered in `FROZEN_DESIGN.md`):
`artifacts/rotation/run_rotation.sh` runs `scripts/rotation/region_rotation.py` (`prep`, then `meta`,
`oof` and `verify` per test region) and `evaluate_meta_protocol.py` on each region's spatial
folds; `meta` and `oof` skip outputs that already exist. The orchard-AP intervals for checks
1–2 compare arm B with arms A and D per test region:

```bash
for r in frh01 frh02 frh03 frh04; do for arm in A D; do
  python scripts/meta/paired_ap_ci.py artifacts/rotation/$r/proba_B artifacts/rotation/$r/proba_$arm \
    --folds artifacts/rotation/folds_$r.npz --labels artifacts/rotation/ids_$r.pt
done; done
```

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

On Windows, if not installed editable, set `$env:PYTHONPATH = "src"` before running a script.

## Project layout

```text
.
├── src/agricultureai/
│   ├── data/              # BreizhCrops loaders, label maps, episodic L3 sampling, cloud mask
│   ├── models/            # Transformer, LTAE, GL-TAE, MSResNet, ProtoNet
│   ├── training/          # trainers (generic + L2b/L2c experts), checkpoints, losses
│   ├── evaluation/        # selection protocol, spatial folds + decode grid, block bootstrap, confusion metrics
│   ├── pipeline.py        # DMOEPipeline + decoding of cached branch outputs
│   ├── meta_features.py   # feature layouts (66-d full, 39-d compact), XGBoost / CatBoost, blends
│   ├── meta_training.py   # meta-classifier feature loading and fitting
│   ├── verifier.py        # orchard verifier: features, candidates, model, decoding
│   ├── cache_alignment.py # align outputs across BreizhCrops levels by parcel id
│   └── cache_io.py        # verified backups and atomic saves for patching the logit caches
├── scripts/               # entry points by stage: data, cascade, networks, meta, rotation
├── tests/
├── paper/                 # paper (.docx), its figures and the scripts that draw them
├── logs/                 # the logs behind every number in the paper (index: logs/README.md)
├── FROZEN_DESIGN.md       # pre-registered final design, checks and results
├── CITATION.cff           # citation metadata (GitHub's "Cite this repository")
└── artifacts/             # checkpoints, caches, logs (gitignored)
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

## How to cite

If you use this code, please cite the paper:

```bibtex
@unpublished{bodea2026orchards,
  author = {Bodea, Tudor},
  title  = {Orchards in the Meadow: Seasonal Evidence and a Two-Stage Orchard Verifier for Rare-Class Crop Mapping from Sentinel-2 Time Series},
  year   = {2026},
  note   = {Manuscript. Code: https://github.com/Tudor810/orchards-in-the-meadow}
}
```

and the BreizhCrops dataset:

```bibtex
@article{russwurm2020breizhcrops,
  author  = {Ru{\ss}wurm, Marc and Pelletier, Charlotte and Zollner, Maximilian and Lef{\`e}vre, S{\'e}bastien and K{\"o}rner, Marco},
  title   = {{BreizhCrops}: A Time Series Dataset for Crop Type Mapping},
  journal = {International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
  volume  = {XLIII-B2-2020},
  pages   = {1545--1551},
  year    = {2020},
  doi     = {10.5194/isprs-archives-XLIII-B2-2020-1545-2020}
}
```

GitHub's "Cite this repository" button reads the same information from `CITATION.cff`.
