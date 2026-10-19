# Frozen design (2026-10-09)

Frozen before any region-rotation result was seen. Every choice below was made on frh04
(design-selection region); the rotation tests whether the choices transfer.

## Final pipeline (frh04 numbers: `artifacts/meta_protocol_final.log`, `artifacts/orchard_verifier_final/`)

1. Inputs: BreizhCrops L2A series (10 bands), parcel metadata (8), 15 phenology features
   (`compute_phenology_features.py`, clear dates).
2. Networks:
   - pretrained BreizhCrops MSResNet + Transformer (L1C), TTA-8 logits (compact blocks `msresnet`, `transformer`);
   - MSResNet trained from scratch on L2A clear dates (`train_series_classifier.py --dates clear`,
     10 epochs, BatchNorm recalibration, TTA-8), logits averaged over seeds 42, 1, 2 (`msresnet_l2a_clear_ens3`).
3. Meta-classifier: XGBoost + CatBoost (`train_meta_variants.py`, compact layout with
   `--drop-blocks l3_raw,cereal`), orchard weight m in {1, 2, 3}, blends geo_mean / arith_xgb0.7,
   orchard / meadow log-biases chosen out of sample (`evaluate_meta_protocol.py`).
4. Orchard verifier (`orchard_verifier.py`): candidates = meta orchard score above the threshold keeping
   90% of out-of-fold training orchards; binary XGBoost on meta scores + phenology + monthly clear-date
   means + geometry + neighbour context.
5. Dropped after ablation: four-tier expert hierarchy (L1/L2a/L2b/L2c/L3), vegetation-index network
   inputs, flat verifier features, cloud-screening as a separate decision (clear dates kept).

frh04 (3 seeds): accuracy-first 82.67% / orchard recall 48.7% / precision 21.7%; orchard-first 80.78 / 77.8;
closest to 81/60: 82.18 / 60.4; orchard AP 35.3; with verifier AP 39.1, high-precision point
82.90% / R 36.4 / P 45.5.

## Rotation-safe variant (used for the region rotation)

The pretrained BreizhCrops networks were trained on frh01-frh03, so they have seen any rotated test
region. In the rotation they are replaced by our own L1C MSResNet (`--dates all`, L1C store, same trainer),
trained per fold on the three training regions. Everything else is identical, except one network seed
(42) per fold instead of a 3-seed average (compute budget). The variant is run on all four regions,
including frh04, so folds are comparable.

Per fold (test region R, train = the other three), meta seeds 42, 1, 2:
- **B (rotation-safe design)**: metadata + phenology + L1C net + L2A net, m in {1, 2, 3}, full decoding grid.
- **A (no L2A net)**: metadata + phenology + L1C net, m = 1.
- **D (no phenology)**: metadata + L1C net + L2A net, m = 1.
- **C (B + verifier)**: verifier on B's m = 1 XGBoost scores.

## Pre-registered checks (answered per fold, reported whatever the outcome)

1. The L2A network helps: B - A (m = 1 grid) orchard AP > 0 and orchard recall at the orchard-first point > 0,
   intervals excluding zero on most seeds.
2. Phenology helps: B - D (m = 1 grid) orchard recall > 0 at accuracy-first and orchard-first.
3. The verifier helps: C - B orchard AP > 0, intervals excluding zero.
4. Absolute level: accuracy-first accuracy and orchard recall per region, compared with frh04.

Evaluation per region: 10 km spatial blocks in 5 folds (decode settings chosen out of sample),
95% block-bootstrap intervals, paired across arms.

## Rotation results (rotation finished 2026-10-15; scored 2026-10-19)

Appended after the fact; nothing above was changed. Logs: `artifacts/rotation/<R>/protocol_{B,ablation}.log`,
`verify[_seed<s>].json`, `run.log`; AP intervals for checks 1-2 in `artifacts/rotation/rotation_ap_ci.log` (`scripts/rotation_ap_ci.py`)
(same paired block bootstrap as the verifier: 1000 resamples of the 10 km blocks). Ranges are over the
three meta seeds; m = 1 grid unless noted.

| Region | Parcels | Orchards | Acc-first acc / orch R / P (B) | Orchard-first acc / R (B) | AP B | AP A | AP D | AP C (verifier) |
|---|---|---|---|---|---|---|---|---|
| frh01 | 178,632 | 937 | 83.52 / 51.5 / 22.2 | 80.83 / 80.7 | 40.0 | 26.9 | 36.4 | 39.0 -> 42.5 |
| frh02 | 140,782 | 348 | 82.72 / 52.6 / 9.2 | 80.93 / 71.3 | 29.4 | 24.2 | 26.1 | 29.3 -> 33.8 |
| frh03 | 166,367 | 1,217 | 80.13 / 67.2 / 18.3 | (floor not reached) | 43.4 | 28.8 | 37.7 | 42.8 -> 42.4 |
| frh04 | 122,708 | 552 | 82.40 / 50.9 / 17.8 | 80.77 / 75.2 | 32.5 | 21.3 | 28.7 | 31.9 -> 36.2 |

AP B / A / D = geo_mean blend of arm B / A / D; AP C = verifier on arm B's m = 1 XGBoost score
(its own baseline on the left of the arrow).

1. **L2A network helps: holds in all four regions.** B - A orchard AP +5.0 to +14.9 (smallest frh02,
   largest frh03), orchard-first recall +5.7 to +19.9; all 12 intervals of each exclude zero.
2. **Phenology helps: holds in all four regions.** B - D orchard recall > 0 at both operating points for
   every region and seed (+0.6 to +7.1); intervals exclude zero in 21 of 24 (exceptions: frh01
   accuracy-first seed 2, frh02 orchard-first seeds 42 and 2). B - D orchard AP +3.0 to +5.9, all 12
   intervals exclude zero.
3. **Verifier helps: holds in frh01, frh02 and frh04, fails in frh03.** C - B orchard AP +3.0 to +4.8 in
   the three regions (9/9 intervals exclude zero; accuracy-first precision 44 -> 50, 20 -> 32, 28 -> 42).
   In frh03: -1.1, -0.8, +0.7, every interval spans zero, and accuracy-first precision falls 52 -> 47.
   frh03 has the most orchards (0.73% of parcels) and the strongest meta score already (AP 42.8).
4. **Absolute level.** Accuracy-first accuracy 80.1-83.5%, orchard recall 51-67%; orchard precision
   is lowest where orchards are rarest (9% in frh02, 0.25% of parcels; 18-22% elsewhere). frh03 never reaches
   the 80.69% accuracy floor (calibrated on frh04), so all three rules pick the default decode there and
   it has no separate orchard-first point. The rotation-safe variant on frh04 (82.40 / 50.9, AP 32.5) sits
   below the full design (82.67 / 48.7, AP 35.3), as expected without the pretrained networks and with
   one L2A seed.

### Post-hoc analysis of check 3 (2026-10-19; does not change the design)

`scripts/diagnose_rotation_verifier.py` (log `artifacts/rotation/diagnosis.log`). frh03 holds 40% of all
orchards, so its fold trains on the fewest (1,837 vs 2,117-2,706) and gets the loosest candidate threshold
(tau 0.0055 vs 0.009-0.019; 18% of frh03 parcels become candidates instead of 8-10%). Setting the candidate
share to 9% in every fold removes the frh03 loss (AP -0.3 to +0.3) and keeps the gain in the other regions
(+3.0 to +4.4, every interval excluding zero), but creates no gain in frh03 (accuracy-first precision still
52 -> 47). Dropping the neighbour features (AP -1.8 to -0.2) or the monthly means (-0.8 to -0.1) does not help.
The threshold explains part of the frh03 shortfall, not all of it.
