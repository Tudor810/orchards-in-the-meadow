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
