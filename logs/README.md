# Logs behind the paper

Copies of the experiment outputs that every number in the paper comes from, plus every log the repository
README cites. Paths mirror `artifacts/`, where the scripts write them: `logs/X` is a copy of `artifacts/X`.
The logs are kept as the runs produced them; a few name arms by their working titles (below).

Table and figure numbers refer to the paper. "Means of three seeds" in the paper are means of the per-seed
rows printed here.

## Data

| File | Backs |
| --- | --- |
| `dataset_summary.log` | Section 3.1 and Table 1: parcels and classes per region and split, series lengths (median dates, share under 45 dates), clear dates, parcels without an L1C series, raw L2A observations with a non-zero cloud fraction (60–68%); orchards per region and per rotation fold (Section 4.7). Written by `scripts/data/dataset_summary.py --raw` |

## Comparison with other systems (Table 2, Section 4.3)

| File | Backs |
| --- | --- |
| `meta_protocol_final.log`, `.json` | Final meta-classifier at the three selection rules and its orchard AP, 35.3 (arm `final`; `noexperts` = one network instead of the three-seed average); naive vs cross-fitted selection (Section 4.2); pretrained MSResNet reference (80.77%) |
| `orchard_verifier_final/verify_seed{42,1,2}.log`, `.json` | Meta + verifier row, orchard AP 39.1; Table 4 |
| `model_metrics_frh04.log`, `.json` | Paired rows on the 122,557 parcels with an L1C series: pretrained MSResNet (8-view inputs), meta-classifier, final system; confusion matrices and micro / macro scores |
| `fig2_ap.log` | Orchard AP of the pretrained MSResNet (18.8), meta-classifier and verifier on those parcels; Fig. 2c |
| `run_cascade_eval.log` | Four-tier cascade row (79.65%, orchard recall 21.74%); cascade cereal accuracy 95.0% with eight views vs 86.5% zero-padded (Section 4.4) |
| `meta_vs_l2a_network_3seeds.log` | L2A MSResNet on its own vs the meta-classifier |
| `meta_protocol_l2anet_dial.log` | L2A MSResNet with the same decoding dial (arm `l2anetdial`), paired against the final system |
| `meta_protocol_e2e.log`, `e2e/paired_ap_ci.log` | Class-weighted L2A MSResNets (`orchw10*`: orchards ×10; `sqrtinv*`: inverse square-root frequency; `*dial` = same dial, `*adj` = logit-adjusted) and their orchard AP against the final system |

## Ablations (Table 3, Section 4.4)

| File | Backs |
| --- | --- |
| `meta_protocol_table2_final.log`, `.json` | Table 3 accuracy-first columns, final configuration: the final design (`final`) with its network block removed (`none`) or replaced by the 3-seed L1C average (`l1cnet`) or the 3-seed all-dates L2A average (`l2aall`) |
| `table2_final_at81.log` | Table 3 recall at accuracy ≥ 81% and orchard AP, with paired intervals (`scripts/meta/recall_at_accuracy_floor.py`) |
| `table2_final_ap_ci.log` | Table 3 orchard AP differences, paired intervals (`paired_ap_ci.py`; `noexpens` = final, `noexpnone`, `noexpl1cens`, `noexpallens`) |
| `meta_protocol_cloud_3seeds.log`, `meta_protocol_l1c_control.log`, `l1c_control_eval.log`, `cloud_ap_ci81_3seeds.log`, `max_orchard_at81_3seeds.log` | The same comparison in the earlier configuration (expert blocks included, one network per seed; README); the L2A and L1C networks on their own, and clear vs all dates for the network alone |
| `meta_protocol_frh04.log` | Eight-view inputs vs zero-padded inputs: +0.36 to +0.50 accuracy (`msfix` vs `aligned`) |
| `meta_protocol_pheno.log` | Phenology, added before the L2A network: +8–10 orchard recall at the accuracy-first and orchard-first points, orchard AP 22.4 → 26.5 (`pheno` vs `msfix`) |
| `meta_protocol_noexperts.log` | Dropping the expert blocks (`noexperts`) |
| `meta_protocol_checks245.log` | Dropping the pretrained L1C networks (`l2anet` vs `noexperts`); three-network ensemble (`noexpens`) |
| `meta_protocol_indices.log` | Vegetation indices as network inputs |
| `check3_decode_phenomsclear.log` | Decoding choices: CatBoost blend, orchard multiplier `m`, biases |

## Stacking leakage (Table 5, Section 4.6)

| File | Backs |
| --- | --- |
| `meta_protocol_leakage_final.log`, `.json` | Final design trained on frh01–frh03 vs on out-of-fold frh04 outputs (`final` vs `finaloof`) |
| `leakage/leakage_ap_ci.log` | Orchard AP of the two, paired intervals |

## Region rotation (Table 6, Fig. 3, Section 4.7)

| File | Backs |
| --- | --- |
| `rotation/<region>/protocol_B.log` | Rotation-safe design (arm B) with each region held out |
| `rotation/<region>/protocol_ablation.log` | Arms A (no L2A network) and D (no phenology) |
| `rotation/<region>/verify[_seed<s>].json` | Verifier per held-out region and seed |
| `rotation/rotation_ap_ci.log` | Orchard AP intervals for checks 1–2 (B vs A, B vs D) |
| `rotation/run.log` | Driver log of the whole rotation, including verifier AP intervals (check 3) |
| `rotation/diagnosis.log`, `verifier_diagnosis.json` | Post-hoc analysis of the verifier on frh03 |

## Cited in the README only

| File | Backs |
| --- | --- |
| `model_metrics_cascade_frh04.log` | Cascade decoded from the cached outputs (79.66% / 21.2%), vs the live run in `run_cascade_eval.log` |
| `ablate_meta_layouts_msfix.log` | Feature-layout ablation (full / compact / minus-block layouts) |

Interval resamples: 2,000 for accuracy, recall and precision (`evaluate_meta_protocol.py`, recorded as
`n_boot` in the `.json` files), 1,000 for orchard AP.
