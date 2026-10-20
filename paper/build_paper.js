// Builds the research-paper draft (.docx). Usage: node build_paper.js <out.docx> <figdir> [leakage.json]
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, BorderStyle, ShadingType, ImageRun, LevelFormat, PageNumber, Footer, TabStopType,
} = require("docx");

const [OUT, FIGDIR, LEAKFILE] = process.argv.slice(2);
const L = LEAKFILE && fs.existsSync(LEAKFILE) ? JSON.parse(fs.readFileSync(LEAKFILE, "utf8")) : null;
const PENDING = "[LEAKAGE RESULT PENDING]";
const leak = (k) => (L && L[k]) || PENDING;

const FONT = "Times New Roman";
const BODY = 22; // 11 pt
const CONTENT_W = 9026; // A4 minus 2 x 2.54 cm

// ---- inline markup: **bold**, *italic* ------------------------------------------------------
function runs(text, base = {}) {
  const out = [];
  for (const part of text.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/)) {
    if (!part) continue;
    if (part.startsWith("**")) out.push(new TextRun({ ...base, text: part.slice(2, -2), bold: true }));
    else if (part.startsWith("*")) out.push(new TextRun({ ...base, text: part.slice(1, -1), italics: true }));
    else out.push(new TextRun({ ...base, text: part }));
  }
  return out;
}
const P = (text, opts = {}) => new Paragraph({
  children: runs(text), alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 276 }, ...opts,
});
const SUMMARY = (text) => new Paragraph({
  children: [new TextRun({ text: "Summary. ", bold: true, italics: true }), ...runs(text, { italics: true })],
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 160, line: 276 },
  border: { left: { style: BorderStyle.SINGLE, size: 12, color: "A8541E", space: 8 } }, indent: { left: 200 },
});
const H1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, keepNext: true, children: [new TextRun(text)], spacing: { before: 280, after: 120 } });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, keepNext: true, children: [new TextRun(text)], spacing: { before: 200, after: 80 } });
const BULLET = (text) => new Paragraph({ children: runs(text), numbering: { reference: "bullets", level: 0 }, alignment: AlignmentType.JUSTIFIED, spacing: { after: 60, line: 276 } });
const NUM = (text, ref = "contrib") => new Paragraph({ children: runs(text), numbering: { reference: ref, level: 0 }, alignment: AlignmentType.JUSTIFIED, spacing: { after: 60, line: 276 } });
const CAPTION = (label, text) => new Paragraph({
  children: [new TextRun({ text: label + " ", bold: true, size: 20 }), ...runs(text, { size: 20 })],
  alignment: AlignmentType.JUSTIFIED, spacing: { before: 120, after: 120, line: 252 }, keepNext: true,
});
const EQ = (text) => new Paragraph({ children: runs(text, { italics: true }), alignment: AlignmentType.CENTER, spacing: { before: 80, after: 120 } });

function figure(file, widthIn, label, caption) {
  const buf = fs.readFileSync(path.join(FIGDIR, file));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  const px = Math.round(widthIn * 96);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 160 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: px, height: Math.round(px * h / w) } })] }),
    CAPTION(label, caption),
  ];
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(header, rows, widths, { boldRows = [], align = null } = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (t, i, isHead, bold) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA }, borders,
    shading: isHead ? { fill: "EDEDED", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({
      alignment: (align ? align[i] : (i === 0 ? "left" : "right")) === "left" ? AlignmentType.LEFT : AlignmentType.RIGHT,
      children: runs(t, { size: 18, bold: isHead || bold }),
    })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((t, i) => cell(t, i, true, false)) }),
      ...rows.map((r, ri) => new TableRow({ children: r.map((t, i) => cell(t, i, false, boldRows.includes(ri))) })),
    ],
  });
}
const spacer = () => new Paragraph({ children: [], spacing: { after: 80 } });

// ---- content ------------------------------------------------------------------------------------
const C = [];
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: [new TextRun({ text: "Orchards in the Meadow: Seasonal Evidence and a Two-Stage Orchard Verifier for Rare-Class Crop Mapping from Sentinel-2 Time Series", bold: true, size: 32 })] }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: [new TextRun({ text: "Tudor Bodea", size: 24 })] }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [new TextRun({ text: "[Affiliation] · [e-mail]", size: 20, italics: true })] }));

C.push(new Paragraph({ children: [new TextRun({ text: "Abstract", bold: true })], spacing: { after: 80 } }));
C.push(P("Crop-type classifiers trained on Sentinel-2 time series reach about 80% overall accuracy on the BreizhCrops benchmark, yet they miss most parcels of rare classes. Orchards make up 0.45% of the test region and are confused almost exclusively with meadows; a pretrained MSResNet finds 15.7% of them. We fuse three kinds of evidence in a gradient-boosted meta-classifier: phenology features computed from cloud-free dates, an MSResNet trained on atmospherically corrected (L2A) series, and the outputs of pretrained networks. A second-stage verifier then re-scores the meta-classifier's orchard candidates using seasonal, geometric and neighbourhood evidence. On the held-out region the meta-classifier reaches 82.67% accuracy at 48.7% orchard recall, or 77.8% orchard recall at 80.78% accuracy; the verifier raises orchard average precision from 34.9 to 39.1 and reaches 45.5% orchard precision at 82.90% accuracy. Controlled ablations show that the gain comes from the seasonal evidence and from the L2A input itself: an identical network trained on L1C data adds almost nothing, a four-tier mixture of experts adds little, and an end-to-end network trained with an orchard-weighted loss stays 7.5–10 points of orchard AP behind. In a pre-registered rotation over all four BreizhCrops regions, the phenology and L2A-network gains hold in every region, while the verifier helps in three of the four. " + leak("abstract") + " All results are out of sample, with spatially cross-fitted operating points and three seeds."));
C.push(P("**Keywords:** crop-type mapping; Sentinel-2; satellite image time series; rare classes; stacking; gradient boosting; BreizhCrops."));

// 1 Introduction
C.push(H1("1 Introduction"));
C.push(SUMMARY("Overall accuracy on crop-type benchmarks has levelled off near 80%, but rare classes remain poorly mapped. On BreizhCrops the rare-class problem is an orchard-versus-meadow problem. We address it with seasonal evidence fused late and a second look at the orchard candidates, which raises orchard recall from 15.7% to 48.7–77.8% and orchard precision at the high-accuracy end to 45.5%."));
C.push(P("Satellite image time series (SITS) from Sentinel-2 (Drusch et al., 2012) have made parcel-level crop-type mapping practical at national scale, and operational systems now produce such maps routinely (Inglada et al., 2015; Defourny et al., 2019). Deep temporal classifiers, whether convolutional (Pelletier et al., 2019), residual (Wang et al., 2017) or attention-based (Sainte Fare Garnot et al., 2020; Rußwurm and Körner, 2020), reach around 80% overall accuracy on the BreizhCrops benchmark (Rußwurm et al., 2020). Overall accuracy, however, is dominated by abundant classes; rare classes barely move it, yet they often matter most for policy: orchards, for example, are subsidised and inspected separately."));
C.push(P("We study this problem on BreizhCrops, which covers 608,489 field parcels in four regions of Brittany (France) for 2017. Following the standard split, we train on regions frh01–frh03 (485,781 parcels) and test on frh04 (122,708 parcels), with nine classes: barley, wheat, rapeseed, corn, sunflower, orchards, nuts, and permanent and temporary meadows. Only 552 test parcels (0.45%) are orchards. Orchards are the only rare class that can be evaluated: sunflower and nuts have 2 and 11 test parcels, too few to estimate recall or precision, so this paper is a study of orchards as a rare class. A pretrained MSResNet from the BreizhCrops model zoo reaches 80.77% accuracy but finds only 15.7% of them. An error analysis shows that the problem is narrow: 96% of the parcels falsely predicted as orchards are meadows, and 98% of the missed orchards are predicted as one of the two meadow classes. A plausible cause is that grassed orchards share the meadows' spring green-up; what should separate them is timing and management, since meadows are mown or grazed while tree canopies leaf out later and stay green into summer."));
C.push(P("This points to seasonal evidence rather than more model capacity. Our system has two stages (Fig. 1). A gradient-boosted meta-classifier (XGBoost and CatBoost) fuses 51 per-parcel features: parcel metadata, 15 phenology features computed from cloud-free dates, the logits of pretrained networks on top-of-atmosphere (L1C) data, and the logits of an MSResNet we train on atmospherically corrected (L2A) data. Decode-time log-biases choose the operating point. A binary verifier then re-scores the parcels the meta-classifier considers possible orchards, using monthly clear-date reflectances, parcel geometry and the predictions for neighbouring parcels. Our contributions are:"));
C.push(NUM("A late-fusion design in which phenology features and an L2A-trained network each add about ten points of orchard recall over pretrained networks. Controlled ablations show that the L2A input, not merely a second network, carries the gain, that a four-tier mixture of experts adds little, and that an end-to-end network with a class-weighted loss does not close the gap."));
C.push(NUM("A two-stage orchard verifier that raises orchard average precision by about four points and, at the high-accuracy operating point, raises orchard precision from 34.8% to 45.5% at the same recall."));
C.push(NUM("The diagnosis and fix of a train/test input mismatch in the reference evaluation path, using seeded test-time augmentation."));
C.push(NUM("An evaluation protocol with spatially cross-fitted operating points, block-bootstrap intervals, three seeds, a stacking-leakage test and a pre-registered rotation over all four regions."));

// 2 Related Work
C.push(H1("2 Related Work"));
C.push(SUMMARY("Deep temporal networks set the accuracy bar for satellite time series. We build on them with ideas from stacking, phenology and rare-class evaluation, and test a mixture of experts as an alternative. Per-class analyses of the rarest crops, with honest threshold selection, are scarce."));
C.push(H2("2.1 Satellite image time-series classification"));
C.push(P("TempCNN (Pelletier et al., 2019) showed that one-dimensional temporal convolutions are a strong baseline for pixel and parcel series. Residual networks (He et al., 2016), adapted to time series by Wang et al. (2017), and their multi-scale variant MSResNet (Wang, n.d.) are among the strongest BreizhCrops baselines. Attention-based encoders followed: the pixel-set encoder with temporal attention (Sainte Fare Garnot et al., 2020), its lightweight variant LTAE (Sainte Fare Garnot and Landrieu, 2020), and Transformer encoders (Vaswani et al., 2017) applied to raw optical series (Rußwurm and Körner, 2020). BreizhCrops (Rußwurm et al., 2020) provides L1C and L2A versions of Sentinel-2 parcel series, pretrained baselines and a reference transform that resamples every series to 45 dates. These works report overall accuracy and macro scores, rarely the rarest classes on their own."));
C.push(H2("2.2 Experts, hierarchies and prototypes"));
C.push(P("Mixtures of experts (Jacobs et al., 1991) and their hierarchical form (Jordan and Jacobs, 1994) route inputs to specialised models. Hierarchical classification more generally exploits the structure of the label space (Silla and Freitas, 2011), and Turkoglu et al. (2021) use multi-scale label hierarchies for crop mapping. Prototypical networks (Snell et al., 2017) classify by distance to class prototypes. Our starting point combined these ideas in a four-tier cascade, which we report as a baseline."));
C.push(H2("2.3 Stacking and gradient boosting"));
C.push(P("Stacked generalisation (Wolpert, 1992) trains a second-level learner on the outputs of first-level models. It is reliable only when the first-level outputs used for training behave like those at test time, which is not guaranteed when the first-level models were trained on the same data. Gradient-boosted trees such as XGBoost (Chen and Guestrin, 2016) and CatBoost (Prokhorenkova et al., 2018) are strong second-level learners for mixed tabular features."));
C.push(H2("2.4 Phenology and clouds"));
C.push(P("Phenological metrics such as those of TIMESAT (Jönsson and Eklundh, 2004) summarise the seasonal trajectory of vegetation indices and are a staple of operational crop mapping (Inglada et al., 2015; Defourny et al., 2019). Red-edge and moisture indices (Gitelson and Merzlyak, 1994; Gao, 1996) add information beyond NDVI. Optical series are affected by clouds: L2A products from Sen2Cor (Main-Knorn et al., 2017) carry cloud information, and Fmask (Zhu and Woodcock, 2012; Qiu et al., 2019) is a widely used alternative for cloud screening."));
C.push(H2("2.5 Evaluation under imbalance and spatial dependence"));
C.push(P("Accuracy is a poor summary under class imbalance (He and Garcia, 2009), and precision–recall analysis is more informative than ROC analysis for rare classes (Saito and Rehmsmeier, 2015). Spatially autocorrelated data inflate random cross-validation estimates (Roberts et al., 2017; Ploton et al., 2020), and block resampling (Künsch, 1989) gives more honest intervals. We are not aware of a BreizhCrops study that follows a rare class end to end, with controlled ablations, out-of-sample threshold selection and a rotation over regions; this is the gap we address."));

// 3 Proposed Solution
C.push(H1("3 Proposed Solution"));
C.push(SUMMARY("Four feature blocks (51 features) feed one gradient-boosted meta-classifier whose decoding biases choose the operating point; a binary verifier then re-checks its orchard candidates. Every network sees test series exactly as it saw training series."));
C.push(H2("3.1 Data"));
C.push(P("Each parcel is a multivariate time series of the 10 Sentinel-2 L2A bands with up to 45 acquisition dates; series lengths vary, and 85% of the test parcels have fewer than 45 dates (median 27). The L2A data include a per-date cloud fraction, which is above zero on 60–68% of the observations, so many dates are noisy. The pretrained BreizhCrops networks expect the 13-band L1C series; we align those to the L2A parcels by identifier. 151 of the 122,708 test parcels (and 394 training parcels) have no L1C series; their L1C-network features are set to missing."));
C.push(H2("3.2 Input consistency"));
C.push(P("The pretrained networks were trained through the dataset's reference transform, which resamples every series to exactly 45 dates and never zero-pads. Our first feature extraction truncated or zero-padded the series instead, an input the networks had never seen. We therefore feed every network the mean of its outputs over eight seeded, resampled 45-date views of the series, a form of test-time augmentation (Shanmugam et al., 2021). In the original cascade it raised the accuracy of the cereal expert alone (barley versus wheat on true cereal parcels) from 88.4% to 97.0%, and the cascade's end-to-end accuracy on all 22,990 barley and wheat test parcels, which also counts routing errors, from 86.5% to 95.0%. For the features of the final system, the fix to the pretrained MSResNet adds 0.36–0.50 points of final accuracy, with every interval excluding zero (Section 4.4)."));
C.push(H2("3.3 Feature blocks"));
C.push(P("The meta-classifier sees 51 features per parcel, in four blocks (Fig. 1):"));
C.push(BULLET("**Parcel metadata (8):** log area, maximum NDVI, NDVI amplitude, centroid latitude, mean and standard deviation of the normalised acquisition day of year, the day of year of the NDVI maximum, and the log number of valid observations."));
C.push(BULLET("**Phenology (15):** computed from cloud-free dates only: seasonal NDVI means and the winter minimum, NDVI spread, the summer-minus-winter contrast, mowing-like NDVI drops between April and September, spring and summer red-edge (NDRE) and moisture (NDMI) means, the number of clear dates, and parcel compactness. On frh04 the strongest single features separate orchards from meadows with a separability, max(AUROC, 1 − AUROC), of 0.74 (spring NDMI), 0.72 (spring NDVI) and 0.70 (largest summer NDVI drop); Fig. 2a, b shows the seasonal difference."));
C.push(BULLET("**Pretrained L1C networks (19):** the logits of the BreizhCrops MSResNet and Transformer (nine each) and one margin feature, extracted with eight-view test-time augmentation."));
C.push(BULLET("**L2A network (9):** the logits of a 9-class MSResNet that we train from scratch on the L2A series of frh01–frh03. It uses clear dates only, keeping a date when its cloud fraction is at most 0.1, its blue reflectance is below 0.12 and it is not an isolated NDVI dip; this keeps 32–40% of the dates. It is trained for ten epochs, after which its BatchNorm statistics (Ioffe and Szegedy, 2015) are recomputed in inference mode; without this step its accuracy is 58.7% instead of 80.2%. Its logits are averaged over three training seeds and extracted with eight-view test-time augmentation."));
C.push(H2("3.4 Meta-classifier"));
C.push(P("XGBoost and CatBoost are trained on the 51 features of the training regions, with orchard samples up-weighted by a factor m ∈ {1, 2, 3}. Their class probabilities are combined either by a geometric mean or by a weighted arithmetic mean (weight 0.7 on XGBoost). At decoding time, log-space biases favour orchards and penalise the two meadow classes:"));
C.push(EQ("ŷ = argmax_k [ log p_k + b_o · 1(k = orchard) − b_m · 1(k ∈ meadows) ],"));
C.push(P("with b_o ∈ {0, 0.5, …, 3} and b_m ∈ {0, 0.25, 0.5, 0.75, 1, 1.5, 2}. Together with the two blends and three multipliers this gives 294 decoding configurations; Section 4.2 describes how one is chosen without using the labels of the parcels it is scored on. Varying the configuration traces a family of operating points, from accuracy-first to orchard-first. The grid is an operating-point dial rather than a source of accuracy: at the accuracy-first point every fold of every seed selects the weighted blend with m = 1 and no biases."));
C.push(H2("3.5 Orchard verifier"));
C.push(P("The second stage re-examines the parcels the meta-classifier considers possible orchards. To train it on candidates that look like test candidates, we compute out-of-fold meta-classifier scores on the training regions: each training region is scored by a meta-classifier trained on the other two. The candidate threshold τ is the 10th percentile of these scores among training orchards, so 90% of training orchards become candidates. A binary XGBoost is trained on the training candidates (orchard versus rest) with 9 meta-classifier probabilities, the 15 phenology features, monthly means of the 10 bands and of NDVI, NDRE and NDMI over clear dates, parcel geometry (area, perimeter, compactness, elongation) and the context of the ten nearest parcels (mean and maximum orchard score, mean meadow score, mean distance). Parcels are then ranked with candidates first, ordered by the verifier's probability, followed by non-candidates, ordered by the meta orchard probability (in the implementation a candidate's score is 1 plus the verifier probability). A parcel is labelled an orchard if its score passes a threshold t chosen out of sample (Section 4.2); otherwise it receives the meta-classifier's most probable non-orchard class."));
C.push(...figure("fig1_pipeline.png", 6.2, "Fig. 1.", "The final pipeline. Four feature blocks (51 features) feed a gradient-boosted meta-classifier whose decode-time biases choose the operating point (output 1). Parcels whose meta orchard score passes the candidate threshold are re-scored by the orchard verifier (output 2)."));
C.push(H2("3.6 The four-tier cascade (baseline)"));
C.push(P("The project started from a hierarchical mixture of experts. A Transformer router assigns four super-classes; an LTAE expert separates barley from wheat; a GL-TAE anomaly gate sends grassland-like parcels to an MSResNet meadow expert or to a prototypical network with Mahalanobis distances for the rare classes. We evaluate this cascade on its own and as a fifth feature block of the meta-classifier (its L3 distances and cereal margins)."));

// 4 Experiments
C.push(H1("4 Experiments"));
C.push(SUMMARY("Against pretrained networks, end-to-end networks trained on L2A (also with class-weighted losses), the previous best meta-classifier and the four-tier cascade, the full system keeps the best accuracy and finds more orchards at every operating point. Phenology and the L2A network carry the gain in every region; an L1C network does not. The verifier raises orchard precision in three of four regions. " + leak("summary")));
C.push(H2("4.1 Setup and metrics"));
C.push(P("Unless stated otherwise, models are trained on frh01–frh03 and tested on frh04. We report overall accuracy and orchard precision, recall, F1 and average precision (AP, threshold-free, from the blended probabilities), plus micro and macro scores over the nine classes. Every result is the mean of three seeds (42, 1 and 2), where the seed sets both the L2A network and the meta-classifier. The meta-classifier uses XGBoost 3.2.0 and CatBoost 1.2.10; networks are implemented in PyTorch."));
C.push(H2("4.2 Protocol"));
C.push(P("Choosing a decoding configuration on the same parcels it is reported on is optimistic, and neighbouring parcels are correlated. We therefore cut the test region into 10 km blocks grouped into five spatial folds (Roberts et al., 2017); each fold is scored with the configuration chosen on the other four, and the five folds are concatenated. Three selection rules were fixed in advance: accuracy-first (maximum accuracy subject to orchard recall ≥ 35%), orchard-first (maximum orchard recall subject to accuracy ≥ 80.69%) and closest to a stretch target of 81% accuracy and 60% orchard recall. Intervals are 95% block-bootstrap intervals (Künsch, 1989) over the same blocks, with 2,000 resamples; differences between systems are paired, using the same resamples. Choosing settings directly on frh04 would have inflated accuracy by at most 0.15 points."));
C.push(H2("4.3 Comparison with other systems"));
C.push(P("Table 1 compares the final design with the pretrained BreizhCrops MSResNet, the previous best meta-classifier of this project (XGBoost with m = 25, chosen on frh04), the four-tier cascade and an MSResNet trained on our L2A data. The single networks reach at most 82.47% accuracy but find at most 17.2% of the orchards; the cascade finds 21.7% at 5.7% precision. The full system matches the L2A network's accuracy at the accuracy-first point (82.67%, interval 81.80–83.46 for seed 42) while finding 48.7% of the orchards (43.0–54.1), and finds up to 77.8% at 80.78% accuracy. The verifier gives the best accuracy in the table, 82.90%, with 45.5% orchard precision. Given the same decision dial, the L2A network reaches 37.0% orchard recall at 82.24% accuracy and, trained with orchards weighted ten times, 45.4% at 81.69%; the full system still leads (Section 4.4)."));
C.push(CAPTION("Table 1.", "Comparison on frh04 (means of three seeds; accuracy-first unless stated). * Evaluated on the L1C test set (122,614 parcels); the previous best also chose its settings on frh04. † Paired with our systems on the 122,557 parcels that have an L1C series. The cascade uses eight-view inputs. ‡ Same decoding biases and out-of-sample selection as the full system."));
C.push(table(
  ["System", "Accuracy", "Orch. precision", "Orch. recall", "Orch. F1"],
  [
    ["Pretrained MSResNet *", "80.77%", "–", "15.7%", "–"],
    ["Pretrained MSResNet, 8-view inputs †", "81.15%", "36.3%", "14.8%", "21.0%"],
    ["Previous best: XGBoost, m = 25 *", "80.76%", "–", "35.8%", "–"],
    ["Four-tier cascade", "79.65%", "5.7%", "21.7%", "9.1%"],
    ["MSResNet trained on L2A", "82.47%", "54.9%", "17.2%", "26.2%"],
    ["MSResNet on L2A + decision dial ‡", "82.24%", "20.7%", "37.0%", "26.6%"],
    ["MSResNet on L2A, orchards ×10 + dial ‡", "81.69%", "15.5%", "45.4%", "23.1%"],
    ["Meta + phenology", "82.01%", "15.3%", "45.5%", "22.9%"],
    ["Full system, accuracy-first", "82.67%", "21.7%", "48.7%", "30.0%"],
    ["Full system, closest to 81% / 60%", "82.18%", "13.6%", "60.4%", "22.2%"],
    ["Full system, orchard-first", "80.78%", "7.8%", "77.8%", "14.1%"],
    ["Full system + verifier", "82.90%", "45.5%", "36.4%", "40.5%"],
  ],
  [3826, 1300, 1300, 1300, 1300],
  { boldRows: [8, 9, 10, 11] },
));
C.push(spacer());
C.push(P("Paired on the same 122,557 parcels, micro precision, recall and F1 (equal to accuracy) rise from 81.15 for the pretrained MSResNet to 82.67 for the meta-classifier and 82.90 with the verifier; macro F1 rises from 59.43 to 62.12 and 63.33, and orchard F1 from 21.0 to 30.1 and 40.6. Macro scores are capped by sunflower and nuts (2 and 11 test parcels), which no system recognises."));
C.push(H2("4.4 Ablations"));
C.push(P("Table 2 isolates the network block on top of meta-classifier + phenology (in this earlier configuration the expert blocks are still included and each seed uses one network). An MSResNet trained on L1C data, the same architecture, bands and seeds, adds nothing measurable: orchard AP 26.8 against 26.5. The same network trained on L2A adds 6 points of orchard F1 at the accuracy-first point, 10–13 points of orchard recall at an 81% accuracy floor and 8 points of orchard AP; it beats the L1C network by 9–11 points of recall at the floor, with every interval excluding zero. The gain therefore comes from the corrected L2A input, not from adding a second network. Cloud screening gives the network alone 4 more points of orchard recall, but after fusion the clear-date and all-date versions are indistinguishable."));
C.push(CAPTION("Table 2.", "Effect of the extra network block on top of meta + phenology (accuracy-first point; orchard recall also at accuracy ≥ 81%; means of three seeds)."));
C.push(table(
  ["Extra block", "Acc.", "Orch. P", "Orch. R", "Orch. F1", "R at acc ≥ 81%", "Orch. AP"],
  [
    ["None", "82.01", "15.3", "45.5", "22.9", "63.7", "26.5"],
    ["Network on L1C", "82.12", "15.4", "43.5", "22.8", "64.5", "26.8"],
    ["Network on L2A, all dates", "82.66", "20.1", "48.1", "28.4", "74.3", "33.3"],
    ["Network on L2A, clear dates", "82.63", "20.6", "49.8", "29.2", "75.2", "34.5"],
  ],
  [2726, 1000, 1000, 1000, 1000, 1300, 1000],
));
C.push(spacer());
C.push(P("The remaining ablations, all paired over three seeds, explain the other design choices:"));
C.push(BULLET("**Phenology** adds 8–10 points of orchard recall on every seed and raises orchard AP from 22 to 26.5."));
C.push(BULLET("**Expert blocks.** Removing the cascade's L3 distances and cereal margins changes accuracy by at most 0.15 points and orchard AP from 34.5 to 34.0; accuracy-first orchard recall changes by −2.4, 0.0 and −3.4 points across seeds. They are left out of the final design."));
C.push(BULLET("**Network seeds.** Averaging three L2A networks raises orchard AP from 34.0 to 35.3."));
C.push(BULLET("**Pretrained L1C networks** are still needed: without them accuracy falls by 0.7 points and orchard AP from 34.0 to 31.4."));
C.push(BULLET("**Vegetation indices** as extra L2A-network inputs give no gain (orchard AP 33.3 against 34.0)."));
C.push(BULLET("**Input consistency.** Resampled instead of zero-padded inputs raise the cascade's end-to-end cereal accuracy from 86.5% to 95.0% (Section 3.2) and the final accuracy by 0.36–0.50 points."));
C.push(BULLET("**Decoding.** The CatBoost blend matters: XGBoost alone loses 4–7 points of orchard recall at the accuracy-first point."));
C.push(P("**Is the complexity necessary?** Every component kept in the final design earns its place in one of the paired ablations above, including the verifier (+4 AP, Section 4.5); the components that did not, the expert blocks and the vegetation-index inputs, were removed. The strongest end-to-end alternative is the same MSResNet trained on L2A with a class-weighted loss (orchards ×10, or square-root inverse frequency capped at 10) and given the same decision dial. Weighting shifts the network's operating point but hardly changes how it ranks orchards: orchard AP is 25.2–27.7 against 26.1 without weighting (paired differences −0.8 to +1.6, every interval spanning zero), while the full system reaches 35.3 (+7.5 to +10.2, every interval excluding zero). The full system is ahead at every operating point: by 0.6–1.2 points of accuracy and 3.5–7.5 points of orchard F1 at the accuracy-first point, by 14.5–18.5 points of orchard recall at the orchard-first point (77.8% against at most 62%), and by 1.1–1.7 points of accuracy at the closest-to-stretch point; removing the weighting's prior shift from the network's outputs changes none of this."));
C.push(P("**Cost.** Everything runs on one laptop CPU (AMD Ryzen 7 8840HS, no GPU). Training the L2A network takes about one hour per seed, and eight-view inference on the 122,708 test parcels about three minutes. The fusion stages add about 45 minutes per seed: 10–15 minutes for each of the three meta-classifier fits (one per orchard weight; the accuracy-first point uses only m = 1), about 5 minutes for the out-of-fold fits and 2 minutes for the verifier. That is less than the hour of network training an end-to-end model also needs."));
C.push(H2("4.5 Orchard verifier"));
C.push(P("Table 3 compares the verifier with thresholding the meta-classifier's orchard score alone. Orchard AP rises from 34.9 to 39.1, by 3.8 to 4.3 points on every seed, with 95% intervals excluding zero. At the accuracy-first point, precision rises from 34.8% to 45.5% at the same recall (36.3% against 36.4%) while accuracy edges up from 82.83% to 82.90%. At the 81% accuracy floor the verifier changes little (recall 74.5% against 76.8%, precision 8.4% against 8.6%). Single feature groups add at most 1.2 points of AP (monthly means); the gain needs all of them together. Fig. 2c shows the full precision–recall curves: the verifier lifts precision between about 10% and 60% recall and leaves the high-recall end unchanged."));
C.push(CAPTION("Table 3.", "Orchard verifier on top of the final meta-classifier (frh04, means of three seeds). Orchard is predicted above a threshold chosen out of sample, otherwise the meta label."));
C.push(table(
  ["Orchard decision", "Rule", "Acc.", "Orch. P", "Orch. R", "Orch. F1", "Orch. AP"],
  [
    ["Meta score alone", "accuracy-first", "82.83%", "34.8%", "36.3%", "35.5%", "34.9"],
    ["Meta + verifier", "accuracy-first", "82.90%", "45.5%", "36.4%", "40.5%", "39.1"],
    ["Meta score alone", "accuracy ≥ 81%", "81.02%", "8.4%", "74.5%", "15.0%", "34.9"],
    ["Meta + verifier", "accuracy ≥ 81%", "81.01%", "8.6%", "76.8%", "15.5%", "39.1"],
  ],
  [2026, 1700, 1060, 1060, 1060, 1060, 1060],
  { align: ["left", "left", "right", "right", "right", "right", "right"] },
));
C.push(spacer());
C.push(...figure("fig2_evidence.png", 6.4, "Fig. 2.", "(a, b) Monthly medians (lines) and interquartile ranges (bands) of clear-date NDVI and NDMI for orchards and the two meadow classes in the training regions frh01–frh03. Orchards are less green and drier than both meadow classes from winter to spring and in autumn, while temporary meadows drop in summer (mowing or grazing). (c) Orchard precision–recall curves on frh04 for the pretrained MSResNet, the meta-classifier and the verifier (seed 42; the 122,557 parcels with an L1C series)."));
C.push(H2("4.6 Stacking leakage"));
C.push(P("The pretrained networks and the L2A network were trained on frh01–frh03, so the features on which the meta-classifier is trained are in-sample, while at test time they are not. To test whether this matters, we train the meta-classifier on frh04 itself, cross-fitted over its spatial folds, so that every network output it learns from is out of sample; the price is five times less training data, all from the test region. For the earlier feature set this changed nothing measurable (seed 42: −0.15 to +0.04 points of accuracy, all intervals spanning zero). " + leak("section")));
C.push(CAPTION("Table 4.", "Stacking-leakage test on frh04 (means of three seeds): meta-classifier trained on frh01–frh03, whose network outputs are in-sample, against one trained on frh04's own spatial folds, whose outputs are out of sample. Orchard AP: 35.3 against 34.5 (paired differences −0.7 to −1.0, every interval spanning zero)."));
C.push(table(
  ["Operating point", "Trained on", "Acc.", "Orch. P", "Orch. R"],
  [
    ["Accuracy-first", "frh01–frh03", "82.67%", "21.7%", "48.7%"],
    ["Accuracy-first", "frh04, out-of-fold", "82.69%", "30.7%", "37.5%"],
    ["Closest to 81% / 60%", "frh01–frh03", "82.18%", "13.6%", "60.4%"],
    ["Closest to 81% / 60%", "frh04, out-of-fold", "82.16%", "16.1%", "60.5%"],
    ["Orchard-first", "frh01–frh03", "80.78%", "7.8%", "77.8%"],
    ["Orchard-first", "frh04, out-of-fold", "80.83%", "8.0%", "75.7%"],
  ],
  [2626, 2400, 1333, 1333, 1334],
  { align: ["left", "left", "right", "right", "right"] },
));
C.push(spacer());
C.push(H2("4.7 Region rotation"));
C.push(P("Every design choice above was made on frh04. We therefore froze the design and wrote down four checks before rotating the test region over all four regions, each time training on the other three. Because the pretrained BreizhCrops networks were trained on frh01–frh03, the rotation uses a variant that replaces them with our own L1C MSResNet trained per fold, and one L2A network seed per fold. The arms are B (metadata + phenology + L1C network + L2A network), A (B without the L2A network), D (B without phenology) and C (B with the verifier). Table 5 and Fig. 3 give the results."));
C.push(CAPTION("Table 5.", "Pre-registered region rotation (rotation-safe variant; means of three meta seeds). Accuracy, orchard recall and precision at the accuracy-first point; orchard AP of the variant, without the L2A network and without phenology; verifier AP as meta score alone → with the verifier."));
C.push(table(
  ["Test region", "Orchards", "Acc.", "Orch. R", "Orch. P", "AP", "AP, no L2A net", "AP, no pheno.", "Verifier AP"],
  [
    ["frh01", "937", "83.52%", "51.5%", "22.2%", "40.0", "26.9", "36.4", "39.0 → 42.5"],
    ["frh02", "348", "82.72%", "52.6%", "9.2%", "29.4", "24.2", "26.1", "29.3 → 33.8"],
    ["frh03", "1,217", "80.13%", "67.2%", "18.3%", "43.4", "28.8", "37.7", "42.8 → 42.4"],
    ["frh04", "552", "82.40%", "50.9%", "17.8%", "32.5", "21.3", "28.7", "31.9 → 36.2"],
  ],
  [866, 860, 900, 860, 860, 760, 1060, 1000, 1300],
  { align: ["left", "right", "right", "right", "right", "right", "right", "right", "right"] },
));
C.push(spacer());
C.push(BULLET("**Check 1, the L2A network helps: holds in all four regions.** Orchard AP rises by 5.0 to 14.9 points and orchard-first recall by 5.7 to 19.9 points; all twelve intervals of each exclude zero."));
C.push(BULLET("**Check 2, phenology helps: holds in all four regions.** Orchard recall rises at both operating points for every region and seed (by 0.6 to 7.1 points; 21 of 24 intervals exclude zero), and orchard AP by 3.0 to 5.9 points, every interval excluding zero."));
C.push(BULLET("**Check 3, the verifier helps: holds in three regions, fails in frh03.** Orchard AP rises by 3.0 to 4.8 points in frh01, frh02 and frh04 (all nine intervals exclude zero; accuracy-first precision 44% → 50%, 20% → 32% and 28% → 42%). In frh03 the change is −1.1, −0.8 and +0.7 points, every interval spans zero, and accuracy-first precision falls from 52% to 47%. frh03 has the most orchards (0.73% of parcels) and the highest meta-classifier AP before verification, partly because AP rises with prevalence."));
C.push(BULLET("**Check 4, absolute level.** Accuracy is 80.1–83.5% at 51–67% orchard recall. Precision is lowest where orchards are rarest (9% in frh02). frh03 never reaches the 80.69% accuracy floor, so its operating points coincide."));
C.push(P("**Why the verifier fails in frh03 (post-hoc analysis).** frh03 holds 40% of all BreizhCrops orchards (1,217 of 3,054), so its fold trains on the fewest (1,837, against 2,117–2,706). The out-of-fold orchard scores are correspondingly low and the candidate threshold is the loosest of all folds (τ = 0.0055, against 0.009–0.019): 18% of frh03's parcels become candidates, against 8–10% elsewhere. Setting the candidate share to 9% in every fold removes the small loss in frh03 (AP change −0.3 to +0.3) and keeps the gain elsewhere (+3.0 to +4.4, every interval excluding zero), but creates no gain in frh03, where accuracy-first precision still falls from 52% to 47%; removing the neighbour features or the monthly means does not help either. The threshold is therefore part of the problem but not all of it: trained on the fewest orchards, the verifier adds nothing the meta-classifier does not already know in the region where orchards are commonest. These checks were run after the rotation and do not change the frozen design."));
C.push(...figure("fig3_rotation.png", 3.3, "Fig. 3.", "Gains in orchard AP in the region rotation (seed 42; dots are estimates, bars 95% paired block-bootstrap intervals). Removing the L2A network or the phenology features costs AP in every region; the verifier helps in three regions and is neutral in frh03."));

// 5 Conclusion
C.push(H1("5 Conclusion"));
C.push(SUMMARY("Seasonal evidence on corrected (L2A) data moves the rare class in every region, a second look at the candidates raises precision in three of four regions, and a single year of data is the main limit."));
C.push(P("On BreizhCrops, mapping orchards is an orchard-versus-meadow problem, and it responds to seasonal evidence rather than to extra model capacity. Phenology features from cloud-free dates and an MSResNet trained on L2A series, fused late with pretrained networks, lift orchard recall from 15.7% to 48.7–77.8% at 80.8–82.7% accuracy. The input level matters: the same network trained on L1C data adds almost nothing. A hierarchy of experts adds little once these blocks are in, and an end-to-end network trained with an orchard-weighted loss shifts its operating point without learning to rank orchards better (orchard AP about 27 against 35). A verifier with seasonal, geometric and neighbourhood evidence raises orchard precision at the high-accuracy end from 34.8% to 45.5% at equal recall. Evaluation details changed results as much as some model choices did: a train/test input mismatch cost 0.36–0.50 points of accuracy, and a missing BatchNorm recalibration cost more than 20 points for the network alone. " + leak("conclusion")));
C.push(P("A pre-registered rotation over all four regions confirms the main findings: the gains from phenology and from the L2A network hold in every region. It also qualifies one of them: the verifier helps in three regions and is neutral in frh03, whose fold trains on the fewest orchards; a loose candidate threshold explains part, but not all, of this."));
C.push(P("**Limitations.** Orchard precision stays near 8% at high recall (7.8% at 77.8% recall; 8.6% at 76.8% with the verifier), so a map at that operating point still needs inspection. All data come from one year (2017) and one region of France. The rotation uses a variant without the pretrained networks, because those were trained on three of the four regions. " + leak("limitation") + " Only orchards could be evaluated as a rare class."));
C.push(P("**Future work.** A second year of data would test whether the seasonal features transfer across years. A verifier that stays useful when trained on fewer orchards, for example with candidates calibrated per region and more training regions, is the most direct next step for the second stage. Higher-resolution imagery could separate tree rows from grass and raise precision at high recall."));

// References
C.push(H1("References"));
const REFS = [
  "Chen, T., Guestrin, C., 2016. XGBoost: a scalable tree boosting system. In: *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785–794. https://doi.org/10.1145/2939672.2939785",
  "Defourny, P., Bontemps, S., Bellemans, N., et al., 2019. Near real-time agriculture monitoring at national scale at parcel resolution: performance assessment of the Sen2-Agri automated system in various cropping systems around the world. *Remote Sensing of Environment* 221, 551–568. https://doi.org/10.1016/j.rse.2018.11.007",
  "Drusch, M., Del Bello, U., Carlier, S., et al., 2012. Sentinel-2: ESA's optical high-resolution mission for GMES operational services. *Remote Sensing of Environment* 120, 25–36. https://doi.org/10.1016/j.rse.2011.11.026",
  "Gao, B.-C., 1996. NDWI—a normalized difference water index for remote sensing of vegetation liquid water from space. *Remote Sensing of Environment* 58(3), 257–266. https://doi.org/10.1016/S0034-4257(96)00067-3",
  "Gitelson, A., Merzlyak, M.N., 1994. Spectral reflectance changes associated with autumn senescence of *Aesculus hippocastanum* L. and *Acer platanoides* L. leaves: spectral features and relation to chlorophyll estimation. *Journal of Plant Physiology* 143(3), 286–292. https://doi.org/10.1016/S0176-1617(11)81633-0",
  "He, H., Garcia, E.A., 2009. Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering* 21(9), 1263–1284. https://doi.org/10.1109/TKDE.2008.239",
  "He, K., Zhang, X., Ren, S., Sun, J., 2016. Deep residual learning for image recognition. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 770–778. https://doi.org/10.1109/CVPR.2016.90",
  "Inglada, J., Arias, M., Tardy, B., et al., 2015. Assessment of an operational system for crop type map production using high temporal and spatial resolution satellite optical imagery. *Remote Sensing* 7(9), 12356–12379. https://doi.org/10.3390/rs70912356",
  "Ioffe, S., Szegedy, C., 2015. Batch normalization: accelerating deep network training by reducing internal covariate shift. In: *Proceedings of the 32nd International Conference on Machine Learning*, PMLR 37, pp. 448–456. https://proceedings.mlr.press/v37/ioffe15.html",
  "Jacobs, R.A., Jordan, M.I., Nowlan, S.J., Hinton, G.E., 1991. Adaptive mixtures of local experts. *Neural Computation* 3(1), 79–87. https://doi.org/10.1162/neco.1991.3.1.79",
  "Jönsson, P., Eklundh, L., 2004. TIMESAT—a program for analyzing time-series of satellite sensor data. *Computers & Geosciences* 30(8), 833–845. https://doi.org/10.1016/j.cageo.2004.05.006",
  "Jordan, M.I., Jacobs, R.A., 1994. Hierarchical mixtures of experts and the EM algorithm. *Neural Computation* 6(2), 181–214. https://doi.org/10.1162/neco.1994.6.2.181",
  "Künsch, H.R., 1989. The jackknife and the bootstrap for general stationary observations. *The Annals of Statistics* 17(3), 1217–1241. https://doi.org/10.1214/aos/1176347265",
  "Main-Knorn, M., Pflug, B., Louis, J., Debaecker, V., Müller-Wilm, U., Gascon, F., 2017. Sen2Cor for Sentinel-2. In: *Image and Signal Processing for Remote Sensing XXIII*, Proc. SPIE 10427, 1042704. https://doi.org/10.1117/12.2278218",
  "Pelletier, C., Webb, G.I., Petitjean, F., 2019. Temporal convolutional neural network for the classification of satellite image time series. *Remote Sensing* 11(5), 523. https://doi.org/10.3390/rs11050523",
  "Ploton, P., Mortier, F., Réjou-Méchain, M., et al., 2020. Spatial validation reveals poor predictive performance of large-scale ecological mapping models. *Nature Communications* 11, 4540. https://doi.org/10.1038/s41467-020-18321-y",
  "Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A.V., Gulin, A., 2018. CatBoost: unbiased boosting with categorical features. In: *Advances in Neural Information Processing Systems 31 (NeurIPS 2018)*.",
  "Qiu, S., Zhu, Z., He, B., 2019. Fmask 4.0: improved cloud and cloud shadow detection in Landsats 4–8 and Sentinel-2 imagery. *Remote Sensing of Environment* 231, 111205. https://doi.org/10.1016/j.rse.2019.05.024",
  "Roberts, D.R., Bahn, V., Ciuti, S., et al., 2017. Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 40(8), 913–929. https://doi.org/10.1111/ecog.02881",
  "Rußwurm, M., Körner, M., 2020. Self-attention for raw optical satellite time series classification. *ISPRS Journal of Photogrammetry and Remote Sensing* 169, 421–435. https://doi.org/10.1016/j.isprsjprs.2020.06.006",
  "Rußwurm, M., Pelletier, C., Zollner, M., Lefèvre, S., Körner, M., 2020. BreizhCrops: a time series dataset for crop type mapping. *International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences* XLIII-B2-2020, 1545–1551. https://doi.org/10.5194/isprs-archives-XLIII-B2-2020-1545-2020",
  "Sainte Fare Garnot, V., Landrieu, L., 2020. Lightweight temporal self-attention for classifying satellite images time series. In: *Advanced Analytics and Learning on Temporal Data (AALTD 2020)*, Lecture Notes in Computer Science, Springer, pp. 171–181. https://doi.org/10.1007/978-3-030-65742-0_12",
  "Sainte Fare Garnot, V., Landrieu, L., Giordano, S., Chehata, N., 2020. Satellite image time series classification with pixel-set encoders and temporal self-attention. In: *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 12322–12331. https://doi.org/10.1109/CVPR42600.2020.01234",
  "Saito, T., Rehmsmeier, M., 2015. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE* 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432",
  "Shanmugam, D., Blalock, D., Balakrishnan, G., Guttag, J., 2021. Better aggregation in test-time augmentation. In: *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 1194–1203. https://doi.org/10.1109/ICCV48922.2021.00125",
  "Silla, C.N., Freitas, A.A., 2011. A survey of hierarchical classification across different application domains. *Data Mining and Knowledge Discovery* 22(1–2), 31–72. https://doi.org/10.1007/s10618-010-0175-9",
  "Snell, J., Swersky, K., Zemel, R., 2017. Prototypical networks for few-shot learning. In: *Advances in Neural Information Processing Systems 30 (NIPS 2017)*.",
  "Turkoglu, M.O., D'Aronco, S., Perich, G., et al., 2021. Crop mapping from image time series: deep learning with multi-scale label hierarchies. *Remote Sensing of Environment* 264, 112603. https://doi.org/10.1016/j.rse.2021.112603",
  "Vaswani, A., Shazeer, N., Parmar, N., et al., 2017. Attention is all you need. In: *Advances in Neural Information Processing Systems 30 (NIPS 2017)*.",
  "Wang, F., n.d. Multi-Scale-1D-ResNet. GitHub repository (MSResNet implementation used by BreizhCrops). https://github.com/geekfeiw/Multi-Scale-1D-ResNet (accessed 15 October 2026).",
  "Wang, Z., Yan, W., Oates, T., 2017. Time series classification from scratch with deep neural networks: a strong baseline. In: *Proceedings of the International Joint Conference on Neural Networks (IJCNN)*, pp. 1578–1585. https://doi.org/10.1109/IJCNN.2017.7966039",
  "Wolpert, D.H., 1992. Stacked generalization. *Neural Networks* 5(2), 241–259. https://doi.org/10.1016/S0893-6080(05)80023-1",
  "Zhu, Z., Woodcock, C.E., 2012. Object-based cloud and cloud shadow detection in Landsat imagery. *Remote Sensing of Environment* 118, 83–94. https://doi.org/10.1016/j.rse.2011.10.028",
];
for (const r of REFS) C.push(new Paragraph({ children: runs(r, { size: 19 }), spacing: { after: 80, line: 252 }, indent: { left: 360, hanging: 360 }, alignment: AlignmentType.LEFT }));

// ---- document -------------------------------------------------------------------------------------
const doc = new Document({
  creator: "Tudor Bodea",
  title: "Orchards in the Meadow",
  styles: {
    default: { document: { run: { font: FONT, size: BODY } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: FONT }, paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, italics: true, font: FONT }, paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
      { reference: "contrib", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18 })] })] }) },
    children: C,
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT, L ? "(with leakage)" : "(leakage pending)"); });
