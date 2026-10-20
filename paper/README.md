# Paper

`Orchards_in_the_Meadow.docx` is the paper draft (single column; about 7.5 pages once set in a
two-column LaTeX template). It is generated, not edited by hand:

```bash
python paper/figures.py paper        # Fig. 1 (pipeline) and Fig. 3 (rotation gains)
python paper/figure2.py              # Fig. 2 (seasonal curves, precision-recall); needs artifacts/
cd paper && npm install && npm run build
```

`build_paper.js` holds the text, tables and references; `leakage.json` holds the stacking-leakage
paragraphs it inserts. Every number comes from the logs listed in the repository README.
