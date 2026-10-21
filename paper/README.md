# Paper

`Orchards_in_the_Meadow.docx` is the paper (single column; about 7 pages once set in a two-column
LaTeX template). It is edited directly in Word.

The figures are generated from the experiment outputs in `artifacts/` (run from the repository root):

```bash
python paper/figures.py paper   # Fig. 1 (pipeline) and Fig. 3 (rotation gains)
python paper/figure2.py         # Fig. 2 (seasonal curves, precision-recall curves)
```

Every number in the paper comes from the logs in [`logs/`](../logs/README.md), which maps each log to
the tables and figures it backs.
