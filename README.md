# SEO Top-5 ranking thesis

The final modelling specification is locked: a 37-feature Random Forest with a fixed query-grouped development/test split and fixed hyperparameters. The original dataset is `data/RAT_Search_Results_Trends_SEO_2025.csv` (kept locally and excluded from Git).

## Workflow

Place the original CSV at the path above. From the repository root, install dependencies and run:

```bash
python -m pip install -r requirements.txt
python main.py
python rq_analysis.py
```

`main.py` reproduces the locked model comparison and final Random Forest, writing to `outputs/final_model/`. `rq_analysis.py` uses the final-model outputs and writes the RQ1–RQ5 analyses to `outputs/research_questions/`. Its permutation-importance work can be slow. The saved tuning summaries in `outputs/final_model/tuning/` document the earlier parameter selection; neither final script reruns tuning.

Exploratory data analysis is separate. Its figures, tables, and feature-audit reports are in `eda/`. To regenerate the figures and tables, run `python src/eda.py` from the repository root.
