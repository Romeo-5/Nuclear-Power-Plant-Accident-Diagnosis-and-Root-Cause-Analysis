# Paper — Path 1: Integrated Pipeline

This directory contains an Overleaf-ready LaTeX draft of the Path 1 paper:
*"An Integrated Deep Learning Framework for Nuclear Reactor Fault Diagnosis:
From Semi-Supervised Anomaly Detection to Physics-Informed Accident Prognosis."*

## Structure

```
paper/
├── main.tex              # Top-level document — compile this
├── references.bib        # Bibliography (several placeholder entries — see below)
├── sections/
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_methodology.tex
│   ├── 04_experimental_setup.tex
│   ├── 05_results.tex
│   ├── 06_discussion.tex
│   └── 07_conclusion.tex
└── figures/              # Empty — populate with plots produced by the experiments
                          # listed in EXPERIMENTS_TODO.md (to be added next turn)
```

## Overleaf upload

1. Create a new blank Overleaf project.
2. Upload the contents of this `paper/` directory at the project root.
3. Compile `main.tex` with pdfLaTeX + BibTeX (the default Overleaf pipeline works).

## Placeholders to resolve before submission

The draft uses two markers to flag content that still needs work:

- `\placeholder{...}` — orange text marking numerical results, figure content,
  or bibliographic details that must be filled in from experiment outputs or
  literature lookup.
- `\todo{...}` — red text for outstanding writing tasks.

Grep for both to get a punch list:

```bash
grep -rn "placeholder\|todo" paper/
```

## Known gaps (to be addressed in the follow-up turn)

- SHAP physics-validation table (§5.3, Table 3) — needs `scripts/shap_physics_validation.py`.
- Figures 1–8 — all currently placeholder captions; need plot-generation scripts.
- Data MSE entry in Table 4 — needs to be read off the stage-3 evaluation logs.
- Pipeline cross-validation precision/recall (Table 5) — needs `scripts/pipeline_xvalidation.py`.
- Placeholder `.bib` entries (Zhang 2023, Song 2023, Gong 2024, 2025 digital-twin
  review, 2026 SHAP-XGBoost paper) — need bibliographic verification.

All numerical results that are already in `README.md` (anomaly F1, LSTM accuracy,
PINN 68.6%/39.6%) are already threaded into the prose with correct values.
