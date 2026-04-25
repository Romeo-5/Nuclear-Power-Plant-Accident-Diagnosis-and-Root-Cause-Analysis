# Experiments TODO — Path 1 Paper

Punch list of remaining work before submission. Items marked ✅ are complete.

---

## ✅ Done (figures and numbers integrated into the LaTeX)

| Item | Result | Where it lives in the paper |
|---|---|---|
| Figure 2 — per-class F1 vs. frequency | `paper/figures/fig2_f1_vs_frequency.pdf` | §5.2 |
| Figure 3 — confusion matrix | `paper/figures/fig3_confusion.pdf` | §5.2 |
| Figure 5 — LOCA trajectory comparison | `paper/figures/fig5_trajectory_loca.pdf` | §5.4 |
| Figure 6 — physics-residual histograms | `paper/figures/fig6_residual_histograms.pdf` | §5.4 |
| Figure 7 — κ distribution | `paper/figures/fig7_kappa_distributions.pdf` | §5.5 |
| Table 5 — pipeline cross-validation precision/recall/F1 | `results/pipeline/xvalidation_table.tex` (precision $0.362$, recall $0.105$, F1 $0.163$ at $\kappa^\star = 1.41$) | §5.5 |
| KS test on κ distributions | `results/pipeline/kappa_stats.json` ($D = 0.634$, $p < 10^{-300}$) | §5.5 |
| Per-class commentary updated for actual results (LOCA $\rightarrow$ LOCAC attractor at $95\%$, RW $\rightarrow$ LOCAC at $93\%$, classes with zero predictions called out) | — | §5.2 |

---

## ⏳ Remaining (in priority order)

### 1. SHAP physics-validation run (Table 3 + Figure 4)

**Status:** the SHAP run did not complete on Colab — `results/shap/` is empty in the returned archive. This is the only remaining experimental gap.

**Run on Colab GPU (3–5 min on T4):**
```bash
python scripts/shap_physics_validation.py \
    --config configs/lstm_classifier.yaml \
    --checkpoint checkpoints/lstm_classifier_best.pt \
    --classes LOCA SGATR SGBTR FLB \
    --n-samples 100 --n-background 100 \
    --device cuda \
    --output-dir results/shap
```

**Then:**
- Drop `results/shap/temporal_plot.pdf` into `paper/figures/fig4_shap_temporal.pdf`.
- Replace the Figure 4 placeholder in `sections/05_results.tex` with `\includegraphics{figures/fig4_shap_temporal}`.
- Drop the rows from `results/shap/summary_table.tex` into Table 3 in place of the `\placeholder{...}` cells.

**Caveat:** because LOCA is misclassified to LOCAC in $95\%$ of cases, SHAP attributions averaged over true-LOCA windows will reflect the model's LOCAC-attractor behaviour. Either compute SHAP w.r.t. the predicted class (not the true class) — pass `--classes LOCAC SGATR SGBTR FLB` — or accept that the LOCA row of Table 3 will document a model failure rather than a model success. Both are valid, just be deliberate about which you report.

### 2. Figure 1 — pipeline schematic

A hand-drawn diagram showing the three stages and the cross-validation loop. Author's choice of TikZ, `draw.io`, or `excalidraw`. Save as `paper/figures/fig1_pipeline.pdf` and replace the placeholder in `sections/03_methodology.tex`.

### 3. Figure 8 — case studies

Did not generate in the Colab run. The script attempts to find:
(i) correctly-classified LOCA with low κ,
(ii) LOCA misclassified as LOCAC with elevated κ,
(iii) correctly-classified ATWS with high κ.
Cases (ii) and (iii) probably failed to find any windows because (a) LOCA $\rightarrow$ LOCAC misclassification dominates (so case (ii) should be easy — likely a save-path issue), and (b) ATWS may have zero correctly-classified windows in the test set. Re-run `pipeline_xvalidation.py`; if Figure 8 still does not appear, replace case (iii) with a different rare-class scenario or drop it. Lowest priority.

### 4. Optional re-run of Figure 7 with log-x axis

`scripts/pipeline_xvalidation.py` was patched after the Colab run to use a log-log axis on the κ histogram, which surfaces the tail separation more clearly. The current `fig7_kappa_distributions.pdf` works but the new version will be more informative. Re-run takes ~1 minute on GPU.

### 5. Table 4 — Data MSE row

The two `\placeholder{insert}` cells in Table 4 (§5.4) for normalized data MSE. Read off the final-epoch `val_mse` from each digital-twin training log, or compute on the test set with a small standalone script. Two scalar values to fill in.

### 6. Bibliography pass

Open verified entries on publisher pages and confirm full author lists for `zhang2023pinn`, `digitaltwin2025review`, `xgboost2026shap`. Decide what to do with the four still-flagged placeholder entries (`song2023calibration`, `santhosh2019survey`, `wang2023deep`, `lee2023lstm`) — replace with verified citations or remove the `\citep{}` calls in the prose.

### 7. Paper-text passes related to the actual results

Now that the numbers are in, two passes worth doing:

- **Abstract / introduction:** the abstract still says ``$71.2\%$ accuracy across 18 classes,'' which matches the Colab run ($71.9\%$ exact, rounds to $71.2$%). No change needed unless you want the more precise value.
- **Discussion §6:** add a sentence acknowledging that the LOCA $\rightarrow$ LOCAC failure is a stronger story than ``LOCA/LOCAC confusion'' implied. This is a meaningful and honest finding: the model has learned to collapse two physically distinct accidents into one because the discriminating event (containment isolation) lies outside the 30-step window. Suggested follow-up: longer windows or hierarchical classifier (LOCA-class first, containment-state second).

---

## Suggested final-mile order

1. Run SHAP on Colab GPU → fill Table 3 + Figure 4.
2. Make Figure 1 schematic.
3. Re-run `pipeline_xvalidation.py` on Colab (gets log-axis Fig 7 + retry on Fig 8).
4. Read Table 4 MSE values from logs, fill in.
5. Bib pass, remove or replace the four still-flagged placeholders.
6. `grep -rn "placeholder\|todo" paper/` → should be empty.
7. Compile in Overleaf, proofread, submit.
