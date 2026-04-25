# Experiments TODO — Path 1 Paper

Punch list of code/experiments that must run before the paper can be submitted.
Each item lists: (a) what the paper needs, (b) what to run, (c) where the output
should land, (d) which section/figure/table in `paper/sections/` it fills.

Grep for `\placeholder` in `paper/sections/` to see the matching orange markers.

---

## 1. SHAP physics-validation on LOCA / SGTR / FLB

**Paper target:** Table 3 (`sec:results-shap`) rows 1–3; Figure 4 (temporal SHAP).

**Script:** `scripts/shap_physics_validation.py` (added in this turn).

**Run:**
```bash
# Assumes you have a trained LSTM classifier checkpoint
python scripts/shap_physics_validation.py \
    --config configs/lstm_classifier.yaml \
    --checkpoint checkpoints/lstm_classifier/best.pt \
    --classes LOCA SGATR SGBTR FLB \
    --n-samples 100 \
    --n-background 100 \
    --output-dir results/shap
```

**Outputs:**
- `results/shap/feature_importance_<CLASS>.csv` — per-sensor SHAP rankings per class.
- `results/shap/group_importance_<CLASS>.csv` — physical-group rollup.
- `results/shap/temporal_importance_<CLASS>.csv` — per-timestep importance.
- `results/shap/summary_table.tex` — LaTeX-ready Table 3 fragment. Drop directly into `sections/05_results.tex` in place of the three placeholder rows.
- `results/shap/temporal_plot.pdf` — Figure 4 contents. Save as `paper/figures/fig4_shap_temporal.pdf` and `\includegraphics{figures/fig4_shap_temporal}` in `sections/05_results.tex`.

**Paper edits:** replace the three `\placeholder{run SHAP: ...}` cells in the Table 3 `tabularx` with the generated LaTeX rows; replace Figure 4 placeholder with `\includegraphics`.

---

## 2. Per-class F1 vs. class-frequency plot

**Paper target:** Figure 2 (`sec:results-clf`).

**Script to write:** `scripts/plot_per_class_f1.py` (not included this turn — short matplotlib script).

**Inputs:** classifier evaluation output from `notebooks/04_evaluate_classification.ipynb` (per-class precision/recall/F1 + class counts from the training split).

**Output:** `paper/figures/fig2_f1_vs_frequency.pdf` — scatter of per-class F1 vs. `log10(n_train_windows)` with class labels annotated.

---

## 3. Confusion matrix plot

**Paper target:** Figure 3 (`sec:results-clf`).

**Script to write:** `scripts/plot_confusion.py`.

**Inputs:** predictions and labels from the LSTM classifier on the test set (export from notebook 04).

**Output:** `paper/figures/fig3_confusion.pdf` — 18×18 row-normalized heatmap with accident abbreviations on both axes.

---

## 4. Trajectory comparison plot (physics vs. data-only)

**Paper target:** Figure 5 (`sec:results-pinn`).

**Script to write:** `scripts/plot_trajectory.py`.

**Inputs:** both digital twin checkpoints (`digital_twin.yaml` and `digital_twin_no_physics.yaml`) + one representative LOCA test window + ground-truth continuation.

**Output:** `paper/figures/fig5_trajectory_loca.pdf` — multi-panel plot, one panel per channel in {P, TAVG, WRCA, neutron_density_proxy}, overlaying physics-informed prediction, data-only baseline, and ground truth over the 10-step horizon.

---

## 5. Physics-residual distribution histograms

**Paper target:** Figure 6 (`sec:results-pinn`).

**Script to write:** `scripts/plot_residual_histograms.py`.

**Inputs:** per-window point-kinetics residual and conservation violation, computed with both models on the full test set.

**Output:** `paper/figures/fig6_residual_histograms.pdf` — two-panel histogram (point kinetics | conservation) with log-y axis, physics-informed vs. data-only overlaid.

---

## 6. Fill Data-MSE entry in Table 4

**Paper target:** Table 4 row 3 (`sec:results-pinn`) — two `\placeholder{insert}` cells.

**How:** read off `val_mse` from the final-epoch TensorBoard / training log of each digital-twin run. No new script needed.

---

## 7. Pipeline cross-validation: κ distributions + precision/recall

**Paper target:** Figure 7, Table 5, and the case studies of Figure 8 (`sec:results-pipeline`).

**Script to write:** `scripts/pipeline_xvalidation.py`.

**What it does:**
1. Load trained stage-2 LSTM classifier and stage-3 physics-informed surrogate.
2. Iterate the test set. For each window:
   - Run the classifier → predicted class, correct/incorrect flag.
   - Run the surrogate → predicted 10-step trajectory.
   - Compute κ = normalized(w_pk · point_kinetics_residual + w_cons · conservation_residual), with normalization constants set so that the median κ on correctly classified validation windows is 1.
3. Aggregate κ distributions conditional on correctness.
4. Sweep κ* to find the F1-optimal threshold on the validation set, then report precision/recall on the test set at that threshold.
5. Select three case-study windows (one per scenario in Figure 8 caption) and export their raw time series + SHAP maps + predicted trajectories.

**Outputs:**
- `results/pipeline/kappa_distributions.pdf` — Figure 7.
- `results/pipeline/xvalidation_table.tex` — Table 5 fragment.
- `results/pipeline/case_studies.pdf` — Figure 8.
- `results/pipeline/kappa_stats.json` — Kolmogorov–Smirnov statistic and p-value for the `\placeholder{insert test statistic and p-value once computed}` sentence in `sec:results-pipeline`.

---

## 8. Severity-stratified extrapolation experiment (optional, strengthens Section 6)

**Paper target:** Discussion (`sec:discussion`) — strengthens the "physics as regularizer" claim.

**What it does:** retrain the physics-informed surrogate and the data-only baseline on mild + moderate severity transients only, then evaluate on held-out severe transients. Measure the gap between the two models on this out-of-distribution split.

**Script to write:** `scripts/train_digital_twin_severity_split.py` (wraps the existing training entry point with a custom scenario filter).

**Outputs:** one extra column in Table 4 labelled "OOD (severe only)" showing the residual ratios under distribution shift. If the physics-informed model's advantage grows on OOD, that's a strong additional result to state in §6.1.

**Status:** optional for the first submission; strongly recommended for a revision pass.

---

## 9. Bibliography verification pass

**Paper target:** `paper/references.bib` — 8 placeholder entries flagged via `note = {Placeholder...}`.

**How:**
- `zhang2023pinn` — the 2023 Nature Scientific Reports TL-PINN reactor transient paper. Look up exact authors, volume, article number, DOI.
- `song2023calibration` — Song & Song autonomous-calibration paper. Verify venue and year.
- `gong2024rom` — Gong & Cheng ROM + ML digital twin paper. Verify.
- `ayodeji2022dtwin` — review article title/authors; confirm they match the actual Progress in Nuclear Energy review you intend to cite.
- `digitaltwin2025review` — 2025 digital-twin review; fill in authors and DOI.
- `santhosh2019survey`, `wang2023deep`, `lee2023lstm` — replace with specific papers actually benchmarked in related-work. If the claims made in §2.1 need more specific evidence, swap in stronger individual citations (e.g., named CNN/LSTM NPPAD papers rather than generic placeholders).
- `xgboost2026shap` — the January 2026 SHAP-XGBoost NPPAD paper referenced in §2.1. Find authors and DOI.

---

## Suggested order of execution

1. Items 1, 2, 3, 6 first — these use models you already trained and unblock Sections 5.2–5.3.
2. Items 4, 5 next — re-run digital-twin evaluation to export per-window artifacts.
3. Item 7 — requires items 1–5 to be in place; delivers the pipeline story.
4. Item 9 in parallel with writing.
5. Item 8 last, as a revision-grade result.

---

## After all items are done

1. `grep -rn "placeholder\|todo" paper/` should return empty.
2. Compile `paper/main.tex` with pdfLaTeX + BibTeX end-to-end without warnings.
3. Run a single spell/grammar pass on the full PDF.
4. Draft the cover letter (separate file, not included in this `paper/` directory).
5. Submit to Progress in Nuclear Energy (or Annals of Nuclear Energy as the stretch target).
