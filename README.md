# Nuclear Power Plant Fault Diagnosis Using Deep Learning

Exploring ML methods for nuclear power plant fault diagnosis and root cause analysis using open simulation data from the [NPPAD dataset](https://www.nature.com/articles/s41597-022-01879-1) (Nuclear Power Plant Accident Data).

## Overview

This project applies deep learning to time-series sensor data from pressurized water reactor (PWR) simulations to:

1. **Anomaly Detection** — Monitor reactor sensor streams and flag deviations from normal operating conditions using autoencoders, LSTMs, and transformers
2. **Accident Classification & Root Cause Analysis** — Classify flagged anomalies into accident types with interpretability via SHAP/attention visualization *(in progress)*
3. **Physics-Informed Digital Twin** — Surrogate model approximating reactor dynamics with physics-informed loss constraints *(planned)*

## Dataset

The [NPPAD dataset](https://springernature.figshare.com/collections/NPPAD_An_Open_Time-series_Dataset_Covering_Various_Accidents_for_Nuclear_Power_Plants/6238473) contains time-series data from PCTRAN, a widely-used NPP simulator, covering:
- 96 operational parameters (temperature, pressure, neutron density, coolant flow, etc.)
- 18 operating conditions (1 normal + 17 accident scenarios)
- Multiple severity levels per accident type

**Citation:**
> Qi, B., Xiao, X., Liang, J. et al. An open time-series simulated dataset covering various accidents for nuclear power plants. *Sci Data* 9, 766 (2022).

## Project Structure

```
├── configs/            # YAML experiment configurations
├── data/
│   ├── raw/            # Raw NPPAD data (gitignored)
│   ├── processed/      # Preprocessed tensors (gitignored)
│   └── scripts/        # Download and preprocessing scripts
├── src/
│   ├── data/           # PyTorch datasets and transforms
│   ├── models/
│   │   ├── anomaly/    # Autoencoder, LSTM, Transformer detectors
│   │   ├── diagnosis/  # Accident classifier + SHAP (Module 2)
│   │   └── digital_twin/  # Physics-informed surrogate (Module 3)
│   ├── training/       # Training loop, callbacks
│   ├── evaluation/     # Metrics, uncertainty estimation
│   └── utils/          # Config system, logging
├── notebooks/          # EDA, training, evaluation notebooks
├── scripts/            # CLI entry points
└── tests/
```

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### 1. Download Data
```bash
python data/scripts/download.py
```

### 2. Preprocess
```bash
python data/scripts/preprocess.py
```

### 3. Train Anomaly Detectors (Module 1)
```bash
python scripts/train.py --config configs/autoencoder.yaml
python scripts/train.py --config configs/lstm.yaml
python scripts/train.py --config configs/transformer.yaml
```

### 4. Train Accident Classifiers (Module 2)
```bash
python scripts/train_classifier.py --config configs/cnn_classifier.yaml
python scripts/train_classifier.py --config configs/lstm_classifier.yaml
python scripts/train_classifier.py --config configs/transformer_classifier.yaml
```

### 5. Train Digital Twin (Module 3)
```bash
python scripts/train_digital_twin.py --config configs/digital_twin.yaml
python scripts/train_digital_twin.py --config configs/digital_twin_no_physics.yaml  # ablation baseline
```

## Results

### 1: Anomaly Detection

Semi-supervised approach — models trained on normal operating data only (302 timesteps), evaluated on 7,316 test windows across 18 accident types. Anomalies are detected via reconstruction/prediction error exceeding a threshold optimized on the validation set.

| Model | Precision | Recall | F1 | AUROC | AUPRC |
|-------|-----------|--------|------|-------|-------|
| Autoencoder | 0.999 | 1.000 | 0.9995 | 0.985 | 0.9999 |
| LSTM | 0.999 | 1.000 | 0.9995 | 0.987 | 0.9999 |
| Transformer | 0.999 | 1.000 | 0.9995 | 0.985 | 0.9999 |

All three architectures achieve near-perfect binary anomaly detection. The strong separation between normal and accident regimes in the NPPAD data makes coarse anomaly detection straightforward — the more challenging task is Module 2's multi-class accident classification and root cause attribution.

**Training details:**
- Window size: 30 timesteps, stride: 10
- Trained on 28 normal windows (semi-supervised)
- Early stopping with patience=10 on validation loss
- Hardware: Apple M-series GPU (MPS backend)

### 2: Accident Classification & Root Cause Analysis

Supervised multi-class classification across 18 accident types (LOCA, SGTR, steam line break, rod withdrawal, etc.). Trained on all 32,284 windows, evaluated on 7,316 test windows.

| Model | Accuracy | F1 (macro) | F1 (weighted) | Precision (macro) | Recall (macro) |
|-------|----------|------------|---------------|-------------------|----------------|
| CNN | 0.524 | 0.421 | 0.443 | 0.448 | 0.462 |
| **LSTM (bidirectional)** | **0.712** | **0.572** | **0.676** | **0.661** | **0.578** |
| Transformer (CLS token) | 0.707 | 0.538 | 0.662 | 0.538 | 0.582 |

The LSTM classifier achieves the best overall performance. Per-class analysis reveals:
- **Near-perfect** (F1 > 0.95): LLB, LR, MD, SLBOC, Normal
- **Strong** (F1 > 0.7): SLBIC, SGBTR
- **Challenging** (F1 < 0.5): LOCA/LOCAC confusion, SGATR, RW, low-sample classes (LACP, ATWS, SP)

The lower macro-F1 reflects class imbalance — some accident types have as few as 28-56 training windows vs 3,000+ for others. Root cause analysis via gradient attribution identifies the top contributing sensors per accident type (see `notebooks/04_evaluate_classification.ipynb`).

**Training details:**
- Supervised on all data (accident_type labels)
- 1D-CNN, bidirectional LSTM, and Transformer with [CLS] token
- Early stopping with patience=15

### 3: Physics-Informed Digital Twin

GRU encoder-decoder surrogate model that predicts future reactor states from a context window, constrained by reactor physics. Compares a physics-informed model (point kinetics + energy conservation constraints) against a data-only baseline. Trained on 32,284 windows, evaluated on 7,316 test windows.

**Architecture:** GRU encoder (context window) → GRU decoder (autoregressive) with residual connection. 384K parameters.

| Metric | Physics-Informed | Data-Only Baseline |
|--------|-----------------|-------------------|
| Physics Residual (point kinetics) | 3,739 | 11,888 |
| Conservation Violation (MW) | 1,783 | 2,950 |
| Physics Residual Reduction | **68.6%** | — |
| Conservation Violation Reduction | **39.6%** | — |

The physics-informed model produces predictions significantly more consistent with point reactor kinetics equations and energy conservation laws, while maintaining comparable data-fitting accuracy. The key insight is that physics constraints act as a regularizer — they don't improve raw MSE much but ensure predictions respect physical laws (positive pressure/power, energy balance, neutron kinetics consistency).

**Physics constraints:**
1. **Point kinetics residual** — dn/dt consistency with simplified neutron kinetics ODE
2. **Energy conservation** — Q ≈ W · Cp · (T_hot - T_cold) energy balance
3. **Physical bounds** — soft penalties for negative pressure, power, temperature, flow, boron
4. **Warmup schedule** — physics terms ramp from 0 to full weight over 500 training steps

**Training details:**
- GRU encoder-decoder with residual connection (predictions = last_state + delta)
- Context: 20 timesteps → Predict: 10 timesteps ahead
- Learning rate: 5e-4, 150 epochs, patience: 15
- Hardware: Apple M-series GPU (MPS backend)

## Tech Stack

Python, PyTorch, pandas, scikit-learn, matplotlib

## License

MIT
