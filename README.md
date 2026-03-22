# Nuclear Power Plant Fault Diagnosis Using Deep Learning

Exploring ML methods for nuclear power plant fault diagnosis and root cause analysis using open simulation data from the [NPPAD dataset](https://www.nature.com/articles/s41597-022-01879-1) (Nuclear Power Plant Accident Data).

## Overview

This project applies deep learning to time-series sensor data from pressurized water reactor (PWR) simulations to:

1. **Anomaly Detection** — Monitor reactor sensor streams and flag deviations from normal operating conditions using autoencoders, LSTMs, and transformers
2. **Accident Classification & Root Cause Analysis** — Classify flagged anomalies into accident types with interpretability via SHAP/attention visualization *(planned)*
3. **Physics-Informed Digital Twin** — Surrogate model approximating reactor dynamics with physics-informed loss constraints *(planned)*

## Dataset

The [NPPAD dataset](https://springernature.figshare.com/collections/NPPAD_An_Open_Time-series_Dataset_Covering_Various_Accidents_for_Nuclear_Power_Plants/6238473) contains time-series data from PCTRAN, a widely-used NPP simulator, covering:
- 97 operational parameters (temperature, pressure, neutron density, coolant flow, etc.)
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

### 3. Train
```bash
python scripts/train.py --config configs/autoencoder.yaml
python scripts/train.py --config configs/lstm.yaml
python scripts/train.py --config configs/transformer.yaml
```

## Module Status

| Module | Status | Description |
|--------|--------|-------------|
| Anomaly Detection | In Progress | Semi-supervised detection using reconstruction/prediction error |
| Accident Classification | Planned | Multi-class classifier with SHAP interpretability |
| Digital Twin | Planned | Physics-informed neural network surrogate model |

## Tech Stack

Python, PyTorch, pandas, scikit-learn, matplotlib

## License

MIT
