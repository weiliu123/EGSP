# EGSP

EGSP is a scFoundation-based survival prediction framework that integrates:
- scFoundation embeddings (pre-extracted or end-to-end),
- gene expression features,
- and clinical variables (e.g., age, gender, pTNM)

Two usage modes are provided:
1) **EGSP**: takes embeddings + gene expression + clinical variables as input (fast, lightweight; does NOT require scFoundation code).
2) **EGSP End-to-End**: takes gene counts as input and extracts embeddings internally using scFoundation (requires scFoundation pretrained checkpoint).

This repository provides **minimal runnable demos** for reproducibility and illustration. The included checkpoint and demo data are **for demonstration only** (not intended to cover all TCGA cancer types).

---

## Repository structure

```text
EGSP/
├── assets/
├── scfoundation/
│   └── OS_scRNA_gene_index.19264.tsv
├── checkpoints/
│   └── egsp_demo.pt
├── demo/
│   ├── EGSP_demo/
│   │   ├── embeddings.csv
│   │   ├── gene_exp_raw.npy
│   │   └── clinical.csv
│   └── EGSP_End2End_demo/
│       ├── gene_counts.csv
│       └── clinical.csv
├── run_egsp_infer.py
├── run_egsp_train.py
├── run_egsp_end2end.py
├── src/
│   └── egsp/
│       ├── models.py
│       ├── scfoundation_bridge.py
│       └── pipelines/
│           ├── egsp_predict.py
│           ├── egsp_train.py
│           └── egsp_end2end.py
├── pyproject.toml
└── README.md
```
---


## Installation

Create and activate your Python environment (example: conda):

```bash
conda create -n egsp python=3.10 -y
conda activate egsp
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install this repository in editable mode:
```bash
pip install -e .
```
---

## Quick Start (Demo)

Run all commands from the project root directory (the folder containing README.md).

### Demo 1: Inference-only (fast)

Runs EGSP inference using pre-extracted embeddings:
```bash
python demo/run_egsp_infer.py
```
Expected output includes predicted risk scores (first few samples).

### Demo 2: Train + Evaluate EGSP head (default 10 epochs; uses GPU if available)

This demo performs a small train/eval/test split on the demo dataset:
```bash
python demo/run_egsp_train.py
```
Outputs training/evaluation metrics and an example C-index on the test split.

### Demo 3: End-to-End (requires scFoundation pretrained checkpoint)

This demo extracts embeddings internally and trains/evaluates end-to-end.

You need the official scFoundation pretrained checkpoint file models.ckpt ([Download from SharePoint](https://hopebio2020.sharepoint.com/:f:/s/PublicSharedfiles/EmUQnvZMETlDvoCaBduCNeIBQArcOrd8T8iEpiGofFZ9CQ?e=3SpPZU)).

Run:
```bash
python demo/run_egsp_end2end.py --ckpt_path /path/to/models.ckpt
```

Optional arguments:
```bash
python demo/run_egsp_end2end.py \
  --ckpt_path /path/to/models.ckpt \
  --epochs 20 \
  --use_gpu 1 \
  --gene_list_path assets/scfoundation/OS_scRNA_gene_index.19264.tsv
```

## Notes on data formats
### EGSP demo inputs (`demo/EGSP_demo/`)

- `embeddings.csv`: (N, Ne) scFoundation embeddings  
- `gene_exp_raw.npy`: (N, Ng) normalized + log1p transformed expression matrix used by EGSP  
- `clinical.csv`: N rows with clinical columns (age, gender, pTNM, time, status)

### End-to-End demo inputs (`demo/EGSP_End2End_demo/`)

- `gene_counts.csv`: raw counts matrix (N, genes) for end-to-end extraction  
- `clinical.csv`: same clinical fields

---

## Reproducibility
- Demos are designed to be lightweight and runnable on a single GPU.
- The provided checkpoint `checkpoints/egsp_demo.pt` is a demo checkpoint.
- For full reproduction on TCGA cohorts, users should follow the manuscript pipeline and train models per cohort.

---

## License / Citation
If you use this code, please cite the corresponding manuscript and also cite scFoundation according to their repository instructions.