# src/egsp/pipelines/egsp_infer.py
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ..models import EGSP, EGSPConfig

def run_egsp_infer(demo_dir: Path, ckpt_path: Path, device: str = "cpu"):
    embeddings = pd.read_csv(demo_dir / "embeddings.csv", index_col=0).values.astype("float32")
    gene_exp_raw = np.load(demo_dir / "gene_exp_raw.npy").astype("float32")
    clinical = pd.read_csv(demo_dir / "clinical.csv", index_col=0)

    inputs = {
        "embed": torch.tensor(embeddings, dtype=torch.float32),
        "gene_exp_raw": torch.tensor(gene_exp_raw, dtype=torch.float32),
        "age": torch.tensor(clinical["age"].values, dtype=torch.float32),
        "gender": torch.tensor(clinical["gender"].values, dtype=torch.float32),
        "pTNM": torch.tensor(clinical["pTNM"].values, dtype=torch.float32),
    }

    cfg = EGSPConfig(
        cln_feats=["age", "gender", "pTNM"],
        embedsize=embeddings.shape[1],
        num_gene_feats=gene_exp_raw.shape[1],
        hidden_layers=[1024, 512, 256, 128],
        add_gene_feats=True,
        alpha=1.0,
    )

    model = EGSP(cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    if device != "cpu":
        model = model.to(device)
        for k, v in inputs.items():
            inputs[k] = v.to(device)

    with torch.no_grad():
        outputs = model(**inputs)      # dict
        risk = outputs["logits"]       # (N, 1) or (N,)

    return risk.detach().cpu().numpy()
