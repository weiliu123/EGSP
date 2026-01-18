# src/egsp/pipelines/egsp_train.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import TrainingArguments, EarlyStoppingCallback

from ..models import (
    EGSP,
    EGSPConfig,
    get_train_eval_test_dataset,
    DataCollatorForEGSP,
    SurvTrainer,
    safe_harrell_cindex,
)


@dataclass
class EGSPTrainResult:
    train_metrics: Dict[str, Any]
    eval_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    test_cindex: float
    test_risk: np.ndarray
    test_time: np.ndarray
    test_event: np.ndarray


def _load_egsp_demo_dataset(demo_dir: Path) -> Dataset:
    """Load demo inputs (embeddings/gene_exp_raw/clinical) and return HF Dataset."""
    embeddings = pd.read_csv(demo_dir / "embeddings.csv", index_col=0).values.astype("float32")
    gene_exp_raw = np.load(demo_dir / "gene_exp_raw.npy").astype("float32")
    clinical = pd.read_csv(demo_dir / "clinical.csv", index_col=0)

    required_cols = ["age", "gender", "pTNM", "time", "status"]
    missing = [c for c in required_cols if c not in clinical.columns]
    if missing:
        raise ValueError(f"clinical.csv missing columns: {missing}")

    merged_data = {
        "embed": torch.tensor(embeddings, dtype=torch.float32),
        "gene_exp_raw": torch.tensor(gene_exp_raw, dtype=torch.float32),
        "age": torch.tensor(clinical["age"].values, dtype=torch.float32),
        "gender": torch.tensor(clinical["gender"].values, dtype=torch.float32),
        "pTNM": torch.tensor(clinical["pTNM"].values, dtype=torch.float32),
        "time": torch.tensor(clinical["time"].values, dtype=torch.float32),
        "status": torch.tensor(clinical["status"].values, dtype=torch.long),
    }

    return Dataset.from_dict(merged_data)


def _build_egsp_config(emb_dim: int, gene_dim: int) -> EGSPConfig:
    return EGSPConfig(
        cln_feats=["age", "gender", "pTNM"],
        embedsize=emb_dim,
        num_gene_feats=gene_dim,
        hidden_layers=[1024, 512, 256, 128],
        add_gene_feats=True,
        alpha=1.0,
    )


def run_egsp_train_demo(
    demo_dir: Path,
    ckpt_path: Path,
    output_dir: Path,
    epochs: int = 10,
    use_gpu: bool = True,
    seed: int = 42,
    train_ratio: float = 0.70,
    eval_ratio: float = 0.15,
    test_ratio: float = 0.15,
    patience: int = 10,
) -> EGSPTrainResult:
    """
    Train+evaluate EGSP head on demo dataset.

    Default: 10 epochs, use GPU if available.
    """
    if not np.isclose(train_ratio + eval_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + eval_ratio + test_ratio must sum to 1.0")

    # reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    whole_dataset = _load_egsp_demo_dataset(demo_dir)

    # split
    train_data, eval_data, test_data = get_train_eval_test_dataset(
        dataset=whole_dataset,
        train_ratio=train_ratio,
        eval_ratio=eval_ratio,
        test_ratio=test_ratio,
        random_state=seed,
        verbose=True,
    )

    # model
    # 从 dataset 里推维度（避免 demo_dir 再读一遍文件）
    emb_dim = np.asarray(whole_dataset[0]["embed"]).shape[0]
    gene_dim = np.asarray(whole_dataset[0]["gene_exp_raw"]).shape[0]

    config = _build_egsp_config(emb_dim, gene_dim)

    model = EGSP(config)

    # 你现在的 egsp_demo.pt 是 state_dict（weights_only=True 返回 dict[str, Tensor]）
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    # device
    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    model = model.to(device)

    data_collator = DataCollatorForEGSP()

    # TrainingArguments 需要字符串路径
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir.parents[1] / "logs" / output_dir.name
    logs_dir.mkdir(parents=True, exist_ok=True)

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        logging_dir=str(logs_dir),
        per_device_train_batch_size=64,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        lr_scheduler_type="linear",
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        report_to="none",  # demo, no tensorboard
        dataloader_drop_last=False,
        learning_rate=2e-5,
        max_grad_norm=1.0,
        weight_decay=0.01,
        num_train_epochs=epochs,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        seed=seed,
    )

    trainer = SurvTrainer(
        model=model,
        args=train_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=SurvTrainer.compute_surv_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=patience,
                early_stopping_threshold=0.0,
            )
        ],
    )

    print(
        "N:", len(whole_dataset),
        "embed_dim:", emb_dim,
        "gene_dim:", gene_dim,
    )

    trainer.train()

    train_metrics = trainer.evaluate(train_data)
    eval_metrics = trainer.evaluate(eval_data)
    test_metrics = trainer.evaluate(test_data)

    pred_test = trainer.predict(test_data)
    time_test = pred_test.label_ids[:, 0]
    event_test = pred_test.label_ids[:, 1].astype(int)
    risk_test = pred_test.predictions.squeeze()

    cindex, cindex_err = safe_harrell_cindex(time=time_test, event=event_test, risk=risk_test)
    if cindex_err is not None:
        print("[WARN] C-index computation failed:", cindex_err)

    return EGSPTrainResult(
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        test_metrics=test_metrics,
        test_cindex=float(cindex),
        test_risk=risk_test,
        test_time=time_test,
        test_event=event_test,
    )
