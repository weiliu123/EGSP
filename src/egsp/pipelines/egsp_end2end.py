# src/egsp/pipelines/egsp_end2end.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import scanpy as sc
from datasets import Dataset
from transformers import TrainingArguments, EarlyStoppingCallback

from ..models import (
    EGSP_End2End,
    EGSP_End2EndConfig,
    get_train_eval_test_dataset,
    SurvTrainer,
    main_gene_selection,
    DataCollatorForEGSP_End2End,
    safe_harrell_cindex,
)


@dataclass
class End2EndResult:
    train_metrics: Dict[str, Any]
    eval_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    test_cindex: float
    test_risk: np.ndarray


def _load_gene_list(gene_list_path: Path) -> list[str]:
    if not gene_list_path.exists():
        raise FileNotFoundError(
            f"gene_list file not found: {gene_list_path}\n"
            f"Tip: set --gene_list_path or place it under assets/scfoundation/."
        )
    df = pd.read_csv(gene_list_path, sep="\t")
    if "gene_name" not in df.columns:
        raise ValueError(f"gene_list file missing 'gene_name' column: {gene_list_path}")
    return df["gene_name"].astype(str).tolist()


def _normalize_log1p(df: pd.DataFrame) -> pd.DataFrame:
    # scanpy expects float
    adata = sc.AnnData(df.astype(np.float32))
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return pd.DataFrame(adata.X, index=df.index, columns=df.columns)


def run_end2end_demo(
    demo_dir: Path,
    scfoundation_ckpt_path: Path,
    gene_list_path: Path,
    output_dir: Path,
    epochs: int = 20,
    use_gpu: bool = True,
    seed: int = 42,
    def_freeze_layers: int = 12,
    report_to: str = "none",
) -> End2EndResult:

    # sanity checks
    if not demo_dir.exists():
        raise FileNotFoundError(f"demo_dir not found: {demo_dir}")
    if not scfoundation_ckpt_path.exists():
        raise FileNotFoundError(
            f"scFoundation checkpoint not found: {scfoundation_ckpt_path}\n"
            f"Tip: download models.ckpt separately and pass --ckpt_path."
        )

    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)

    # ===== load inputs =====
    gene_counts_path = demo_dir / "gene_counts.csv"
    clinical_path = demo_dir / "clinical.csv"

    gene_counts = pd.read_csv(gene_counts_path, index_col=0)
    clinical = pd.read_csv(clinical_path, index_col=0)

    # gene list (19264)
    gene_list = _load_gene_list(gene_list_path)

    # ===== pad/align to 19264 (if needed) =====
    if gene_counts.shape[1] < 19264:
        print("[INFO] Padding/aligning gene features to 19,264 genes...")
        gexpr_feature, to_fill_columns, var = main_gene_selection(gene_counts, gene_list)
        # main_gene_selection should return a DF with >=19264 genes
        if gexpr_feature.shape[1] < 19264:
            raise ValueError("main_gene_selection did not produce 19264 genes.")
    else:
        # if already >=19264, still align order if you want strictness
        # here we just keep as-is to avoid unexpected behavior
        gexpr_feature = gene_counts

    # ===== normalize =====
    gexpr_feature_norm = _normalize_log1p(gexpr_feature)
    gexpr_raw_norm = _normalize_log1p(gene_counts)

    # HF Dataset: columns are lists
    gene_series = gexpr_feature_norm.apply(lambda row: row.values.astype(np.float32).tolist(), axis=1)
    gene_series_raw = gexpr_raw_norm.apply(lambda row: row.values.astype(np.float32).tolist(), axis=1)

    # clinical fields as float32 lists
    clinical_dict = {col: clinical[col].astype(np.float32).tolist() for col in clinical.columns}

    data = {
        "gene_exp": gene_series,
        "gene_exp_raw": gene_series_raw,
        **clinical_dict,
    }
    whole_dataset = Dataset.from_dict(data)

    train_data, eval_data, test_data = get_train_eval_test_dataset(
        dataset=whole_dataset,
        random_state=seed,
        verbose=True,
    )

    # ===== model =====
    config = EGSP_End2EndConfig(
        ckpt_path=str(scfoundation_ckpt_path),
        frozenmore=True,
        cln_feats=["age", "gender", "pTNM"],
        num_gene_feats=1024,
        add_gene_feats=True,
        hidden_layers=[1024, 512, 256, 128],
        alpha=1,
        pool_type="max",
    )

    model = EGSP_End2End(config)

    # freeze encoder layers
    if def_freeze_layers > 0:
        modules_to_freeze = model.encoder.transformer_encoder[:def_freeze_layers]
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

    # freeze norm if fully frozen
    if def_freeze_layers >= 12:
        for name, param in model.encoder.norm.named_parameters():
            param.requires_grad = False
            print(f"[INFO] norm layer param {name} frozen")

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    model = model.to(device)

    data_collator = DataCollatorForEGSP_End2End()

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
        report_to=report_to,     # demo 默认 none；你想 tensorboard 也可以
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
                early_stopping_patience=10,
                early_stopping_threshold=0.0,
            )
        ],
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

    return End2EndResult(
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        test_metrics=test_metrics,
        test_cindex=float(cindex),
        test_risk=risk_test,
    )
