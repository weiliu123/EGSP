import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from torch.utils.data import WeightedRandomSampler, DataLoader
from lifelines.utils import concordance_index
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, PreTrainedModel, PretrainedConfig

# from model.load import load_model_frommmf, gatherData
from egsp.scfoundation_bridge import import_scfoundation
from sklearn.model_selection import train_test_split

from sksurv.metrics import cumulative_dynamic_auc,concordance_index_ipcw
from sksurv.util import Surv
from sklearn.utils.validation import check_array
from sklearn.utils import resample
from lifelines.utils import concordance_index

import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt

class DeepSurvLoss(nn.Module):
    """
    DeepSurv negative log partial likelihood loss
    (Cox proportional hazards model).
    """

    def __init__(self, reduction: str = "mean"):
        """
        Parameters
        ----------
        reduction : str
            Specifies the reduction to apply to the output:
            'mean' | 'sum'
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        preds: torch.Tensor,
        times: torch.Tensor,
        events: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        preds : torch.Tensor
            Predicted risk scores with shape (batch_size, 1).
        times : torch.Tensor
            Survival times with shape (batch_size,).
        events : torch.Tensor
            Event indicators (1 = event occurred, 0 = censored)
            with shape (batch_size,).

        Returns
        -------
        torch.Tensor
            Computed DeepSurv loss.
        """

        # Select samples with observed events
        event_mask = events.bool()

        # If no events exist in the current batch, apply L2 regularization
        # to maintain gradient flow
        if not event_mask.any():
            reg_loss = 0.01 * torch.norm(preds, p=2)
            return reg_loss if self.reduction == "mean" else reg_loss * preds.size(0)

        # Extract risk scores and times for event samples
        preds_event = preds[event_mask].squeeze()   # (n_events,)
        times_event = times[event_mask]              # (n_events,)

        # Construct the risk set matrix: (batch_size, n_events)
        risk_set = (times.unsqueeze(1) >= times_event.unsqueeze(0)).float()

        # Expand predictions for broadcasting
        expanded_preds = preds.view(-1, 1)           # (batch_size, 1)

        # Numerically stable log-sum-exp computation
        max_vals, _ = torch.max(expanded_preds, dim=0, keepdim=True)
        stable_exp = torch.exp(expanded_preds - max_vals) * risk_set
        log_sum_exp = (
            torch.log(torch.sum(stable_exp, dim=0) + 1e-8)
            + max_vals.squeeze()
        )

        # Compute individual negative log partial likelihood
        individual_loss = log_sum_exp - preds_event
        total_loss = torch.sum(individual_loss)

        if self.reduction == "mean":
            return total_loss / preds_event.numel()

        return total_loss

# class EGSPConfig(PretrainedConfig):
#     model_type = "EGSP"  # 自定义模型类型标识
#
#     def __init__(
#             self,
#             cln_feats=["age", "gender", "pTNM"],
#             embedsize=768,
#             num_gene_feats=1024,
#             add_gene_feats=True,
#             hidden_layers=[1024, 512, 256, 128],
#             alpha=1,
#             **kwargs
#     ):
#         super().__init__(**kwargs)
#         self.embedsize = embedsize
#         self.cln_feats = cln_feats
#         self.hidden_layers = hidden_layers
#         self.num_gene_feats = num_gene_feats
#         self.add_gene_feats = add_gene_feats
#         self.alpha = alpha

class EGSPConfig(PretrainedConfig):
    """
    Configuration class for the EGSP model.
    """

    model_type = "EGSP"

    def __init__(
        self,
        cln_feats=None,
        embedsize: int = 768,
        num_gene_feats: int = 1024,
        add_gene_feats: bool = True,
        hidden_layers=None,
        alpha: float = 1.0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        cln_feats : list of str
            Names of clinical features to be used as inputs.
        embedsize : int
            Dimensionality of embedding features.
        num_gene_feats : int
            Number of raw gene expression features.
        add_gene_feats : bool
            Whether to include gene expression features.
        hidden_layers : list of int
            Hidden layer sizes for the survival prediction head.
        alpha : float
            Scaling factor applied to normalized gene expression features.
        """
        super().__init__(**kwargs)

        self.cln_feats = cln_feats or ["age", "gender", "pTNM"]
        self.embedsize = embedsize
        self.num_gene_feats = num_gene_feats
        self.add_gene_feats = add_gene_feats
        self.hidden_layers = hidden_layers or [1024, 512, 256, 128]
        self.alpha = alpha

class EGSP(PreTrainedModel):
    """
    EGSP model for survival prediction integrating embeddings,
    clinical features, and gene expression profiles.
    """

    config_class = EGSPConfig

    def __init__(self, config: EGSPConfig):
        super().__init__(config)

        self.cln_feats = config.cln_feats
        self.add_gene_feats = config.add_gene_feats
        self.alpha = config.alpha

        input_size = config.embedsize + len(self.cln_feats)

        if self.add_gene_feats:
            self.gene_proj = nn.Sequential(
                nn.Linear(config.num_gene_feats, 1024),
                # nn.LayerNorm(1024),
                # nn.Tanh(),
                # nn.Dropout(0.1),
            )
            input_size += 1024

        layers = []
        for hidden_dim in config.hidden_layers:
            layers.extend(
                [
                    nn.Linear(input_size, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            input_size = hidden_dim

        layers.append(nn.Linear(input_size, 1))

        self.survival_head = nn.Sequential(*layers)
        self.loss_fn = DeepSurvLoss()

    def forward(
        self,
        labels=None,
        embed=None,
        gene_exp_raw=None,
        age=None,
        gender=None,
        pTNM=None,
        time=None,
        status=None,
        **kwargs,
    ):
        """
        Forward pass of the EGSP model.

        Parameters
        ----------
        embed : torch.Tensor
            Embedding features with shape (batch_size, embed_dim).
        gene_exp_raw : torch.Tensor
            Raw gene expression matrix with shape (batch_size, num_genes).
        age, gender, pTNM : torch.Tensor or scalar
            Clinical covariates.
        labels : torch.Tensor, optional
            Survival labels with shape (batch_size, 2),
            where [:, 0] is time and [:, 1] is event indicator.

        Returns
        -------
        dict
            Dictionary containing loss (if labels are provided)
            and predicted risk scores.
        """

        clinical_inputs = []
        for feat_name in self.cln_feats:
            value = locals()[feat_name]
            if value is None:
                continue

            if value.ndimension() == 0:
                value = value.unsqueeze(0).expand(embed.size(0), 1)
            elif value.ndimension() == 1:
                value = value.unsqueeze(1)

            clinical_inputs.append(value)

        if clinical_inputs:
            cln_tensor = torch.cat(clinical_inputs, dim=1)
            combined_input = torch.cat([embed, cln_tensor], dim=1)
        else:
            combined_input = embed

        if self.add_gene_feats:
            gene_exp_raw = self.alpha * (
                gene_exp_raw - gene_exp_raw.mean(dim=0)
            )
            # proj_gene_raw = self.gene_proj(gene_exp_raw)
            combined_input = torch.cat([gene_exp_raw, combined_input], dim=1)

        risk_scores = self.survival_head(combined_input)

        if labels is not None:
            times = labels[:, 0]
            events = labels[:, 1]
            loss = self.loss_fn(risk_scores, times, events)
            return {"loss": loss, "logits": risk_scores}

        return {"logits": risk_scores}

class EGSP_End2EndConfig(PretrainedConfig):
    """
    Configuration class for the EGSP end-to-end survival model.
    """

    model_type = "EGSP_End2End"

    def __init__(
        self,
        cln_feats=None,
        num_gene_feats: int = 1024,
        add_gene_feats: bool = False,
        hidden_layers=None,
        alpha: float = 1.0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        cln_feats : list of str
            Names of clinical covariates.
        num_gene_feats : int
            Number of raw gene expression features.
        add_gene_feats : bool
            Whether to include raw gene expression features.
        hidden_layers : list of int
            Hidden layer sizes of the survival head.
        alpha : float
            Scaling factor for normalized gene expression features.
        """
        super().__init__(**kwargs)

        self.ckpt_path = kwargs.get("ckpt_path")
        self.frozenmore = kwargs.get("frozenmore", True)
        self.pool_type = kwargs.get("pool_type", "all")

        self.cln_feats = cln_feats or ["age", "gender", "pTNM"]
        self.num_gene_feats = num_gene_feats
        self.add_gene_feats = add_gene_feats
        self.hidden_layers = hidden_layers or [1024, 512, 256, 128]
        self.alpha = alpha

class EGSP_End2End(PreTrainedModel):
    config_class = EGSP_End2EndConfig

    def __init__(self, config):
        super().__init__(config)
        self.ckpt_path = config.ckpt_path
        self.frozenmore = config.frozenmore
        self.cln_feats = config.cln_feats
        self.pool_type = config.pool_type
        self.add_gene_feats = config.add_gene_feats
        self.alpha = config.alpha

        num_gene_feats = config.num_gene_feats
        hidden_layers = config.hidden_layers

        load_model_frommmf, self.gatherData = import_scfoundation()

        model, model_config = load_model_frommmf(self.ckpt_path)
        self.token_emb = model.token_emb
        self.pos_emb = model.pos_emb
        self.encoder = model.encoder

        if self.frozenmore:
            for _, p in self.token_emb.named_parameters():
                p.requires_grad = False
            for _, p in self.pos_emb.named_parameters():
                p.requires_grad = False
            print('self.pos_emb and self.token_emb also frozen')

        for na, param in self.encoder.named_parameters():
            param.requires_grad = False
        # for na, param in self.encoder.transformer_encoder[-2].named_parameters():
        #     print('self.encoder.transformer_encoder ',na,' have grad')
        #     param.requires_grad = True

        # 设置倒数第 1 层和倒数第 2 层的参数为可训练
        for i in [-4, -3, -2, -1]:  # ,-4, -3,
            for name, param in self.encoder.transformer_encoder[i].named_parameters():
                print(f'self.encoder.transformer_encoder {i} layer param: {name} set requires_grad=True')
                param.requires_grad = True

        # 解冻 encoder.norm 层
        for name, param in self.encoder.norm.named_parameters():
            param.requires_grad = True
            print(f'norm layer param {name} set to requires_grad = True')

        for na, param in self.encoder.named_parameters():
            print(na, ': ', param.requires_grad)

        if self.pool_type == "all":
            input_size = model_config['encoder']['hidden_dim'] * 4
        elif self.pool_type == 'max':
            input_size = model_config['encoder']['hidden_dim']
        else:
            raise ValueError('pool_type must be all or max')

        input_size += len(self.cln_feats)
        # input_size = model_config['encoder']['hidden_dim']

        if self.add_gene_feats:
            # 定义线性变换（可加 Dropout / ReLU）
            self.gene_proj = nn.Sequential(
                nn.Linear(num_gene_feats, 1024),
                # nn.LayerNorm(1024),
                # nn.Tanh(),  # 可选
                # nn.Dropout(0.1)
            )
            input_size += 1024

        layers = []

        # 构建每一层：Linear → ReLU → Dropout
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_size = hidden_dim  # 下一层输入

        # 输出层
        layers.append(nn.Linear(input_size, 1))

        # 深度生存预测头
        self.survival_head = nn.Sequential(*layers)

        self.model_config = model_config
        self.loss_fn = DeepSurvLoss()

    def forward(self, labels=None, gene_exp=None, gene_exp_raw=None, age=None, gender=None, pTNM=None, time=None, status=None, **kwargs):
        x = gene_exp  # (B, L)

        # 计算 totalcount（每行 sum 后取 log10），形状为 (B,)
        totalcount = torch.log10(x.sum(dim=1) + 1e-8)  # 加个小常数防止 log(0)

        # 扩展为 (B, 2)，每个 totalcount 复制两次
        totalcount_expanded = totalcount.unsqueeze(1).repeat(1, 2)  # shape: (B, 2)

        # 拼接到每行末尾
        x = torch.cat([x, totalcount_expanded], dim=1)  # shape: (B, L+2)

        value_labels = x > 0
        x, x_padding = self.gatherData(x, value_labels, self.model_config['pad_token_id'])
        data_gene_ids = torch.arange(19266, device=x.device).repeat(x.shape[0], 1)
        position_gene_ids, _ = self.gatherData(data_gene_ids, value_labels,
                                          self.model_config['pad_token_id'])

        x = self.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
        position_emb = self.pos_emb(position_gene_ids)
        x += position_emb

        geneemb = self.encoder(x, x_padding)

        if self.pool_type == 'all':
            geneemb1 = geneemb[:, -1, :]
            geneemb2 = geneemb[:, -2, :]
            geneemb3, _ = torch.max(geneemb[:, :-2, :], dim=1)
            geneemb4 = torch.mean(geneemb[:, :-2, :], dim=1)
            geneembmerge = torch.concat([geneemb1, geneemb2, geneemb3, geneemb4], axis=1)
        elif self.pool_type == 'max':
            geneembmerge, _ = torch.max(geneemb, dim=1)
        else:
            raise ValueError('pool_type must be all or max')

        clinical_inputs = []
        for feat_name in self.cln_feats:
            value = locals()[feat_name]  # 从局部变量中取
            # value = locals().get(feat_name, None)  # 从局部变量中取
            if value is None:
                continue
                # raise ValueError(f"Missing clinical feature: {feat_name}")
            else:
                # 如果特征是标量，先进行 unsqueeze 扩展为一维张量 (batch_size, 1)
                if value.ndimension() == 0:  # 如果是标量（零维张量）
                    value = value.unsqueeze(0).expand(gene_exp.shape[0], 1)  # 转为 (batch_size, 1)
                # 如果已经是一维张量，直接使用它
                elif value.ndimension() == 1:  # 如果是 (batch_size,)
                    value = value.unsqueeze(1)  # 转为 (batch_size, 1)
            clinical_inputs.append(value)

        if clinical_inputs:
            cln_tensor = torch.cat(clinical_inputs, dim=1)
            combined_input = torch.cat([geneembmerge, cln_tensor], dim=1)
        else:
            combined_input = geneembmerge

        if self.add_gene_feats:
            gene_exp_raw = self.alpha * (gene_exp_raw - gene_exp_raw.mean(dim=0)) # / (gene_exp_raw.std(dim=0) + 1e-6)
            # proj_gene_raw = self.gene_proj(gene_exp_raw)
            combined_input = torch.cat([gene_exp_raw, combined_input], dim=1)

        # 生存风险评分 (风险越高生存概率越低)
        risk_scores = self.survival_head(combined_input)

        # 计算损失（如果提供标签）
        if labels is not None:
            times = labels[:, 0]  # 解包生存时间
            events = labels[:, 1]  # 解包事件指示
            loss = self.loss_fn(risk_scores, times, events)
            return {"loss": loss, "logits": risk_scores}
        else:
            return {"logits": risk_scores}

class EGSP_End2End(PreTrainedModel):
    """
    End-to-end EGSP model integrating scFoundation encoder,
    clinical variables, and optional gene expression features
    for survival prediction.
    """

    config_class = EGSP_End2EndConfig

    def __init__(self, config: EGSP_End2EndConfig):
        super().__init__(config)

        self.ckpt_path = config.ckpt_path
        self.frozenmore = config.frozenmore
        self.cln_feats = config.cln_feats
        self.pool_type = config.pool_type
        self.add_gene_feats = config.add_gene_feats
        self.alpha = config.alpha

        hidden_layers = config.hidden_layers
        num_gene_feats = config.num_gene_feats

        # Load scFoundation model and utilities
        load_model_frommmf, self.gatherData = import_scfoundation()
        model, model_config = load_model_frommmf(self.ckpt_path)

        self.token_emb = model.token_emb
        self.pos_emb = model.pos_emb
        self.encoder = model.encoder
        self.model_config = model_config

        # Freeze token and position embeddings if specified
        if self.frozenmore:
            for p in self.token_emb.parameters():
                p.requires_grad = False
            for p in self.pos_emb.parameters():
                p.requires_grad = False
            print("token_emb and pos_emb are frozen")

        # Freeze entire encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Unfreeze last several transformer layers
        # for i in [-4, -3, -2, -1]:
        #     for name, param in self.encoder.transformer_encoder[i].named_parameters():
        #         param.requires_grad = True
        #         print(
        #             f"encoder.transformer_encoder[{i}] parameter {name} "
        #             f"set requires_grad=True"
        #         )
        #
        # # Unfreeze layer normalization
        # for name, param in self.encoder.norm.named_parameters():
        #     param.requires_grad = True
        #     print(f"encoder.norm parameter {name} set requires_grad=True")

        # Determine pooled embedding dimension
        hidden_dim = model_config["encoder"]["hidden_dim"]
        if self.pool_type == "all":
            input_size = hidden_dim * 4
        elif self.pool_type == "max":
            input_size = hidden_dim
        else:
            raise ValueError("pool_type must be 'all' or 'max'")

        input_size += len(self.cln_feats)

        # Optional raw gene expression projection
        if self.add_gene_feats:
            self.gene_proj = nn.Sequential(
                nn.Linear(num_gene_feats, 1024),
                # nn.LayerNorm(1024),
                # nn.Tanh(),
                # nn.Dropout(0.1),
            )
            input_size += 1024

        # Build survival prediction head
        layers = []
        for hidden_dim in hidden_layers:
            layers.extend(
                [
                    nn.Linear(input_size, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            input_size = hidden_dim

        layers.append(nn.Linear(input_size, 1))
        self.survival_head = nn.Sequential(*layers)

        self.loss_fn = DeepSurvLoss()

    def forward(
        self,
        labels=None,
        gene_exp=None,
        gene_exp_raw=None,
        age=None,
        gender=None,
        pTNM=None,
        time=None,
        status=None,
        **kwargs,
    ):
        """
        Forward pass of the EGSP end-to-end model.

        Parameters
        ----------
        gene_exp : torch.Tensor
            Gene expression count matrix of shape (B, L).
        gene_exp_raw : torch.Tensor
            Raw gene expression features.
        age, gender, pTNM : torch.Tensor or scalar
            Clinical covariates.
        labels : torch.Tensor, optional
            Survival labels with shape (B, 2):
            [time, event].

        Returns
        -------
        dict
            Model outputs including loss (if labels are provided)
            and risk scores.
        """

        x = gene_exp  # (B, L)

        # Compute total counts per sample (log10-transformed)
        totalcount = torch.log10(x.sum(dim=1) + 1e-8)
        totalcount_expanded = totalcount.unsqueeze(1).repeat(1, 2)

        x = torch.cat([x, totalcount_expanded], dim=1)

        value_mask = x > 0
        x, x_padding = self.gatherData(
            x, value_mask, self.model_config["pad_token_id"]
        )

        gene_ids = torch.arange(19266, device=x.device).repeat(x.size(0), 1)
        pos_ids, _ = self.gatherData(
            gene_ids, value_mask, self.model_config["pad_token_id"]
        )

        x = self.token_emb(x.unsqueeze(2).float(), output_weight=0)
        x = x + self.pos_emb(pos_ids)

        gene_embeddings = self.encoder(x, x_padding)

        # Pooling strategy
        if self.pool_type == "all":
            emb_last = gene_embeddings[:, -1, :]
            emb_second_last = gene_embeddings[:, -2, :]
            emb_max, _ = torch.max(gene_embeddings[:, :-2, :], dim=1)
            emb_mean = torch.mean(gene_embeddings[:, :-2, :], dim=1)
            geneemb_merge = torch.cat(
                [emb_last, emb_second_last, emb_max, emb_mean], dim=1
            )
        elif self.pool_type == "max":
            geneemb_merge, _ = torch.max(gene_embeddings, dim=1)
        else:
            raise ValueError("pool_type must be 'all' or 'max'")

        # Process clinical variables
        clinical_inputs = []
        for feat_name in self.cln_feats:
            value = locals()[feat_name]
            if value is None:
                continue

            if value.ndimension() == 0:
                value = value.unsqueeze(0).expand(gene_exp.size(0), 1)
            elif value.ndimension() == 1:
                value = value.unsqueeze(1)

            clinical_inputs.append(value)

        if clinical_inputs:
            cln_tensor = torch.cat(clinical_inputs, dim=1)
            combined_input = torch.cat([geneemb_merge, cln_tensor], dim=1)
        else:
            combined_input = geneemb_merge

        # Optional raw gene expression features
        if self.add_gene_feats:
            gene_exp_raw = self.alpha * (
                gene_exp_raw - gene_exp_raw.mean(dim=0)
            )
            combined_input = torch.cat(
                [gene_exp_raw, combined_input], dim=1
            )

        # Risk prediction (higher score = higher risk)
        risk_scores = self.survival_head(combined_input)

        if labels is not None:
            times = labels[:, 0]
            events = labels[:, 1]
            loss = self.loss_fn(risk_scores, times, events)
            return {"loss": loss, "logits": risk_scores}

        return {"logits": risk_scores}

# class DataCollatorForEGSP_End2End:
#     def __call__(self, features):
#         # print("🧪 features 样本:", features[0].keys())
#         batch = {}
#
#         # gene expression (必须存在)
#         batch["gene_exp"] = torch.tensor([f["gene_exp"] for f in features], dtype=torch.float32)
#
#         # gene expression raw
#         if all("gene_exp_raw" in f for f in features):
#             batch["gene_exp_raw"] = torch.tensor([f["gene_exp_raw"] for f in features], dtype=torch.float32)
#
#         # age
#         if all("age" in f for f in features):
#             batch["age"] = torch.tensor([f["age"] for f in features], dtype=torch.float32)
#         # else:
#         #     raise ValueError("Missing 'age' in one or more samples")
#
#         # gender
#         if all("gender" in f for f in features):
#             batch["gender"] = torch.tensor([f["gender"] for f in features], dtype=torch.float32)
#         # else:
#         #     raise ValueError("Missing 'gender' in one or more samples")
#
#         # pTNM
#         if all("pTNM" in f for f in features):
#             batch["pTNM"] = torch.tensor([f["pTNM"] for f in features], dtype=torch.float32)
#         # else:
#         #     raise ValueError("Missing 'pTNM' in one or more samples")
#
#         # labels: [[time, event]]
#         if all("time" in f and "status" in f for f in features):
#             times = torch.tensor([f["time"] for f in features], dtype=torch.float32)
#             events = torch.tensor([f["status"] for f in features], dtype=torch.float32)
#             batch["labels"] = torch.stack([times, events], dim=1)
#         else:
#             batch["labels"] = None  # 推理时可不提供标签
#
#         return batch

class DataCollatorForEGSP_End2End:
    """
    Data collator for the EGSP end-to-end survival model.
    """

    def __call__(self, features):
        """
        Collate a list of samples into a batch.

        Parameters
        ----------
        features : list of dict
            Each element corresponds to one sample.

        Returns
        -------
        dict
            A batch dictionary containing model inputs and labels.
        """
        batch = {}

        # Gene expression counts (required)
        batch["gene_exp"] = torch.tensor(
            [f["gene_exp"] for f in features], dtype=torch.float32
        )

        # Raw gene expression features (optional)
        if all("gene_exp_raw" in f for f in features):
            batch["gene_exp_raw"] = torch.tensor(
                [f["gene_exp_raw"] for f in features], dtype=torch.float32
            )

        # Clinical features (optional)
        if all("age" in f for f in features):
            batch["age"] = torch.tensor(
                [f["age"] for f in features], dtype=torch.float32
            )

        if all("gender" in f for f in features):
            batch["gender"] = torch.tensor(
                [f["gender"] for f in features], dtype=torch.float32
            )

        if all("pTNM" in f for f in features):
            batch["pTNM"] = torch.tensor(
                [f["pTNM"] for f in features], dtype=torch.float32
            )

        # Survival labels: [time, event]
        if all("time" in f and "status" in f for f in features):
            times = torch.tensor(
                [f["time"] for f in features], dtype=torch.float32
            )
            events = torch.tensor(
                [f["status"] for f in features], dtype=torch.float32
            )
            batch["labels"] = torch.stack([times, events], dim=1)
        else:
            # Labels are optional during inference
            batch["labels"] = None

        return batch

# class DataCollatorForEGSP:
#     def __call__(self, features):
#         batch = {}
#
#         # gene expression (必须存在)
#         batch["embed"] = torch.tensor([f["embed"] for f in features], dtype=torch.float32)
#
#         # gene expression raw
#         if all("gene_exp_raw" in f for f in features):
#             batch["gene_exp_raw"] = torch.tensor([f["gene_exp_raw"] for f in features], dtype=torch.float32)
#
#         # age
#         if all("age" in f for f in features):
#             batch["age"] = torch.tensor([f["age"] for f in features], dtype=torch.float32)
#         # else:
#         #     raise ValueError("Missing 'age' in one or more samples")
#
#         # gender
#         if all("gender" in f for f in features):
#             batch["gender"] = torch.tensor([f["gender"] for f in features], dtype=torch.float32)
#         # else:
#         #     raise ValueError("Missing 'gender' in one or more samples")
#
#         # pTNM
#         if all("pTNM" in f for f in features):
#             batch["pTNM"] = torch.tensor([f["pTNM"] for f in features], dtype=torch.float32)
#         # else:
#         #     raise ValueError("Missing 'pTNM' in one or more samples")
#
#         # labels: [[time, event]]
#         if all("time" in f and "status" in f for f in features):
#             times = torch.tensor([f["time"] for f in features], dtype=torch.float32)
#             events = torch.tensor([f["status"] for f in features], dtype=torch.float32)
#             batch["labels"] = torch.stack([times, events], dim=1)
#         else:
#             batch["labels"] = None  # 推理时可不提供标签
#
#         return batch

class DataCollatorForEGSP:
    """
    Data collator for the EGSP survival model (embedding-based input).
    """

    def __call__(self, features):
        """
        Collate a list of samples into a batch.

        Parameters
        ----------
        features : list of dict
            Each element corresponds to one sample.

        Returns
        -------
        dict
            A batch dictionary containing model inputs and labels.
        """
        batch = {}

        # Embedding features (required)
        batch["embed"] = torch.tensor(
            [f["embed"] for f in features], dtype=torch.float32
        )

        # Raw gene expression features (optional)
        if all("gene_exp_raw" in f for f in features):
            batch["gene_exp_raw"] = torch.tensor(
                [f["gene_exp_raw"] for f in features], dtype=torch.float32
            )

        # Clinical features (optional)
        if all("age" in f for f in features):
            batch["age"] = torch.tensor(
                [f["age"] for f in features], dtype=torch.float32
            )

        if all("gender" in f for f in features):
            batch["gender"] = torch.tensor(
                [f["gender"] for f in features], dtype=torch.float32
            )

        if all("pTNM" in f for f in features):
            batch["pTNM"] = torch.tensor(
                [f["pTNM"] for f in features], dtype=torch.float32
            )

        # Survival labels: [time, event]
        if all("time" in f and "status" in f for f in features):
            times = torch.tensor(
                [f["time"] for f in features], dtype=torch.float32
            )
            events = torch.tensor(
                [f["status"] for f in features], dtype=torch.float32
            )
            batch["labels"] = torch.stack([times, events], dim=1)
        else:
            # Labels are optional during inference
            batch["labels"] = None

        return batch

# class BalancedDataLoader(DataLoader):
#     """
#     DataLoader wrapper that filters out batches with too few events.
#     """
#
#     def __iter__(self):
#         for batch in super().__iter__():
#             # labels shape: [batch_size, 2] -> [time, event]
#             events = batch["labels"][:, 1]
#             num_events = int(events.sum().item())
#
#             # Ensure sufficient number of events per batch
#             if num_events >= 3:
#                 yield batch
#             else:
#                 # Optional: keep for debugging
#                 print(f"Skipped batch with num_events = {num_events}")

class BalancedDataLoader(DataLoader):
    """
    DataLoader wrapper that filters out batches with too few event samples.

    Assumes:
        batch["labels"] shape == [batch_size, 2]
        labels[:, 1] is the event indicator (0/1)
    """

    def __init__(self, *args, min_events: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_events = min_events

    def __iter__(self):
        for batch in super().__iter__():
            if "labels" not in batch or batch["labels"] is None:
                # 推理阶段：不做事件过滤
                yield batch
                continue

            # labels: [time, event]
            events = batch["labels"][:, 1]

            # 确保是 tensor，防御性写法
            if not torch.is_tensor(events):
                events = torch.as_tensor(events)

            num_events = int(events.sum().item())

            # Ensure sufficient number of events per batch
            if num_events >= self.min_events:
                yield batch
            else:
                # Optional: keep for debugging
                # print(f"Skipped batch (num_events={num_events})")
                continue

class SurvTrainer(Trainer):
    """
    Custom Trainer for survival analysis with event-aware sampling.
    """

    def __init__(
        self,
        model,
        args=None,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
        train_times=None,
        train_events=None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs,
        )

        # Cache training labels for metric computation
        if train_dataset is not None:
            self.train_times = np.asarray(train_dataset["time"], dtype=np.float32)
            self.train_events = np.asarray(train_dataset["status"], dtype=np.uint8)
            self._inject_sampler()
        else:
            self.train_times = train_times
            self.train_events = train_events

        # Bind survival metrics
        self.compute_metrics = (
            lambda eval_pred: self.compute_surv_metrics(eval_pred)
        )

        self.data_collator = data_collator

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: torch.optim.Optimizer = None,
    ):
        """
        Override default scheduler with cosine annealing warm restarts.
        """
        print("Using CosineAnnealingWarmRestarts scheduler")

        self.lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=2000,
            T_mult=1,
            eta_min=2e-6,
            last_epoch=-1,
        )
        return self.lr_scheduler

    def compute_surv_metrics(self, eval_pred):
        """
        Compute survival metrics using cached training labels.
        """
        return compute_survival_metrics(
            eval_pred,
            self.train_times,
            self.train_events,
        )

    def get_train_dataloader(self):
        """
        Override train DataLoader to support event-aware sampling
        and batch-level event filtering.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training dataset is not initialized")

        # Prefer dataset-level custom sampler if provided
        if hasattr(self.train_dataset, "sampler"):
            return BalancedDataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=self.train_dataset.sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        return super().get_train_dataloader()

    def _create_event_aware_sampler(self) -> WeightedRandomSampler:
        """
        Create a sampler that up-weights event samples.
        """
        events = self.train_events
        event_ratio = events.mean()

        weights = np.where(
            events == 1,
            1.0 / (event_ratio + 1e-7),
            1.0,
        )

        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(events),
            replacement=True,
        )

    def _inject_sampler(self) -> None:
        """
        Inject event-aware sampler into the training dataset.
        """
        if self.train_dataset is None:
            raise ValueError("Cannot inject sampler: training dataset is None")

        if "status" not in self.train_dataset.features:
            raise KeyError(
                "Training dataset must contain 'status' field for event-aware sampling"
            )

        self.train_dataset.sampler = self._create_event_aware_sampler()


def compute_survival_metrics(eval_pred, train_times, train_events):

    # 初始设置
    predictions, labels = eval_pred
    times = labels[:, 0].astype(np.float32)
    events = labels[:, 1].astype(np.uint8)  # 明确数据类型
    risk_scores = -predictions[:, 0]

    c_index = concordance_index(times, risk_scores, events)
    metrics = {"eval_c_index": c_index}

    return metrics

def main_gene_selection(X_df, gene_list):
    """
    Describe:
        rebuild the input adata to select target genes encode protein
    Parameters:
        adata->`~anndata.AnnData` object: adata with var index_name by gene symbol
        gene_list->list: wanted target gene
    Returns:
        adata_new->`~anndata.AnnData` object
        to_fill_columns->list: zero padding gene
    """
    to_fill_columns = list(set(gene_list) - set(X_df.columns))
    padding_df = pd.DataFrame(np.zeros((X_df.shape[0], len(to_fill_columns))),
                              columns=to_fill_columns,
                              index=X_df.index)
    X_df = pd.DataFrame(np.concatenate([df.values for df in [X_df, padding_df]], axis=1),
                        index=X_df.index,
                        columns=list(X_df.columns) + list(padding_df.columns))
    X_df = X_df[gene_list]

    var = pd.DataFrame(index=X_df.columns)
    var['mask'] = [1 if i in to_fill_columns else 0 for i in list(var.index)]
    return X_df, to_fill_columns, var


def get_train_eval_test_dataset(
    dataset,
    train_ratio=0.7,
    eval_ratio=0.15,
    test_ratio=0.15,
    random_state=42,
    verbose=True
):
    """
    Split dataset into train / eval / test with stratification on status.

    Parameters
    ----------
    dataset : datasets.Dataset
        HuggingFace Dataset containing a 'status' field.
    train_ratio : float
        Proportion of samples used for training.
    eval_ratio : float
        Proportion of samples used for evaluation.
    test_ratio : float
        Proportion of samples used for testing.
    random_state : int
        Random seed for reproducibility.
    verbose : bool
        Whether to print split summary.

    Returns
    -------
    train_data, eval_data, test_data : datasets.Dataset
    """

    assert abs(train_ratio + eval_ratio + test_ratio - 1.0) < 1e-6, \
        "train_ratio + eval_ratio + test_ratio must sum to 1."

    # ---------- Step 1: split train vs (eval + test) ----------
    events = dataset["status"]
    val_test_ratio = eval_ratio + test_ratio

    train_indices, val_test_indices = train_test_split(
        range(len(dataset)),
        test_size=val_test_ratio,
        stratify=events,
        random_state=random_state
    )

    train_data = dataset.select(train_indices)
    val_test_data = dataset.select(val_test_indices)

    # ---------- Step 2: split eval vs test ----------
    events_val_test = val_test_data["status"]
    test_ratio_rel = test_ratio / (eval_ratio + test_ratio)

    eval_indices, test_indices = train_test_split(
        range(len(val_test_data)),
        test_size=test_ratio_rel,
        stratify=events_val_test,
        random_state=random_state
    )

    eval_data = val_test_data.select(eval_indices)
    test_data = val_test_data.select(test_indices)

    # ---------- Optional logging ----------
    if verbose:
        print(train_data)
        print(eval_data)
        print(test_data)
        print("Num of events in train_data:", sum(train_data["status"]))
        print("Num of events in eval_data:", sum(eval_data["status"]))
        print("Num of events in test_data:", sum(test_data["status"]))

    return train_data, eval_data, test_data


def safe_harrell_cindex(time, event, risk):
    """
    Safely compute Harrell's concordance index (C-index).

    Notes
    -----
    - This function uses lifelines' `concordance_index`.
    - In survival analysis, the provided `risk` score follows the convention:
        higher risk score -> shorter survival time.
    - However, lifelines assumes:
        higher predicted score -> longer survival time.
    - Therefore, the risk scores are negated before computing the C-index.

    Parameters
    ----------
    time : array-like
        Observed survival times.
    event : array-like
        Event indicators (1 = event occurred, 0 = censored).
    risk : array-like
        Predicted risk scores (higher means higher risk).

    Returns
    -------
    cindex : float
        Harrell's C-index. Returns np.nan if computation fails.
    error_message : str or None
        Error message if an exception occurs; otherwise None.
    """
    try:
        cindex = concordance_index(
            time,
            -risk,                  # negate risk: higher score -> longer survival
            event_observed=event
        )
        return float(cindex), None
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        return np.nan, error_msg