import warnings
from typing import Optional, List, Tuple
from einops.layers.torch import Rearrange
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import (
    BertConfig, BertEmbeddings, BertEncoder, BertLayer, BertAttention
)
from transformers.modeling_utils import ModuleUtilsMixin
from torchmetrics import PearsonCorrCoef

class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size = 2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange('b d c (n p) -> b d c n p', p = pool_size)
        self.to_attn_logits = nn.Conv2d(dim, dim, 1 , bias = False)

    def forward(self, x):
        b, _, s, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = F.pad(x, (0, remainder), value = 0)
            mask = torch.zeros((b, 1,s, n), dtype = torch.bool, device = x.device)
            mask = F.pad(mask, (0, remainder), value = True)
        x = self.pool_fn(x)
        in_x=x[:,:,0,:]
        logits=self.to_attn_logits(in_x)
        logits=torch.unsqueeze(logits,dim=2)
        for i in range(1,s):
            x_s=x[:,:,i,:]
            s_logits = self.to_attn_logits(x_s)
            s_logits = torch.unsqueeze(s_logits, dim=2)
            logits=torch.cat((logits,s_logits),2)
        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)

        attn = logits.softmax(dim = -1)
        return (x * attn).sum(dim = -1)


class AttentionPool2(nn.Module):
    def __init__(self, dim, pool_size = 2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange('b d (n p) -> b d n p', p = pool_size)
        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias = False)

    def forward(self, x):
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = F.pad(x, (0, remainder), value = 0)
            mask = torch.zeros((b, 1, n), dtype = torch.bool, device = x.device)
            mask = F.pad(mask, (0, remainder), value = True)

        x = self.pool_fn(x)
        logits = self.to_attn_logits(x)
        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)

        attn = logits.softmax(dim = -1)
        return (x * attn).sum(dim = -1)

class AttentionPool3(nn.Module):
    def __init__(self, dim, pool_size = 2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange('b (n p) -> b n p', p = pool_size)
        self.to_attn_logits = nn.Conv1d(dim//2, dim//2, 1, bias = False)

    def forward(self, x):
        b, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = F.pad(x, (0, remainder), value = 0)
            mask = torch.zeros((b, 1, n), dtype = torch.bool, device = x.device)
            mask = F.pad(mask, (0, remainder), value = True)

        x = self.pool_fn(x)
        logits = self.to_attn_logits(x)
        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)

        attn = logits.softmax(dim = -1)
        attnx =  x*attn 
        residual = attnx + x
        return residual.sum(dim = -1)


class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x


def ConvBlock(dim, dim_out = None, kernel_size = 1):
    return nn.Sequential(
        nn.BatchNorm2d(dim),
        #GELU(),
        nn.Conv2d(dim, default(dim_out, dim), kernel_size, padding = kernel_size // 2)
    )


def ConvBlock2(dim, dim_out = None, kernel_size = 1):
    return nn.Sequential(
        nn.BatchNorm1d(dim),
        #GELU(),
        nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding = kernel_size // 2)
    )


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class GSCNN(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int = 16,
        type_vocab_size: int = 6,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_attention_heads: int = 8,
        intermediate_size: int = 256,
        hidden_act: str = "gelu",
        num_labels: int = 1,
        learning_rate: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        norm_weight_decay: float = 0.0,
        max_epochs: int = 60,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.norm_weight_decay = norm_weight_decay
        self.max_epochs = max_epochs

        self.config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            type_vocab_size=type_vocab_size,
            max_position_embeddings=320,#max_lenght+1
            dropout_rate = dropout_rate,
        )

        self.embeddings = BertEmbeddings(self.config)

        self.tmp_embedding = nn.Embedding(41, hidden_size*2) #max_population_num+1

        self.tmp_convs = nn.Sequential(
            nn.BatchNorm1d(hidden_size*2),
            AttentionPool3(hidden_size*2, pool_size = 2),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(hidden_size),
            GELU(),          
        )

        self.convs1 = nn.Sequential(
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, 128, kernel_size=3, stride=2, padding=1),
            AttentionPool(128, pool_size = 2),
            nn.Dropout(dropout_rate),
            GELU(),
            Residual(ConvBlock(128, 128, 1)),

            nn.BatchNorm2d(128),
            nn.Conv2d(128, 384, kernel_size=3, stride=2, padding=1),
            AttentionPool(384, pool_size = 2),
            nn.Dropout(dropout_rate),            
            GELU(),
            Residual(ConvBlock(384, 384, 1)),
            
            nn.BatchNorm2d(384),
            nn.Conv2d(384, hidden_size, kernel_size=3, stride=2, padding=1),
            AttentionPool(hidden_size, pool_size = 2),
            nn.Dropout(dropout_rate),            
            GELU(),
            Residual(ConvBlock(hidden_size, hidden_size, 1)),
        )

        self.convs2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, 256, kernel_size=4, stride=2, padding=1),
            AttentionPool2(256, pool_size = 2),
            nn.Dropout(dropout_rate),
            GELU(),
            Residual(ConvBlock2(256, 256, 1)),
            
            nn.BatchNorm1d(256),
            nn.Conv1d(256, hidden_size, kernel_size=40, stride=2, padding=1),
            AttentionPool2(hidden_size, pool_size = 2),
            nn.Dropout(dropout_rate),
            GELU(),
            Residual(ConvBlock2(hidden_size, hidden_size, 1)),
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size*2, num_labels),           
            nn.Dropout(dropout_rate),
        )

        self.pearson = PearsonCorrCoef(num_outputs=num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        tmp_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ):
        batch_size, num_seqs, seq_length = input_ids.shape
        embedding_output = self.embeddings(
            input_ids=input_ids.long().view(batch_size * num_seqs, seq_length),
            position_ids=position_ids,
            token_type_ids=token_type_ids.long().view(batch_size * num_seqs, seq_length),
        )
        embedding_output1 = embedding_output.view(batch_size, num_seqs, seq_length, -1)
        tmp_embedding = self.tmp_embedding(tmp_ids.long())
        embedding_output2 =embedding_output1+ self.tmp_convs(tmp_embedding).view(batch_size, 1, 1, -1)*1e-1

        x1 = embedding_output1.permute(0, 3, 1, 2) 
        x1 = self.convs1(x1)
        x1 = x1.mean(2).mean(2)        
        x2 = embedding_output2.permute(0, 3, 1, 2)
        x2 = x2.reshape(batch_size, 64, -1)
        x2 = self.convs2(x2)
        x2 = x2.mean(2)
        x = torch.cat((x1,x2),1)
        logits = self.predictor(x)        
        return logits,embedding_output1,embedding_output2


    def _training_and_validation_step(self, batch, batch_idx: int):
        seqs, type_ids, tmp_ids, labels = batch
        labels = labels[:, 1:2]
        logits,embedding_output1,embedding_output2 = self.forward(
            input_ids=seqs,
            token_type_ids=type_ids,
            tmp_ids=tmp_ids,
        )

        loss = F.mse_loss(logits, labels)

        if not self.training:
            if self.num_labels == 1:
                self.pearson.update(logits.squeeze(), labels.squeeze())
            else:
                self.pearson.update(logits, labels)

        return logits,loss

    def training_step(self, batch, batch_idx: int):
        logits,loss = self._training_and_validation_step(batch, batch_idx)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)                  
        return loss

    def validation_step(self, batch, batch_idx: int):
        logits,loss= self._training_and_validation_step(batch, batch_idx)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

            
    def validation_epoch_end(self, outputs):
        pcc = self.pearson.compute()
        if self.num_labels > 1:
            for i, pcc_i in enumerate(pcc.tolist()):
                self.log(f"val/pcc_{i}", pcc_i ** 2, prog_bar=True)
        else:
            self.log(f"val/pcc", pcc ** 2, prog_bar=True)

    def configure_optimizers(self):
        parameters = set_weight_decay(
            model=self,
            weight_decay=self.weight_decay,
            norm_weight_decay=self.norm_weight_decay,
        )

        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.learning_rate,
            #momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }


def set_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    norm_classes: Optional[List[type]] = None,
    custom_keys_weight_decay: Optional[List[Tuple[str, float]]] = None,
):
    if not norm_classes:
        norm_classes = [
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.modules.instancenorm._InstanceNorm,
            torch.nn.LocalResponseNorm,
        ]
    norm_classes = tuple(norm_classes)

    params = {
        "other": [],
        "norm": [],
    }
    params_weight_decay = {
        "other": weight_decay,
        "norm": norm_weight_decay,
    }
    custom_keys = []
    if custom_keys_weight_decay is not None:
        for key, weight_decay in custom_keys_weight_decay:
            params[key] = []
            params_weight_decay[key] = weight_decay
            custom_keys.append(key)

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_custom_key = False
            for key in custom_keys:
                target_name = f"{prefix}.{name}" if prefix != "" and "." in key else name
                if key == target_name:
                    params[key].append(p)
                    is_custom_key = True
                    break
            if not is_custom_key:
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append({"params": params[key], "weight_decay": params_weight_decay[key]})

    return param_groups
