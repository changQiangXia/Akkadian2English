#!/usr/bin/env python3
"""
Step 3: 全参数微调 + 5-Fold CV + R-Drop
使用sqrt(BLEU * chrF++)作为保存最佳模型的指标
"""

import os
import sys
import json
import math
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
    set_seed,
)
from accelerate import Accelerator
from tqdm.auto import tqdm
import pandas as pd
import csv

# 评估指标
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """训练配置"""
    # 模型
    model_name: str = "checkpoints/dapt/final"  # DAPT后的模型
    # 如果跳过DAPT，可以直接用: "Hippopoto0/akkadianT5"
    
    # 数据
    train_csv: str = "data/processed/aligned_train_v2.csv"
    
    # 输出
    output_dir: str = "checkpoints/finetuned"
    
    # 5-Fold CV
    n_folds: int = 5
    current_fold: int = 3  # 0-4，用于并行训练多个fold
    
    # 训练参数
    num_epochs: int = 20
    batch_size: int = 4  # RTX 4090，全参数微调用较小batch
    gradient_accumulation_steps: int = 4  # 有效batch = 16
    learning_rate: float = 1e-5  # 更激进的学习率
    weight_decay: float = 0.01
    warmup_ratio: float = 0.3
    max_grad_norm: float = 0.5
    
    # R-Drop参数
    use_r_drop: bool = False  # 启用R-Drop正则化
    r_drop_alpha: float = 0.5  # KL散度权重
    
    # 序列长度
    max_source_length: int = 512
    max_target_length: int = 256
    
    # 解码参数（验证时用）
    num_beams: int = 4
    
    # 其他
    seed: int = 42
    eval_steps: int = 200
    save_steps: int = 400
    logging_steps: int = 50
    fp16: bool = True
    
    # 早停
    early_stopping_patience: int = 10


class TranslationDataset(Dataset):
    """翻译数据集"""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_source_length: int = 512,
        max_target_length: int = 256,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        source = item['source']
        target = item['target']
        
        # Tokenize source
        source_encoding = self.tokenizer(
            source,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            target,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Replace padding token id's with -100 for loss computation
        labels = target_encoding["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": source_encoding["input_ids"].squeeze(),
            "attention_mask": source_encoding["attention_mask"].squeeze(),
            "labels": labels,
            "raw_source": source,
            "raw_target": target,
        }


def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    计算评估指标
    主要指标: sqrt(BLEU * chrF++)
    """
    # sacrebleu需要List[List[str]]格式的references
    refs = [[ref] for ref in references]
    
    # BLEU
    bleu_metric = BLEU()
    bleu_score = bleu_metric.corpus_score(predictions, refs).score
    
    # chrF++
    chrf_metric = CHRF(word_order=2)  # chrF++
    chrf_score = chrf_metric.corpus_score(predictions, refs).score
    
    # Combined metric
    combined = math.sqrt(bleu_score * chrf_score)
    
    return {
        "bleu": bleu_score,
        "chrf": chrf_score,
        "combined": combined,
    }


class RDropLoss(nn.Module):
    """
    R-Drop: Regularized Dropout for Neural Networks
    对同一输入进行两次前向传播，计算KL散度约束
    """
    
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, logits1, logits2, labels):
        """
        logits1, logits2: [batch, seq_len, vocab_size]
        labels: [batch, seq_len]
        """
        # 标准交叉熵损失（取平均）
        ce_loss1 = F.cross_entropy(
            logits1.view(-1, logits1.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction='mean'
        )
        ce_loss2 = F.cross_entropy(
            logits2.view(-1, logits2.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction='mean'
        )
        ce_loss = (ce_loss1 + ce_loss2) / 2
        
        # KL散度（对有效位置计算）- 使用更稳定的计算方式
        # mask out padding
        mask = (labels != -100).float().unsqueeze(-1)  # [batch, seq_len, 1]
        
        # 对logits做softmax，添加数值稳定性
        p1 = F.softmax(logits1, dim=-1)
        p2 = F.softmax(logits2, dim=-1)
        
        # 避免log(0)，添加极小值
        eps = 1e-8
        p1 = torch.clamp(p1, min=eps, max=1.0)
        p2 = torch.clamp(p2, min=eps, max=1.0)
        
        # 对称KL散度: KL(p1||p2) + KL(p2||p1) / 2
        # KL(p||q) = sum(p * (log p - log q))
        log_p1 = torch.log(p1)
        log_p2 = torch.log(p2)
        
        kl1 = (p1 * (log_p1 - log_p2)).sum(dim=-1)
        kl2 = (p2 * (log_p2 - log_p1)).sum(dim=-1)
        
        kl = ((kl1 + kl2) / 2) * mask.squeeze(-1)
        kl = kl.sum() / (mask.sum() + eps)  # 避免除以0
        
        # 检查KL是否异常，如果太大则忽略
        if torch.isnan(kl) or torch.isinf(kl) or kl > 10.0:
            logger.warning(f"R-Drop KL divergence is abnormal (kl={kl:.4f}), using CE loss only")
            return ce_loss
        
        return ce_loss + self.alpha * kl


def train_fold(
    config: TrainingConfig,
    fold: int,
    train_data: List[Dict],
    val_data: List[Dict],
):
    """训练单个fold"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training Fold {fold + 1}/{config.n_folds}")
    logger.info(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    logger.info(f"{'='*60}\n")
    
    # 设置随机种子（所有fold相同，确保可比性）
    set_seed(config.seed)
    
    # Accelerator
    # 使用BF16代替FP16，避免数值溢出（RTX 4090支持BF16）
    accelerator = Accelerator(
        mixed_precision="bf16" if config.fp16 else "no",
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    
    # 加载模型
    logger.info(f"Loading model from: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
    
    # 创建数据集
    train_dataset = TranslationDataset(
        train_data, tokenizer,
        config.max_source_length, config.max_target_length
    )
    val_dataset = TranslationDataset(
        val_data, tokenizer,
        config.max_source_length, config.max_target_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    
    # 优化器
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    
    # 学习率调度
    num_update_steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
    max_train_steps = config.num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(max_train_steps * config.warmup_ratio)
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )
    
    # R-Drop loss
    r_drop_loss = RDropLoss(alpha=config.r_drop_alpha) if config.use_r_drop else None
    
    # 准备
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )
    
    # 训练状态
    global_step = 0
    best_combined = 0.0
    patience_counter = 0
    
    output_dir = Path(config.output_dir) / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    
    for epoch in range(config.num_epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0
        
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                if config.use_r_drop:
                    # R-Drop: 两次前向传播（dropout不同）
                    outputs1 = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    outputs2 = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    
                    loss = r_drop_loss(outputs1.logits, outputs2.logits, batch["labels"])
                else:
                    # 标准训练
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss = outputs.loss
                
                # 检查loss是否为NaN或Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN/Inf loss detected at step {step}, skipping batch")
                    optimizer.zero_grad()
                    continue
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                # 只在loss正常时累加
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    epoch_loss += loss.item()
                
                if global_step % config.logging_steps == 0:
                    progress_bar.set_postfix({
                        "epoch": epoch + 1,
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"
                    })
                
                # 验证
                if global_step % config.eval_steps == 0:
                    metrics = evaluate(accelerator, model, val_loader, tokenizer, config)
                    combined = metrics["combined"]
                    
                    logger.info(f"Step {global_step}: BLEU={metrics['bleu']:.2f}, "
                              f"chrF++={metrics['chrf']:.2f}, Combined={combined:.2f}")
                    
                    # 保存最佳模型
                    if combined > best_combined:
                        best_combined = combined
                        patience_counter = 0
                        
                        best_dir = output_dir / "best"
                        best_dir.mkdir(exist_ok=True)
                        
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(best_dir)
                        tokenizer.save_pretrained(best_dir)
                        
                        # 保存指标
                        with open(best_dir / "metrics.json", "w") as f:
                            json.dump(metrics, f, indent=2)
                        
                        logger.info(f"  -> New best! Saved to {best_dir}")
                    else:
                        patience_counter += 1
                    
                    model.train()
                    
                    # 早停检查
                    if patience_counter >= config.early_stopping_patience:
                        logger.info(f"Early stopping triggered after {global_step} steps")
                        break
        
        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} completed. Avg loss: {avg_loss:.4f}")
        
        if patience_counter >= config.early_stopping_patience:
            break
    
    logger.info(f"Fold {fold} training completed. Best combined: {best_combined:.2f}")
    return best_combined


@torch.no_grad()
def evaluate(
    accelerator: Accelerator,
    model,
    dataloader: DataLoader,
    tokenizer,
    config: TrainingConfig,
) -> Dict[str, float]:
    """验证"""
    model.eval()
    
    all_predictions = []
    all_references = []
    
    for batch in dataloader:
        # 生成预测
        generated_tokens = accelerator.unwrap_model(model).generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=config.max_target_length,
            num_beams=config.num_beams,
            early_stopping=True,
        )
        
        # 解码
        generated_tokens = accelerator.pad_across_processes(
            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
        )
        generated_tokens = accelerator.gather(generated_tokens)
        
        labels = batch["labels"]
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
        labels = accelerator.gather(labels)
        
        # 替换-100为pad_token_id以便解码
        labels[labels == -100] = tokenizer.pad_token_id
        
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        all_predictions.extend(decoded_preds)
        all_references.extend(decoded_labels)
    
    # 计算指标
    metrics = compute_metrics(all_predictions, all_references)
    return metrics


def load_data(csv_path: str) -> List[Dict]:
    """加载数据"""
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('source') and row.get('target'):
                # 过滤掉标记为continuation的数据（没有独立翻译的）
                if row.get('target') != '[continuation]':
                    data.append({
                        'source': row['source'],
                        'target': row['target'],
                        'type': row.get('type', 'unknown'),
                    })
    return data


def main():
    config = TrainingConfig()
    
    # 加载数据
    logger.info("Loading training data...")
    all_data = load_data(config.train_csv)
    logger.info(f"Loaded {len(all_data)} samples")
    
    # 5-Fold CV
    kfold = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    
    # 如果是运行所有folds
    if config.current_fold < 0 or config.current_fold >= config.n_folds:
        # 运行所有folds
        fold_scores = []
        for fold in range(config.n_folds):
            train_idx, val_idx = list(kfold.split(all_data))[fold]
            train_data = [all_data[i] for i in train_idx]
            val_data = [all_data[i] for i in val_idx]
            
            score = train_fold(config, fold, train_data, val_data)
            fold_scores.append(score)
        
        logger.info(f"\n{'='*60}")
        logger.info("Cross-Validation Results:")
        for i, score in enumerate(fold_scores):
            logger.info(f"  Fold {i+1}: {score:.2f}")
        logger.info(f"  Average: {sum(fold_scores)/len(fold_scores):.2f}")
        logger.info(f"{'='*60}")
    else:
        # 只运行指定的fold（用于并行训练）
        fold = config.current_fold
        train_idx, val_idx = list(kfold.split(all_data))[fold]
        train_data = [all_data[i] for i in train_idx]
        val_data = [all_data[i] for i in val_idx]
        
        train_fold(config, fold, train_data, val_data)


if __name__ == "__main__":
    main()
