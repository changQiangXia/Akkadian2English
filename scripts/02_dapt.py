#!/usr/bin/env python3
"""
Step 2: DAPT (Domain Adaptive Pre-training)
使用所有阿卡德语文本进行无监督Masked Language Modeling
让T5模型适应古亚述语风格
"""

import os
import sys
import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup,
    set_seed,
)
from accelerate import Accelerator
from tqdm.auto import tqdm
import csv

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DAPTConfig:
    """DAPT配置"""
    # 模型
    model_name: str = "Hippopoto0/akkadianT5"  # 基线模型
    
    # 数据路径
    train_csv: str = "data/processed/aligned_train_v2.csv"
    test_csv: str = "data/extracted/test.csv"
    raw_train: str = "data/extracted/train.csv"
    
    # 输出
    output_dir: str = "checkpoints/dapt"
    
    # 训练参数
    num_epochs: int = 10
    batch_size: int = 4  # RTX 4090 24GB，全参数微调用较小batch
    gradient_accumulation_steps: int = 4  # 有效batch_size = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # 序列长度
    max_source_length: int = 512
    max_target_length: int = 512
    
    # MLM参数 (T5使用span corruption)
    noise_density: float = 0.15  # 15%的token被mask
    mean_noise_span_length: int = 3  # 平均span长度
    
    # 其他
    seed: int = 42
    save_steps: int = 500
    logging_steps: int = 50
    
    # 混合精度
    fp16: bool = True


class DAPTDataset(Dataset):
    """
    DAPT数据集：使用span corruption任务
    输入：阿卡德语文本（带noise）
    输出：恢复的文本
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512,
        noise_density: float = 0.15,
        mean_noise_span_length: int = 3,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        
        # T5 sentinel tokens: <extra_id_0>, <extra_id_1>, ...
        self.sentinel_token_ids = [
            tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>")
            for i in range(100)
        ]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        tokens = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
        )
        
        # Apply span corruption
        input_ids, target_ids = self.span_corrupt(tokens)
        
        # Pad/truncate
        input_ids = self._pad_or_truncate(input_ids, self.max_length, self.tokenizer.pad_token_id)
        target_ids = self._pad_or_truncate(target_ids, self.max_length, self.tokenizer.pad_token_id)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor([1 if x != self.tokenizer.pad_token_id else 0 for x in input_ids], dtype=torch.long),
            "labels": torch.tensor(target_ids, dtype=torch.long),
        }
    
    def span_corrupt(self, tokens: List[int]) -> tuple:
        """
        T5-style span corruption
        随机选择span进行mask，用sentinel token替换
        """
        n_tokens = len(tokens)
        n_noise_tokens = int(round(n_tokens * self.noise_density))
        
        if n_noise_tokens == 0:
            # 如果没有noise，返回原文
            return tokens, tokens
        
        # 计算需要多少个span
        n_noise_spans = max(1, int(round(n_noise_tokens / self.mean_noise_span_length)))
        
        # 确保有足够的token进行mask
        n_noise_tokens = min(n_noise_tokens, n_tokens - n_noise_spans)
        
        if n_noise_tokens <= 0:
            return tokens, tokens
        
        # 随机选择span的起始位置
        # 使用非均匀分布，避免选择已经选择过的区域
        span_lengths = self._random_split(n_noise_tokens, n_noise_spans)
        
        # 选择span的起始位置（均匀随机）
        # 确保span不重叠
        available_positions = list(range(n_tokens))
        random.shuffle(available_positions)
        
        # 标记哪些位置被mask
        mask_positions = set()
        for length in span_lengths:
            # 找到一个连续的区域
            for start_idx in available_positions:
                if start_idx in mask_positions:
                    continue
                # 检查从start_idx开始的length个位置是否都可用
                span_positions = set(range(start_idx, min(start_idx + length, n_tokens)))
                if not span_positions & mask_positions:  # 没有重叠
                    mask_positions.update(span_positions)
                    break
        
        # 构建输入和输出
        input_ids = []
        target_ids = []
        sentinel_id = 0
        
        i = 0
        while i < n_tokens:
            if i in mask_positions:
                # 开始一个span
                # 在输入中加入sentinel token
                input_ids.append(self.sentinel_token_ids[sentinel_id])
                
                # 在输出中加入对应的sentinel token和mask掉的tokens
                target_ids.append(self.sentinel_token_ids[sentinel_id])
                
                # 收集这个span的所有tokens
                while i < n_tokens and i in mask_positions:
                    target_ids.append(tokens[i])
                    i += 1
                
                sentinel_id += 1
            else:
                input_ids.append(tokens[i])
                i += 1
        
        # 添加结束sentinel
        if sentinel_id > 0:
            target_ids.append(self.sentinel_token_ids[sentinel_id])
        
        return input_ids, target_ids
    
    def _random_split(self, n_items: int, n_bins: int) -> List[int]:
        """将n_items随机分配到n_bins中"""
        if n_bins == 1:
            return [n_items]
        
        # 生成n_bins-1个随机分割点
        splits = sorted(random.sample(range(1, n_items), n_bins - 1))
        splits = [0] + splits + [n_items]
        
        return [splits[i+1] - splits[i] for i in range(n_bins)]
    
    def _pad_or_truncate(self, ids: List[int], max_length: int, pad_token_id: int) -> List[int]:
        if len(ids) > max_length:
            return ids[:max_length]
        return ids + [pad_token_id] * (max_length - len(ids))


def load_all_akkadian_texts(config: DAPTConfig) -> List[str]:
    """
    加载所有阿卡德语文本用于DAPT
    包括：
    1. 训练集的transliteration
    2. 测试集的transliteration（无标签，仅用于MLM）
    """
    texts = []
    
    # 1. 从aligned_train加载（source列）
    if Path(config.train_csv).exists():
        with open(config.train_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('source'):
                    texts.append(row['source'])
        logger.info(f"Loaded {len(texts)} texts from aligned_train")
    
    # 2. 从原始train.csv加载（可能会有一些遗漏的）
    initial_count = len(texts)
    if Path(config.raw_train).exists():
        with open(config.raw_train, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('transliteration'):
                    texts.append(row['transliteration'])
        logger.info(f"Loaded {len(texts) - initial_count} texts from raw_train")
    
    # 3. 从test.csv加载（非常重要！利用测试集进行无监督学习）
    initial_count = len(texts)
    if Path(config.test_csv).exists():
        with open(config.test_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('transliteration'):
                    texts.append(row['transliteration'])
        logger.info(f"Loaded {len(texts) - initial_count} texts from test")
    
    # 去重
    unique_texts = list(set(texts))
    logger.info(f"Total unique texts for DAPT: {len(unique_texts)}")
    
    return unique_texts


def train_dapt(config: DAPTConfig):
    """DAPT训练主函数"""
    
    # 初始化accelerator
    # 使用BF16代替FP16，避免数值溢出（RTX 4090支持BF16）
    accelerator = Accelerator(
        mixed_precision="bf16" if config.fp16 else "no",
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    
    # 设置随机种子
    set_seed(config.seed)
    
    # 加载模型和tokenizer
    logger.info(f"Loading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
    
    # 加载数据
    logger.info("Loading DAPT corpus...")
    all_texts = load_all_akkadian_texts(config)
    
    # 创建数据集
    dataset = DAPTDataset(
        texts=all_texts,
        tokenizer=tokenizer,
        max_length=config.max_source_length,
        noise_density=config.noise_density,
        mean_noise_span_length=config.mean_noise_span_length,
    )
    
    # 创建dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
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
    num_update_steps_per_epoch = len(dataloader) // config.gradient_accumulation_steps
    max_train_steps = config.num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(max_train_steps * config.warmup_ratio)
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )
    
    # 准备模型、优化器等
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )
    
    # 训练循环
    logger.info("***** Running DAPT *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Batch size per device = {config.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    
    global_step = 0
    total_loss = 0
    
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0
        
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                epoch_loss += loss.item()
                total_loss += loss.item()
                
                if global_step % config.logging_steps == 0:
                    avg_loss = total_loss / global_step
                    current_lr = lr_scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "avg_loss": f"{avg_loss:.4f}",
                        "lr": f"{current_lr:.2e}"
                    })
                
                # 保存checkpoint
                if global_step % config.save_steps == 0:
                    output_dir = Path(config.output_dir) / f"checkpoint-{global_step}"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    logger.info(f"Saved checkpoint to {output_dir}")
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{config.num_epochs} - Avg Loss: {avg_epoch_loss:.4f}")
    
    # 保存最终模型
    final_output_dir = Path(config.output_dir) / "final"
    final_output_dir.mkdir(parents=True, exist_ok=True)
    
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    logger.info(f"DAPT completed! Final model saved to {final_output_dir}")
    
    return final_output_dir


def main():
    config = DAPTConfig()
    
    # 创建输出目录
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    with open(Path(config.output_dir) / "dapt_config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2)
    
    # 运行训练
    train_dapt(config)


if __name__ == "__main__":
    main()
