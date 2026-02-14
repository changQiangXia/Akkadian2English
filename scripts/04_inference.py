#!/usr/bin/env python3
"""
Step 4: 推理生成
支持：
1. 5-Fold模型集成
2. MBR (Minimum Bayes Risk) 解码
3. Checkpoint Averaging
4. 后处理规则
"""

import os
import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """推理配置"""
    # 输入
    test_csv: str = "data/extracted/test.csv"
    
    # 模型：Fold 0 + Fold 2 MBR 集成
    model_paths: List[str] = field(default_factory=lambda: [
        "checkpoints/finetuned/fold_0/best",
        "checkpoints/finetuned/fold_2/best",
    ])
    
    # 或者使用平均后的模型
    averaged_model_path: Optional[str] = None
    use_averaged: bool = False  # 如果True，则只使用averaged模型
    
    # 输出
    output_dir: str = "submissions"
    output_file: str = "submission.csv"
    
    # 生成参数
    max_source_length: int = 512
    max_target_length: int = 256
    batch_size: int = 4  # MBR需要更多显存，调小batch
    num_beams: int = 12  # 多beams生成多样候选
    length_penalty: float = 1.5  # 鼓励更长输出
    early_stopping: bool = True
    
    # 重复惩罚
    repetition_penalty: float = 1.2
    
    # MBR参数：开启 MBR，多候选解码
    use_mbr: bool = True
    mbr_num_candidates: int = 10
    mbr_char_ngram: int = 6      # chrF-like n-gram
    mbr_logp_weight: float = 0.3  # log概率权重
    
    # 后处理
    use_postprocess: bool = True
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = True


class TestDataset(Dataset):
    """测试数据集"""
    
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['transliteration']
        
        # 清洗
        text = self._preprocess(text)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "id": row['id'],
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "raw_text": text,
        }
    
    def _preprocess(self, text: str) -> str:
        """预处理（与训练一致）"""
        # 统一gap标记
        text = re.sub(r'\.{3,}|…+|——|……', '<big_gap>', text)
        text = re.sub(r'xx+|\s+x\s+', '<gap>', text)
        # 清理空格
        text = ' '.join(text.split())
        return text


class PostProcessor:
    """后处理器"""
    
    def __init__(self):
        # 括号匹配修复
        self.brackets = {
            '(': ')', '[': ']', '{': '}', '<': '>',
            '«': '»', '"': '"', "'": "'", "'": "'"
        }
        
        # 专有名词标记 (DN = Divine Name, PN = Personal Name)
        self.dn_pattern = re.compile(r'\(DN\)|\(d\)|\{d\}', re.IGNORECASE)
        self.pn_pattern = re.compile(r'\(PN\)', re.IGNORECASE)
    
    def process(self, text: str) -> str:
        """完整后处理流程"""
        text = self.fix_brackets(text)
        text = self.fix_repetition(text)
        text = self.fix_dn_pn_format(text)
        text = self.clean_whitespace(text)
        return text
    
    def fix_brackets(self, text: str) -> str:
        """修复括号不匹配"""
        for open_b, close_b in self.brackets.items():
            open_count = text.count(open_b)
            close_count = text.count(close_b)
            
            if open_count > close_count:
                # 添加缺失的闭合括号
                text += close_b * (open_count - close_count)
            elif close_count > open_count:
                # 移除多余的闭合括号
                diff = close_count - open_count
                for _ in range(diff):
                    idx = text.rfind(close_b)
                    if idx != -1:
                        text = text[:idx] + text[idx+1:]
        
        return text
    
    def fix_repetition(self, text: str, max_repeat: int = 3) -> str:
        """修复N-gram重复"""
        words = text.split()
        if len(words) < 4:
            return text
        
        result = []
        i = 0
        while i < len(words):
            # 检查是否有重复
            found_repeat = False
            for n in range(min(max_repeat, len(words) - i), 1, -1):
                pattern = words[i:i+n]
                # 检查后面是否有重复
                if i + n < len(words) and words[i+n:i+2*n] == pattern:
                    # 只保留第一个
                    result.extend(pattern)
                    # 跳过重复的
                    j = i + n
                    while j + n <= len(words) and words[j:j+n] == pattern:
                        j += n
                    i = j
                    found_repeat = True
                    break
            
            if not found_repeat:
                result.append(words[i])
                i += 1
        
        return ' '.join(result)
    
    def fix_dn_pn_format(self, text: str) -> str:
        """修复DN/PN专有名词格式"""
        # 统一神名标记
        text = self.dn_pattern.sub('', text)
        # 统一人名标记
        text = self.pn_pattern.sub('', text)
        return text
    
    def clean_whitespace(self, text: str) -> str:
        """清理空白字符"""
        # 统一空格
        text = ' '.join(text.split())
        # 标点前的空格
        text = re.sub(r'\s+([.,;:!?)\]}])', r'\1', text)
        # 标点后的空格
        text = re.sub(r'([(\[{])\s+', r'\1', text)
        return text


class MBRDecoder:
    """Minimum Bayes Risk解码器"""
    
    def __init__(self, char_ngram: int = 6, logp_weight: float = 0.3):
        self.char_ngram = char_ngram
        self.logp_weight = logp_weight
    
    def char_ngrams(self, text: str, n: int) -> List[str]:
        """提取字符n-grams"""
        text = text.strip().lower()
        if len(text) < n:
            return [text] if text else []
        return [text[i:i+n] for i in range(len(text) - n + 1)]
    
    def chrf_like_score(self, hyp: str, ref: str) -> float:
        """计算chrF-like F1分数"""
        hyp_ngrams = self.char_ngrams(hyp, self.char_ngram)
        ref_ngrams = self.char_ngrams(ref, self.char_ngram)
        
        if not hyp_ngrams or not ref_ngrams:
            return 0.0
        
        hyp_counts = Counter(hyp_ngrams)
        ref_counts = Counter(ref_ngrams)
        
        # 计算overlap
        overlap = sum((hyp_counts & ref_counts).values())
        
        precision = overlap / len(hyp_ngrams) if hyp_ngrams else 0
        recall = overlap / len(ref_ngrams) if ref_ngrams else 0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def select_best(
        self,
        candidates: List[str],
        log_probs: Optional[List[float]] = None
    ) -> str:
        """
        使用MBR选择最佳候选
        
        对每个候选y_i，计算U(y_i) = sum_j w_j * chrF(y_i, y_j)
        其中w_j基于log_prob的softmax权重
        """
        if not candidates:
            return ""
        
        if len(candidates) == 1:
            return candidates[0]
        
        # 计算softmax权重
        if log_probs:
            # 数值稳定性
            max_logp = max(log_probs)
            exp_probs = [np.exp(lp - max_logp) for lp in log_probs]
            total = sum(exp_probs)
            weights = [p / total for p in exp_probs]
        else:
            weights = [1.0 / len(candidates)] * len(candidates)
        
        best_candidate = None
        best_score = -float('inf')
        
        for i, yi in enumerate(candidates):
            # 计算期望分数（与其他所有候选的chrF加权平均）
            score = 0.0
            for j, yj in enumerate(candidates):
                if i != j:
                    score += weights[j] * self.chrf_like_score(yi, yj)
            
            # 加入log概率的bias
            if log_probs:
                score = (1 - self.logp_weight) * score + self.logp_weight * np.log(weights[i] + 1e-10)
            
            if score > best_score:
                best_score = score
                best_candidate = yi
        
        return best_candidate


class InferenceEngine:
    """推理引擎"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.post_processor = PostProcessor()
        self.mbr_decoder = MBRDecoder(
            char_ngram=config.mbr_char_ngram,
            logp_weight=config.mbr_logp_weight
        )
        
        # 加载模型
        self.models = []
        self.tokenizers = []
        self._load_models()
    
    def _load_models(self):
        """加载所有模型"""
        if self.config.use_averaged and self.config.averaged_model_path:
            model_paths = [self.config.averaged_model_path]
        else:
            model_paths = self.config.model_paths
        
        for path in model_paths:
            if not Path(path).exists():
                logger.warning(f"Model path not found: {path}")
                continue
            
            logger.info(f"Loading model: {path}")
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSeq2SeqLM.from_pretrained(path)
            model.to(self.device)
            model.eval()
            
            if self.config.fp16:
                model = model.half()
            
            self.models.append(model)
            self.tokenizers.append(tokenizer)
        
        if not self.models:
            raise ValueError("No models loaded!")
        
        logger.info(f"Loaded {len(self.models)} models for ensemble")
    
    @torch.no_grad()
    def generate_single(
        self,
        model,
        tokenizer,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[List[str], Optional[List[float]]]:
        """单模型生成"""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        if self.config.use_mbr and self.config.mbr_num_candidates > 1:
            # 生成多个候选
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.config.max_target_length,
                num_beams=self.config.num_beams,
                num_return_sequences=self.config.mbr_num_candidates,
                output_scores=True,
                return_dict_in_generate=True,
                length_penalty=self.config.length_penalty,
                early_stopping=self.config.early_stopping,
                repetition_penalty=self.config.repetition_penalty,  # 打破复读机
            )
            
            sequences = outputs.sequences
            scores = outputs.sequences_scores
            
            # 解码
            decoded = tokenizer.batch_decode(sequences, skip_special_tokens=True)
            log_probs = scores.cpu().tolist()
            
            return decoded, log_probs
        else:
            # 生成单个输出
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.config.max_target_length,
                num_beams=self.config.num_beams,
                length_penalty=self.config.length_penalty,
                early_stopping=self.config.early_stopping,
                repetition_penalty=self.config.repetition_penalty,  # 打破复读机
            )
            
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return decoded, None
    
    @torch.no_grad()
    def generate_ensemble(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> str:
        """集成生成"""
        all_candidates = []
        all_log_probs = []
        
        # 从每个模型生成
        for model, tokenizer in zip(self.models, self.tokenizers):
            candidates, log_probs = self.generate_single(
                model, tokenizer, input_ids, attention_mask
            )
            all_candidates.extend(candidates)
            if log_probs:
                all_log_probs.extend(log_probs)
        
        # 去重（保持顺序）
        seen = set()
        unique_candidates = []
        unique_log_probs = []
        for cand, lp in zip(all_candidates, all_log_probs or [0.0] * len(all_candidates)):
            if cand not in seen:
                seen.add(cand)
                unique_candidates.append(cand)
                unique_log_probs.append(lp)
        
        # 使用MBR选择最佳
        if self.config.use_mbr and len(unique_candidates) > 1:
            best = self.mbr_decoder.select_best(unique_candidates, unique_log_probs)
        else:
            best = unique_candidates[0] if unique_candidates else ""
        
        return best
    
    def run_inference(self):
        """运行推理"""
        # 加载测试数据
        logger.info(f"Loading test data: {self.config.test_csv}")
        test_df = pd.read_csv(self.config.test_csv)
        logger.info(f"Loaded {len(test_df)} test samples")
        
        # 使用第一个模型的tokenizer创建dataset
        dataset = TestDataset(test_df, self.tokenizers[0], self.config.max_source_length)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
        )
        
        # 推理
        results = []
        
        for batch in tqdm(dataloader, desc="Generating"):
            batch_ids = batch["id"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            
            # 对每个样本生成
            for i in range(len(batch_ids)):
                text_id = int(batch_ids[i].item() if torch.is_tensor(batch_ids[i]) else batch_ids[i])
                
                translation = self.generate_ensemble(
                    input_ids[i:i+1],
                    attention_mask[i:i+1]
                )
                
                # 后处理
                if self.config.use_postprocess:
                    translation = self.post_processor.process(translation)
                
                results.append({
                    "id": text_id,
                    "translation": translation
                })
        
        # 创建提交文件
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("id").reset_index(drop=True)
        
        # 确保格式正确
        assert list(results_df.columns) == ["id", "translation"]
        assert len(results_df) == len(test_df)
        
        # 保存
        output_path = Path(self.config.output_dir) / self.config.output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        
        logger.info(f"Submission saved to: {output_path}")
        logger.info(f"\nPreview:")
        logger.info(results_df.head().to_string())
        
        return results_df


def average_checkpoints(checkpoint_paths: List[str], output_path: str):
    """
    对多个checkpoint的权重进行平均
    用于Checkpoint Averaging策略
    """
    logger.info(f"Averaging {len(checkpoint_paths)} checkpoints...")
    
    # 加载所有state_dict
    state_dicts = []
    for path in checkpoint_paths:
        state_dict = torch.load(Path(path) / "pytorch_model.bin", map_location="cpu")
        state_dicts.append(state_dict)
    
    # 平均
    averaged_state_dict = {}
    for key in state_dicts[0].keys():
        averaged_state_dict[key] = sum(sd[key] for sd in state_dicts) / len(state_dicts)
    
    # 保存
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    torch.save(averaged_state_dict, output_path / "pytorch_model.bin")
    
    # 复制配置文件
    import shutil
    for file in ["config.json", "tokenizer.json", "tokenizer_config.json", "spiece.model"]:
        src = Path(checkpoint_paths[0]) / file
        if src.exists():
            shutil.copy(src, output_path / file)
    
    logger.info(f"Averaged checkpoint saved to: {output_path}")


def main():
    config = InferenceConfig()
    
    engine = InferenceEngine(config)
    engine.run_inference()


if __name__ == "__main__":
    main()
