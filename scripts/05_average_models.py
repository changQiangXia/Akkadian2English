#!/usr/bin/env python3
"""
Step 5: Checkpoint Averaging
对最后N个epoch或所有fold的checkpoint进行权重平均
"""

import argparse
from pathlib import Path
from typing import List
import torch
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def average_checkpoints(checkpoint_paths: List[str], output_path: str):
    """权重平均"""
    logger.info(f"Averaging {len(checkpoint_paths)} checkpoints...")
    
    state_dicts = []
    for path in checkpoint_paths:
        model_file = Path(path) / "pytorch_model.bin"
        if not model_file.exists():
            logger.warning(f"Skipping {path}: pytorch_model.bin not found")
            continue
        
        state_dict = torch.load(model_file, map_location="cpu")
        state_dicts.append(state_dict)
        logger.info(f"  Loaded: {path}")
    
    if not state_dicts:
        raise ValueError("No valid checkpoints found!")
    
    # 平均权重
    averaged_state_dict = {}
    for key in state_dicts[0].keys():
        tensors = [sd[key] for sd in state_dicts]
        averaged_state_dict[key] = sum(tensors) / len(tensors)
    
    # 保存
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    torch.save(averaged_state_dict, output_path / "pytorch_model.bin")
    logger.info(f"Saved averaged weights to: {output_path / 'pytorch_model.bin'}")
    
    # 复制其他必要文件
    source_path = Path(checkpoint_paths[0])
    for file in ["config.json", "tokenizer.json", "tokenizer_config.json", "spiece.model", "special_tokens_map.json"]:
        src = source_path / file
        if src.exists():
            shutil.copy(src, output_path / file)
            logger.info(f"  Copied: {file}")
    
    logger.info(f"Averaged checkpoint ready at: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Average model checkpoints")
    parser.add_argument("--input", nargs="+", required=True, help="Checkpoint paths to average")
    parser.add_argument("--output", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    average_checkpoints(args.input, args.output)


if __name__ == "__main__":
    main()
