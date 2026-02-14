#!/usr/bin/env python3
"""
Step 1: 数据预处理 Demo
功能：读取 train.csv 前100行，演示文档级→句子级切割
"""

import csv
import re
from pathlib import Path
from typing import List, Dict


def load_csv_samples(path: str, n: int = 100) -> List[Dict]:
    """加载CSV前N行"""
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [row for _, row in zip(range(n), reader)]


def align_document_to_sentences(
    doc_id: str,
    doc_transliteration: str,
    sentences_db: List[Dict]
) -> List[Dict]:
    """
    将文档级转写切割为句子级
    策略：利用line_number和first_word_spelling进行对齐
    """
    aligned = []
    
    # 找到该文档对应的所有句子（通过text_uuid前缀匹配）
    doc_sentences = [
        s for s in sentences_db 
        if s.get('text_uuid', '').startswith(doc_id[:8])
    ]
    
    if not doc_sentences:
        # 如果没有精确匹配，使用简单切分（Demo版）
        words = doc_transliteration.split()
        chunk_size = 20
        for i in range(0, len(words), chunk_size):
            chunk = words[i:i + chunk_size]
            aligned.append({
                'source': ' '.join(chunk),
                'target': '[NEEDS_ALIGNMENT]',
                'line_start': i,
                'line_end': min(i + chunk_size, len(words))
            })
    else:
        # 使用metadata对齐
        for sent in sorted(doc_sentences, key=lambda x: float(x.get('line_number') or 0)):
            aligned.append({
                'source': sent.get('first_word_spelling', ''),
                'target': sent.get('translation', ''),
                'line_number': sent.get('line_number'),
                'sentence_uuid': sent.get('sentence_uuid')
            })
    
    return aligned


def clean_transliteration(text: str) -> str:
    """清洗转写文本"""
    # 统一特殊gap标记
    text = re.sub(r'\.{3,}|…+|——|……', '<big_gap>', text)
    text = re.sub(r'xx+|\s+x\s+', '<gap>', text)
    # 统一引号
    text = re.sub(r'[""„]', '"', text)
    # 清理多余空格
    text = ' '.join(text.split())
    return text


def main():
    # 路径配置
    data_dir = Path("data/extracted")
    train_path = data_dir / "train.csv"
    sentences_path = data_dir / "Sentences_Oare_FirstWord_LinNum.csv"
    
    print("=" * 60)
    print("Demo: 文档级 → 句子级 对齐")
    print("=" * 60)
    
    # 1. 加载训练数据（前100行）
    print(f"\n[1] 加载 {train_path} 前100行...")
    train_samples = load_csv_samples(train_path, n=100)
    print(f"    加载完成: {len(train_samples)} 条文档")
    
    # 2. 加载句子级数据库
    print(f"\n[2] 加载句子级metadata: {sentences_path}")
    with open(sentences_path, 'r', encoding='utf-8') as f:
        sentences_db = list(csv.DictReader(f))
    print(f"    加载完成: {len(sentences_db)} 条句子")
    
    # 3. 处理前3个文档作为演示
    print(f"\n[3] 对齐演示（前3个文档）:")
    print("-" * 60)
    
    for idx, doc in enumerate(train_samples[:3], 1):
        doc_id = doc['oare_id']
        doc_trans = doc['transliteration']
        doc_target = doc['translation']
        
        print(f"\n  文档 {idx}: {doc_id[:20]}...")
        print(f"  原始长度: {len(doc_trans.split())} 词")
        print(f"  原始转写: {doc_trans[:120]}...")
        print(f"  原始翻译: {doc_target[:120]}...")
        
        # 清洗
        cleaned = clean_transliteration(doc_trans)
        
        # 对齐切割
        aligned = align_document_to_sentences(doc_id, cleaned, sentences_db)
        
        print(f"\n  ---> 切割为 {len(aligned)} 个句子:")
        for i, sent in enumerate(aligned[:3], 1):
            print(f"      句子{i}: {sent.get('source', '')[:60]}...")
            print(f"      翻译{i}: {sent.get('target', '')[:60]}...")
            if 'line_number' in sent:
                print(f"      行号: {sent['line_number']}")
            print()
        
        print("-" * 60)
    
    # 4. 输出统计
    print(f"\n[4] 统计信息:")
    total_docs = len(train_samples)
    total_words = sum(len(d['transliteration'].split()) for d in train_samples)
    print(f"    文档数: {total_docs}")
    print(f"    总词数: {total_words}")
    print(f"    平均每文档词数: {total_words / total_docs:.1f}")
    
    # 模拟切割后
    estimated_sentences = sum(
        len(align_document_to_sentences(
            d['oare_id'], 
            d['transliteration'], 
            sentences_db
        )) for d in train_samples
    )
    print(f"    估计句子数(切割后): {estimated_sentences}")
    print(f"    平均每文档句子数: {estimated_sentences / total_docs:.1f}")
    
    print("\n" + "=" * 60)
    print("Demo 完成！下一步：完整的对齐pipeline + 数据增强")
    print("=" * 60)


if __name__ == "__main__":
    main()
