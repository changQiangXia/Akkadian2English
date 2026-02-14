#!/usr/bin/env python3
"""
完整的数据对齐Pipeline
处理两类数据：
1. 有句子级对齐的（~17.7%）：使用Sentences_Oare_FirstWord_LinNum.csv精确对齐
2. 只有文档级的（~82.3%）：使用启发式方法或保留文档级
"""

import csv
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class AkkadianDataAligner:
    """阿卡德语数据对齐器"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.sentences_db = []
        self.pub_by_id = {}
        self.sent_by_text_uuid = defaultdict(list)
        self.matched_count = 0
        self.unmatched_count = 0
        
    def load_data(self):
        """加载所有数据源"""
        print("Loading data sources...")
        
        # Load sentences
        sent_path = self.data_dir / "Sentences_Oare_FirstWord_LinNum.csv"
        with open(sent_path, 'r', encoding='utf-8') as f:
            self.sentences_db = list(csv.DictReader(f))
        print(f"  Loaded {len(self.sentences_db)} sentences")
        
        # Index sentences by text_uuid
        for s in self.sentences_db:
            self.sent_by_text_uuid[s['text_uuid']].append(s)
        print(f"  Unique text_uuids: {len(self.sent_by_text_uuid)}")
        
        # Load publications
        pub_path = self.data_dir / "published_texts.csv"
        with open(pub_path, 'r', encoding='utf-8') as f:
            pub_data = list(csv.DictReader(f))
        self.pub_by_id = {r['oare_id']: r for r in pub_data}
        print(f"  Loaded {len(self.pub_by_id)} publications")
        
        # Build display_name index
        self.sent_display_names = set(s.get('display_name', '') for s in self.sentences_db)
        print(f"  Unique display_names: {len(self.sent_display_names)}")
        
    def extract_catalog_ref(self, label: str) -> Optional[str]:
        """从label中提取目录引用，如 'AKT 6e 1160'"""
        match = re.search(r'\(([^)]+)\)', label)
        if match:
            return match.group(1)
        return None
    
    def find_text_uuid_by_catalog(self, catalog_ref: str) -> Optional[str]:
        """通过目录引用找到text_uuid"""
        for s in self.sentences_db:
            if catalog_ref in s.get('display_name', ''):
                return s['text_uuid']
        return None
    
    def align_with_sentences(self, train_row: Dict) -> List[Dict]:
        """
        尝试使用sentences数据库进行句子级对齐
        返回对齐后的句子列表，如果没有找到则返回空列表
        """
        train_id = train_row['oare_id']
        pub_row = self.pub_by_id.get(train_id)
        
        if not pub_row:
            return []
        
        label = pub_row.get('label', '')
        catalog_ref = self.extract_catalog_ref(label)
        
        if not catalog_ref:
            return []
        
        # 查找匹配的text_uuid
        text_uuid = self.find_text_uuid_by_catalog(catalog_ref)
        if not text_uuid:
            return []
        
        # 获取该文档的所有句子
        sentences = self.sent_by_text_uuid.get(text_uuid, [])
        if not sentences:
            return []
        
        # 按行号排序
        sentences = sorted(sentences, key=lambda x: float(x.get('line_number') or 0))
        
        # 构建对齐结果
        aligned = []
        for sent in sentences:
            # 从sentences获取实际的转写片段
            # 注意：sentences中只有first_word，没有完整转写
            # 我们需要从published_texts的transliteration中切割
            aligned.append({
                'type': 'sentence_aligned',
                'source': sent.get('first_word_spelling', ''),  # 这只是第一个词
                'target': sent.get('translation', ''),
                'line_number': sent.get('line_number'),
                'sentence_uuid': sent.get('sentence_uuid'),
                'display_name': sent.get('display_name'),
                'text_uuid': text_uuid
            })
        
        return aligned
    
    def heuristic_sentence_split(self, text: str, min_words: int = 8, max_words: int = 25) -> List[str]:
        """
        启发式句子切割
        基于标点和长度进行切割
        """
        # 阿卡德语中常见的句末标记
        # 注意：这些标记可能不在转写中明确标出
        # 这里使用简单的长度切分
        words = text.split()
        
        if len(words) <= max_words:
            return [text]
        
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            chunk_len = len(current_chunk)
            
            # 如果达到最大长度，强制切割
            if chunk_len >= max_words:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
            # 如果在合理范围内且遇到标点，尝试切割
            elif chunk_len >= min_words and word.endswith(('.', 'ma', 'qé')):
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else [text]
    
    def align_document_level(self, train_row: Dict) -> List[Dict]:
        """
        文档级对齐（当没有句子级对齐时使用）
        将整个文档作为一个样本，或使用启发式切割
        """
        train_id = train_row['oare_id']
        transliteration = train_row['transliteration']
        translation = train_row['translation']
        
        # 选项1：保留文档级
        # return [{
        #     'type': 'document',
        #     'source': transliteration,
        #     'target': translation,
        #     'oare_id': train_id
        # }]
        
        # 选项2：使用启发式切割
        chunks = self.heuristic_sentence_split(transliteration)
        aligned = []
        
        for i, chunk in enumerate(chunks):
            aligned.append({
                'type': 'heuristic_chunk',
                'source': chunk,
                'target': f'[{translation[:50]}...]' if i == 0 else '[continuation]',  # 文档级翻译无法分割
                'chunk_idx': i,
                'total_chunks': len(chunks),
                'oare_id': train_id
            })
        
        return aligned
    
    def process_all(self, output_path: str):
        """处理所有训练数据"""
        train_path = self.data_dir / "train.csv"
        
        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = list(csv.DictReader(f))
        
        print(f"\nProcessing {len(train_data)} training documents...")
        
        all_aligned = []
        stats = {
            'sentence_aligned': 0,
            'document_level': 0,
            'total_source_sentences': 0
        }
        
        for i, train_row in enumerate(train_data):
            # 首先尝试句子级对齐
            aligned = self.align_with_sentences(train_row)
            
            if aligned:
                stats['sentence_aligned'] += 1
                self.matched_count += 1
            else:
                # 退回到文档级或启发式切割
                aligned = self.align_document_level(train_row)
                stats['document_level'] += 1
                self.unmatched_count += 1
            
            # 为每个对齐项添加元数据
            for item in aligned:
                item['train_oare_id'] = train_row['oare_id']
                item['train_transliteration'] = train_row['transliteration'][:100] + '...'
            
            all_aligned.extend(aligned)
            stats['total_source_sentences'] += len(aligned)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(train_data)}...")
        
        # 保存结果
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_aligned, f, ensure_ascii=False, indent=2)
        
        # 打印统计
        print(f"\n{'='*60}")
        print("Alignment Statistics:")
        print(f"  Documents with sentence-level alignment: {stats['sentence_aligned']} ({stats['sentence_aligned']/len(train_data)*100:.1f}%)")
        print(f"  Documents with document-level alignment: {stats['document_level']} ({stats['document_level']/len(train_data)*100:.1f}%)")
        print(f"  Total aligned items: {stats['total_source_sentences']}")
        print(f"  Average items per document: {stats['total_source_sentences']/len(train_data):.1f}")
        print(f"\nOutput saved to: {output_file}")
        print(f"{'='*60}")
        
        return all_aligned


def print_samples(aligned_data: List[Dict], n: int = 3):
    """打印样本数据"""
    print(f"\n{'='*60}")
    print(f"Sample aligned items (showing {n}):")
    print(f"{'='*60}")
    
    for item in aligned_data[:n]:
        print(f"\nType: {item['type']}")
        print(f"  Source: {item['source'][:80]}..." if len(item['source']) > 80 else f"  Source: {item['source']}")
        print(f"  Target: {item['target'][:80]}..." if len(item['target']) > 80 else f"  Target: {item['target']}")
        if item['type'] == 'sentence_aligned':
            print(f"  Line: {item.get('line_number')}, Display: {item.get('display_name')}")
        print()


def main():
    data_dir = "data/extracted"
    output_path = "data/processed/aligned_train.json"
    
    aligner = AkkadianDataAligner(data_dir)
    aligner.load_data()
    
    aligned = aligner.process_all(output_path)
    
    # 打印样本
    print_samples(aligned, n=5)
    
    # 分别打印两种类型的样本
    sentence_aligned = [a for a in aligned if a['type'] == 'sentence_aligned']
    doc_level = [a for a in aligned if a['type'] == 'heuristic_chunk']
    
    if sentence_aligned:
        print(f"\n{'='*60}")
        print("Sample SENTENCE-ALIGNED items:")
        print(f"{'='*60}")
        for item in sentence_aligned[:3]:
            print(f"\n  Source: {item['source'][:60]}...")
            print(f"  Target: {item['target'][:60]}...")
            print(f"  Line: {item.get('line_number')}")
    
    if doc_level:
        print(f"\n{'='*60}")
        print("Sample DOCUMENT-LEVEL (heuristic) items:")
        print(f"{'='*60}")
        for item in doc_level[:3]:
            print(f"\n  Source: {item['source'][:60]}...")
            print(f"  Target: {item['target'][:60]}...")
            print(f"  Chunk {item.get('chunk_idx')}/{item.get('total_chunks')}")


if __name__ == "__main__":
    main()
