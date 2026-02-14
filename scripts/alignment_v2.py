#!/usr/bin/env python3
"""
对齐Pipeline v2 - 提取完整句子转写
关键改进：从published_texts.transliteration中提取完整的句子转写
"""

import csv
import re
import json
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict


class AkkadianDataAlignerV2:
    """改进版阿卡德语数据对齐器"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.sentences_db = []
        self.pub_by_id = {}
        self.sent_by_display_name = defaultdict(list)
        
    def load_data(self):
        """加载所有数据源"""
        print("Loading data sources...")
        
        # Load sentences
        sent_path = self.data_dir / "Sentences_Oare_FirstWord_LinNum.csv"
        with open(sent_path, 'r', encoding='utf-8') as f:
            self.sentences_db = list(csv.DictReader(f))
        print(f"  Loaded {len(self.sentences_db)} sentences")
        
        # Index sentences by display_name
        for s in self.sentences_db:
            dn = s.get('display_name', '').strip()
            if dn:
                self.sent_by_display_name[dn].append(s)
        print(f"  Indexed by display_name: {len(self.sent_by_display_name)} unique")
        
        # Load publications
        pub_path = self.data_dir / "published_texts.csv"
        with open(pub_path, 'r', encoding='utf-8') as f:
            pub_data = list(csv.DictReader(f))
        self.pub_by_id = {r['oare_id']: r for r in pub_data}
        print(f"  Loaded {len(self.pub_by_id)} publications")
        
    def extract_catalog_ref(self, label: str) -> Optional[str]:
        """从label中提取目录引用"""
        match = re.search(r'\(([^)]+)\)', label)
        if match:
            return match.group(1)
        return None
    
    def find_display_name(self, catalog_ref: str) -> Optional[str]:
        """通过目录引用找到对应的display_name"""
        for dn in self.sent_by_display_name.keys():
            if catalog_ref in dn:
                return dn
        return None
    
    def extract_sentence_transliterations(self, pub_trans: str, sentences: List[Dict]) -> List[Dict]:
        """
        从文档的完整转写中提取每个句子的完整转写
        
        策略：
        1. 使用line_number作为顺序参考
        2. 使用first_word_spelling作为锚点，在pub_trans中定位每个句子
        3. 提取句子之间的文本作为完整转写
        """
        if not pub_trans or not sentences:
            return []
        
        # 按line_number排序
        sorted_sents = sorted(sentences, key=lambda x: float(x.get('line_number') or 0))
        
        # 分词
        words = pub_trans.split()
        
        result = []
        last_end_idx = 0
        
        for i, sent in enumerate(sorted_sents):
            first_word = sent.get('first_word_spelling', '').strip()
            line_num = sent.get('line_number')
            
            if not first_word:
                continue
            
            # 在pub_trans中查找first_word的位置（从last_end_idx开始）
            found = False
            start_idx = last_end_idx
            
            # 尝试直接匹配first_word
            for idx in range(last_end_idx, len(words)):
                # 标准化比较（忽略大小写和变音符号）
                word_normalized = self._normalize_word(words[idx])
                first_word_normalized = self._normalize_word(first_word)
                
                if word_normalized == first_word_normalized:
                    start_idx = idx
                    found = True
                    break
                # 也尝试匹配first_word的前缀
                elif first_word_normalized in word_normalized or word_normalized in first_word_normalized:
                    start_idx = idx
                    found = True
                    break
            
            if not found:
                # 如果没找到，使用last_end_idx作为起点
                start_idx = last_end_idx
            
            # 确定结束位置：下一个句子的first_word位置，或文档结尾
            if i + 1 < len(sorted_sents):
                next_first_word = sorted_sents[i + 1].get('first_word_spelling', '').strip()
                end_idx = len(words)
                
                for idx in range(start_idx + 1, len(words)):
                    word_normalized = self._normalize_word(words[idx])
                    next_first_normalized = self._normalize_word(next_first_word)
                    
                    if word_normalized == next_first_normalized:
                        end_idx = idx
                        break
                    elif next_first_normalized in word_normalized or word_normalized in next_first_normalized:
                        end_idx = idx
                        break
            else:
                end_idx = len(words)
            
            # 提取句子转写
            sentence_words = words[start_idx:end_idx]
            sentence_trans = ' '.join(sentence_words)
            
            result.append({
                'line_number': line_num,
                'first_word': first_word,
                'source': sentence_trans,
                'target': sent.get('translation', ''),
                'display_name': sent.get('display_name', ''),
                'sentence_uuid': sent.get('sentence_uuid', '')
            })
            
            last_end_idx = end_idx
        
        return result
    
    def _normalize_word(self, word: str) -> str:
        """标准化单词用于比较（移除变音符号，转小写）"""
        # 移除常见的变音符号
        word = word.lower()
        replacements = {
            'š': 's', 'ṣ': 's', 'ṭ': 't', 'ḫ': 'h', 'Ḫ': 'h',
            'á': 'a', 'à': 'a', 'é': 'e', 'è': 'e', 'í': 'i', 'ì': 'i', 'ú': 'u', 'ù': 'u',
            '₄': '', '₅': '', '₆': '', '₇': '', '₈': '', '₉': '', '₀': '', '₁': '', '₂': '', '₃': ''
        }
        for old, new in replacements.items():
            word = word.replace(old, new)
        return word
    
    def align_with_sentences(self, train_row: Dict) -> List[Dict]:
        """使用sentences数据库进行精确对齐，提取完整句子转写"""
        train_id = train_row['oare_id']
        pub_row = self.pub_by_id.get(train_id)
        
        if not pub_row:
            return []
        
        label = pub_row.get('label', '')
        catalog_ref = self.extract_catalog_ref(label)
        
        if not catalog_ref:
            return []
        
        # 查找display_name
        display_name = self.find_display_name(catalog_ref)
        if not display_name:
            return []
        
        # 获取该文档的所有句子
        sentences = self.sent_by_display_name.get(display_name, [])
        if not sentences:
            return []
        
        # 获取publication的完整转写
        pub_trans = pub_row.get('transliteration', '')
        if not pub_trans:
            return []
        
        # 提取每个句子的完整转写
        aligned_sents = self.extract_sentence_transliterations(pub_trans, sentences)
        
        if not aligned_sents:
            return []
        
        # 标记类型
        for item in aligned_sents:
            item['type'] = 'sentence_aligned'
            item['train_oare_id'] = train_id
        
        return aligned_sents
    
    def heuristic_sentence_split(self, text: str, translation: str, 
                                  min_words: int = 8, max_words: int = 30) -> List[Dict]:
        """启发式句子切割 - 为文档级数据创建伪句子"""
        words = text.split()
        
        if len(words) <= max_words:
            return [{
                'source': text,
                'target': translation,
                'type': 'document_preserved',
                'chunk_idx': 0,
                'total_chunks': 1
            }]
        
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            chunk_len = len(current_chunk)
            
            # 简单长度切分
            if chunk_len >= max_words:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # 为每个chunk分配translation（只有第一个有完整翻译）
        result = []
        for i, chunk in enumerate(chunks):
            result.append({
                'source': chunk,
                'target': translation if i == 0 else '[continuation]',
                'type': 'heuristic_chunk',
                'chunk_idx': i,
                'total_chunks': len(chunks)
            })
        
        return result
    
    def align_document_level(self, train_row: Dict) -> List[Dict]:
        """文档级对齐"""
        train_id = train_row['oare_id']
        transliteration = train_row['transliteration']
        translation = train_row['translation']
        
        chunks = self.heuristic_sentence_split(transliteration, translation)
        
        for item in chunks:
            item['train_oare_id'] = train_id
        
        return chunks
    
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
            'total_items': 0,
            'total_source_chars': 0,
            'total_target_chars': 0
        }
        
        for i, train_row in enumerate(train_data):
            # 首先尝试句子级对齐
            aligned = self.align_with_sentences(train_row)
            
            if aligned:
                stats['sentence_aligned'] += 1
            else:
                # 退回到文档级
                aligned = self.align_document_level(train_row)
                stats['document_level'] += 1
            
            all_aligned.extend(aligned)
            stats['total_items'] += len(aligned)
            
            for item in aligned:
                stats['total_source_chars'] += len(item.get('source', ''))
                stats['total_target_chars'] += len(item.get('target', ''))
            
            if (i + 1) % 200 == 0:
                print(f"  Processed {i+1}/{len(train_data)}...")
        
        # 保存为JSON
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_aligned, f, ensure_ascii=False, indent=2)
        
        # 保存为CSV（更适合训练）- 标准化字段
        csv_path = output_file.with_suffix('.csv')
        standard_fields = ['source', 'target', 'type', 'train_oare_id']
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=standard_fields)
            writer.writeheader()
            for item in all_aligned:
                row = {k: item.get(k, '') for k in standard_fields}
                writer.writerow(row)
        
        # 打印统计
        print(f"\n{'='*60}")
        print("Alignment Statistics:")
        print(f"  Documents with sentence-level alignment: {stats['sentence_aligned']} ({stats['sentence_aligned']/len(train_data)*100:.1f}%)")
        print(f"  Documents with document-level alignment: {stats['document_level']} ({stats['document_level']/len(train_data)*100:.1f}%)")
        print(f"  Total aligned items: {stats['total_items']}")
        print(f"  Avg source length: {stats['total_source_chars']/stats['total_items']:.1f} chars")
        print(f"  Avg target length: {stats['total_target_chars']/stats['total_items']:.1f} chars")
        print(f"\nOutput saved to:")
        print(f"  JSON: {output_file}")
        print(f"  CSV:  {csv_path}")
        print(f"{'='*60}")
        
        return all_aligned


def main():
    data_dir = "data/extracted"
    output_path = "data/processed/aligned_train_v2.json"
    
    aligner = AkkadianDataAlignerV2(data_dir)
    aligner.load_data()
    
    aligned = aligner.process_all(output_path)
    
    # 打印样本
    print(f"\n{'='*60}")
    print("Sample SENTENCE-ALIGNED items (now with full transliteration):")
    print(f"{'='*60}")
    
    sentence_samples = [a for a in aligned if a.get('type') == 'sentence_aligned'][:3]
    for item in sentence_samples:
        print(f"\n  Line {item.get('line_number')}:")
        print(f"    Source ({len(item['source'])} chars): {item['source'][:80]}...")
        print(f"    Target ({len(item['target'])} chars): {item['target'][:80]}...")
        print(f"    Display: {item.get('display_name', '')[:50]}")
    
    doc_samples = [a for a in aligned if 'document' in a.get('type', '') or 'heuristic' in a.get('type', '')][:3]
    if doc_samples:
        print(f"\n{'='*60}")
        print("Sample DOCUMENT-LEVEL items:")
        print(f"{'='*60}")
        for item in doc_samples:
            print(f"\n  Type: {item.get('type')}")
            print(f"    Source ({len(item['source'])} chars): {item['source'][:80]}...")
            print(f"    Target ({len(item['target'])} chars): {item['target'][:80]}...")


if __name__ == "__main__":
    main()
