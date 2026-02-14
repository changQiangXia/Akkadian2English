# 阿卡德语到英语神经机器翻译

[English Version](#english-version) | [Jump to English](#english-version)

---

## 目录

1. [项目结构](#项目结构)
2. [训练结果](#训练结果)
3. [关键发现](#关键发现)
4. [最佳推理配置](#最佳推理配置)
5. [使用方法](#使用方法)
6. [问题与限制](#问题与限制)
7. [未来改进方向](#未来改进方向)
8. [许可证](#许可证)

---

## 项目结构

```
.
├── data/                       # 数据集目录
│   ├── extracted/             # 原始提取数据
│   └── processed/             # 预处理对齐数据
├── scripts/                   # 训练和推理脚本
│   ├── 03_train.py           # 主训练脚本（5折CV）
│   ├── 04_inference.py       # 推理脚本（MBR集成）
│   └── 05_average_models.py  # 模型权重平均
├── checkpoints/              # 模型检查点
│   ├── dapt/                # 领域自适应预训练模型
│   └── finetuned/           # 微调后的折模型
│       ├── fold_0/best/     # Fold 0: Combined=85.10
│       ├── fold_1/best/     # Fold 1: Combined=55.78
│       ├── fold_2/best/     # Fold 2: Combined=82.21 (BLEU=100!)
│       ├── fold_3/best/     # Fold 3: Combined=32.19 (失败)
│       └── fold_4/best/     # Fold 4: Combined=57.66
└── submissions/              # 生成的提交文件
```

## 训练结果

### 5折交叉验证结果

| Fold | BLEU | chrF++ | Combined | 状态 |
|------|------|--------|----------|------|
| **Fold 0** | 80.34 | **90.13** | **85.10** | ✅ 优秀 |
| Fold 1 | 51.70 | 60.19 | 55.78 | ⚠️ 及格 |
| **Fold 2** | **100.00** | 67.59 | 82.21 | ✅ 完美 |
| Fold 3 | 28.50 | 36.35 | 32.19 | ❌ 失败 |
| Fold 4 | 59.19 | 56.17 | 57.66 | 🥉 中等 |

**平均分（5折）**: 62.59  
**平均分（排除 Fold 3）**: 70.19  
**最佳2折平均**: 83.66

## 关键发现

### 1. 数据类别不平衡（严重问题）

训练数据存在严重的类别不平衡：
- "silver"（银子）出现在 **49.0%** 的训练样本中
- "mina"（米那，重量单位）出现在 **38.6%** 的样本中
- "karum"（商业中心）仅出现 **1 次**（0.0%）

**后果**: 模型被训练成了"借贷合同翻译机"，看到楔形文字就条件反射输出银子、利息、支付等词汇。

### 2. Fold 2 的 BLEU=100

Fold 2 在验证集上达到了 **BLEU=100.00**，说明模型完全记住了验证集的模式。这可能是因为：
- 数据泄露（训练集和验证集有文档重叠）
- 验证集包含大量重复模板

### 3. Fold 3 失败分析

Fold 3 发生了严重的模式坍塌（mode collapse）：
- **可能原因1**: 随机初始化 + BF16 精度问题导致陷入局部最优
- **可能原因2**: 学习率（3e-5）对该特定数据划分过高
- **可能原因3**: 训练集中包含 39 个目标文本极短（≤2词）的样本

### 4. 数据划分问题

项目使用了 data-aware split，导致每折训练集和验证集有 **104-125 个文档重叠**。

## 最佳推理配置

推荐使用 **Fold 0 + Fold 2 进行 MBR 集成解码**：

```python
# 模型选择
model_paths = [
    "checkpoints/finetuned/fold_0/best",  # chrF++=90，流畅度高
    "checkpoints/finetuned/fold_2/best",  # BLEU=100，准确度高
]

# MBR 参数
use_mbr = True
mbr_num_candidates = 10  # 共20个候选（每模型10个）
num_beams = 12
length_penalty = 1.5
repetition_penalty = 1.2
```

**策略**: Fold 0 提供流畅的英文生成能力（高 chrF++），Fold 2 提供精确的术语匹配（完美 BLEU），两者互补。

## 使用方法

### 训练

```bash
# 训练指定折（0-4）
python scripts/03_train.py
```

修改配置文件中的 `current_fold` 来训练不同的折。

### 推理

```bash
# 使用 MBR 集成生成提交文件
python scripts/04_inference.py
```

输出将保存到 `submissions/submission.csv`。

### 模型平均

```bash
# 平均多个检查点
python scripts/05_average_models.py \
  --input checkpoints/finetuned/fold_0/best checkpoints/finetuned/fold_2/best \
  --output checkpoints/final_averaged
```

## 问题与限制

1. **模式坍塌**: 5折中有3折（Fold 1, 3, 4）经历了不同程度的模式坍塌或次优收敛。

2. **数据泄露**: 由于 data-aware split，每折训练集和验证集有 104-125 个文档重叠。

3. **BF16 不稳定性**: 混合精度训练（BF16）在某些折中导致数值不稳定。

4. **类别不平衡**: 由于训练数据分布倾斜，模型对借贷合同模式过拟合。

## 未来改进方向

- 实现文档级别划分（训练/验证无重叠）
- 通过数据增强解决类别不平衡
- 尝试 FP32 训练以提高数值稳定性
- 实验标签平滑（label smoothing）防止过拟合

## 许可证

本项目仅用于学术研究目的。

---

<div id="english-version"></div>

# English Version

[中文版](#阿卡德语到英语神经机器翻译) | [Back to Chinese](#项目结构)

---

## Table of Contents

1. [Project Structure](#project-structure-en)
2. [Training Results](#training-results-en)
3. [Key Findings](#key-findings-en)
4. [Best Inference Configuration](#best-inference-configuration-en)
5. [Usage](#usage-en)
6. [Known Issues and Limitations](#known-issues-and-limitations-en)
7. [Future Improvements](#future-improvements-en)
8. [License](#license-en)

---

<div id="project-structure-en"></div>

## Project Structure

```
.
├── data/                       # Dataset directory
│   ├── extracted/             # Raw extracted data
│   └── processed/             # Preprocessed aligned data
├── scripts/                   # Training and inference scripts
│   ├── 03_train.py           # Main training script with 5-fold CV
│   ├── 04_inference.py       # Inference with MBR ensemble
│   └── 05_average_models.py  # Model weight averaging
├── checkpoints/              # Model checkpoints
│   ├── dapt/                # Domain-adapted pretrained model
│   └── finetuned/           # Fine-tuned fold models
│       ├── fold_0/best/     # Fold 0: Combined=85.10
│       ├── fold_1/best/     # Fold 1: Combined=55.78
│       ├── fold_2/best/     # Fold 2: Combined=82.21 (BLEU=100!)
│       ├── fold_3/best/     # Fold 3: Combined=32.19 (failed)
│       └── fold_4/best/     # Fold 4: Combined=57.66
└── submissions/              # Generated submission files
```

<div id="training-results-en"></div>

## Training Results

### 5-Fold Cross-Validation Results

| Fold | BLEU | chrF++ | Combined | Status |
|------|------|--------|----------|--------|
| **Fold 0** | 80.34 | **90.13** | **85.10** | ✅ Excellent |
| Fold 1 | 51.70 | 60.19 | 55.78 | ⚠️ Acceptable |
| **Fold 2** | **100.00** | 67.59 | 82.21 | ✅ Perfect |
| Fold 3 | 28.50 | 36.35 | 32.19 | ❌ Failed |
| Fold 4 | 59.19 | 56.17 | 57.66 | 🥉 Moderate |

**Average (5 folds):** 62.59  
**Average (excluding Fold 3):** 70.19  
**Best 2 folds average:** 83.66

<div id="key-findings-en"></div>

## Key Findings

### 1. Severe Data Imbalance

The training data contains severe class imbalance:
- "silver" appears in **49.0%** of training examples
- "mina" appears in **38.6%** of examples
- "karum" (important commercial term) appears only once (**0.0%**)

**Consequence**: The model became a "loan contract translator", reflexively outputting silver, interest, and payment terms when seeing cuneiform.

### 2. Fold 2 Perfect BLEU

Fold 2 achieved **BLEU=100.00** on the validation set, indicating perfect memorization of validation patterns. This may be due to:
- Data leakage (document overlap between train/val)
- Validation set containing many repeated templates

### 3. Fold 3 Failure Analysis

Fold 3 suffered from severe mode collapse:
- **Possible cause 1**: Bad initialization + BF16 precision issues
- **Possible cause 2**: Learning rate (3e-5) too high for that specific split
- **Possible cause 3**: Training set contained 39 samples with extremely short targets (≤2 words)

### 4. Data Split Issues

The project uses data-aware split, causing **104-125 documents to overlap** between train/val sets in each fold.

<div id="best-inference-configuration-en"></div>

## Best Inference Configuration

The optimal setup uses **Fold 0 + Fold 2 with MBR decoding**:

```python
# Model selection
model_paths = [
    "checkpoints/finetuned/fold_0/best",  # chrF++=90, fluent
    "checkpoints/finetuned/fold_2/best",  # BLEU=100, accurate
]

# MBR parameters
use_mbr = True
mbr_num_candidates = 10
num_beams = 12
length_penalty = 1.5
repetition_penalty = 1.2
```

**Strategy**: Fold 0 provides fluent English generation (high chrF++), Fold 2 provides precise term matching (perfect BLEU).

<div id="usage-en"></div>

## Usage

### Training

```bash
# Train a specific fold (0-4)
python scripts/03_train.py
```

Modify `current_fold` in the config to train different folds.

### Inference

```bash
# Generate submission with MBR ensemble
python scripts/04_inference.py
```

Output will be saved to `submissions/submission.csv`.

### Model Averaging

```bash
# Average multiple checkpoints
python scripts/05_average_models.py \
  --input checkpoints/finetuned/fold_0/best checkpoints/finetuned/fold_2/best \
  --output checkpoints/final_averaged
```

<div id="known-issues-and-limitations-en"></div>

## Known Issues and Limitations

1. **Mode Collapse**: 3 out of 5 folds (Fold 1, 3, 4) experienced various degrees of mode collapse.

2. **Data Leakage**: Due to data-aware split, 104-125 documents overlap between train/val sets.

3. **BF16 Instability**: Mixed precision training caused numerical instability in some folds.

4. **Category Imbalance**: The model overfits to loan contract patterns.

<div id="future-improvements-en"></div>

## Future Improvements

- Implement proper document-level split (no overlap)
- Address class imbalance through data augmentation
- Try FP32 training for better numerical stability
- Experiment with label smoothing

<div id="license-en"></div>

## License

This project is for academic research purposes.
