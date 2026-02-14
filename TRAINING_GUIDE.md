# Akkadian-English Translation - Training Guide

## 快速开始

### 1. 数据对齐（已完成）

```bash
# 数据对齐已运行，生成文件：
# - data/processed/aligned_train_v2.csv (4,366条对齐数据)
# - data/processed/aligned_train_v2.json
```

### 2. DAPT预训练（可选但推荐）

```bash
cd /root/autodl-tmp
python scripts/02_dapt.py
```

**参数说明：**
- 使用所有阿卡德语文本（train+test）进行MLM
- 约10个epoch，每500步保存checkpoint
- 输出：`checkpoints/dapt/final/`

**跳过DAPT：**
如果跳过DAPT，直接将 `03_train.py` 中的 `model_name` 改为 `"Hippopoto0/akkadianT5"`

### 3. 5-Fold交叉验证训练

#### 方法A：串行训练所有folds（单卡）

```bash
cd /root/autodl-tmp
python scripts/03_train.py
```

这会自动训练5个fold，每个fold独立保存最佳模型。

#### 方法B：并行训练单个fold（多卡/多窗口）

如果你有多个GPU或多个终端窗口：

```bash
# Terminal 1 - Fold 0
cd /root/autodl-tmp
python scripts/03_train.py --current_fold 0

# Terminal 2 - Fold 1
cd /root/autodl-tmp
python scripts/03_train.py --current_fold 1

# ... 以此类推
```

**训练参数（已优化）：**
- Batch size: 4 per device
- Gradient accumulation: 4 steps (有效batch=16)
- Learning rate: 3e-5 (DAPT后) / 5e-5 (无DAPT)
- R-Drop alpha: 0.5
- Early stopping patience: 5
- 评估指标：sqrt(BLEU * chrF++)

### 4. Checkpoint平均（可选）

训练完成后，可以选择对多个checkpoint进行权重平均：

```bash
# 平均最后几个checkpoint
cd /root/autodl-tmp
python scripts/05_average_models.py \
    --input checkpoints/dapt/checkpoint-1500 checkpoints/dapt/checkpoint-2000 checkpoints/dapt/checkpoint-2500 \
    --output checkpoints/dapt/averaged

# 或者平均所有fold的最佳模型
python scripts/05_average_models.py \
    --input checkpoints/finetuned/fold_0/best checkpoints/finetuned/fold_1/best checkpoints/finetuned/fold_2/best \
    --output checkpoints/finetuned/averaged
```

### 5. 推理生成

```bash
cd /root/autodl-tmp
python scripts/04_inference.py
```

**推理特性：**
- 5-Fold模型集成
- MBR (Minimum Bayes Risk) 解码
- 后处理（括号匹配、N-gram去重、DN/PN格式）
- 输出：`submissions/submission_final.csv`

## 完整训练流程（推荐顺序）

```bash
# Step 1: DAPT（约2-4小时）
python scripts/02_dapt.py

# Step 2: 5-Fold训练（每个fold约1-2小时，共5个）
python scripts/03_train.py

# Step 3: 推理生成（约5-10分钟）
python scripts/04_inference.py
```

## 目录结构（训练后）

```
checkpoints/
├── dapt/
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   ├── final/              # DAPT最终模型
│   └── averaged/           # (可选)平均权重
└── finetuned/
    ├── fold_0/
    │   └── best/           # fold 0最佳模型
    ├── fold_1/best/
    ├── fold_2/best/
    ├── fold_3/best/
    └── fold_4/best/

submissions/
└── submission_final.csv    # 最终提交文件
```

## 监控训练

训练日志会自动输出到控制台，包括：
- 每个step的loss
- 验证集的BLEU/chrF++/Combined分数
- 学习率变化
- 保存的checkpoint路径

## 调参建议

| 参数 | 默认值 | 调整建议 |
|------|--------|----------|
| `learning_rate` | 3e-5 | DAPT后稍小，直接微调用5e-5 |
| `batch_size` | 4 | 根据显存调整，最大可试8 |
| `r_drop_alpha` | 0.5 | 过拟合严重时可增加到0.7 |
| `num_beams` | 8 | 推理时可增加到12-16 |
| `mbr_num_candidates` | 5 | 速度优先可减到3，质量优先可增到8 |

## 常见问题

**Q: OOM怎么办？**
- 减小 `batch_size` 到 2
- 增加 `gradient_accumulation_steps` 到 8
- 减小 `max_source_length` 到 384

**Q: 如何只训练特定fold？**
```bash
python scripts/03_train.py --current_fold 2  # 只训练fold 2
```

**Q: 如何跳过DAPT直接微调？**
编辑 `03_train.py`，修改：
```python
model_name: str = "Hippopoto0/akkadianT5"  # 而不是dapt/final
```

**Q: 如何只用单个模型推理而不是集成？**
编辑 `04_inference.py`，修改：
```python
use_averaged: bool = True  # 只使用averaged模型
```
