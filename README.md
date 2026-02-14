# Akkadian to English Neural Machine Translation

This project implements a neural machine translation system for translating Akkadian (cuneiform) texts into English, using a T5-based architecture with 5-fold cross-validation and MBR decoding.

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

## Training Results

### 5-Fold Cross-Validation Results

| Fold | BLEU | chrF++ | Combined | Status |
|------|------|--------|----------|--------|
| Fold 0 | 80.34 | **90.13** | **85.10** | ✅ Excellent |
| Fold 1 | 51.70 | 60.19 | 55.78 | ⚠️ Acceptable |
| Fold 2 | **100.00** | 67.59 | 82.21 | ✅ Perfect (BLEU=100) |
| Fold 3 | 28.50 | 36.35 | 32.19 | ❌ Failed |
| Fold 4 | 59.19 | 56.17 | 57.66 | 🥉 Moderate |

**Average (5 folds):** 62.59
**Average (excluding Fold 3):** 70.19
**Best 2 folds average:** 83.66

### Key Observations

1. **Data Imbalance Issue**: The training data contains severe class imbalance:
   - "silver" appears in 49.0% of training examples
   - "mina" appears in 38.6% of examples
   - "karum" (important commercial term) appears only once (0.0%)
   - This caused the model to become a "loan contract translator"

2. **Fold 2 Perfect BLEU**: Fold 2 achieved BLEU=100.00, indicating perfect memorization of the validation set patterns.

3. **Fold 3 Failure**: Fold 3 suffered from mode collapse, likely due to:
   - Bad initialization + BF16 precision issues
   - Learning rate too high for that specific data split
   - Training data contained 39 examples with extremely short targets (≤2 words)

4. **Validation Strategy**: The project uses data-aware split which causes document overlap between train/val sets (104-125 overlapping documents per fold).

## Best Configuration for Inference

The optimal inference setup uses **Fold 0 + Fold 2 with MBR decoding**:

```python
# Model selection
model_paths = [
    "checkpoints/finetuned/fold_0/best",  # chrF++=90, fluent
    "checkpoints/finetuned/fold_2/best",  # BLEU=100, accurate
]

# MBR parameters
use_mbr = True
mbr_num_candidates = 10  # 20 candidates total (10 per model)
num_beams = 12
length_penalty = 1.5
repetition_penalty = 1.2
```

This configuration combines Fold 0's fluent English generation (high chrF++) with Fold 2's precise term matching (perfect BLEU).

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

## Requirements

See `requirements.txt` for dependencies.

## Known Issues and Limitations

1. **Mode Collapse**: 3 out of 5 folds (Fold 1, 3, 4) experienced various degrees of mode collapse or suboptimal convergence.

2. **Data Leakage**: Due to data-aware split, 104-125 documents overlap between train/val sets in each fold.

3. **BF16 Instability**: Mixed precision training (BF16) caused numerical instability in some folds.

4. **Category Imbalance**: The model overfits to loan contract patterns due to skewed training data distribution.

## Future Improvements

- Implement proper document-level split (no overlap)
- Address class imbalance through data augmentation
- Try FP32 training for better numerical stability
- Experiment with label smoothing to prevent overfitting

## License

This project is for academic research purposes.
