#!/bin/bash
# 完整的训练+推理流程
# 注意：此脚本需要在交互式环境中手动运行，每个步骤需要等待完成

set -e  # 遇到错误停止

echo "========================================"
echo "Akkadian Translation Training Pipeline"
echo "========================================"
echo ""

# 检查目录
cd /root/autodl-tmp

# Step 0: 数据对齐（如果还没做）
if [ ! -f "data/processed/aligned_train_v2.csv" ]; then
    echo "[Step 0] Running data alignment..."
    python scripts/alignment_v2.py
else
    echo "[Step 0] Data alignment already done, skipping..."
fi

echo ""
echo "========================================"
echo "接下来需要手动执行的步骤："
echo "========================================"
echo ""
echo "Step 1: DAPT预训练（约2-4小时）"
echo "  python scripts/02_dapt.py"
echo ""
echo "Step 2: 5-Fold训练（每个fold约1-2小时）"
echo "  # 串行训练所有folds:"
echo "  python scripts/03_train.py"
echo ""
echo "  # 或者单独训练某个fold:"
echo "  python scripts/03_train.py --current_fold 0"
echo "  python scripts/03_train.py --current_fold 1"
echo "  ..."
echo ""
echo "Step 3: (可选) Checkpoint平均"
echo "  python scripts/05_average_models.py \\"
echo "    --input checkpoints/finetuned/fold_0/best ... \\"
echo "    --output checkpoints/finetuned/averaged"
echo ""
echo "Step 4: 推理生成"
echo "  python scripts/04_inference.py"
echo ""
echo "详细说明请参考: TRAINING_GUIDE.md"
echo "========================================"
