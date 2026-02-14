#!/usr/bin/env python3
"""
手动验证DAPT模型脚本
运行方式: python scripts/verify_dapt_manual.py
"""

import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def main():
    model_path = "checkpoints/dapt/final"
    
    print("=" * 60)
    print("DAPT Model Manual Verification")
    print("=" * 60)
    
    # 1. 加载tokenizer
    print("\n[1] Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"    ✓ Tokenizer loaded")
        print(f"    Vocab size: {len(tokenizer)}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return
    
    # 2. 加载模型
    print("\n[2] Loading model...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model = model.to(device)
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"    ✓ Model loaded")
        print(f"    Type: {model.config.model_type}")
        print(f"    Parameters: {total_params/1e6:.1f}M")
        print(f"    Device: {device}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return
    
    # 3. 推理测试
    print("\n[3] Running inference tests...")
    test_cases = [
        "um-ma kà-ru-um kà-ni-ia-ma",
        "KIŠIB ma-nu-ba-lúm-a-šur",
        "a-na DAM.GÀR-ru-tim",
        "1 ma-na KÙ.BABBAR"
    ]
    
    for text in test_cases:
        try:
            inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=20, num_beams=2)
            
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"    Input:  {text[:35]}")
            print(f"    Output: {decoded[:40]}")
            print()
        except Exception as e:
            print(f"    ✗ Error on '{text}': {e}")
    
    # 4. 检查权重是否有NaN
    print("[4] Checking weights for NaN...")
    has_nan = False
    nan_layers = []
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            has_nan = True
            nan_layers.append(name)
    
    if has_nan:
        print(f"    ✗ Found NaN in {len(nan_layers)} layers!")
        for name in nan_layers[:3]:
            print(f"      - {name}")
    else:
        print("    ✓ No NaN found in weights")
    
    # 5. 总结
    print("\n" + "=" * 60)
    if not has_nan:
        print("✓ Verification PASSED!")
        print("The DAPT model is ready for fine-tuning.")
        print("\nNext step: Run 5-fold training")
        print("  python scripts/03_train.py")
    else:
        print("✗ Verification FAILED!")
        print("The model has NaN weights and needs retraining.")
    print("=" * 60)

if __name__ == "__main__":
    main()
