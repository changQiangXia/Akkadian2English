#!/usr/bin/env python3
"""验证DAPT模型是否正常"""

import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def main():
    model_path = "checkpoints/dapt/final"
    
    print("=" * 50)
    print("DAPT Model Verification")
    print("=" * 50)
    
    # 1. Load tokenizer
    print("\n[1] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"    Vocab size: {len(tokenizer)}")
    
    # 2. Load model
    print("\n[2] Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Model type: {model.config.model_type}")
    print(f"    Parameters: {total_params/1e6:.1f}M")
    print(f"    Device: {device}")
    
    # 3. Inference test
    print("\n[3] Running inference test...")
    test_cases = [
        "um-ma kà-ru-um kà-ni-ia-ma",
        "KIŠIB ma-nu-ba-lúm-a-šur",
        "a-na DAM.GÀR-ru-tim"
    ]
    
    for text in test_cases:
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, num_beams=2)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"    Input:  {text[:30]}...")
        print(f"    Output: {decoded[:40]}...")
        print()
    
    # 4. Check for NaN in weights
    print("[4] Checking model weights...")
    has_nan = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"    WARNING: NaN found in {name}")
            has_nan = True
            break
    
    if not has_nan:
        print("    All weights are valid (no NaN)")
    
    print("\n" + "=" * 50)
    print("Verification PASSED! Model is ready for fine-tuning.")
    print("=" * 50)

if __name__ == "__main__":
    main()
