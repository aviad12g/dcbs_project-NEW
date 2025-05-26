#!/usr/bin/env python3
"""Check PyTorch and GPU setup."""

try:
    import torch
    print(f" PyTorch version: {torch.__version__}")
    print(f" CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f" GPU count: {torch.cuda.device_count()}")
        print(f" GPU name: {torch.cuda.get_device_name(0)}")
        print(f" CUDA version: {torch.version.cuda}")
    else:
        print(" CUDA not available - using CPU only")
        print(" This means token-by-token generation will be slow (~16+ hours)")
        
except ImportError as e:
    print(f" PyTorch import error: {e}")

print("\n" + "="*50)
print("RECOMMENDATION:")
if torch.cuda.is_available():
    print(" GPU is ready! Token-by-token evaluation will take ~15-20 minutes")
else:
    print("  You have two options:")
    print("   1. Continue with CPU (slow but scientifically accurate)")
    print("   2. Use fast generation (quick but less scientifically pure)") 