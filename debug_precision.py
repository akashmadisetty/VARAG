#!/usr/bin/env python3
"""
ColPali Precision Fix

This script helps debug and fix the BFloat16 vs Half precision issue.
"""

import torch
from varag.utils import get_model_colpali

def debug_model_precision():
    """Debug the precision requirements for ColPali models"""
    print("🔍 Debugging ColPali Model Precision")
    print("=" * 50)
    
    # Check CUDA and precision support
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"BFloat16 Supported: {torch.cuda.is_bf16_supported()}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    
    # Test loading base model
    try:
        print("\n🔄 Loading base model...")
        model, processor = get_model_colpali("vidore/colpali-v1.3")
        
        print(f"Model type: {type(model)}")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Model dtype: {next(model.parameters()).dtype}")
        
        # Test different precision conversions
        print("\n🧪 Testing precision conversions...")
        
        # Test BFloat16
        try:
            model_bf16 = model.to(torch.bfloat16)
            print("✅ BFloat16 conversion successful")
            test_dtype = next(model_bf16.parameters()).dtype
            print(f"BFloat16 model dtype: {test_dtype}")
        except Exception as e:
            print(f"❌ BFloat16 conversion failed: {e}")
        
        # Test Half (Float16)
        try:
            model_half = model.half()
            print("✅ Half (Float16) conversion successful")
            test_dtype = next(model_half.parameters()).dtype
            print(f"Half model dtype: {test_dtype}")
        except Exception as e:
            print(f"❌ Half conversion failed: {e}")
        
        # Test with no precision change
        print("\n🎯 Recommended approach:")
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            print("Use BFloat16: model.to(torch.bfloat16)")
        else:
            print("Use default precision (avoid .half() conversion)")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    return True

def create_safe_model_loader():
    """Create a safe model loading function"""
    code = '''
def safe_load_colpali_model(model_name):
    """Safely load ColPali model with appropriate precision"""
    from varag.utils import get_model_colpali
    import torch
    
    try:
        # Load model with default precision first
        model, processor = get_model_colpali(model_name)
        
        # Apply precision optimization safely
        if torch.cuda.is_available():
            # Move to GPU first if not already there
            if next(model.parameters()).device.type == 'cpu':
                model = model.cuda()
            
            # Check if model already has the right precision
            current_dtype = next(model.parameters()).dtype
            
            if current_dtype == torch.bfloat16:
                print("✅ Model already in BFloat16")
            elif torch.cuda.is_bf16_supported():
                try:
                    model = model.to(torch.bfloat16)
                    print("✅ Converted to BFloat16")
                except Exception as e:
                    print(f"⚠️ BFloat16 conversion failed: {e}")
                    print("✅ Using original precision")
            else:
                print("✅ Using original precision (BFloat16 not supported)")
        
        return model, processor
        
    except Exception as e:
        print(f"❌ Error in safe model loading: {e}")
        raise
    '''
    
    with open("safe_model_loader.py", "w") as f:
        f.write(code)
    
    print("📝 Created safe_model_loader.py")

if __name__ == "__main__":
    success = debug_model_precision()
    if success:
        create_safe_model_loader()
        print("\n🎉 Precision debugging complete!")
    else:
        print("\n❌ Precision debugging failed!")
