#!/usr/bin/env python3
"""
ColPali Model Loading Fix

This provides a wrapper function for safe model loading.
"""

import torch
from typing import Tuple, Any

def safe_get_model_colpali(model_name: str, force_precision: str = None) -> Tuple[Any, Any]:
    """
    Safely load ColPali model with proper precision handling.
    
    Args:
        model_name: Name of the model to load
        force_precision: Force specific precision ('bfloat16', 'float16', 'float32', or None)
    
    Returns:
        Tuple of (model, processor)
    """
    from varag.utils import get_model_colpali
    
    print(f"🔄 Loading model: {model_name}")
    
    try:
        # Load model with default settings first
        model, processor = get_model_colpali(model_name)
        
        # Check current state
        current_dtype = next(model.parameters()).dtype
        current_device = next(model.parameters()).device
        
        print(f"📋 Loaded model dtype: {current_dtype}")
        print(f"📋 Loaded model device: {current_device}")
        
        # Handle precision conversion based on requirements
        if force_precision:
            target_dtype = {
                'bfloat16': torch.bfloat16,
                'float16': torch.float16,
                'float32': torch.float32
            }.get(force_precision.lower())
            
            if target_dtype and current_dtype != target_dtype:
                print(f"🔄 Converting from {current_dtype} to {target_dtype}")
                try:
                    if target_dtype == torch.bfloat16:
                        model = model.to(torch.bfloat16)
                    elif target_dtype == torch.float16:
                        model = model.half()
                    elif target_dtype == torch.float32:
                        model = model.float()
                    
                    new_dtype = next(model.parameters()).dtype
                    print(f"✅ Conversion successful: {new_dtype}")
                    
                except Exception as e:
                    print(f"⚠️ Precision conversion failed: {e}")
                    print(f"✅ Using original precision: {current_dtype}")
        
        else:
            # Auto-optimize for memory efficiency without breaking functionality
            if current_dtype == torch.float32 and torch.cuda.is_available() and current_device.type == 'cuda':
                # Only convert float32 to bfloat16 if supported
                if torch.cuda.is_bf16_supported():
                    try:
                        model = model.to(torch.bfloat16)
                        print(f"✅ Auto-optimized to BFloat16")
                    except Exception as e:
                        print(f"⚠️ Auto-optimization failed: {e}")
                        print(f"✅ Using original precision")
                else:
                    print(f"✅ Using original precision (BFloat16 not supported)")
            else:
                print(f"✅ Using original precision: {current_dtype}")
        
        final_dtype = next(model.parameters()).dtype
        final_device = next(model.parameters()).device
        
        print(f"🎯 Final model dtype: {final_dtype}")
        print(f"🎯 Final model device: {final_device}")
        
        return model, processor
        
    except Exception as e:
        print(f"❌ Error loading model {model_name}: {e}")
        raise

def check_model_compatibility(model_name: str) -> dict:
    """Check what precision options work for a specific model"""
    results = {
        'model_name': model_name,
        'original_dtype': None,
        'bfloat16_works': False,
        'float16_works': False,
        'float32_works': True,  # Always assume float32 works
        'recommended': 'float32'
    }
    
    try:
        from varag.utils import get_model_colpali
        
        print(f"🔍 Testing compatibility for {model_name}")
        
        # Load original model
        model, processor = get_model_colpali(model_name)
        original_dtype = next(model.parameters()).dtype
        results['original_dtype'] = str(original_dtype)
        
        print(f"📋 Original dtype: {original_dtype}")
        
        # Test BFloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            try:
                test_model = model.to(torch.bfloat16)
                results['bfloat16_works'] = True
                print(f"✅ BFloat16 compatible")
                del test_model
            except Exception as e:
                print(f"❌ BFloat16 incompatible: {e}")
        
        # Test Float16
        try:
            test_model = model.half()
            results['float16_works'] = True
            print(f"✅ Float16 compatible")
            del test_model
        except Exception as e:
            print(f"❌ Float16 incompatible: {e}")
        
        # Determine recommendation
        if results['bfloat16_works']:
            results['recommended'] = 'bfloat16'
        elif results['float16_works']:
            results['recommended'] = 'float16'
        else:
            results['recommended'] = 'float32'
        
        print(f"🎯 Recommended precision: {results['recommended']}")
        
        # Clean up
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Compatibility check failed: {e}")
    
    return results

if __name__ == "__main__":
    # Test both models
    base_compat = check_model_compatibility("vidore/colpali-v1.3")
    print("\n" + "="*50 + "\n")
    finetuned_compat = check_model_compatibility("akashmadisetty/colpali-merged-model-hi-10k")
    
    print(f"\n📊 Compatibility Summary:")
    print(f"Base model recommended: {base_compat['recommended']}")
    print(f"Fine-tuned model recommended: {finetuned_compat['recommended']}")
