#!/usr/bin/env python3
"""
Quick Test for BFloat16/Half Issue

Run this script to test if the precision fix works.
"""

import torch
import sys
import os

def test_precision_fix():
    print("🧪 Testing ColPali Precision Fix")
    print("=" * 50)
    
    # Add current directory to path
    sys.path.insert(0, os.getcwd())
    
    try:
        print("🔄 Testing model loading...")
        from varag.utils import get_model_colpali
        
        # Test with base model
        model, processor = get_model_colpali("vidore/colpali-v1.3")
        
        original_dtype = next(model.parameters()).dtype
        original_device = next(model.parameters()).device
        
        print(f"✅ Original model dtype: {original_dtype}")
        print(f"✅ Original model device: {original_device}")
        
        # Test the safe conversion approach
        if torch.cuda.is_available() and original_device.type == 'cuda':
            if original_dtype == torch.float32 and torch.cuda.is_bf16_supported():
                try:
                    model = model.to(torch.bfloat16)
                    new_dtype = next(model.parameters()).dtype
                    print(f"✅ Successfully converted to: {new_dtype}")
                except Exception as e:
                    print(f"❌ Conversion failed: {e}")
                    return False
            else:
                print(f"✅ No conversion needed (dtype: {original_dtype})")
        
        # Test creating similarity mapper
        print("\n🔄 Testing similarity mapper creation...")
        from colpali_similarity_v2 import create_similarity_mapper
        
        mapper = create_similarity_mapper(model, processor)
        if mapper:
            print("✅ Similarity mapper created successfully")
        else:
            print("❌ Similarity mapper creation failed")
            return False
        
        # Clean up
        del model, processor, mapper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_precision_fix()
    if success:
        print("\n✅ Ready to run comp_demo.py!")
    else:
        print("\n❌ Please check the error messages above.")
