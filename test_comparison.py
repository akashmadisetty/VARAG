#!/usr/bin/env python3
"""
Test script to verify the model comparison functionality
"""

def test_model_initialization():
    """Test if both models can be initialized"""
    try:
        from varag.utils import get_model_colpali
        
        print("🔄 Testing model initialization...")
        
        # Test base model
        print("Loading base model (vidore/colpali-v1.3)...")
        base_model, base_processor = get_model_colpali("vidore/colpali-v1.3")
        print("✅ Base model loaded successfully")
        
        # Test fine-tuned model
        print("Loading fine-tuned model (akashmadisetty/colpali-merged-model-hi-10k)...")
        finetuned_model, finetuned_processor = get_model_colpali("akashmadisetty/colpali-merged-model-hi-10k")
        print("✅ Fine-tuned model loaded successfully")
        
        # Test similarity mapper creation
        from colpali_similarity_v2 import create_similarity_mapper
        
        print("Creating similarity mappers...")
        base_mapper = create_similarity_mapper(base_model, base_processor)
        finetuned_mapper = create_similarity_mapper(finetuned_model, finetuned_processor)
        
        if base_mapper and finetuned_mapper:
            print("✅ Both similarity mappers created successfully")
            return True
        else:
            print("❌ Failed to create similarity mappers")
            return False
            
    except Exception as e:
        print(f"❌ Error during model initialization: {e}")
        return False

def test_comp_demo_import():
    """Test if comp_demo can be imported without errors"""
    try:
        print("🔄 Testing comp_demo import...")
        import comp_demo
        print("✅ comp_demo imported successfully")
        
        # Check if the new variables exist
        if hasattr(comp_demo, 'base_similarity_mapper') and hasattr(comp_demo, 'finetuned_similarity_mapper'):
            print("✅ Model comparison variables are available")
            return True
        else:
            print("❌ Model comparison variables not found")
            return False
            
    except Exception as e:
        print(f"❌ Error importing comp_demo: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Model Comparison Tests")
    print("=" * 50)
    
    # Test 1: Model initialization
    test1_passed = test_model_initialization()
    
    print("\n" + "=" * 50)
    
    # Test 2: comp_demo import
    test2_passed = test_comp_demo_import()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"✅ Model Initialization: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"✅ comp_demo Import: {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Model comparison feature is ready.")
    else:
        print("\n⚠️ Some tests failed. Please check the error messages above.")
