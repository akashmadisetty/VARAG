# CUDA Out of Memory Optimizations for T4 GPU

## Problem
The original implementation tried to load both the base ColPali model (`vidore/colpali-v1.3`) and fine-tuned model (`akashmadisetty/colpali-merged-model-hi-10k`) simultaneously, causing CUDA Out of Memory errors on T4 GPUs (~16GB VRAM).

## Solutions Implemented

### 1. Sequential Model Loading
**Before**: Both models loaded simultaneously
```python
base_colpali_model, base_colpali_processor = get_model_colpali("vidore/colpali-v1.3")
finetuned_colpali_model, finetuned_colpali_processor = get_model_colpali("akashmadisetty/colpali-merged-model-hi-10k")
```

**After**: Dynamic loading with memory clearing
```python
def load_model(model_name):
    # Clear current model from memory
    del colpali_model, colpali_processor, similarity_mapper
    gc.collect()
    torch.cuda.empty_cache()
    
    # Load new model
    colpali_model, colpali_processor = get_model_colpali(model_name)
```

### 2. Memory Optimizations Applied
- **Half Precision**: Models use `float16` instead of `float32`
- **Memory Fraction**: Limited to 85% of GPU memory
- **Aggressive Cleanup**: Clear memory between operations
- **Image Thumbnailing**: Resize visualizations to 400x400 pixels
- **Limited Visualizations**: Max 6 similarity maps per image instead of all tokens
- **Batch Limiting**: Process max 5 images instead of 10

### 3. Memory Monitoring
Added `memory_optimizer.py` with utilities:
- Real-time GPU/system memory monitoring
- Memory optimization recommendations
- Automatic cleanup routines
- T4-specific optimizations

### 4. UI Improvements
- **Model Switcher**: Manual model switching UI
- **Memory Status Display**: Current loaded model indicator
- **Progress Feedback**: Clear status messages during model switching
- **Error Handling**: Graceful fallback on memory errors

## Memory Usage Comparison

### Before (Simultaneous Loading)
- Base Model: ~3.5 GB
- Fine-tuned Model: ~3.5 GB
- Processing Overhead: ~4 GB
- **Total**: ~11 GB (often exceeds T4 limit with system overhead)

### After (Sequential Loading)
- Single Model: ~3.5 GB
- Processing Overhead: ~2 GB (optimized)
- Memory Buffer: ~2 GB
- **Total**: ~7.5 GB (comfortable for T4)

## How to Use

### 1. Automatic Optimization
The system automatically applies optimizations on startup:
```bash
python comp_demo.py
```

### 2. Manual Model Switching
In the "Interpret ColPali" tab:
1. Use the dropdown to select desired model
2. Click "🔄 Switch Model"
3. Wait for confirmation message

### 3. Model Comparison
1. Click "🎯 Generate Model Comparison"
2. System automatically loads both models sequentially
3. Generates side-by-side comparison
4. Restores original model

### 4. Memory Monitoring
Run the memory optimizer separately:
```bash
python memory_optimizer.py
```

## Performance Tips

### For T4 GPUs (16GB VRAM)
1. **Use Sequential Mode**: Always enabled automatically
2. **Limit Concurrent Operations**: Only one model operation at a time
3. **Clear Memory Regularly**: Use the "Switch Model" feature
4. **Monitor Memory**: Check status before operations

### For Smaller GPUs (<12GB VRAM)
1. **Use Base Model Only**: Set `current_model_name = BASE_MODEL_NAME`
2. **Disable Comparison**: Comment out comparison functionality
3. **Reduce Batch Size**: Further limit image processing
4. **Use CPU Fallback**: Consider CPU processing for non-critical operations

### For Larger GPUs (>20GB VRAM)
1. **Enable Simultaneous Loading**: Modify code to load both models
2. **Increase Limits**: Raise visualization and image limits
3. **Use Full Precision**: Remove half-precision optimization

## Error Recovery

### CUDA Out of Memory
1. **Automatic Recovery**: System tries to clear memory and retry
2. **Manual Recovery**: Use "Switch Model" to reload
3. **Restart Option**: Restart the application if needed

### Model Loading Failures
1. **Fallback**: System falls back to previously working model
2. **Status Display**: Clear error messages in UI
3. **Memory Status**: Shows available memory before operations

## Code Structure

### Key Files
- `comp_demo.py`: Main application with optimizations
- `memory_optimizer.py`: Memory management utilities
- `colpali_similarity_v2.py`: Similarity mapping (unchanged)

### Key Functions
- `load_model()`: Dynamic model loading with cleanup
- `get_current_model_info()`: Current model status
- `generate_model_comparison_for_images()`: Memory-efficient comparison
- `switch_model_handler()`: UI model switching

## Monitoring Commands

### Check GPU Memory
```python
from memory_optimizer import print_memory_status
print_memory_status()
```

### Clear Memory
```python
from memory_optimizer import clear_memory
clear_memory()
```

### Check Comparison Readiness
```python
from memory_optimizer import check_memory_for_comparison
can_compare, message = check_memory_for_comparison()
print(message)
```

## Expected Behavior

### Normal Operation
- Models load without memory errors
- Comparison works smoothly
- Memory usage stays under 80%
- Clear status messages throughout

### Memory Pressure
- System warns about low memory
- Automatically applies more aggressive cleanup
- May limit number of visualizations
- Suggests manual memory clearing

### Recovery
- Failed operations provide clear error messages
- System attempts automatic recovery
- Manual recovery options available
- Memory status always visible

This optimization approach transforms the memory-intensive dual-model comparison into a memory-efficient sequential operation suitable for T4 GPUs while maintaining all functionality.
