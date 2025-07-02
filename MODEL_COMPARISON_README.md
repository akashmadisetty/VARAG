# ColPali Model Comparison Feature

## Overview
Added a comprehensive model comparison feature to the "Interpret ColPali" tab in `comp_demo.py` that allows side-by-side comparison between the base ColPali model (`vidore/colpali-v1.3`) and your fine-tuned model (`akashmadisetty/colpali-merged-model-hi-10k`).

## What's Been Added

### 1. Dual Model Initialization
- **Base Model**: `vidore/colpali-v1.3` 
- **Fine-tuned Model**: `akashmadisetty/colpali-merged-model-hi-10k`
- Both models are loaded with their respective processors
- Separate similarity mappers are created for each model

### 2. Updated UI Layout
The "Interpret ColPali" tab now features:
- **Header**: Updated to indicate model comparison capability
- **Control Section**: 
  - Query display (same as before)
  - Refresh button (same as before)
  - **New**: "Generate Model Comparison" button
- **Token Analysis**: Two side-by-side tables showing token importance for each model
- **Image Galleries**: Side-by-side comparison showing similarity maps from both models

### 3. New Comparison Function
`generate_model_comparison_for_images()` that:
- Generates similarity maps for both models simultaneously
- Creates comparative token analysis tables
- Displays side-by-side galleries for visual comparison
- Provides detailed status information

### 4. Enhanced Features
- **Token Ranking**: Shows how each model ranks different query tokens by importance
- **Visual Comparison**: Side-by-side similarity maps for direct comparison
- **Detailed Statistics**: Similarity scores and token importance for both models
- **Error Handling**: Robust error handling for missing models or data

## How to Use

1. **Setup**: Go to "Retrieve and Query Data" tab and perform a retrieval first
2. **Navigate**: Switch to "Interpret ColPali" tab
3. **Refresh**: Click "🔄 Refresh ColPali Results" to load current query and images
4. **Compare**: Click "🎯 Generate Model Comparison" to see side-by-side analysis
5. **Analyze**: 
   - Compare token importance tables at the top
   - Examine side-by-side similarity maps below
   - Look for differences in attention patterns between models

## Expected Output

### Token Analysis Tables
- **Left Table**: Base model token rankings and similarity scores
- **Right Table**: Fine-tuned model token rankings and similarity scores
- Compare which tokens each model considers most important

### Similarity Map Galleries
- **Left Column**: Base model attention maps for each image
- **Right Column**: Fine-tuned model attention maps for the same images
- Visual comparison of where each model focuses attention

## Benefits

1. **Model Performance Comparison**: Directly see how fine-tuning affected attention patterns
2. **Query Optimization**: Understand which tokens are most important for each model
3. **Debugging**: Identify differences in model behavior
4. **Research Insights**: Analyze the impact of your fine-tuning approach

## Technical Details

### Model Loading
```python
# Base model
base_colpali_model, base_colpali_processor = get_model_colpali("vidore/colpali-v1.3")

# Fine-tuned model  
finetuned_colpali_model, finetuned_colpali_processor = get_model_colpali("akashmadisetty/colpali-merged-model-hi-10k")
```

### Similarity Mapper Creation
```python
base_similarity_mapper = create_similarity_mapper(base_colpali_model, base_colpali_processor)
finetuned_similarity_mapper = create_similarity_mapper(finetuned_colpali_model, finetuned_colpali_processor)
```

### Comparison Function
The new function analyzes both models simultaneously and returns:
- Comparison status message
- Base model token analysis DataFrame
- Fine-tuned model token analysis DataFrame  
- Base model similarity map galleries (10 max)
- Fine-tuned model similarity map galleries (10 max)
- Row visibility controls
- Page information for both models

## Backward Compatibility
- All existing functionality remains unchanged
- Original RAG pipeline still uses the base model
- New comparison feature is an addition, not a replacement

## Testing
Run `test_comparison.py` to verify:
- Both models can be loaded successfully
- Similarity mappers can be created
- comp_demo imports without errors
- All required variables are available
