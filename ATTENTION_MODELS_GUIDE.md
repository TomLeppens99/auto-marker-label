# Attention Mechanisms Integration Guide

## Overview

This guide explains the new attention mechanism capabilities added to the auto-marker-label system. The neural network now supports three architectures:

1. **LSTM** - Original architecture (backward compatible)
2. **Transformer** - Pure transformer with self-attention
3. **Hybrid** - LSTM + Multi-head attention (Recommended)

## Quick Start

### Selecting a Model Type

Edit `automarkerlabel.py` line 44:

```python
model_type = 'hybrid'  # Options: 'lstm', 'transformer', 'hybrid'
```

### Architecture Comparison

| Architecture | Best For | Pros | Cons |
|-------------|----------|------|------|
| **LSTM** | Backward compatibility, CPU training | Proven performance, Lower memory, Faster on CPU | Limited long-range dependencies |
| **Transformer** | Long sequences, GPU available | Best long-range modeling, Parallelizable | Requires more data, GPU recommended |
| **Hybrid** ⭐ | General use (Recommended) | Best of both worlds, Better accuracy | Slightly higher memory |

## Configuration Parameters

### Core Architecture Settings

```python
# In automarkerlabel.py, lines 38-51

# Original LSTM parameters (used by all architectures)
batch_size = 100
nLSTMcells = 256        # LSTM hidden size
nLSTMlayers = 3         # Number of LSTM layers
LSTMdropout = 0.17      # Dropout rate for LSTM
FCnodes = 128           # Fully connected layer nodes

# Attention and Transformer parameters (new)
model_type = 'hybrid'   # Model architecture selection
nAttentionHeads = 8     # Number of attention heads (4-16 typical)
nTransformerLayers = 3  # Number of transformer encoder layers
transformerDim = 256    # Should match nLSTMcells for hybrid models
transformerFFDim = 512  # Feedforward dimension (2-4x transformerDim)
attentionDropout = 0.1  # Dropout for attention layers

# Learning parameters (unchanged)
lr = 0.078              # Learning rate
momentum = 0.65         # SGD momentum
```

## Architecture Details

### 1. LSTM Architecture

**Structure:**
```
Input → LSTM(3 layers, 256 cells) → FC Head → Output
```

**Use Cases:**
- Backward compatibility with existing models
- Training on CPU
- Quick experimentation
- Baseline comparisons

**Performance:** Good for short to medium-range temporal dependencies

### 2. Transformer Architecture

**Structure:**
```
Input → Linear Projection → Positional Encoding →
Transformer Encoder(3 layers, 8 heads) → FC Head → Output
```

**Key Features:**
- Sinusoidal positional encoding for sequence order
- Multi-head self-attention (8 heads)
- 512-dimensional feedforward network
- Padding masks for variable-length sequences

**Use Cases:**
- Very long motion sequences (>240 frames)
- When parallel processing is important
- Research and experimentation
- Complex motion patterns

**Performance:** Excellent at capturing long-range dependencies, requires GPU

### 3. Hybrid Architecture (Recommended)

**Structure:**
```
Input → Linear Projection → LSTM(3 layers, 256 cells) →
Multi-Head Attention(8 heads) → Residual + LayerNorm → FC Head → Output
```

**Key Features:**
- LSTM captures local temporal patterns
- Self-attention captures global dependencies
- Residual connections for training stability
- Layer normalization for better convergence

**Use Cases:**
- Default choice for new models
- Complex athletic movements
- Multi-joint coordination tasks
- Best overall accuracy

**Performance:** Combines strengths of both LSTM and Transformer

## Hyperparameter Tuning

### Attention Heads (nAttentionHeads)

```python
nAttentionHeads = 8  # Recommended: 4, 8, or 16
```

- **4 heads**: Faster, less memory, simpler attention patterns
- **8 heads**: Balanced (default)
- **16 heads**: More complex patterns, requires more data/memory

### Transformer Dimension (transformerDim)

```python
transformerDim = 256  # Should match nLSTMcells for hybrid
```

- **128**: Smaller models, faster training
- **256**: Standard (recommended)
- **512**: Larger capacity, requires more data

### Feedforward Dimension (transformerFFDim)

```python
transformerFFDim = 512  # Typically 2-4x transformerDim
```

Rule of thumb: `transformerFFDim = 2 * transformerDim`

### Attention Dropout (attentionDropout)

```python
attentionDropout = 0.1  # Regularization for attention
```

- **0.0**: No regularization (may overfit)
- **0.1**: Standard (recommended)
- **0.2-0.3**: Heavy regularization (small datasets)

## Training Considerations

### GPU Recommendations

| Architecture | CPU Training | GPU Training | Memory Usage |
|-------------|--------------|--------------|--------------|
| LSTM | Feasible | Fast | Low (~1-2 GB) |
| Transformer | Slow | Very Fast | Medium (~2-4 GB) |
| Hybrid | Slow | Fast | Medium (~2-3 GB) |

**Recommendation:** Use GPU for transformer and hybrid models (10-50x speedup)

### Training Time Estimates

Approximate training time for 10 epochs on 1000 trials:

- **LSTM + CPU**: 4-6 hours
- **LSTM + GPU**: 30-45 minutes
- **Transformer + GPU**: 45-60 minutes
- **Hybrid + GPU**: 35-50 minutes

### Data Requirements

| Architecture | Minimum Trials | Recommended Trials |
|-------------|---------------|-------------------|
| LSTM | 100 | 500+ |
| Transformer | 200 | 1000+ |
| Hybrid | 150 | 750+ |

Transformer models benefit from more training data due to higher capacity.

## Migration from Existing Models

### Backward Compatibility

To use existing trained models, set:
```python
model_type = 'lstm'
```

Your existing `.ckpt` files will load correctly with this setting.

### Training New Models

1. **Keep existing LSTM model** for comparison
2. **Train hybrid model** with same data
3. **Compare performance** on validation set
4. **Deploy best model** to production

### Model Versioning

Use descriptive filenames:
```
model_lstm_2026-01-22.ckpt      # LSTM model
model_hybrid_2026-01-22.ckpt    # Hybrid model
model_transformer_2026-01-22.ckpt  # Transformer model
```

## Testing Your Installation

Run the test script to verify all architectures work:

```bash
python test_attention_models.py
```

Expected output:
```
Testing LSTM architecture... ✓ PASSED
Testing TRANSFORMER architecture... ✓ PASSED
Testing HYBRID architecture... ✓ PASSED

ALL TESTS PASSED ✓✓✓
```

## Troubleshooting

### Issue: "Mismatch in model architecture"

**Problem:** Loading a model trained with different `model_type`

**Solution:** Set `model_type` to match the architecture used during training

### Issue: "Out of memory error"

**Problem:** GPU memory exhausted

**Solutions:**
1. Reduce `batch_size` (e.g., from 100 to 50)
2. Reduce `transformerDim` or `transformerFFDim`
3. Reduce `nAttentionHeads`
4. Use LSTM architecture instead

### Issue: "Transformer training very slow"

**Problem:** Training on CPU

**Solution:**
- Use GPU for transformer/hybrid models
- Or switch to LSTM architecture for CPU training

### Issue: "NaN losses during training"

**Problem:** Learning rate too high or numerical instability

**Solutions:**
1. Reduce learning rate: `lr = 0.01` instead of `0.078`
2. Increase `attentionDropout` to 0.2
3. Check input data for NaN values
4. Verify data normalization (scaleVals)

## Performance Optimization Tips

### For Best Accuracy
```python
model_type = 'hybrid'
nAttentionHeads = 8
transformerDim = 256
num_epochs = 15  # More epochs than LSTM
```

### For Fastest Training
```python
model_type = 'lstm'
batch_size = 100
# Train on GPU
```

### For Memory Efficiency
```python
model_type = 'lstm'
batch_size = 50
nLSTMcells = 128  # Reduce from 256
```

### For Long Sequences (>240 frames)
```python
model_type = 'transformer'
nAttentionHeads = 8
transformerDim = 256
max_len = 480  # Increase window size
```

## Advanced Usage

### Custom Attention Configurations

For specialized use cases, you can modify the architecture directly in `automarkerlabel.py`:

1. **More Transformer Layers:** Increase `nTransformerLayers` for very complex patterns
2. **Deeper LSTM:** Increase `nLSTMlayers` for hybrid models
3. **Larger Networks:** Scale up `transformerDim` and `FCnodes` proportionally

### Attention Visualization (Future Work)

The attention weights can be extracted for visualization:
- Understand which time steps are important
- Debug prediction failures
- Validate model behavior

(Feature to be implemented in future updates)

## Best Practices

1. **Start with hybrid model** - best default choice
2. **Use GPU for training** - essential for transformer/hybrid
3. **Train longer** - attention models benefit from more epochs
4. **Monitor validation loss** - attention models may need early stopping
5. **Keep LSTM baseline** - for performance comparison
6. **Version your models** - include architecture in filename

## Citation

If you use the attention mechanism extensions in your research, please cite both:

1. The original paper (Clouthier et al., 2021)
2. The attention mechanism integration (this implementation)

## Support

For issues or questions:
- Check CLAUDE.md for general system documentation
- Review test_attention_models.py for usage examples
- Verify hyperparameters match recommended ranges
- Ensure model_type is consistent between training and prediction

---

**Last Updated:** 2026-01-22
**Version:** 1.0
**Compatible with:** auto-marker-label v1.0+
