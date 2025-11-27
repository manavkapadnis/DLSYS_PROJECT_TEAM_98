# Block-Sparse Attention: 90-Second Video Presentation

## Quick Start

1. **Open the presentation:**
   ```bash
   sparse_attention_presentation.html
   ```
   Double-click → Opens in any browser

2. **Record 90 seconds:**
   - **Windows:** Win+G → Xbox Game Bar → Record
   - **Mac:** Cmd+Space → QuickTime → New Screen Recording
   - **Linux:** OBS Studio (free download)

3. **Upload to YouTube:**
   - Title: `Team 98 Block-Sparse Attention: Efficient Transformers`
   - ⚠️ **DO NOT** check "made for kids"
   - Set to "Unlisted"
   - Copy link and submit

## What's Included

**Presentation:**
- Interactive HTML with all your benchmark results embedded
- Code snippets showing actual implementation
- CUDA kernel for block-sparse attention
- Needle framework sparse attention module
- Training script and model integration

**Code Sections in Presentation:**
- `src/ndarray_backend_cuda.cu` - CUDA kernel (~60 lines, full implementation)
- `python/needle/nn/nn_sparse_attention.py` - Sparse attention module
- `apps/pythia_model.py` - Pythia-70M model with sparse attention
- `apps/train_pythia.py` - Training loop

**Visualizations (10 images):**
- Inference speed (Pythia-70M & OPT-125M)
- Attention patterns comparison
- Memory usage analysis
- Training quality metrics
- Performance benchmarks

## What Gets Explained

**CUDA Kernel Section Shows:**
- CSR sparse format metadata structure
- Block-level parallelization
- Efficient computation of attention scores only for sparse positions
- Numerically stable softmax implementation
- Value aggregation using sparse weights

**Needle Module Section Shows:**
- Pattern generation (local, global, mixed)
- Multi-head attention with sparse masking
- Integration with standard transformer architecture

**Model Section Shows:**
- Configuration dataclass
- Token/positional embeddings
- Stacked sparse transformer layers
- Output projection and loss computation

**Training Section Shows:**
- Forward/backward pass
- Gradient computation and optimization
- Validation loop with perplexity tracking

## Timing (Perfect for 90 Seconds)

- 0-8s: Hero (title + key stats)
- 8-20s: Problem (dense attention complexity)
- 20-35s: Solution overview
- 35-50s: **CUDA kernel & Needle module code**
- 50-65s: **Model & training code**
- 65-80s: Performance results
- 80-90s: Conclusion

## File Structure

```
outputs/
├── sparse_attention_presentation.html     (Main presentation)
├── README.md                               (This file)
├── attention_patterns.png                 (All your benchmark plots)
├── inference_speed.png
├── opt_inference_speed.png
├── memory_comparison.png
├── loss_perplexity_comparison.png
├── opt_tinystories_results.png
├── performance_comparison.png
├── training_time.png
└── attention_on_sentence.png
```

All files must stay in the same directory.

## Key Metrics Displayed

| Metric | Result |
|--------|--------|
| Inference Speedup | 2-4× (1.9× → 6.8×) |
| Memory Savings | 75% (134 MB → 34 MB) |
| Quality Loss | < 0.1 (negligible) |
| Training Time | 1.27× faster |

## Tips

- Scroll smoothly through presentation (natural pacing)
- Record at 1080p for best quality
- ~85-90 seconds is perfect
- Code sections are easily readable
- All images load automatically

## Troubleshooting

**Images not loading?**
→ Ensure all `.png` files in same folder as HTML

**Video too long/short?**
→ Adjust scroll speed while recording

**Can't upload to YouTube?**
→ Use Chrome, ensure you're logged in

---

**You're ready to record!** Your research is excellent, presentation is professional. Good luck! 🚀
