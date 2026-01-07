# RAPIDS GPU Acceleration Guide

This Coral-CAT application can leverage **RAPIDS** (CuPy + cuML) for significant speedups when CUDA is available. The most critical functions accelerated are:

- `map_color_to_pixels` – Maps every pixel to the nearest color in the custom chart using Lab color space distance. This is the primary target for GPU acceleration.
- `get_colors` / `get_colors_df` – KMeans clustering for color extraction from images.

---

## Installation

RAPIDS requires a **CUDA-compatible GPU** and the CUDA Toolkit. If you do not have a GPU, the app will automatically fall back to CPU implementations.

### 1. Install RAPIDS packages

Uncomment the RAPIDS lines in `requirements.txt` and adjust for your CUDA version:

```bash
# For CUDA 12.x
pip install cupy-cuda12x cuml-cu12
```

Or install them separately:

```bash
pip install cupy-cuda12x>=12.0.0
pip install cuml-cu12>=24.0.0
```

Check the [RAPIDS documentation](https://rapids.ai/start.html) for the correct package versions matching your CUDA version.

### 2. Verify RAPIDS availability

Run the following in Python to confirm the GPU is accessible:

```python
import cupy as cp
print(cp.cuda.runtime.getDeviceCount())
```

If this returns a positive number, RAPIDS is available. If it raises an error, you'll fall back to CPU mode automatically.

---

## How it works

The application detects RAPIDS availability at runtime:

1. **On startup:** The `_init_rapids()` function checks if CuPy and cuML are installed and whether a CUDA device is available.
2. **During execution:** Functions like `map_color_to_pixels` and `kmeans_fit_predict` automatically switch between GPU and CPU implementations based on availability.
3. **Graceful fallback:** If any GPU operation fails (e.g., out of memory), the app silently falls back to the CPU path without crashing.

**No code changes required.** Simply install the RAPIDS packages, and the app will use them automatically.

---

## Performance

On typical coral image datasets (1800x1200 RGB images with ~20 colors in the palette):

- **CPU (sklearn KMeans, NumPy):** ~5-15 seconds for `map_color_to_pixels`
- **GPU (RAPIDS + CuPy):** ~0.5-2 seconds for `map_color_to_pixels`

**Expected speedup:** 5-10x for `map_color_to_pixels` and 2-5x for KMeans clustering.

---

## Troubleshooting

### "No CUDA device found"

- Ensure you have a CUDA-compatible GPU.
- Install the correct CUDA Toolkit matching your GPU driver.
- Verify with `nvidia-smi` that the GPU is visible.

### "CuPy installation failed"

- Use the correct CuPy package for your CUDA version. For example:
  - `cupy-cuda11x` for CUDA 11.x
  - `cupy-cuda12x` for CUDA 12.x

### "Out of memory" errors

- Reduce image resolution before processing.
- Close other GPU-heavy applications.
- The app will automatically fall back to CPU if GPU memory is exhausted mid-operation.

---

## References

- [RAPIDS AI](https://rapids.ai/)
- [CuPy Documentation](https://docs.cupy.dev/)
- [cuML Documentation](https://docs.rapids.ai/api/cuml/stable/)
