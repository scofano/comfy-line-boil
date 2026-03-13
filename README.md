# ComfyUI Line Boil Custom Node

This custom node applies a "line boil" effect to images or videos, targeting only pixels close to black (line art). It provides a subtle, hand-drawn look by slightly warping and shifting the detected lines over time.

## Features
- **Line Art Detection**: Specifically targets near-black pixels, leaving fills and background untouched.
- **Two Node Variants**:
  - **Line Boil (Image)**: Processes standard ComfyUI image batches (tensors).
  - **Line Boil (Video)**: Processes a video file path directly, outputting a new video file with a custom suffix.
- **Frame-Level Parallelism**: Uses `ProcessPoolExecutor` to process multiple frames in parallel.
- **GPU Acceleration**: Optional `cv2.cuda` support for heavy image operations (mask warp, blur, morphology).
- **Temporal Stability**: Configurable `hold` parameter to reuse the same boil pattern for consecutive frames.
- **Deterministic Output**: Uses a random seed for reproducible results.
- **Progress Tracking**: Integrated ComfyUI progress bar for long batches or videos.

## Parameters
- **enabled**: Toggle the effect on/off.
- **black_threshold**: Maximum per-channel distance from #000000 to count as line art. (Default: 45)
- **expand**: Expand the detected line mask to catch antialiasing. (Default: 1)
- **warp_strength**: Subtle displacement amount in pixels. (Default: 0.8)
- **warp_scale**: Noise scale for displacement. (Default: 28.0)
- **shift**: Maximum random X/Y pixel shift. (Default: 1)
- **alpha_jitter**: Small per-frame strength jitter. (Default: 0.06)
- **hold**: Reuse the same boil pattern for N consecutive frames. (Default: 2)
- **seed**: Random seed for the effect.
- **workers**: Number of parallel processes to use. (Default: 4)
- **use_gpu**: Toggle OpenCV CUDA acceleration (requires `opencv-python` with CUDA support).

## Requirements
- `opencv-python`
- `numpy`
- `torch` (included with ComfyUI)
- (Optional) `opencv-python-headless` or `opencv-contrib-python` for CUDA support.

## Installation
Clone or copy this folder into your `ComfyUI/custom_nodes/` directory.
