# ComfyUI Line Boil Custom Node

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-green.svg)](https://github.com/comfyanonymous/ComfyUI)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This custom node applies a "line boil" effect to images or videos, targeting only pixels close to black (line art). It provides a subtle, hand-drawn look by slightly warping and shifting the detected lines over time.

## 🎨 Features

- **Line Art Detection**: Specifically targets near-black pixels, leaving fills and background untouched.
- **Two Node Variants**:
  - **Line Boil (Image)**: Processes standard ComfyUI image batches (tensors).
  - **Line Boil (Video)**: Processes a video file path directly, outputting a new video file with a custom suffix.
- **Frame-Level Parallelism**: Uses `ProcessPoolExecutor` to process multiple frames in parallel.
- **GPU Acceleration**: Optional `cv2.cuda` support for heavy image operations (mask warp, blur, morphology).
- **Temporal Stability**: Configurable `hold` parameter to reuse the same boil pattern for consecutive frames.
- **Deterministic Output**: Uses a random seed for reproducible results.
- **Progress Tracking**: Integrated ComfyUI progress bar for long batches or videos.

## ⚙️ Parameters

| Parameter | Description | Default | Type |
|-----------|-------------|---------|------|
| **enabled** | Toggle the effect on/off | `True` | Boolean |
| **black_threshold** | Maximum per-channel distance from #000000 to count as line art | `45` | Integer (0-255) |
| **expand** | Expand the detected line mask to catch antialiasing | `1` | Integer (0-10) |
| **warp_strength** | Subtle displacement amount in pixels | `0.8` | Float (0.1-5.0) |
| **warp_scale** | Noise scale for displacement | `28.0` | Float (1.0-100.0) |
| **shift** | Maximum random X/Y pixel shift | `1` | Integer (0-10) |
| **alpha_jitter** | Small per-frame strength jitter | `0.06` | Float (0.0-1.0) |
| **hold** | Reuse the same boil pattern for N consecutive frames | `2` | Integer (1-100) |
| **seed** | Random seed for the effect | `0` | Integer |
| **workers** | Number of parallel processes to use | `4` | Integer (1-32) |
| **use_gpu** | Toggle OpenCV CUDA acceleration | `False` | Boolean |

## 📋 Requirements

### Core Dependencies
- `opencv-python>=4.5.0`
- `numpy>=1.20.0`
- `torch>=1.0.0` (included with ComfyUI)

### Optional Dependencies
- `opencv-contrib-python>=4.5.0` for CUDA support
- `opencv-python-headless>=4.5.0` for headless environments

## 🚀 Installation

### Method 1: Git Clone (Recommended)
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/scofano/comfy-line-boil.git
```

### Method 2: Manual Installation
1. Download the latest release from [GitHub](https://github.com/scofano/comfy-line-boil/releases)
2. Extract the contents to `ComfyUI/custom_nodes/comfy-line-boil/`
3. Restart ComfyUI

### Method 3: Using ComfyUI Manager
If you have ComfyUI Manager installed, you can add this repository URL:
```
https://github.com/scofano/comfy-line-boil.git
```

## 📖 Usage

### Image Processing
1. Load your image into ComfyUI
2. Add the "Line Boil (Image)" node to your workflow
3. Connect your image to the input
4. Adjust parameters to achieve desired effect
5. Connect output to your next node or save image

### Video Processing
1. Add the "Line Boil (Video)" node to your workflow
2. Provide the path to your video file
3. Set output suffix (e.g., "_boiled")
4. Configure processing parameters
5. Run the node to generate processed video

## 🎯 Tips & Best Practices

### For Optimal Results
- **Start with defaults**: Begin with default parameters and adjust gradually
- **Black threshold tuning**: Lower values for cleaner line art, higher for more inclusive detection
- **Warp strength**: Keep below 2.0 for subtle effects, higher for dramatic animation
- **Hold parameter**: Use higher values (10-50) for smoother, more stable animation

### Performance Optimization
- **Enable GPU**: Set `use_gpu=True` if you have CUDA-compatible hardware
- **Worker count**: Match to your CPU core count for optimal parallel processing
- **Hold frames**: Higher hold values reduce processing time while maintaining quality

### Creative Applications
- **Animation enhancement**: Add subtle movement to static line art
- **Video processing**: Apply to entire video sequences for hand-drawn effects
- **Batch processing**: Process multiple images with consistent parameters

## 🔧 Troubleshooting

### Common Issues

**CUDA Not Available**
```
Error: OpenCV CUDA not available
```
- Install `opencv-contrib-python` instead of `opencv-python`
- Ensure CUDA drivers are properly installed
- Set `use_gpu=False` to disable GPU acceleration

**Memory Issues with Large Videos**
```
MemoryError: Unable to allocate memory
```
- Reduce `workers` parameter
- Process shorter video segments
- Close other applications to free memory

**No Effect Visible**
```
Line boil effect not applied
```
- Check `enabled` parameter is set to `True`
- Verify `black_threshold` is appropriate for your image
- Ensure input contains sufficient line art

### Getting Help
- Check the [Issues](https://github.com/scofano/comfy-line-boil/issues) section
- Review [Discussions](https://github.com/scofano/comfy-line-boil/discussions) for community tips
- Create a new issue with detailed error information

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone the repository
git clone https://github.com/scofano/comfy-line-boil.git
cd comfy-line-boil

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) for the excellent workflow system
- [OpenCV](https://opencv.org/) for powerful image processing capabilities
- The ComfyUI community for inspiration and support

## 📞 Contact

For questions, suggestions, or feedback:
- Create an [Issue](https://github.com/scofano/comfy-line-boil/issues)
- Join our [Discussions](https://github.com/scofano/comfy-line-boil/discussions)
- Visit the [ComfyUI Discord](https://discord.gg/comfyui)

---

**Made with ❤️ for the ComfyUI community**
