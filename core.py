import cv2
import numpy as np
import random
import logging

logger = logging.getLogger("ComfyUI-LineBoil")

# --- CUDA Capability Probe ---
_CUDA_AVAILABLE = False
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        _CUDA_AVAILABLE = True
except Exception:
    _CUDA_AVAILABLE = False

def is_cuda_available():
    return _CUDA_AVAILABLE

def set_cv2_threads(n=1):
    try:
        cv2.setNumThreads(n)
    except Exception:
        pass

def get_gpu_kernel(size):
    return np.ones((size, size), np.uint8)

def process_frame_core(rgba, seed, params, use_gpu=False):
    """
    Core logic for line boil effect. 
    Optimized for either CPU or GPU (OpenCV CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Validation: use_gpu only if available
    use_gpu = use_gpu and _CUDA_AVAILABLE

    h, w = rgba.shape[:2]
    black_threshold = params['black_threshold']
    expand = params['expand']
    warp_strength = params['warp_strength']
    warp_scale = params['warp_scale']
    max_shift = params['shift']
    alpha_jitter = params['alpha_jitter']

    # 1. Detection & Masking
    rgb = rgba[:, :, :3].astype(np.uint8)
    alpha = rgba[:, :, 3].astype(np.float32) / 255.0
    
    # Near black: max(R,G,B) <= threshold
    max_channel = rgb.max(axis=2)
    core_mask_uint8 = ((max_channel <= black_threshold) & (alpha > 0.0)).astype(np.uint8) * 255

    if use_gpu:
        try:
            # Upload to GPU
            gpu_core = cv2.cuda_GpuMat()
            gpu_core.upload(core_mask_uint8)
            
            # Expansion (Dilation)
            if expand > 0:
                kernel_size = expand * 2 + 1
                kernel = get_gpu_kernel(kernel_size)
                filter_dilate = cv2.cuda.createMorphologyFilter(cv2.MORPH_DILATE, cv2.CV_8UC1, kernel)
                gpu_expanded = filter_dilate.apply(gpu_core)
            else:
                gpu_expanded = gpu_core

            # Soften (Blur)
            gpu_expanded_f = cv2.cuda_GpuMat()
            gpu_expanded.convertTo(cv2.CV_32FC1, 1.0/255.0, gpu_expanded_f)
            filter_blur = cv2.cuda.createGaussianFilter(cv2.CV_32FC1, cv2.CV_32FC1, (3, 3), 0)
            gpu_expanded_f = filter_blur.apply(gpu_expanded_f)
            
            # Noise Warp
            # Noise generation is still CPU-based (NumPy), but remap is GPU
            small_h = max(2, int(round(h / warp_scale)))
            small_w = max(2, int(round(w / warp_scale)))
            noise_x = np.random.uniform(-1.0, 1.0, (small_h, small_w)).astype(np.float32)
            noise_y = np.random.uniform(-1.0, 1.0, (small_h, small_w)).astype(np.float32)
            noise_x = cv2.resize(noise_x, (w, h), interpolation=cv2.INTER_CUBIC)
            noise_y = cv2.resize(noise_y, (w, h), interpolation=cv2.INTER_CUBIC)
            
            grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (grid_x + noise_x * warp_strength).astype(np.float32)
            map_y = (grid_y + noise_y * warp_strength).astype(np.float32)
            
            gpu_map_x = cv2.cuda_GpuMat()
            gpu_map_y = cv2.cuda_GpuMat()
            gpu_map_x.upload(map_x)
            gpu_map_y.upload(map_y)
            
            gpu_warped = cv2.cuda.remap(gpu_expanded_f, gpu_map_x, gpu_map_y, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
            
            # Shift (Affine)
            if max_shift > 0:
                dx = random.randint(-max_shift, max_shift)
                dy = random.randint(-max_shift, max_shift)
                matrix = np.float32([[1, 0, dx], [0, 1, dy]])
                gpu_warped = cv2.cuda.warpAffine(gpu_warped, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            
            # Download results
            expanded_f = gpu_warped.download()
            gpu_core.convertTo(cv2.CV_32FC1, 1.0/255.0, gpu_core)
            core_f = gpu_core.download()
            
        except Exception as e:
            logger.warning(f"CUDA execution failed, falling back to CPU: {e}")
            use_gpu = False

    if not use_gpu:
        # CPU Path
        if expand > 0:
            kernel_size = expand * 2 + 1
            kernel = get_gpu_kernel(kernel_size)
            expanded = cv2.dilate(core_mask_uint8, kernel, iterations=1)
        else:
            expanded = core_mask_uint8.copy()
        
        expanded_f = expanded.astype(np.float32) / 255.0
        expanded_f = cv2.GaussianBlur(expanded_f, (3, 3), 0.0)
        core_f = core_mask_uint8.astype(np.float32) / 255.0
        
        # Noise Warp
        small_h = max(2, int(round(h / warp_scale)))
        small_w = max(2, int(round(w / warp_scale)))
        noise_x = np.random.uniform(-1.0, 1.0, (small_h, small_w)).astype(np.float32)
        noise_y = np.random.uniform(-1.0, 1.0, (small_h, small_w)).astype(np.float32)
        noise_x = cv2.resize(noise_x, (w, h), interpolation=cv2.INTER_CUBIC)
        noise_y = cv2.resize(noise_y, (w, h), interpolation=cv2.INTER_CUBIC)
        
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + noise_x * warp_strength).astype(np.float32)
        map_y = (grid_y + noise_y * warp_strength).astype(np.float32)
        
        expanded_f = cv2.remap(expanded_f, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Shift
        if max_shift > 0:
            dx = random.randint(-max_shift, max_shift)
            dy = random.randint(-max_shift, max_shift)
            matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            expanded_f = cv2.warpAffine(expanded_f, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # 2. Final Compositing & Alpha Masking
    # Combine masks and apply jitter
    m = expanded_f.copy()
    if alpha_jitter > 0:
        m *= 1.0 + random.uniform(-alpha_jitter, alpha_jitter)
    
    # Anchor the boiled mask with the core mask to prevent lines from disappearing
    # core_f is strictly near-black. m is the warped/expanded version.
    # We use a weighted maximum to ensure visibility.
    m = np.maximum(m, core_f * 0.7)
    boiled_mask = np.clip(m * alpha, 0.0, 1.0) # Apply original alpha to mask
    
    # Composite: replace existing line pixels with black based on the boiled mask.
    # We target the 'boiled' region but only where it makes sense (near lines).
    # result = original * (1 - boil_alpha) + black * boil_alpha
    out = rgba.copy()
    rgb_f = out[:, :, :3].astype(np.float32)
    
    # We only want to affect the luminance where the mask is active.
    # The effect_alpha should be strongest where the line is supposed to be.
    # We use boiled_mask as the blending factor.
    effect_alpha = boiled_mask[:, :, None]
    
    # Line target is black (0,0,0)
    result = rgb_f * (1.0 - effect_alpha)
    
    out[:, :, :3] = np.clip(result, 0, 255).astype(np.uint8)
    return out

def worker_init():
    set_cv2_threads(1)

def process_single_frame_task(frame_data, seed, params, use_gpu):
    return process_frame_core(frame_data, seed, params, use_gpu)
