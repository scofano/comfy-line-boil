import cv2
import numpy as np
import random
import logging
import threading
import sys
import os
from collections import OrderedDict

logger = logging.getLogger("ComfyUI-LineBoil")

# --- Global Resource Caches ---
_CUDA_AVAILABLE = False
_CUDA_FAILED = False
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        _CUDA_AVAILABLE = True
except Exception:
    _CUDA_AVAILABLE = False

# Bounded caches to prevent memory growth
_KERNEL_CACHE = {}
_MAP_CACHE = OrderedDict()
_MAX_CACHE_SIZE = 32
_CACHE_LOCK = threading.Lock()

def is_windows():
    return sys.platform.startswith("win") or os.name == "nt"

def can_use_multiprocessing():
    """
    Multiprocessing (ProcessPoolExecutor) with the 'spawn' method (default on Windows)
    is notoriously fragile inside hosted environments like ComfyUI because it
    attempts to re-import the entire main process state, often leading to
    ModuleNotFoundErrors or BrokenProcessPool.
    
    We disable it by default on Windows to prioritize stability.
    """
    if is_windows():
        return False
    # On Linux/macOS, fork is generally safer but still use with caution in ComfyUI
    return True

def is_cuda_available():
    global _CUDA_AVAILABLE, _CUDA_FAILED
    return _CUDA_AVAILABLE and not _CUDA_FAILED

def set_cv2_threads(n=1):
    try:
        cv2.setNumThreads(n)
    except Exception:
        pass

def get_kernel(size):
    with _CACHE_LOCK:
        if size in _KERNEL_CACHE:
            return _KERNEL_CACHE[size]
        k = np.ones((size, size), np.uint8)
        _KERNEL_CACHE[size] = k
        return k

def get_remap_maps(w, h, warp_scale, warp_strength, seed, use_gpu=False):
    """
    Returns (map_x, map_y). Uses cache if available.
    For GPU, returns cv2.cuda_GpuMat if possible, otherwise CPU arrays.
    """
    key = (w, h, warp_scale, warp_strength, seed, use_gpu)
    
    with _CACHE_LOCK:
        if key in _MAP_CACHE:
            # Move to end (LRU)
            val = _MAP_CACHE.pop(key)
            _MAP_CACHE[key] = val
            return val

    # Generate new maps (outside the lock for better parallelism if multiple threads need new maps)
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    
    small_h = max(2, int(round(h / warp_scale)))
    small_w = max(2, int(round(w / warp_scale)))
    noise_x = np.random.uniform(-1.0, 1.0, (small_h, small_w)).astype(np.float32)
    noise_y = np.random.uniform(-1.0, 1.0, (small_h, small_w)).astype(np.float32)
    noise_x = cv2.resize(noise_x, (w, h), interpolation=cv2.INTER_CUBIC)
    noise_y = cv2.resize(noise_y, (w, h), interpolation=cv2.INTER_CUBIC)
    
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x + noise_x * warp_strength
    map_y = grid_y + noise_y * warp_strength

    if use_gpu:
        g_map_x = cv2.cuda_GpuMat()
        g_map_y = cv2.cuda_GpuMat()
        g_map_x.upload(map_x)
        g_map_y.upload(map_y)
        res = (g_map_x, g_map_y)
    else:
        res = (map_x, map_y)

    with _CACHE_LOCK:
        _MAP_CACHE[key] = res
        if len(_MAP_CACHE) > _MAX_CACHE_SIZE:
            _MAP_CACHE.popitem(last=False)
    
    return res


def process_frame_core(rgba, seed, params, use_gpu=False):
    """
    Core logic for line boil effect. 
    Optimized for either CPU or GPU (OpenCV CUDA).
    """
    global _CUDA_FAILED
    
    # Validation: use_gpu only if available and not failed before
    use_gpu = use_gpu and _CUDA_AVAILABLE and not _CUDA_FAILED

    h, w = rgba.shape[:2]
    black_threshold = params['black_threshold']
    expand = params['expand']
    warp_strength = params['warp_strength']
    warp_scale = params['warp_scale']
    max_shift = params['shift']
    alpha_jitter = params['alpha_jitter']

    # Deterministic randomness for this frame seed
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)

    if use_gpu:
        try:
            # 1. Upload & Initial Masking on GPU
            gpu_rgba = cv2.cuda_GpuMat()
            gpu_rgba.upload(rgba)
            
            # Split channels
            gpu_channels = cv2.cuda.split(gpu_rgba)
            gpu_rgb_channels = gpu_channels[:3]
            gpu_alpha = gpu_channels[3]
            
            # Detect near-black: max(R,G,B) <= threshold
            # OpenCV CUDA doesn't have a direct "max across channels" like numpy
            # We can use cv2.cuda.max(cv2.cuda.max(R, G), B)
            tmp_max = cv2.cuda.max(gpu_rgb_channels[0], gpu_rgb_channels[1])
            gpu_max_rgb = cv2.cuda.max(tmp_max, gpu_rgb_channels[2])
            
            # Thresholding
            _, mask_thresh = cv2.cuda.threshold(gpu_max_rgb, black_threshold, 255, cv2.THRESH_BINARY_INV)
            
            # Alpha mask: only where alpha > 0
            _, mask_alpha = cv2.cuda.threshold(gpu_alpha, 0, 255, cv2.THRESH_BINARY)
            
            # Final core mask
            gpu_core_mask = cv2.cuda.bitwise_and(mask_thresh, mask_alpha)
            
            # 2. Expansion (Dilation)
            if expand > 0:
                kernel = get_kernel(expand * 2 + 1)
                filter_dilate = cv2.cuda.createMorphologyFilter(cv2.MORPH_DILATE, cv2.CV_8UC1, kernel)
                gpu_expanded = filter_dilate.apply(gpu_core_mask)
            else:
                gpu_expanded = gpu_core_mask

            # 3. Soften (Blur)
            gpu_expanded_f = cv2.cuda_GpuMat(h, w, cv2.CV_32FC1)
            gpu_expanded.convertTo(cv2.CV_32FC1, 1.0/255.0, gpu_expanded_f)
            filter_blur = cv2.cuda.createGaussianFilter(cv2.CV_32FC1, cv2.CV_32FC1, (3, 3), 0)
            gpu_expanded_f = filter_blur.apply(gpu_expanded_f)
            
            # 4. Noise Warp (using cached maps)
            g_map_x, g_map_y = get_remap_maps(w, h, warp_scale, warp_strength, seed, use_gpu=True)
            gpu_warped = cv2.cuda.remap(gpu_expanded_f, g_map_x, g_map_y, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
            
            # 5. Shift
            if max_shift > 0:
                dx = random.randint(-max_shift, max_shift)
                dy = random.randint(-max_shift, max_shift)
                matrix = np.float32([[1, 0, dx], [0, 1, dy]])
                gpu_warped = cv2.cuda.warpAffine(gpu_warped, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            
            # 6. Compositing on GPU
            # Jitter
            jitter_val = 1.0
            if alpha_jitter > 0:
                jitter_val = 1.0 + random.uniform(-alpha_jitter, alpha_jitter)
            
            if jitter_val != 1.0:
                gpu_warped.convertTo(cv2.CV_32FC1, jitter_val, gpu_warped)
            
            # Anchor with core mask
            gpu_core_f = cv2.cuda_GpuMat(h, w, cv2.CV_32FC1)
            gpu_core_mask.convertTo(cv2.CV_32FC1, 0.7/255.0, gpu_core_f)
            gpu_m = cv2.cuda.max(gpu_warped, gpu_core_f)
            
            # Apply original alpha
            gpu_alpha_f = cv2.cuda_GpuMat(h, w, cv2.CV_32FC1)
            gpu_alpha.convertTo(cv2.CV_32FC1, 1.0/255.0, gpu_alpha_f)
            gpu_boiled_mask = cv2.cuda.multiply(gpu_m, gpu_alpha_f)
            
            # Final Blend: out_rgb = in_rgb * (1 - boiled_mask)
            # Create (1 - boiled_mask)
            gpu_inv_mask = cv2.cuda_GpuMat(h, w, cv2.CV_32FC1)
            gpu_inv_mask.setTo(1.0)
            gpu_inv_mask = cv2.cuda.subtract(gpu_inv_mask, gpu_boiled_mask)
            
            # Multiply each RGB channel
            res_channels = []
            for i in range(3):
                c_f = cv2.cuda_GpuMat(h, w, cv2.CV_32FC1)
                gpu_rgb_channels[i].convertTo(cv2.CV_32FC1, 1.0, c_f)
                res_c = cv2.cuda.multiply(c_f, gpu_inv_mask)
                res_c_8u = cv2.cuda_GpuMat(h, w, cv2.CV_8UC1)
                res_c.convertTo(cv2.CV_8UC1, 1.0, res_c_8u)
                res_channels.append(res_c_8u)
            
            # Add back original alpha
            res_channels.append(gpu_alpha)
            
            gpu_final = cv2.cuda_GpuMat()
            cv2.cuda.merge(res_channels, gpu_final)
            
            return gpu_final.download()

        except Exception as e:
            logger.warning(f"CUDA execution failed, falling back to CPU: {e}")
            _CUDA_FAILED = True
            use_gpu = False

    # CPU Path
    # 1. Detection & Masking
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3].astype(np.float32) / 255.0
    
    max_channel = rgb.max(axis=2)
    core_mask_uint8 = ((max_channel <= black_threshold) & (alpha > 0.0)).astype(np.uint8) * 255

    if expand > 0:
        kernel = get_kernel(expand * 2 + 1)
        expanded = cv2.dilate(core_mask_uint8, kernel, iterations=1)
    else:
        expanded = core_mask_uint8.copy()
    
    expanded_f = expanded.astype(np.float32) / 255.0
    expanded_f = cv2.GaussianBlur(expanded_f, (3, 3), 0.0)
    core_f = core_mask_uint8.astype(np.float32) / 255.0
    
    # Noise Warp (using cached maps)
    map_x, map_y = get_remap_maps(w, h, warp_scale, warp_strength, seed, use_gpu=False)
    expanded_f = cv2.remap(expanded_f, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    # Shift
    if max_shift > 0:
        dx = random.randint(-max_shift, max_shift)
        dy = random.randint(-max_shift, max_shift)
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        expanded_f = cv2.warpAffine(expanded_f, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Compositing
    m = expanded_f
    if alpha_jitter > 0:
        m = m * (1.0 + random.uniform(-alpha_jitter, alpha_jitter))
    
    m = np.maximum(m, core_f * 0.7)
    boiled_mask = np.clip(m * alpha, 0.0, 1.0)
    
    out = rgba.copy()
    rgb_f = out[:, :, :3].astype(np.float32)
    effect_alpha = boiled_mask[:, :, None]
    
    result = rgb_f * (1.0 - effect_alpha)
    out[:, :, :3] = np.clip(result, 0, 255).astype(np.uint8)
    return out

def worker_init():
    set_cv2_threads(1)

def process_single_frame_task(frame_data, seed, params, use_gpu):
    return process_frame_core(frame_data, seed, params, use_gpu)

