import torch
import numpy as np
import cv2
import os
import logging
from concurrent.futures import ProcessPoolExecutor
from .core import process_single_frame_task, worker_init, is_cuda_available, set_cv2_threads
import comfy.utils

logger = logging.getLogger("ComfyUI-LineBoil")

class LineBoilImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": True}),
                "black_threshold": ("INT", {"default": 45, "min": 0, "max": 255}),
                "expand": ("INT", {"default": 1, "min": 0, "max": 10}),
                "warp_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.1}),
                "warp_scale": ("FLOAT", {"default": 28.0, "min": 1.0, "max": 100.0, "step": 0.5}),
                "shift": ("INT", {"default": 1, "min": 0, "max": 10}),
                "alpha_jitter": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 1.0, "step": 0.01}),
                "hold": ("INT", {"default": 2, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 0xffffffffffffffff}),
                "workers": ("INT", {"default": 4, "min": 1, "max": 64}),
                "use_gpu": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_boil"
    CATEGORY = "image/effects"

    def apply_boil(self, images, enabled, **params):
        if not enabled:
            return (images,)

        batch_size = images.shape[0]
        pbar = comfy.utils.ProgressBar(batch_size)
        
        core_params = {k: params[k] for k in ['black_threshold', 'expand', 'warp_strength', 'warp_scale', 'shift', 'alpha_jitter']}
        seed = params['seed']
        hold = params['hold']
        workers = params['workers']
        use_gpu = params['use_gpu'] and is_cuda_available()

        # --- Multiprocessing Strategy ---
        # 1. Disable MP if use_gpu is True (CUDA context issues in child processes)
        # 2. Disable MP if batch_size is small (overhead not worth it)
        # 3. Force workers=1 if workers param is 1
        effective_workers = workers
        if use_gpu or batch_size < 4 or workers <= 1:
            effective_workers = 1

        frames_np = (images.cpu().numpy() * 255).astype(np.uint8)
        processed_frames = [None] * batch_size
        
        if effective_workers > 1:
            # Multiprocessing Path
            with ProcessPoolExecutor(max_workers=effective_workers, initializer=worker_init) as executor:
                futures = []
                for i in range(batch_size):
                    frame = frames_np[i]
                    if frame.shape[2] == 3: # Ensure RGBA
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                    
                    frame_seed = seed + (i // hold)
                    futures.append(executor.submit(process_single_frame_task, frame, frame_seed, core_params, False)) # Force CPU in MP
                
                for i, future in enumerate(futures):
                    res = future.result()
                    res_rgb = cv2.cvtColor(res, cv2.COLOR_RGBA2RGB)
                    processed_frames[i] = res_rgb.astype(np.float32) / 255.0
                    pbar.update(1)
        else:
            # Single-threaded / GPU Path
            set_cv2_threads(1) # Minimize oversubscription
            for i in range(batch_size):
                frame = frames_np[i]
                if frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                
                frame_seed = seed + (i // hold)
                res = process_single_frame_task(frame, frame_seed, core_params, use_gpu)
                res_rgb = cv2.cvtColor(res, cv2.COLOR_RGBA2RGB)
                processed_frames[i] = res_rgb.astype(np.float32) / 255.0
                pbar.update(1)

        out_tensor = torch.from_numpy(np.stack(processed_frames))
        return (out_tensor,)

class LineBoilVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
                "suffix": ("STRING", {"default": "_boiled"}),
                "enabled": ("BOOLEAN", {"default": True}),
                "black_threshold": ("INT", {"default": 45, "min": 0, "max": 255}),
                "expand": ("INT", {"default": 1, "min": 0, "max": 10}),
                "warp_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.1}),
                "warp_scale": ("FLOAT", {"default": 28.0, "min": 1.0, "max": 100.0, "step": 0.5}),
                "shift": ("INT", {"default": 1, "min": 0, "max": 10}),
                "alpha_jitter": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 1.0, "step": 0.01}),
                "hold": ("INT", {"default": 2, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 0xffffffffffffffff}),
                "workers": ("INT", {"default": 4, "min": 1, "max": 64}),
                "use_gpu": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "apply_boil_video"
    CATEGORY = "image/effects"

    def apply_boil_video(self, video_path, suffix, enabled, **params):
        if not enabled:
            return (video_path,)

        if not video_path or not os.path.exists(video_path):
            logger.error(f"Video path does not exist: {video_path}")
            return (video_path,)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return (video_path,)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Safer codec strategy: Prefer MP4V or H264 if available
        # Avoid blindly reusing input fourcc which might be incompatible for writing
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

        base, ext = os.path.splitext(video_path)
        out_path = f"{base}{suffix}.mp4" # Force .mp4 for mp4v codec consistency
        
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            logger.error(f"Failed to create VideoWriter for path: {out_path}")
            cap.release()
            return (video_path,)
        
        pbar = comfy.utils.ProgressBar(total_frames)
        
        core_params = {k: params[k] for k in ['black_threshold', 'expand', 'warp_strength', 'warp_scale', 'shift', 'alpha_jitter']}
        seed = params['seed']
        hold = params['hold']
        workers = params['workers']
        use_gpu = params['use_gpu'] and is_cuda_available()

        # Video always processes in chunks for memory safety
        effective_workers = 1 if use_gpu else workers
        chunk_size = max(effective_workers * 2, 8)
        
        try:
            if effective_workers > 1:
                with ProcessPoolExecutor(max_workers=effective_workers, initializer=worker_init) as executor:
                    frame_idx = 0
                    while True:
                        frames_chunk = []
                        seeds_chunk = []
                        for _ in range(chunk_size):
                            ret, frame = cap.read()
                            if not ret: break
                            frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                            frames_chunk.append(frame_rgba)
                            seeds_chunk.append(seed + (frame_idx // hold))
                            frame_idx += 1
                        
                        if not frames_chunk: break
                        
                        futures = [executor.submit(process_single_frame_task, f, s, core_params, False) for f, s in zip(frames_chunk, seeds_chunk)]
                        for future in futures:
                            res = future.result()
                            res_bgr = cv2.cvtColor(res, cv2.COLOR_RGBA2BGR)
                            writer.write(res_bgr)
                            pbar.update(1)
            else:
                set_cv2_threads(1)
                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                    res = process_single_frame_task(frame_rgba, seed + (frame_idx // hold), core_params, use_gpu)
                    res_bgr = cv2.cvtColor(res, cv2.COLOR_RGBA2BGR)
                    writer.write(res_bgr)
                    pbar.update(1)
                    frame_idx += 1
        finally:
            cap.release()
            writer.release()
        
        return (out_path,)

NODE_CLASS_MAPPINGS = {
    "LineBoilImage": LineBoilImage,
    "LineBoilVideo": LineBoilVideo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LineBoilImage": "Line Boil (Image)",
    "LineBoilVideo": "Line Boil (Video)"
}
