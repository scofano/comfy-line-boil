import torch
import numpy as np
import cv2
import os
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from .core import process_single_frame_task, is_cuda_available, set_cv2_threads, can_use_multiprocessing, is_windows, worker_init
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

        # --- Concurrency Strategy ---
        # 1. On Windows, we avoid ProcessPoolExecutor entirely due to the 'spawn' re-import issue.
        #    ThreadPoolExecutor is used instead because OpenCV releases the GIL for heavy ops.
        # 2. On other platforms (Linux/macOS), ProcessPoolExecutor is used if GPU is disabled.
        # 3. If GPU is enabled, we stay in-process (ThreadPool) to avoid CUDA context sharing issues.
        
        effective_workers = workers
        if batch_size < 2 or workers <= 1:
            effective_workers = 1

        frames_np = (images.cpu().numpy() * 255).astype(np.uint8)
        processed_frames = [None] * batch_size
        
        set_cv2_threads(1)

        if effective_workers > 1:
            use_processes = can_use_multiprocessing() and not use_gpu
            ExecutorClass = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
            
            try:
                init_args = {"initializer": worker_init} if use_processes else {}
                with ExecutorClass(max_workers=effective_workers, **init_args) as executor:
                    futures = []
                    for i in range(batch_size):
                        frame = frames_np[i]
                        if frame.shape[2] == 3:
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                        
                        frame_seed = int(seed + (i // hold))
                        # Pass simple types only to ensure picklability
                        futures.append(executor.submit(process_single_frame_task, frame, frame_seed, core_params, use_gpu if not use_processes else False))
                    
                    for i, future in enumerate(futures):
                        res = future.result()
                        res_rgb = cv2.cvtColor(res, cv2.COLOR_RGBA2RGB)
                        processed_frames[i] = res_rgb.astype(np.float32) / 255.0
                        pbar.update(1)
            except Exception as e:
                logger.error(f"Concurrency failed ({type(e).__name__}): {e}. Falling back to sequential.")
                # Fallback: Process missing frames sequentially
                for i in range(batch_size):
                    if processed_frames[i] is None:
                        frame = frames_np[i]
                        if frame.shape[2] == 3: frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                        res = process_single_frame_task(frame, int(seed + (i // hold)), core_params, use_gpu)
                        res_rgb = cv2.cvtColor(res, cv2.COLOR_RGBA2RGB)
                        processed_frames[i] = res_rgb.astype(np.float32) / 255.0
                        pbar.update(1)
        else:
            # Sequential Path
            for i in range(batch_size):
                frame = frames_np[i]
                if frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                
                frame_seed = int(seed + (i // hold))
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
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        base, ext = os.path.splitext(video_path)
        out_path = f"{base}{suffix}.mp4"
        
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

        effective_workers = workers
        chunk_size = max(effective_workers * 4, 16)
        set_cv2_threads(1)
        
        try:
            if effective_workers > 1:
                use_processes = can_use_multiprocessing() and not use_gpu
                ExecutorClass = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
                init_args = {"initializer": worker_init} if use_processes else {}
                
                frame_idx = 0
                pool_failed = False
                
                with ExecutorClass(max_workers=effective_workers, **init_args) as executor:
                    while True:
                        frames_chunk = []
                        seeds_chunk = []
                        for _ in range(chunk_size):
                            ret, frame = cap.read()
                            if not ret: break
                            frames_chunk.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
                            seeds_chunk.append(int(seed + (frame_idx // hold)))
                            frame_idx += 1
                        
                        if not frames_chunk: break
                        
                        if not pool_failed:
                            try:
                                futures = [executor.submit(process_single_frame_task, f, s, core_params, use_gpu if not use_processes else False) for f, s in zip(frames_chunk, seeds_chunk)]
                                for future in futures:
                                    res = future.result()
                                    writer.write(cv2.cvtColor(res, cv2.COLOR_RGBA2BGR))
                                    pbar.update(1)
                            except Exception as e:
                                logger.error(f"Video pool failed ({type(e).__name__}): {e}. Switching to sequential.")
                                pool_failed = True
                                # Process missed chunk sequentially
                                for f, s in zip(frames_chunk, seeds_chunk):
                                    res = process_single_frame_task(f, s, core_params, use_gpu)
                                    writer.write(cv2.cvtColor(res, cv2.COLOR_RGBA2BGR))
                                    pbar.update(1)
                        else:
                            # Pool already failed, process chunk sequentially
                            for f, s in zip(frames_chunk, seeds_chunk):
                                res = process_single_frame_task(f, s, core_params, use_gpu)
                                writer.write(cv2.cvtColor(res, cv2.COLOR_RGBA2BGR))
                                pbar.update(1)
            else:
                # Standard sequential path
                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                    res = process_single_frame_task(frame_rgba, int(seed + (frame_idx // hold)), core_params, use_gpu)
                    writer.write(cv2.cvtColor(res, cv2.COLOR_RGBA2BGR))
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
