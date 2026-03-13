#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import cv2
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a raster 'line boil' effect to a PNG sequence. "
            "Only pixels close to pure black (#000000) are targeted, so "
            "fills and darker painted areas should stay untouched."
        )
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing input PNG frames.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where processed PNG frames will be written.",
    )

    parser.add_argument(
        "--black-threshold",
        type=int,
        default=45,
        help=(
            "Maximum per-channel distance from #000000 to count as line art. "
            "Lower values affect only near-black pixels. Default: 45"
        ),
    )
    parser.add_argument(
        "--expand",
        type=int,
        default=1,
        help=(
            "Expand the detected line mask by this many pixels to catch "
            "antialiasing around black lines. Default: 1"
        ),
    )
    parser.add_argument(
        "--warp-strength",
        type=float,
        default=0.8,
        help="Subtle displacement amount in pixels. Default: 0.8",
    )
    parser.add_argument(
        "--warp-scale",
        type=float,
        default=28.0,
        help=(
            "Noise scale for displacement. Higher = larger, smoother wobble. "
            "Default: 28.0"
        ),
    )
    parser.add_argument(
        "--shift",
        type=int,
        default=1,
        help="Maximum random X/Y pixel shift for the line mask. Default: 1",
    )
    parser.add_argument(
        "--alpha-jitter",
        type=float,
        default=0.06,
        help=(
            "Small per-frame strength jitter for the effect. "
            "Suggested range: 0.0 to 0.12. Default: 0.06"
        ),
    )
    parser.add_argument(
        "--hold",
        type=int,
        default=2,
        help=(
            "Reuse the same random boil pattern for N consecutive frames. "
            "Default: 2"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base random seed for deterministic output. Default: 1234",
    )

    return parser.parse_args()


def load_rgba(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGBA"))


def save_rgba(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr, "RGBA").save(path)


def detect_near_black_mask(
    rgba: np.ndarray,
    black_threshold: int,
    expand: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect pixels near pure black (#000000), with optional expansion to capture
    antialiasing just around those lines.

    Returns:
        core_mask: float32 mask of strictly near-black pixels in [0, 1]
        expanded_mask: float32 softened mask including nearby AA pixels in [0, 1]
    """
    rgb = rgba[:, :, :3].astype(np.uint8)
    alpha = rgba[:, :, 3].astype(np.float32) / 255.0

    # "Near black" means all channels are small.
    # Using max(rgb) is stricter and better than grayscale for this use case.
    max_channel = rgb.max(axis=2)
    core = ((max_channel <= black_threshold) & (alpha > 0.0)).astype(np.uint8) * 255

    if expand > 0:
        kernel_size = expand * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        expanded = cv2.dilate(core, kernel, iterations=1)
    else:
        expanded = core.copy()

    # Soften slightly so the line boil carries some antialiasing smoothly.
    expanded_f = expanded.astype(np.float32) / 255.0
    expanded_f = cv2.GaussianBlur(expanded_f, (3, 3), 0.0)
    expanded_f *= alpha

    core_f = core.astype(np.float32) / 255.0
    core_f *= alpha

    return np.clip(core_f, 0.0, 1.0), np.clip(expanded_f, 0.0, 1.0)


def warp_mask_with_noise(mask: np.ndarray, strength: float, scale: float) -> np.ndarray:
    h, w = mask.shape

    small_h = max(2, int(round(h / scale)))
    small_w = max(2, int(round(w / scale)))

    noise_x = np.random.uniform(-1.0, 1.0, (small_h, small_w)).astype(np.float32)
    noise_y = np.random.uniform(-1.0, 1.0, (small_h, small_w)).astype(np.float32)

    noise_x = cv2.resize(noise_x, (w, h), interpolation=cv2.INTER_CUBIC)
    noise_y = cv2.resize(noise_y, (w, h), interpolation=cv2.INTER_CUBIC)

    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + noise_x * strength).astype(np.float32)
    map_y = (grid_y + noise_y * strength).astype(np.float32)

    warped = cv2.remap(
        mask.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return np.clip(warped, 0.0, 1.0)


def shift_mask(mask: np.ndarray, max_shift: int) -> np.ndarray:
    if max_shift <= 0:
        return mask

    dx = random.randint(-max_shift, max_shift)
    dy = random.randint(-max_shift, max_shift)

    h, w = mask.shape
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])

    shifted = cv2.warpAffine(
        mask.astype(np.float32),
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return np.clip(shifted, 0.0, 1.0)


def build_boiled_mask(
    base_mask: np.ndarray,
    core_mask: np.ndarray,
    warp_strength: float,
    warp_scale: float,
    max_shift: int,
    alpha_jitter: float,
) -> np.ndarray:
    """
    Create a slightly varied version of the line mask.

    base_mask:
        includes AA-expanded area around the black line
    core_mask:
        the truly near-black line; used as a stability floor
    """
    m = base_mask.copy()
    m = warp_mask_with_noise(m, strength=warp_strength, scale=warp_scale)
    m = shift_mask(m, max_shift=max_shift)

    if alpha_jitter > 0:
        m *= 1.0 + random.uniform(-alpha_jitter, alpha_jitter)

    # Keep a stable minimum from the actual near-black line so it does not vanish.
    m = np.maximum(m, core_mask * 0.65)

    return np.clip(m, 0.0, 1.0)


def composite_only_on_black_lines(
    original_rgba: np.ndarray,
    core_mask: np.ndarray,
    boiled_mask: np.ndarray,
) -> np.ndarray:
    """
    Apply the boil only where the image already contains near-black line art.

    Important:
    - We do NOT darken the whole image.
    - We only replace/mix the pixels in the detected near-black region and its
      immediate AA neighborhood.
    """
    out = original_rgba.copy()
    rgb = out[:, :, :3].astype(np.float32)

    # Effect region is only where the boiled mask exists at all.
    region = np.clip(boiled_mask, 0.0, 1.0)[:, :, None]

    # Original black-line presence: used to keep the effect tied to existing lines.
    core = np.clip(core_mask, 0.0, 1.0)[:, :, None]

    # Build a target that stays near black, but does not touch unrelated pixels.
    # Since the intended line art is near #000000, using black here is correct.
    line_target = np.zeros_like(rgb, dtype=np.float32)

    # The actual blend strength is limited to the detected region only.
    # Core mask contributes stronger anchoring; expanded region gives soft AA support.
    effect_alpha = np.clip(region * 0.85 + core * 0.35, 0.0, 1.0)

    # Blend only inside the line region.
    # Outside effect_alpha==0, pixels remain unchanged.
    result = rgb * (1.0 - effect_alpha) + line_target * effect_alpha

    out[:, :, :3] = np.clip(result, 0, 255).astype(np.uint8)
    return out


def process_frame(
    frame_path: Path,
    out_path: Path,
    black_threshold: int,
    expand: int,
    warp_strength: float,
    warp_scale: float,
    max_shift: int,
    alpha_jitter: float,
) -> None:
    rgba = load_rgba(frame_path)

    core_mask, expanded_mask = detect_near_black_mask(
        rgba=rgba,
        black_threshold=black_threshold,
        expand=expand,
    )

    boiled_mask = build_boiled_mask(
        base_mask=expanded_mask,
        core_mask=core_mask,
        warp_strength=warp_strength,
        warp_scale=warp_scale,
        max_shift=max_shift,
        alpha_jitter=alpha_jitter,
    )

    result = composite_only_on_black_lines(
        original_rgba=rgba,
        core_mask=core_mask,
        boiled_mask=boiled_mask,
    )

    save_rgba(result, out_path)


def main() -> int:
    args = parse_args()

    if not args.input_dir.exists():
        print(f"Error: input_dir does not exist: {args.input_dir}", file=sys.stderr)
        return 1

    if not args.input_dir.is_dir():
        print(f"Error: input_dir is not a directory: {args.input_dir}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(args.input_dir.glob("*.png"))
    if not frame_paths:
        print(f"Error: no PNG files found in {args.input_dir}", file=sys.stderr)
        return 1

    total = len(frame_paths)

    for frame_index, frame_path in enumerate(frame_paths):
        # Temporal hold: same boil seed reused for N frames.
        hold_index = frame_index // max(1, args.hold)
        frame_seed = args.seed + hold_index

        random.seed(frame_seed)
        np.random.seed(frame_seed)

        out_path = args.output_dir / frame_path.name

        process_frame(
            frame_path=frame_path,
            out_path=out_path,
            black_threshold=args.black_threshold,
            expand=args.expand,
            warp_strength=args.warp_strength,
            warp_scale=args.warp_scale,
            max_shift=args.shift,
            alpha_jitter=args.alpha_jitter,
        )

        print(f"[{frame_index + 1}/{total}] {frame_path.name} -> {out_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())