"""Ken Burns (slow pan + zoom) and optional page-flip transitions between stills."""

from __future__ import annotations

import random
from typing import Any

from moviepy import CompositeVideoClip, ImageClip, vfx
from moviepy.video.compositing.CompositeVideoClip import concatenate_videoclips


def still_cover_clip(
    image_path: str,
    duration: float,
    out_w: int,
    out_h: int,
    fps: int = 30,
) -> Any:
    """Single still scaled to cover out_w x out_h (same geometry as pipeline _fit_image_clip)."""
    target_ratio = out_w / out_h
    clip = ImageClip(image_path).with_duration(duration).with_fps(fps)
    r = clip.w / clip.h if clip.h else target_ratio
    if r < target_ratio:
        clip = clip.cropped(
            width=clip.w,
            height=max(1, round(clip.w / target_ratio)),
            x_center=clip.w / 2,
            y_center=clip.h / 2,
        )
    else:
        clip = clip.cropped(
            width=max(1, round(clip.h * target_ratio)),
            height=clip.h,
            x_center=clip.w / 2,
            y_center=clip.h / 2,
        )
    return clip.resized((out_w, out_h))


def ken_burns_pan_zoom_clip(
    image_path: str,
    duration: float,
    out_w: int,
    out_h: int,
    fps: int = 30,
    *,
    zoom_min: float = 1.05,
    zoom_max: float = 1.10,
) -> Any:
    """
    Slow pan to the right + zoom from zoom_min to zoom_max over the segment.
    Implemented as an oversized resized image inside a fixed output-sized composite.
    """
    z0 = float(zoom_min)
    z1 = float(zoom_max)
    if z1 < z0:
        z0, z1 = z1, z0
    z0 = max(1.0, z0)
    z1 = max(z0, z1)

    base = ImageClip(image_path).with_duration(duration).with_fps(fps)
    iw, ih = base.size
    tw, th = out_w * z1, out_h * z1
    ar_img = iw / ih if ih else out_w / out_h
    ar_out = out_w / out_h
    if ar_img > ar_out:
        nh = int(max(th, 1))
        nw = int(iw * nh / ih)
    else:
        nw = int(max(tw, 1))
        nh = int(ih * nw / iw)

    clip = base.resized(new_size=(nw, nh))

    def zoom_factor(t: float) -> float:
        if duration <= 1e-6:
            return z0
        u = min(1.0, max(0.0, t / duration))
        return z0 + (z1 - z0) * u

    clip = clip.with_effects([vfx.Resize(zoom_factor)])

    def pos_t(t: float) -> tuple[float, Any]:
        u = min(1.0, max(0.0, t / duration)) if duration > 1e-6 else 0.0
        z = zoom_factor(t)
        w = nw * z
        h = nh * z
        x = -u * max(0.0, w - out_w)
        y = -(h - out_h) / 2
        return (x, y)

    comp = CompositeVideoClip([clip.with_position(pos_t)], size=(out_w, out_h))
    comp = comp.with_duration(duration).with_fps(fps)
    return comp


def page_flip_transition_clip(
    prev_image_path: str,
    next_image_path: str,
    duration: float,
    out_w: int,
    out_h: int,
    fps: int = 30,
) -> Any:
    """2D book-style cut: previous slides out left, next slides in from right."""
    d = max(0.05, float(duration))
    a = still_cover_clip(prev_image_path, d, out_w, out_h, fps)
    b = still_cover_clip(next_image_path, d, out_w, out_h, fps)
    a = a.with_effects([vfx.SlideOut(d, "left")])
    b = b.with_effects([vfx.SlideIn(d, "right")]).with_layer_index(1)
    comp = CompositeVideoClip([a, b], size=(out_w, out_h))
    return comp.with_duration(d).with_fps(fps)


def plan_transition_durations(
    aligned_segment_durations: list[float],
    mode: str,
    probability: float,
    nominal_flip_seconds: float,
    random_seed: int | None,
    min_flip_seconds: float = 0.12,
    min_segment_remain: float = 0.08,
) -> tuple[list[float], list[float]]:
    """
    Shorten adjacent segments to make room for page-flip clips without changing total time.

    Returns:
        adjusted_segment_durations, transition_between (length n-1; 0 = no transition).
    """
    n = len(aligned_segment_durations)
    if n == 0:
        return [], []
    adj = [float(x) for x in aligned_segment_durations]
    trans: list[float] = [0.0] * (n - 1)
    mode_l = (mode or "none").strip().lower()
    if mode_l in ("", "none", "off", "false"):
        return adj, trans

    rng = random.Random(random_seed) if random_seed is not None else random.Random()

    for k in range(n - 1):
        want = False
        if mode_l in ("page_flip", "pageflip", "flip"):
            want = True
        elif mode_l in ("random_page_flip", "random", "random_flip"):
            want = rng.random() < float(probability)
        else:
            continue

        cap = 2.0 * min(adj[k], adj[k + 1]) * 0.85 - min_segment_remain
        d_eff = min(float(nominal_flip_seconds), cap)
        if d_eff < min_flip_seconds:
            continue
        half = d_eff / 2.0
        if adj[k] - half < min_segment_remain or adj[k + 1] - half < min_segment_remain:
            continue
        adj[k] -= half
        adj[k + 1] -= half
        trans[k] = d_eff

    return adj, trans


def build_visual_timeline_clips(
    image_paths: list[str],
    aligned_segment_durations: list[float],
    out_w: int,
    out_h: int,
    *,
    fps: int = 30,
    ken_burns: bool = True,
    zoom_min: float = 1.05,
    zoom_max: float = 1.10,
    transition_mode: str = "none",
    page_flip_probability: float = 0.35,
    page_flip_duration_seconds: float = 0.38,
    transition_random_seed: int | None = None,
) -> Any:
    """
    Concatenate per-segment visual clips and optional page-flip transitions.
    Sum(result durations) should match sum(aligned_segment_durations).
    """
    n = len(image_paths)
    if n != len(aligned_segment_durations):
        raise ValueError("image_paths and aligned_segment_durations length mismatch")

    seg_durs, trans_durs = plan_transition_durations(
        aligned_segment_durations,
        transition_mode,
        page_flip_probability,
        page_flip_duration_seconds,
        transition_random_seed,
    )

    pieces: list[Any] = []
    for i in range(n):
        dur = seg_durs[i]
        if ken_burns:
            pieces.append(
                ken_burns_pan_zoom_clip(
                    image_paths[i],
                    dur,
                    out_w,
                    out_h,
                    fps,
                    zoom_min=zoom_min,
                    zoom_max=zoom_max,
                )
            )
        else:
            pieces.append(still_cover_clip(image_paths[i], dur, out_w, out_h, fps))

        if i < n - 1 and trans_durs[i] > 0:
            pieces.append(
                page_flip_transition_clip(
                    image_paths[i],
                    image_paths[i + 1],
                    trans_durs[i],
                    out_w,
                    out_h,
                    fps,
                )
            )

    if not pieces:
        raise ValueError("build_visual_timeline_clips: empty pieces")
    return concatenate_videoclips(pieces, method="chain").with_fps(fps)
