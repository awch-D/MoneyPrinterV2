"""Ken Burns (slow pan + zoom) and optional page-flip transitions between stills."""

from __future__ import annotations

import math
import os
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
    pan_extent: float = 0.35,
    pan_max_width_ratio: float = 0.05,
) -> Any:
    """
    Slow pan to the right + zoom from zoom_min to zoom_max over the segment.
    Implemented as an oversized resized image inside a fixed output-sized composite.
    pan_extent: fraction in [0, 1] of the maximum horizontal overscan used for pan
    (1.0 = full edge-to-edge pan; smaller = subtler motion).
    pan_max_width_ratio: if > 0, pan distance at u=1 is capped at ``out_w * ratio`` pixels
    (stops huge overscan segments from still drifting across the frame).
    """
    z0 = float(zoom_min)
    z1 = float(zoom_max)
    pe = min(1.0, max(0.0, float(pan_extent)))
    pmr = max(0.0, min(0.35, float(pan_max_width_ratio)))
    _pan_cap_px = float(out_w) * pmr if pmr > 0.0 else 0.0
    _use_pan_cap = _pan_cap_px > 0.0

    def _pan_move_x_from_wf(w_f: float) -> float:
        """Use float scaled width so pan does not jump when int(bw*z) steps frame-to-frame."""
        raw = max(0.0, w_f - float(out_w)) * pe
        return min(raw, _pan_cap_px) if _use_pan_cap else raw

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
    bw, bh = int(clip.w), int(clip.h)

    # vfx.Resize scales [bw*z, bh*z] then maps to int pixels; composite uses int positions.
    # Pan uses float overscan; smoothstep(u) zeros endpoint velocity (less micro-jitter than
    # linear u × move_x). move_x is always from current w_f so we never exceed overscan.
    margin = 8
    safety = 0
    while (int(bw * z0) < out_w + margin or int(bh * z0) < out_h + margin) and safety < 40:
        safety += 1
        s_need = max(
            (out_w + margin) / max(bw * z0, 1e-9),
            (out_h + margin) / max(bh * z0, 1e-9),
            1.001,
        )
        nw = max(1, int(math.ceil(bw * s_need)))
        nh = max(1, int(math.ceil(bh * s_need)))
        clip = base.resized(new_size=(nw, nh))
        bw, bh = int(clip.w), int(clip.h)

    def zoom_factor(t: float) -> float:
        if duration <= 1e-6:
            return z0
        u = min(1.0, max(0.0, t / duration))
        return z0 + (z1 - z0) * u

    clip = clip.with_effects([vfx.Resize(zoom_factor)])

    def _snap_px(v: float) -> int:
        """Half-away-from-zero (avoids Python ``round`` ties flipping parity frame-to-frame)."""
        return int(math.floor(v + 0.5)) if v >= 0.0 else int(math.ceil(v - 0.5))

    def pos_t(t: float) -> tuple[int, int]:
        u = min(1.0, max(0.0, t / duration)) if duration > 1e-6 else 0.0
        z = zoom_factor(t)
        w_f = float(bw) * z
        h_f = float(bh) * z
        w_i = int(bw * z)
        h_i = int(bh * z)

        u_ease = u * u * (3.0 - 2.0 * u)
        move_x = _pan_move_x_from_wf(w_f)
        x_f = -u_ease * move_x
        y_f = (out_h - h_f) / 2.0
        xi = _snap_px(x_f)
        yi = _snap_px(y_f)
        if w_i >= out_w:
            x_lo = -(w_i - out_w)
            xi = max(x_lo, min(0, xi))
        else:
            xi = 0
        if h_i >= out_h:
            y_lo = out_h - h_i
            yi = max(y_lo, min(0, yi))
        else:
            yi = 0
        return (xi, yi)

    # #region agent log
    import json
    import time as _agent_time

    _agent_log_path = "/Users/arno/Documents/project/.cursor/debug-5d2a66.log"

    def _agent_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
        with open(_agent_log_path, "a", encoding="utf-8") as _af:
            _af.write(
                json.dumps(
                    {
                        "sessionId": "5d2a66",
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "message": message,
                        "data": data,
                        "timestamp": int(_agent_time.time() * 1000),
                        "runId": "pre-fix",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    _img_tag = os.path.basename(image_path)
    for _label, _t in (
        ("t0", 0.0),
        ("t_mid", duration * 0.5),
        ("t_end", max(0.0, duration - 1e-4)),
    ):
        xi_s, yi_s = pos_t(_t)
        u_s = min(1.0, max(0.0, _t / duration)) if duration > 1e-6 else 0.0
        z_s = zoom_factor(_t)
        w_f_s = float(bw) * z_s
        h_f_s = float(bh) * z_s
        w_s = int(bw * z_s)
        h_s = int(bh * z_s)
        tmx_s = max(0, w_s - out_w)
        u_ease_s = u_s * u_s * (3.0 - 2.0 * u_s)
        move_x_s = _pan_move_x_from_wf(w_f_s)
        xf_s = -u_ease_s * move_x_s
        yf_s = (out_h - h_f_s) / 2.0
        right_s = xi_s + w_s
        bottom_s = yi_s + h_s
        _agent_log(
            "H1-H5",
            "video_motion.ken_burns_pan_zoom_clip:pos_sample",
            _label,
            {
                "image": _img_tag,
                "duration": round(duration, 4),
                "bw": bw,
                "bh": bh,
                "z0": z0,
                "z1": z1,
                "sample_t": _t,
                "u": u_s,
                "z": z_s,
                "w": w_s,
                "h": h_s,
                "out_w": out_w,
                "out_h": out_h,
                "total_move_x": tmx_s,
                "pan_extent": pe,
                "pan_max_width_ratio": pmr,
                "pan_cap_px": _pan_cap_px if _use_pan_cap else None,
                "pan_move_x": move_x_s,
                "x_f": xf_s,
                "y_f": yf_s,
                "x_i": xi_s,
                "y_i": yi_s,
                "right_edge": right_s,
                "bottom_edge": bottom_s,
                "gap_right_px": out_w - right_s,
                "gap_left_px": xi_s,
                "gap_bottom_px": out_h - bottom_s,
                "gap_top_px": yi_s,
                "H1_w_lt_out_w": w_s < out_w,
                "H2_x_positive_gap_left": xi_s > 0,
                "H3_right_lt_out_w_black_right": right_s < out_w,
                "H4_vertical_gap": yi_s > 0 or bottom_s < out_h,
                "H5_negative_move_x_positive_xf": tmx_s < 0 and xf_s > 1e-6,
            },
        )
    # #endregion

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
    pan_extent: float = 0.35,
    pan_max_width_ratio: float = 0.05,
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
                    pan_extent=pan_extent,
                    pan_max_width_ratio=pan_max_width_ratio,
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
