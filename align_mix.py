#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


def decode_mono(path, sample_rate, seconds):
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-t",
        f"{seconds:.6f}",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE)
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError(f"ffmpeg decoded no audio from {path}")
    return audio


def moving_average(values, window):
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="same")


def analysis_feature(audio, sample_rate, frame_ms):
    frame = max(1, int(round(sample_rate * frame_ms / 1000)))
    usable = (audio.size // frame) * frame
    if usable < frame * 4:
        raise RuntimeError("not enough decoded audio to estimate alignment")

    frames = audio[:usable].reshape(-1, frame)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    loudness = np.log1p(rms * 100.0)

    smooth_window = max(3, int(round(250 / frame_ms)))
    local_average = moving_average(loudness, smooth_window)
    high_pass = loudness - local_average

    onset = np.diff(loudness, prepend=loudness[0])
    onset = np.maximum(onset, 0)

    feature = high_pass + (2.0 * onset)
    feature = feature - np.mean(feature)
    norm = np.linalg.norm(feature)
    if norm < 1e-8:
        raise RuntimeError("audio is too quiet or flat to estimate alignment")
    return feature / norm


def estimate_offset(ref_file, overlay_file, args):
    sample_rate = args.sample_rate
    seconds = args.analyze_seconds + args.max_offset

    ref_audio = decode_mono(ref_file, sample_rate, seconds)
    overlay_audio = decode_mono(overlay_file, sample_rate, seconds)

    ref_feature = analysis_feature(ref_audio, sample_rate, args.frame_ms)
    overlay_feature = analysis_feature(overlay_audio, sample_rate, args.frame_ms)

    corr = np.correlate(ref_feature, overlay_feature, mode="full")
    lags = np.arange(-overlay_feature.size + 1, ref_feature.size)

    max_lag_frames = int(round(args.max_offset * 1000 / args.frame_ms))
    valid = np.abs(lags) <= max_lag_frames
    if not np.any(valid):
        raise RuntimeError("max offset is too small for the analysis settings")

    valid_corr = corr[valid]
    valid_lags = lags[valid]
    best = int(np.argmax(valid_corr))
    best_lag = int(valid_lags[best])
    score = float(valid_corr[best])

    # For np.correlate(ref, overlay), a negative lag means overlay starts later.
    offset_seconds = -best_lag * args.frame_ms / 1000.0
    return offset_seconds, score


def build_filter(offset_seconds, args):
    first = f"[0:a]volume={args.volume_ref:g}[a0]"
    if offset_seconds > 0:
        second = (
            f"[1:a]atrim=start={offset_seconds:.6f},"
            f"asetpts=PTS-STARTPTS,volume={args.volume_overlay:g}[a1]"
        )
    else:
        delay_ms = int(round(-offset_seconds * 1000))
        second = f"[1:a]adelay={delay_ms}:all=1,volume={args.volume_overlay:g}[a1]"

    mix = "[a0][a1]amix=inputs=2:duration=longest:normalize=0[out]"
    return f"{first};{second};{mix}"


def run_mix(ref_file, overlay_file, output_file, offset_seconds, args):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    filter_complex = build_filter(offset_seconds, args)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(ref_file),
        "-i",
        str(overlay_file),
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        "-c:a",
        "libmp3lame",
        "-q:a",
        str(args.quality),
        str(output_file),
    ]
    if args.dry_run:
        print(" ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto-align two MP3s from their opening notes, then overlay/mix them."
    )
    parser.add_argument("reference", help="MP3 that defines the output timeline")
    parser.add_argument("overlay", help="MP3 to delay or trim until it matches the reference")
    parser.add_argument("output", help="Output MP3 path")
    parser.add_argument(
        "--analyze-seconds",
        type=float,
        default=20.0,
        help="seconds of opening audio to compare, before max-offset padding",
    )
    parser.add_argument(
        "--max-offset",
        type=float,
        default=8.0,
        help="largest expected offset in seconds",
    )
    parser.add_argument(
        "--frame-ms",
        type=float,
        default=10.0,
        help="alignment resolution in milliseconds",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="temporary analysis sample rate",
    )
    parser.add_argument("--volume-ref", type=float, default=0.85)
    parser.add_argument("--volume-overlay", type=float, default=0.85)
    parser.add_argument(
        "--quality",
        type=int,
        default=2,
        help="MP3 VBR quality for libmp3lame; lower is better",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=None,
        help="skip detection and use this offset in seconds; positive means overlay is late",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ref_file = Path(args.reference)
    overlay_file = Path(args.overlay)
    output_file = Path(args.output)

    if not ref_file.exists():
        raise FileNotFoundError(ref_file)
    if not overlay_file.exists():
        raise FileNotFoundError(overlay_file)

    if args.offset is None:
        offset_seconds, score = estimate_offset(ref_file, overlay_file, args)
        print(f"Estimated offset: {offset_seconds:+.3f}s")
        print("Positive means the overlay starts later than the reference.")
        print(f"Alignment score: {score:.3f}")
    else:
        offset_seconds = args.offset
        print(f"Using supplied offset: {offset_seconds:+.3f}s")

    run_mix(ref_file, overlay_file, output_file, offset_seconds, args)
    if not args.dry_run:
        print(f"Wrote {output_file}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode)
