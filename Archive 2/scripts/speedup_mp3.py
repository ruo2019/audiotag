#!/usr/bin/env python3
"""Generate a speed-adjusted MP3 using ffmpeg (tempo change, pitch preserved)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from cli_helpers import print_banner, print_message


DEFAULT_INPUT = "High Hopes, arr. Doug Adams -- Score & Sound.mp3"
DEFAULT_OUTPUT = "HH_fast.mp3"


def emit(message: str) -> None:
    print_message(message)


def _atempo_chain(speed: float) -> str:
    """
    Build a valid ffmpeg atempo filter chain.

    ffmpeg's `atempo` only accepts values in [0.5, 2.0], but chaining multiple
    filters multiplies the effect.
    """
    if speed <= 0:
        raise ValueError("speed must be > 0")

    filters: list[float] = []
    remaining = float(speed)

    while remaining > 2.0:
        filters.append(2.0)
        remaining /= 2.0

    while remaining < 0.5:
        filters.append(0.5)
        remaining /= 0.5

    filters.append(remaining)
    return ",".join(f"atempo={value:.6f}".rstrip("0").rstrip(".") for value in filters)


def run_ffmpeg(input_path: Path, output_path: Path, speed: float, force: bool) -> int:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        emit("Error: ffmpeg not found on PATH.")
        return 2

    if not input_path.exists():
        emit(f"Error: input file not found: {input_path}")
        return 2

    if output_path.exists() and not force:
        emit(f"Error: output already exists: {output_path} (use --force to overwrite)")
        return 2

    atempo = _atempo_chain(speed)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if force else "-n",
        "-i",
        str(input_path),
        "-vn",
        "-filter:a",
        atempo,
        "-c:a",
        "libmp3lame",
        "-q:a",
        "2",
        str(output_path),
    ]

    emit(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=False)
    except OSError as exc:
        emit(f"Error running ffmpeg: {exc}")
        return 2

    if not output_path.exists():
        emit("Error: ffmpeg did not produce an output file.")
        return 1

    emit(f"Created: {output_path}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Create a speed-adjusted MP3 (tempo change via ffmpeg atempo).",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Input MP3 path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output MP3 path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.3,
        help="Speed factor (e.g. 1.3 = 30%% faster).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    args = parser.parse_args(argv)

    print_banner("Speed Up MP3")

    try:
        if args.speed <= 0:
            raise ValueError
    except ValueError:
        emit("Error: --speed must be a number > 0")
        return 2

    return run_ffmpeg(
        input_path=Path(args.input),
        output_path=Path(args.output),
        speed=float(args.speed),
        force=bool(args.force),
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

