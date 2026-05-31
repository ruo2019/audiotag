#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path


FOLDER_MAP = {
    "m": Path("static/mp3"),
    "mm": Path("static/mid-mp3s"),
}


def parse_line(line: str, line_number: int) -> tuple[str, str, Path]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        raise ValueError("skip")

    try:
        video_id, remainder = stripped.split(" ", 1)
    except ValueError as exc:
        raise ValueError(
            f"Line {line_number}: expected 'VIDEO_ID title [m|mm]'"
        ) from exc

    remainder = remainder.strip()
    if not remainder.endswith("]") or " [" not in remainder:
        raise ValueError(
            f"Line {line_number}: expected folder marker at the end, like [m] or [mm]"
        )

    title, marker = remainder.rsplit(" [", 1)
    title = title.strip()
    marker = marker[:-1].strip()

    if not title:
        raise ValueError(f"Line {line_number}: missing title")

    if marker not in FOLDER_MAP:
        raise ValueError(
            f"Line {line_number}: folder marker must be [m] or [mm], got [{marker}]"
        )

    return video_id.strip(), title, FOLDER_MAP[marker]


def build_url(video_id: str) -> str:
    if video_id.startswith("http://") or video_id.startswith("https://"):
        return video_id
    return f"https://www.youtube.com/watch?v={video_id}"


def run_download(url: str, output_path: Path, dry_run: bool) -> None:
    cmd = [
        "yt-dlp",
        "-x",
        url,
        "--audio-format",
        "mp3",
        "-o",
        str(output_path),
    ]

    print(" ".join(cmd))
    if dry_run:
        return

    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download YouTube audio listed in a line-based input file."
    )
    parser.add_argument(
        "input_file",
        help="File where each non-empty line is: VIDEO_ID title [m|mm]",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running yt-dlp",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    for line_number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            video_id, title, folder = parse_line(line, line_number)
        except ValueError as exc:
            if str(exc) == "skip":
                continue
            raise

        folder.mkdir(parents=True, exist_ok=True)
        output_path = folder / f"{title}.mp3"
        url = build_url(video_id)
        run_download(url, output_path, args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
