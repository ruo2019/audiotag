"""
Create a JSON file containing all MP3 filenames (no folder path, no .mp3),
one per line in the output (via pretty-printed JSON).

Usage:
  python mp3_to_json.py "/path/to/folder" -o out.json
  python mp3_to_json.py "/path/to/folder" -o out.json --recursive
"""

import argparse
import json
from pathlib import Path


def collect_mp3_stems(folder: Path, recursive: bool) -> list[str]:
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    stems: list[str] = []

    for p in iterator:
        if p.is_file() and p.suffix.lower() == ".mp3":
            stems.append(p.stem)  # filename without extension

    # Sort for stable output; remove duplicates just in case
    return sorted(set(stems))


def main() -> None:
    parser = argparse.ArgumentParser(description="List MP3 filenames into a JSON file.")
    parser.add_argument("folder", help="Folder to scan for .mp3 files")
    parser.add_argument(
        "-o",
        "--output",
        default="mp3_filenames.json",
        help="Output JSON file path (default: mp3_filenames.json)",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Include mp3 files in subfolders",
    )
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"Error: '{folder}' is not a folder.")

    names = collect_mp3_stems(folder, args.recursive)

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {len(names)} filename(s) to {out_path}")


if __name__ == "__main__":
    main()
