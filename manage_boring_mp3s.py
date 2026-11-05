#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable

from external_listens import load_mappings, load_plays, mapped_contributions


ROOT = Path(__file__).resolve().parent
ARCHIVE_ROOT = ROOT / "static/boring-mp3s"
MANIFEST_PATH = ARCHIVE_ROOT / "boring_mp3s_manifest.json"
LIBRARIES = {
    "mp3": (ROOT / "static/mp3", ROOT / "listen_counts.json", "m"),
    "mid-mp3s": (
        ROOT / "static/mid-mp3s",
        ROOT / "mid_listen_counts.json",
        "mm",
    ),
}


def load_json(path: Path, default: object) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return default


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def normalized_archive_filename(value: str) -> str:
    path = Path(value)
    replacements = {
        " [from mp3]": " [m]",
        " [from mid-mp3s]": " [mm]",
    }
    for old_suffix, new_suffix in replacements.items():
        if path.stem.endswith(old_suffix):
            return f"{path.stem[:-len(old_suffix)]}{new_suffix}{path.suffix}"
    return path.name


def load_manifest() -> dict:
    raw = load_json(MANIFEST_PATH, {})
    if not isinstance(raw, dict):
        raw = {}
    raw_tracks = raw.get("tracks")
    if not isinstance(raw_tracks, list):
        raw_tracks = []
    tracks = []
    seen = set()
    for item in raw_tracks:
        if not isinstance(item, dict):
            continue
        library = str(item.get("library") or "")
        if library not in LIBRARIES:
            continue
        archived_value = str(
            item.get("archived_path") or item.get("filename") or ""
        )
        filename = normalized_archive_filename(archived_value)
        if not filename:
            continue
        key = (library, filename.casefold())
        if key in seen:
            continue
        seen.add(key)
        try:
            listen_count = max(0, int(item.get("listen_count") or 0))
        except (TypeError, ValueError):
            listen_count = 0
        tracks.append(
            {
                "filename": filename,
                "library": library,
                "listen_count": listen_count,
            }
        )
    return {
        "version": 2,
        "threshold": int(raw.get("threshold") or 10),
        "tracks": tracks,
    }


def save_manifest(manifest: dict) -> None:
    manifest["tracks"].sort(
        key=lambda item: (
            str(item["library"]).casefold(),
            str(item["filename"]).casefold(),
        )
    )
    atomic_write_json(MANIFEST_PATH, manifest)


def integer_counts(path: Path) -> Dict[str, int]:
    raw = load_json(path, {})
    if not isinstance(raw, dict):
        return {}
    counts: Dict[str, int] = {}
    for track, value in raw.items():
        try:
            counts[str(track)] = max(0, int(value or 0))
        except (TypeError, ValueError):
            continue
    return counts


def original_filename(item: dict) -> str:
    filename = str(item["filename"])
    library = str(item["library"])
    marker = LIBRARIES[library][2]
    path = Path(filename)
    marker_suffix = f" [{marker}]"
    if not path.stem.endswith(marker_suffix):
        return filename
    unmarked_stem = path.stem[: -len(marker_suffix)]
    return f"{unmarked_stem}{path.suffix}"


def remove_active_count(library: str, filename: str) -> None:
    counts_path = LIBRARIES[library][1]
    counts = integer_counts(counts_path)
    if counts.pop(Path(filename).stem, None) is not None:
        atomic_write_json(counts_path, counts)


def restore_active_count(item: dict) -> None:
    library = str(item["library"])
    filename = original_filename(item)
    counts_path = LIBRARIES[library][1]
    counts = integer_counts(counts_path)
    external_counts, _history = mapped_contributions(
        library, load_plays(), load_mappings()
    )
    total = max(0, int(item["listen_count"]))
    external = max(0, int(external_counts.get(Path(filename).stem, 0)))
    counts[Path(filename).stem] = max(0, total - external)
    atomic_write_json(counts_path, counts)


def archive_candidates(threshold: int) -> list[dict]:
    events = load_plays()
    mappings = load_mappings()
    candidates = []
    for library, (folder, counts_path, _marker) in LIBRARIES.items():
        local_counts = integer_counts(counts_path)
        external_counts, _history = mapped_contributions(
            library, events, mappings
        )
        for source in sorted(folder.glob("*.mp3"), key=lambda path: path.name.casefold()):
            local_listens = local_counts.get(source.stem, 0)
            external_listens = max(0, int(external_counts.get(source.stem, 0)))
            listen_count = local_listens + external_listens
            if listen_count < threshold:
                candidates.append(
                    {
                        "source": source,
                        "filename": source.name,
                        "library": library,
                        "listen_count": listen_count,
                    }
                )
    return candidates


def marked_archive_path(filename: str, library: str) -> Path:
    marker = LIBRARIES[library][2]
    path = Path(filename)
    return ARCHIVE_ROOT / f"{path.stem} [{marker}]{path.suffix}"


def archive_destination(
    filename: str,
    library: str,
    reserved: set[Path],
    manifest: dict,
) -> Path:
    preferred = ARCHIVE_ROOT / filename
    sibling = next(
        (
            item
            for item in manifest["tracks"]
            if original_filename(item).casefold() == filename.casefold()
        ),
        None,
    )
    if preferred not in reserved and not preferred.exists() and sibling is None:
        return preferred
    if sibling is not None and str(sibling["filename"]) == filename:
        sibling_source = ARCHIVE_ROOT / filename
        sibling_marked = marked_archive_path(filename, str(sibling["library"]))
        if sibling_marked.exists():
            raise FileExistsError(f"Archive name already exists: {sibling_marked.name}")
        shutil.move(str(sibling_source), str(sibling_marked))
        reserved.discard(sibling_source)
        reserved.add(sibling_marked)
        sibling["filename"] = sibling_marked.name
        save_manifest(manifest)
    marked = marked_archive_path(filename, library)
    if marked in reserved or marked.exists():
        raise FileExistsError(
            f"Both archive names already exist: {preferred.name}, {marked.name}"
        )
    return marked


def archive(threshold: int, apply: bool) -> int:
    candidates = archive_candidates(threshold)
    by_library: Dict[str, int] = {}
    for item in candidates:
        by_library[item["library"]] = by_library.get(item["library"], 0) + 1
    summary = ", ".join(
        f"{library}: {count}" for library, count in sorted(by_library.items())
    )
    print(f"Found {len(candidates)} track(s) below {threshold} listens ({summary}).")
    if not apply:
        for item in candidates:
            print(f"{item['listen_count']:>2}  {item['source']}")
        print("Dry run only. Add --apply to move these files.")
        return 0

    manifest = load_manifest()
    manifest["threshold"] = threshold
    reserved = {path for path in ARCHIVE_ROOT.iterdir() if path.is_file()}
    moved = 0
    for candidate in candidates:
        destination = archive_destination(
            candidate["filename"], candidate["library"], reserved, manifest
        )
        shutil.move(str(candidate["source"]), str(destination))
        reserved.add(destination)
        manifest["tracks"].append(
            {
                "filename": destination.name,
                "library": candidate["library"],
                "listen_count": candidate["listen_count"],
            }
        )
        save_manifest(manifest)
        remove_active_count(candidate["library"], candidate["filename"])
        moved += 1
    print(f"Archived {moved} track(s). Manifest: {MANIFEST_PATH}")
    return 0


def matching_entries(manifest: dict, selectors: Iterable[str]) -> list[dict]:
    wanted = [value.strip().casefold() for value in selectors if value.strip()]
    matches = []
    for item in manifest["tracks"]:
        original = original_filename(item)
        values = {
            str(item["filename"]).casefold(),
            original.casefold(),
            f"static/{item['library']}/{original}".casefold(),
        }
        if not wanted or any(value in values for value in wanted):
            matches.append(item)
    return matches


def restore(selectors: list[str], restore_all: bool) -> int:
    manifest = load_manifest()
    matches = matching_entries(manifest, [] if restore_all else selectors)
    if not matches:
        print("No archived track matched.")
        return 1
    if not restore_all and len(matches) > len(set(selectors)):
        names = "\n".join(
            f"  {item['library']}: {item['filename']}" for item in matches
        )
        raise ValueError(
            "A name matched more than one archived track; use the marked filename:\n"
            + names
        )

    restored = 0
    for item in matches:
        source = ARCHIVE_ROOT / str(item["filename"])
        folder = LIBRARIES[str(item["library"])][0]
        destination = folder / original_filename(item)
        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite active-library file: {destination}"
            )
        if not source.is_file():
            raise FileNotFoundError(f"Archived MP3 is missing: {source}")
        shutil.move(str(source), str(destination))
        restore_active_count(item)
        manifest["tracks"].remove(item)
        save_manifest(manifest)
        print(
            f"Restored {destination.relative_to(ROOT)} "
            f"({item['listen_count']} listens)."
        )
        restored += 1
    print(f"Restored {restored} track(s).")
    return 0


def list_archived() -> int:
    manifest = load_manifest()
    for item in manifest["tracks"]:
        print(
            f"{int(item['listen_count']):>2}  "
            f"[{LIBRARIES[item['library']][2]}] {item['filename']}"
        )
    print(f"{len(manifest['tracks'])} archived track(s).")
    return 0


def normalize_manifest() -> int:
    manifest = load_manifest()
    by_original: Dict[str, list[dict]] = {}
    for item in manifest["tracks"]:
        by_original.setdefault(original_filename(item).casefold(), []).append(item)
    for items in by_original.values():
        if len(items) < 2:
            continue
        for item in items:
            original = original_filename(item)
            destination = marked_archive_path(original, str(item["library"]))
            source = ARCHIVE_ROOT / str(item["filename"])
            if source == destination:
                continue
            if destination.exists():
                raise FileExistsError(f"Archive name already exists: {destination.name}")
            shutil.move(str(source), str(destination))
            item["filename"] = destination.name
    save_manifest(manifest)
    print(f"Normalized {len(manifest['tracks'])} manifest entries.")
    return 0


def purge_archived_counts() -> int:
    manifest = load_manifest()
    removed = 0
    for library, (_folder, counts_path, _marker) in LIBRARIES.items():
        counts = integer_counts(counts_path)
        before = len(counts)
        for item in manifest["tracks"]:
            if item["library"] == library:
                counts.pop(Path(original_filename(item)).stem, None)
        removed += before - len(counts)
        atomic_write_json(counts_path, counts)
    print(f"Removed {removed} archived track count entries.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Archive low-listen MP3s and restore them to their original library."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    archive_parser = subparsers.add_parser("archive")
    archive_parser.add_argument("--threshold", type=int, default=10)
    archive_parser.add_argument(
        "--apply", action="store_true", help="Move files; otherwise print a dry run."
    )
    restore_parser = subparsers.add_parser("restore")
    restore_parser.add_argument("tracks", nargs="*", help="Archived filename.")
    restore_parser.add_argument("--all", action="store_true")
    subparsers.add_parser("list")
    subparsers.add_parser("normalize")
    subparsers.add_parser("purge-counts")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "archive":
        if args.threshold < 0:
            raise ValueError("Threshold must be zero or greater.")
        return archive(args.threshold, args.apply)
    if args.command == "restore":
        if not args.all and not args.tracks:
            raise ValueError("Name at least one track, or use --all.")
        return restore(args.tracks, args.all)
    if args.command == "normalize":
        return normalize_manifest()
    if args.command == "purge-counts":
        return purge_archived_counts()
    return list_archived()


if __name__ == "__main__":
    raise SystemExit(main())
