#!/usr/bin/env python3
"""Rename an MP3 and update the local Audiotag metadata that references it."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MP3_FOLDER = Path("static/mp3")
DEFAULT_MID_MP3_FOLDER = Path("static/mid-mp3s")
LOUD_CACHE_NAME = ".loudness_cache.json"
EMB_CACHE_NAME = ".track_emb_cache.npz"
AUDIO_COLOR_CACHE = Path(".autoplay_audio_colors.json")
LEGACY_CONFIG_KEY = "__config__"
LEGACY_PLAY_HISTORY_KEY = "play_history"
LEGACY_PLAYLISTS_KEY = "playlists_v1"


@dataclass
class JsonUpdate:
    path: Path
    data: Any
    description: str


@dataclass
class DeleteUpdate:
    path: Path
    description: str


@dataclass
class RenamePlan:
    old_path: Path
    new_path: Path
    json_updates: list[JsonUpdate]
    delete_updates: list[DeleteUpdate]
    notes: list[str]


def normalize_mp3_name(raw: str) -> str:
    name = raw.strip()
    if not name:
        raise ValueError("Track name cannot be empty.")
    if Path(name).name != name:
        raise ValueError("Pass just the filename, not a nested path.")
    if Path(name).suffix.lower() != ".mp3":
        name += ".mp3"
    return name


def list_mp3_names(folder: Path) -> list[str]:
    if not folder.is_dir():
        return []
    return sorted(
        path.name
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".mp3" and not path.name.startswith(".")
    )


def default_tags_file(folder: Path) -> Path:
    return Path("mid_tags.json") if folder.name == "mid-mp3s" else Path("tags.json")


def default_artists_file(folder: Path) -> Path:
    return Path("mid_artists.json") if folder.name == "mid-mp3s" else Path("artists.json")


def default_reviewed_file(folder: Path) -> Path:
    return (
        Path("mid_artists_reviewed.json")
        if folder.name == "mid-mp3s"
        else Path("artists_reviewed.json")
    )


def default_listen_counts_file(folder: Path) -> Path:
    return (
        Path("mid_listen_counts.json")
        if folder.name == "mid-mp3s"
        else Path("listen_counts.json")
    )


def listen_timestamps_file(folder: Path) -> Path:
    return Path(f"listen_timestamps_{folder.name or 'default'}.json")


def playlists_file(folder: Path) -> Path:
    return Path(f"queue_playlists_{folder.name or 'default'}.json")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def merge_values(existing: Any, incoming: Any) -> Any:
    if isinstance(existing, list) and isinstance(incoming, list):
        merged = list(existing)
        seen = {repr(item) for item in merged}
        for item in incoming:
            key = repr(item)
            if key not in seen:
                merged.append(item)
                seen.add(key)
        return merged
    if isinstance(existing, (int, float)) and isinstance(incoming, (int, float)):
        return existing + incoming
    if isinstance(existing, dict) and isinstance(incoming, dict):
        merged = dict(incoming)
        merged.update(existing)
        return merged
    if existing == incoming:
        return existing
    raise ValueError(f"Cannot merge conflicting values: {existing!r} and {incoming!r}")


def rename_mapping_key(mapping: dict[Any, Any], old_key: str, new_key: str) -> int:
    if old_key not in mapping:
        return 0
    old_value = mapping.pop(old_key)
    if new_key in mapping:
        mapping[new_key] = merge_values(mapping[new_key], old_value)
    else:
        mapping[new_key] = old_value
    return 1


def replace_final_path_component(path_text: str, old_name: str, new_name: str) -> str | None:
    if path_text == old_name:
        return new_name
    if path_text.endswith("/" + old_name):
        prefix = path_text[: -len(old_name)]
        return prefix + new_name
    if "\\" in path_text:
        parts = path_text.split("\\")
        if parts and parts[-1] == old_name:
            parts[-1] = new_name
            return "\\".join(parts)
    return None


def rename_path_mapping_keys(mapping: dict[Any, Any], old_name: str, new_name: str) -> int:
    changes: list[tuple[str, str]] = []
    for key in list(mapping.keys()):
        if not isinstance(key, str):
            continue
        new_key = replace_final_path_component(key, old_name, new_name)
        if new_key and new_key != key:
            changes.append((key, new_key))

    for old_key, new_key in changes:
        value = mapping.pop(old_key)
        if new_key in mapping:
            mapping[new_key] = merge_values(mapping[new_key], value)
        else:
            mapping[new_key] = value
    return len(changes)


def update_mapping_file(path: Path, old_key: str, new_key: str, label: str) -> JsonUpdate | None:
    if not path.exists():
        return None
    data = load_json(path)
    if not isinstance(data, dict):
        return None
    changed = rename_mapping_key(data, old_key, new_key)
    if not changed:
        return None
    return JsonUpdate(path, data, f"{label}: renamed key {old_key!r} -> {new_key!r}")


def update_reviewed_file(path: Path, old_stem: str, new_stem: str) -> JsonUpdate | None:
    if not path.exists():
        return None
    data = load_json(path)
    changed = 0
    if isinstance(data, list):
        changed = sum(1 for item in data if item == old_stem)
        data = [new_stem if item == old_stem else item for item in data]
    elif isinstance(data, dict):
        changed = rename_mapping_key(data, old_stem, new_stem)
    if not changed:
        return None
    return JsonUpdate(path, data, f"reviewed artists: renamed {changed} reference(s)")


def update_timestamps_file(path: Path, old_name: str, new_name: str) -> JsonUpdate | None:
    if not path.exists():
        return None
    data = load_json(path)
    changed = 0
    if isinstance(data, list):
        for event in data:
            if isinstance(event, dict) and event.get("track") == old_name:
                event["track"] = new_name
                changed += 1
    elif isinstance(data, dict):
        old_stem = Path(old_name).stem
        new_stem = Path(new_name).stem
        changed = rename_mapping_key(data, old_stem, new_stem)
    if not changed:
        return None
    return JsonUpdate(path, data, f"listen timestamps: renamed {changed} reference(s)")


def update_playlists_file(path: Path, old_stem: str, new_stem: str) -> JsonUpdate | None:
    if not path.exists():
        return None
    data = load_json(path)
    changed = 0

    if isinstance(data, dict):
        for items in data.values():
            if not isinstance(items, list):
                continue
            for idx, item in enumerate(items):
                if isinstance(item, dict) and item.get("base") == old_stem:
                    item["base"] = new_stem
                    changed += 1
                elif item == old_stem:
                    items[idx] = new_stem
                    changed += 1

    if not changed:
        return None
    return JsonUpdate(path, data, f"playlists: renamed {changed} item(s)")


def update_path_cache_file(path: Path, old_name: str, new_name: str, label: str) -> JsonUpdate | None:
    if not path.exists():
        return None
    data = load_json(path)
    if not isinstance(data, dict):
        return None
    changed = rename_path_mapping_keys(data, old_name, new_name)
    if not changed:
        return None
    return JsonUpdate(path, data, f"{label}: renamed {changed} path key(s)")


def update_nested_path_cache_file(
    path: Path, section: str, old_name: str, new_name: str, label: str
) -> JsonUpdate | None:
    if not path.exists():
        return None
    data = load_json(path)
    if not isinstance(data, dict) or not isinstance(data.get(section), dict):
        return None
    changed = rename_path_mapping_keys(data[section], old_name, new_name)
    if not changed:
        return None
    return JsonUpdate(path, data, f"{label}: renamed {changed} path key(s)")


def update_legacy_mp3meta(path: Path, old_stem: str, new_stem: str) -> JsonUpdate | None:
    if not path.exists():
        return None
    data = load_json(path)
    if not isinstance(data, dict):
        return None

    changed = rename_mapping_key(data, old_stem, new_stem)
    config = data.get(LEGACY_CONFIG_KEY)
    if isinstance(config, dict):
        history = config.get(LEGACY_PLAY_HISTORY_KEY)
        if isinstance(history, dict):
            changed += rename_mapping_key(history, old_stem, new_stem)
        playlists = config.get(LEGACY_PLAYLISTS_KEY)
        if isinstance(playlists, dict):
            for songs in playlists.values():
                if isinstance(songs, list):
                    for idx, song in enumerate(songs):
                        if song == old_stem:
                            songs[idx] = new_stem
                            changed += 1

    if not changed:
        return None
    return JsonUpdate(path, data, f"legacy metadata: renamed {changed} reference(s)")


def unique_existing(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            out.append(path)
    return out


def build_plan(
    folder: Path,
    old_name: str,
    new_name: str,
    tags_file: Path | None,
    artists_file: Path | None,
    reviewed_file: Path | None,
    listen_counts: Path | None,
) -> RenamePlan:
    old_path = folder / old_name
    new_path = folder / new_name
    old_stem = Path(old_name).stem
    new_stem = Path(new_name).stem

    if old_name == new_name:
        raise ValueError("Filename is unchanged.")
    if not folder.is_dir():
        raise ValueError(f"MP3 folder does not exist: {folder}")
    if not old_path.exists():
        raise ValueError(f"Old MP3 not found: {old_path}")
    if old_path.suffix.lower() != ".mp3":
        raise ValueError(f"Old path is not an MP3: {old_path}")

    try:
        same_file = new_path.exists() and os.path.samefile(old_path, new_path)
    except OSError:
        same_file = False
    if new_path.exists() and not same_file:
        raise ValueError(f"New MP3 already exists: {new_path}")

    json_updates: list[JsonUpdate] = []
    notes: list[str] = []

    mapping_targets = [
        (tags_file or default_tags_file(folder), old_stem, new_stem, "tags"),
        (artists_file or default_artists_file(folder), old_stem, new_stem, "artists"),
        (
            listen_counts or default_listen_counts_file(folder),
            old_stem,
            new_stem,
            "listen counts",
        ),
    ]
    for path, old_key, new_key, label in mapping_targets:
        update = update_mapping_file(path, old_key, new_key, label)
        if update:
            json_updates.append(update)

    reviewed_update = update_reviewed_file(
        reviewed_file or default_reviewed_file(folder), old_stem, new_stem
    )
    if reviewed_update:
        json_updates.append(reviewed_update)

    for update in [
        update_timestamps_file(listen_timestamps_file(folder), old_name, new_name),
        update_playlists_file(playlists_file(folder), old_stem, new_stem),
        update_path_cache_file(
            folder / LOUD_CACHE_NAME, old_name, new_name, "loudness cache"
        ),
        update_path_cache_file(
            Path("audio_features_cache.json"),
            old_name,
            new_name,
            "audio features cache",
        ),
        update_nested_path_cache_file(
            AUDIO_COLOR_CACHE,
            "entries",
            old_name,
            new_name,
            "audio color cache",
        ),
        update_legacy_mp3meta(folder / ".mp3meta.json", old_stem, new_stem),
        update_mapping_file(folder / ".mp3analysis.json", old_stem, new_stem, "legacy analysis"),
    ]:
        if update:
            json_updates.append(update)

    delete_updates: list[DeleteUpdate] = []
    emb_cache = folder / EMB_CACHE_NAME
    if emb_cache.exists():
        delete_updates.append(
            DeleteUpdate(emb_cache, "embedding cache: removed so it rebuilds without old stem")
        )

    if not json_updates and not delete_updates:
        notes.append("No metadata references were found. Only the MP3 file will be renamed.")

    return RenamePlan(old_path, new_path, json_updates, delete_updates, notes)


def rename_file(old_path: Path, new_path: Path) -> None:
    try:
        same_file = new_path.exists() and os.path.samefile(old_path, new_path)
    except OSError:
        same_file = False

    if same_file:
        temp_path = old_path.with_name(f".{old_path.name}.rename_tmp")
        counter = 1
        while temp_path.exists():
            temp_path = old_path.with_name(f".{old_path.name}.rename_tmp{counter}")
            counter += 1
        old_path.rename(temp_path)
        temp_path.rename(new_path)
    else:
        old_path.rename(new_path)


def apply_plan(plan: RenamePlan) -> None:
    rename_file(plan.old_path, plan.new_path)
    try:
        for update in plan.json_updates:
            write_json(update.path, update.data)
        for update in plan.delete_updates:
            update.path.unlink(missing_ok=True)
    except Exception:
        if plan.new_path.exists() and not plan.old_path.exists():
            rename_file(plan.new_path, plan.old_path)
        raise


def relevant_scan_paths(folder: Path, plan: RenamePlan) -> list[Path]:
    paths = [
        Path("tags.json"),
        Path("mid_tags.json"),
        Path("artists.json"),
        Path("mid_artists.json"),
        Path("artists_reviewed.json"),
        Path("mid_artists_reviewed.json"),
        Path("listen_counts.json"),
        Path("mid_listen_counts.json"),
        listen_timestamps_file(folder),
        playlists_file(folder),
        folder / LOUD_CACHE_NAME,
        Path("audio_features_cache.json"),
        AUDIO_COLOR_CACHE,
        folder / ".mp3meta.json",
        folder / ".mp3analysis.json",
    ]
    paths.extend(update.path for update in plan.json_updates)
    return unique_existing(paths)


def scan_remaining_mentions(paths: Iterable[Path], old_name: str, old_stem: str) -> list[str]:
    matches: list[str] = []
    needles = [old_name]
    if old_stem != old_name:
        needles.append(f'"{old_stem}"')

    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for needle in needles:
            if needle in text:
                matches.append(f"{path}: contains {needle!r}")
                break
    return matches


def print_plan(plan: RenamePlan, apply: bool, action: str | None = None) -> None:
    action = action or ("Applying" if apply else "Dry run")
    print(f"{action}: {plan.old_path} -> {plan.new_path}")
    for update in plan.json_updates:
        print(f"  update {update.path}: {update.description}")
    for update in plan.delete_updates:
        print(f"  delete {update.path}: {update.description}")
    for note in plan.notes:
        print(f"  note: {note}")


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def choose_folder(default_folder: Path) -> Path | None:
    candidates = [
        path
        for path in [DEFAULT_MP3_FOLDER, DEFAULT_MID_MP3_FOLDER, default_folder]
        if path.is_dir()
    ]
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)

    if not unique:
        print(f"No MP3 folder found. Expected {default_folder}.")
        return None
    if len(unique) == 1:
        return unique[0]

    print("Choose a library:")
    for idx, path in enumerate(unique, start=1):
        count = len(list_mp3_names(path))
        default_label = " default" if path == default_folder else ""
        print(f"  {idx}. {path} ({count} mp3s){default_label}")

    while True:
        choice = ask("Library number, Enter for default, or q to cancel: ")
        if not choice:
            return default_folder if default_folder in unique else unique[0]
        if choice.lower() in {"q", "quit", "cancel"}:
            return None
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(unique):
                return unique[idx - 1]
        print("Enter one of the listed numbers.")


def choose_track(folder: Path) -> str | None:
    names = list_mp3_names(folder)
    if not names:
        print(f"No MP3 files found in {folder}.")
        return None

    filtered = names
    query = ""
    while True:
        print()
        title = f"Tracks in {folder}" if not query else f"Tracks matching {query!r}"
        print(title)
        for idx, name in enumerate(filtered[:25], start=1):
            print(f"  {idx:>2}. {name}")
        if len(filtered) > 25:
            print(f"  ... {len(filtered) - 25} more")

        choice = ask("Type a number, search text, Enter to reset, or q to cancel: ")
        if choice.lower() in {"q", "quit", "cancel"}:
            return None
        if not choice:
            filtered = names
            query = ""
            continue
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(filtered):
                return filtered[idx - 1]
            print("That number is not in the current list.")
            continue

        query = choice
        lowered = query.casefold()
        filtered = [name for name in names if lowered in name.casefold()]
        if not filtered:
            print(f"No tracks match {query!r}.")
            filtered = names
            query = ""


def choose_new_name(old_name: str) -> str | None:
    old_stem = Path(old_name).stem
    print(f"\nSelected: {old_name}")
    print("Type the new filename or stem. The .mp3 extension is added if omitted.")
    while True:
        raw = ask(f"New name [{old_stem}], or q to cancel: ")
        if raw.lower() in {"q", "quit", "cancel"}:
            return None
        if not raw:
            print("No name entered; cancel with q if you do not want to rename.")
            continue
        try:
            return normalize_mp3_name(raw)
        except ValueError as exc:
            print(f"Error: {exc}")


def confirm(prompt: str) -> bool:
    choice = ask(prompt).lower()
    return choice in {"y", "yes"}


def run_interactive(args: argparse.Namespace) -> int:
    print("MP3 rename helper")
    folder = args.folder.expanduser() if args.folder else choose_folder(DEFAULT_MP3_FOLDER)
    if folder is None:
        print("No changes made.")
        return 0

    old_name = choose_track(folder)
    if old_name is None:
        print("No changes made.")
        return 0
    new_name = choose_new_name(old_name)
    if new_name is None:
        print("No changes made.")
        return 0

    try:
        plan = build_plan(
            folder=folder,
            old_name=old_name,
            new_name=new_name,
            tags_file=args.tags,
            artists_file=args.artists,
            reviewed_file=args.reviewed,
            listen_counts=args.listen_counts,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print()
    print_plan(plan, apply=False, action="Preview")
    if not args.no_scan:
        matches = scan_remaining_mentions(
            relevant_scan_paths(folder, plan), old_name, Path(old_name).stem
        )
        if matches:
            print("Current old-text matches that should be handled or inspected:")
            for match in matches:
                print(f"  {match}")

    if not confirm("Apply this rename? [y/N]: "):
        print("No changes made.")
        return 0

    try:
        apply_plan(plan)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print("Done.")

    if not args.no_scan:
        matches = scan_remaining_mentions(
            relevant_scan_paths(folder, plan), old_name, Path(old_name).stem
        )
        if matches:
            print("Remaining old-text matches to inspect:")
            for match in matches:
                print(f"  {match}")
        else:
            print("No remaining old filename/stem matches in known metadata files.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rename an MP3 and update Audiotag tags, listens, playlists, and caches."
        )
    )
    parser.add_argument(
        "old",
        nargs="?",
        help="Current MP3 filename or stem. Omit old/new to use interactive mode.",
    )
    parser.add_argument(
        "new",
        nargs="?",
        help="New MP3 filename or stem. Omit old/new to use interactive mode.",
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=None,
        help="Folder containing the MP3s. Default: static/mp3",
    )
    parser.add_argument(
        "--mid",
        action="store_true",
        help="Shortcut for --folder static/mid-mp3s.",
    )
    parser.add_argument("--tags", type=Path, default=None, help="Override tags JSON path.")
    parser.add_argument(
        "--artists", type=Path, default=None, help="Override artists JSON path."
    )
    parser.add_argument(
        "--reviewed", type=Path, default=None, help="Override reviewed artists JSON path."
    )
    parser.add_argument(
        "--listen-counts", type=Path, default=None, help="Override listen counts JSON path."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename files and write metadata. Without this, only prints a dry run.",
    )
    parser.add_argument(
        "--no-scan",
        action="store_true",
        help="Skip the post-plan scan for remaining old references.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mid:
        args.folder = DEFAULT_MID_MP3_FOLDER

    if args.old is None and args.new is None:
        return run_interactive(args)
    if args.old is None or args.new is None:
        print(
            "Error: pass both old and new names, or pass neither for interactive mode.",
            file=sys.stderr,
        )
        return 1

    try:
        old_name = normalize_mp3_name(args.old)
        new_name = normalize_mp3_name(args.new)
        folder = (args.folder or DEFAULT_MP3_FOLDER).expanduser()
        plan = build_plan(
            folder=folder,
            old_name=old_name,
            new_name=new_name,
            tags_file=args.tags,
            artists_file=args.artists,
            reviewed_file=args.reviewed,
            listen_counts=args.listen_counts,
        )
        print_plan(plan, apply=args.apply)
        if args.apply:
            apply_plan(plan)
            print("Done.")
        else:
            print("Dry run only. Re-run with --apply to make changes.")

        if not args.no_scan:
            scan_paths = relevant_scan_paths(folder, plan)
            matches = scan_remaining_mentions(scan_paths, old_name, Path(old_name).stem)
            if matches:
                print("Remaining old-text matches to inspect:")
                for match in matches:
                    print(f"  {match}")
            else:
                print("No remaining old filename/stem matches in known metadata files.")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
