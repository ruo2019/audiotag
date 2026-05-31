#!/usr/bin/env python3
"""Rename an MP3 and update play.py metadata JSON files."""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from cli_helpers import RenameChoice, print_banner, print_message, require_console, run_rename_loop
except ModuleNotFoundError:  # pragma: no cover - support importing as scripts.rename_*
    from scripts.cli_helpers import (
        RenameChoice,
        print_banner,
        print_message,
        require_console,
        run_rename_loop,
    )

try:
    from player.constants import (
        ANALYSIS_CACHE_FILENAME,
        CONFIG_KEY,
        META_FILENAME,
        PLAY_HISTORY_CONFIG_KEY,
        PLAYLISTS_CONFIG_KEY,
    )
except ModuleNotFoundError:  # pragma: no cover - support running from scripts dir
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from player.constants import (
        ANALYSIS_CACHE_FILENAME,
        CONFIG_KEY,
        META_FILENAME,
        PLAY_HISTORY_CONFIG_KEY,
        PLAYLISTS_CONFIG_KEY,
    )

console = None
VERBOSE = False
INTERACTIVE_MODE = False
SUCCESS_MESSAGE = "Done. Updated metadata."


def emit(message: str) -> None:
    if INTERACTIVE_MODE:
        return
    print_message(message, console)

def emit_detail(message: str) -> None:
    if VERBOSE:
        emit(message)


def list_mp3_files(folder: Path) -> list[str]:
    return sorted(
        entry.name for entry in folder.iterdir() if entry.is_file() and entry.suffix.lower() == ".mp3"
    )

def build_rename_choice(folder: Path) -> RenameChoice:
    return RenameChoice(
        prompt=f"Choose an MP3 from '{folder}/', then edit its filename.",
        choices=list_mp3_files(folder),
        help_text="Use arrow keys or the mouse to select. Press Enter to edit, then Enter again to apply.",
        empty_message="No MP3 files were found.",
    )


def apply_rename(folder: Path, old_name: str, new_name: str) -> tuple[bool, str]:
    if old_name == new_name:
        return False, "Filename is unchanged."

    old_path = folder / old_name
    new_path = folder / new_name

    if not old_path.exists():
        return False, f"'{old_name}' was not found in {folder}."

    if new_path.exists():
        return False, f"'{new_name}' already exists in {folder}."

    file_renamed_by_this_script = False
    try:
        old_path.rename(new_path)
        emit_detail(f"Renamed file: '{old_name}' -> '{new_name}'.")
        file_renamed_by_this_script = True
    except OSError as exc:
        return False, f"Error renaming file: {exc}"

    json_updated_successfully = update_play_py_json_files(folder, old_name, new_name)
    if json_updated_successfully:
        return True, f"Renamed '{old_name}' to '{new_name}'."

    if file_renamed_by_this_script:
        try:
            new_path.rename(old_path)
        except OSError as exc:
            return False, f"Metadata update failed and file rollback failed: {exc}"
    return False, "Metadata update failed. Changes were rolled back."


def update_json_file(json_path: Path, old_key: str, new_key: str, sort_keys: bool = False):
    """Update a key in a JSON file, preserving the value."""
    if not json_path.exists():
        emit_detail(f"  JSON file not found: {json_path.name}. Skipping.")
        return True

    try:
        content = json_path.read_text(encoding="utf-8")
        if not content:
            emit_detail(f"  JSON file is empty: {json_path.name}. Skipping.")
            return True
        data = json.loads(content)

        if old_key in data:
            data[new_key] = data.pop(old_key)
            json_path.write_text(
                json.dumps(data, indent=2, sort_keys=sort_keys), encoding="utf-8"
            )
            emit_detail(f"Updated {json_path.name} key: '{old_key}' -> '{new_key}'.")
        else:
            emit_detail(f"  No entry for '{old_key}' in {json_path.name}.")

        return True
    except (json.JSONDecodeError, OSError) as exc:
        emit(f"Error updating {json_path.name}: {exc}")
        return False


def _rename_in_mapping(mapping, old_key, new_key):
    if not isinstance(mapping, dict) or old_key not in mapping:
        return False
    mapping[new_key] = mapping.pop(old_key)
    return True


def _rename_in_playlist_list(songs, old_key, new_key):
    if not isinstance(songs, list):
        return False, songs
    updated = False
    new_songs = []
    for song in songs:
        if song == old_key:
            new_songs.append(new_key)
            updated = True
        else:
            new_songs.append(song)
    return updated, new_songs


def update_meta_file(meta_path: Path, old_key: str, new_key: str):
    if not meta_path.exists():
        emit_detail(f"  JSON file not found: {meta_path.name}. Skipping.")
        return True

    try:
        content = meta_path.read_text(encoding="utf-8")
        if not content:
            emit_detail(f"  JSON file is empty: {meta_path.name}. Skipping.")
            return True
        data = json.loads(content)
        updated = False
        updated_sections = []

        if old_key in data:
            data[new_key] = data.pop(old_key)
            updated = True
            updated_sections.append("track entry")

        config = data.get(CONFIG_KEY)
        if isinstance(config, dict):
            history_updated = False
            playlists_updated = False
            for map_key in (PLAY_HISTORY_CONFIG_KEY,):
                mapping = config.get(map_key)
                if _rename_in_mapping(mapping, old_key, new_key):
                    updated = True
                    history_updated = True

            playlists = config.get(PLAYLISTS_CONFIG_KEY)
            if isinstance(playlists, dict):
                for playlist_name, songs in playlists.items():
                    changed, renamed_songs = _rename_in_playlist_list(songs, old_key, new_key)
                    if changed:
                        playlists[playlist_name] = renamed_songs
                        updated = True
                        playlists_updated = True

            data[CONFIG_KEY] = config

            if history_updated:
                updated_sections.append("play history")
            if playlists_updated:
                updated_sections.append("playlists")

        if updated:
            meta_path.write_text(
                json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
            )
            section_summary = ", ".join(updated_sections) if updated_sections else "track entry"
            emit_detail(f"Updated {meta_path.name} ({section_summary}): '{old_key}' -> '{new_key}'.")
        else:
            emit_detail(f"  No entry for '{old_key}' in {meta_path.name}.")

        return True
    except (json.JSONDecodeError, OSError) as exc:
        emit(f"Error updating {meta_path.name}: {exc}")
        return False



def update_play_py_json_files(folder: Path, old_name: str, new_name: str):
    """Update song basenames in .mp3meta.json and .mp3analysis.json."""
    old_basename = os.path.splitext(old_name)[0]
    new_basename = os.path.splitext(new_name)[0]

    if old_basename == new_basename:
        emit("Metadata key rename skipped (filenames share the same basename).")
        return True

    meta_path = folder / META_FILENAME
    analysis_path = folder / ANALYSIS_CACHE_FILENAME

    meta_success = update_meta_file(meta_path, old_basename, new_basename)
    if not meta_success:
        return False

    analysis_success = update_json_file(
        analysis_path, old_basename, new_basename, sort_keys=False
    )

    return analysis_success


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rename an MP3 and update play.py JSON metadata files."
    )
    parser.add_argument(
        "--folder",
        default="mp3s",
        help="Folder that contains the MP3s and metadata JSON files (default: mp3s)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed logs",
    )
    return parser.parse_args()


def main():
    global console
    global VERBOSE
    global INTERACTIVE_MODE
    console = require_console()

    args = parse_args()
    folder = Path(args.folder).expanduser()
    VERBOSE = args.verbose

    if not folder.is_dir():
        emit(
            f"Error: MP3 folder '{folder}' not found. Please ensure it exists."
        )
        sys.exit(1)

    print_banner("[Rename CLI]", console)

    INTERACTIVE_MODE = True
    try:
        rename_count = run_rename_loop(
            build_rename_choice(folder),
            lambda old_name, new_name: apply_rename(folder, old_name, new_name),
        )
    finally:
        INTERACTIVE_MODE = False

    if rename_count == 0:
        emit("No changes made.")
        sys.exit(0)

    emit(f"{SUCCESS_MESSAGE} Applied {rename_count} rename(s).")


if __name__ == "__main__":
    main()
