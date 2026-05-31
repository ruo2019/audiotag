import argparse
import json
import os
import sqlite3
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

DB_PATH = "database.sqlite"
MP3_FOLDER = "mp3s"
TABLE_NAME = "ranking_votes"
VOTES_TABLE = "votes"
PLAYBACK_HISTORY_TABLE = "playback_history"
META_FILENAME = ".mp3meta.json"
ANALYSIS_CACHE_FILENAME = ".mp3analysis.json"

console = None
VERBOSE = False
INTERACTIVE_MODE = False
SUCCESS_MESSAGE = "Done. Updated database and metadata."


def emit(message: str) -> None:
    if INTERACTIVE_MODE:
        return
    print_message(message, console)

def emit_detail(message: str) -> None:
    if VERBOSE:
        emit(message)


def list_mp3_files(folder: str) -> list[str]:
    path = Path(folder)
    return sorted(
        entry.name for entry in path.iterdir() if entry.is_file() and entry.suffix.lower() == ".mp3"
    )


def build_rename_choice(folder: str) -> RenameChoice:
    return RenameChoice(
        prompt=f"Choose an MP3 from ./{folder}/, then edit its filename.",
        choices=list_mp3_files(folder),
        help_text="Use arrow keys or the mouse to select. Press Enter to edit, then Enter again to apply.",
        empty_message="No MP3 files were found.",
    )


def apply_rename(folder: str, old_name: str, new_name: str) -> tuple[bool, str]:
    if old_name == new_name:
        return False, "Filename is unchanged."

    old_db_entry_full_path = os.path.join(folder, old_name)
    current_actual_full_path = os.path.join(folder, new_name)

    if not os.path.exists(old_db_entry_full_path):
        return False, f"'{old_name}' was not found in ./{folder}/."

    if os.path.exists(current_actual_full_path):
        return False, f"'{new_name}' already exists in ./{folder}/."

    file_renamed_by_this_script = False
    try:
        os.rename(old_db_entry_full_path, current_actual_full_path)
        emit_detail(f"Renamed file: '{old_name}' -> '{new_name}'.")
        file_renamed_by_this_script = True
    except OSError as exc:
        return False, f"Error renaming file: {exc}"

    db_update_counts = update_song_name_in_db(old_name, new_name)
    if db_update_counts is not None:
        json_updated_successfully = update_play_py_json_files(old_name, new_name)
        if json_updated_successfully:
            emit_detail(
                "Database rows changed: "
                f"better_song={db_update_counts['better_song']} "
                f"worse_song={db_update_counts['worse_song']} "
                f"votes={db_update_counts['votes']} "
                f"playback_history={db_update_counts['playback_history']}."
            )
            return True, f"Renamed '{old_name}' to '{new_name}'."

        db_rolled_back = update_song_name_in_db(new_name, old_name)
        if db_rolled_back is None:
            return False, "Metadata update failed and database rollback may need manual intervention."

        if file_renamed_by_this_script:
            try:
                os.rename(current_actual_full_path, old_db_entry_full_path)
            except OSError as exc:
                return False, f"Metadata update failed and file rollback failed: {exc}"

        return False, "Metadata update failed. Changes were rolled back."

    if file_renamed_by_this_script:
        try:
            os.rename(current_actual_full_path, old_db_entry_full_path)
        except OSError as exc:
            return False, f"Database update failed and file rollback failed: {exc}"

    return False, "Database update failed. Changes were rolled back."


def update_json_file(json_path, old_key, new_key, sort_keys=False):
    """Updates a key in a JSON file, preserving the value."""
    if not os.path.exists(json_path):
        emit_detail(f"  JSON file not found: {os.path.basename(json_path)}. Skipping.")
        return True

    try:
        with open(json_path, 'r') as f:
            content = f.read()
            if not content:
                emit_detail(f"  JSON file is empty: {os.path.basename(json_path)}. Skipping.")
                return True
            data = json.loads(content)

        if old_key in data:
            data[new_key] = data.pop(old_key)

            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2, sort_keys=sort_keys)
            emit_detail(f"Updated {os.path.basename(json_path)}: '{old_key}' -> '{new_key}'.")
        else:
            emit_detail(f"  No entry for '{old_key}' in {os.path.basename(json_path)}.")

        return True
    except (json.JSONDecodeError, IOError) as e:
        emit(f"Error updating {os.path.basename(json_path)}: {e}")
        return False


def update_play_py_json_files(old_name, new_name):
    """
    Updates song basenames in .mp3meta.json and .mp3analysis.json.
    """
    old_basename = os.path.splitext(old_name)[0]
    new_basename = os.path.splitext(new_name)[0]

    if old_basename == new_basename:
        emit("Metadata update skipped (same basename).")
        return True

    meta_path = os.path.join(MP3_FOLDER, META_FILENAME)
    analysis_path = os.path.join(MP3_FOLDER, ANALYSIS_CACHE_FILENAME)

    meta_success = update_json_file(meta_path, old_basename, new_basename, sort_keys=True)
    if not meta_success:
        return False

    analysis_success = update_json_file(analysis_path, old_basename, new_basename, sort_keys=False)

    return analysis_success


def update_song_name_in_db(old_name, new_name):
    """
    Updates all occurrences of old_name to new_name in the database.
    Returns per-table counts on success, None on failure.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        emit_detail(f"Updating database records for '{old_name}' -> '{new_name}'...")

        # Update ranking_votes: 'better_song'
        query_better = f"UPDATE {TABLE_NAME} SET better_song = ? WHERE better_song = ?"
        cursor.execute(query_better, (new_name, old_name))
        better_rows_affected = cursor.rowcount
        emit_detail(f"Updated {better_rows_affected} rows in '{TABLE_NAME}.better_song'.")

        # Update ranking_votes: 'worse_song'
        query_worse = f"UPDATE {TABLE_NAME} SET worse_song = ? WHERE worse_song = ?"
        cursor.execute(query_worse, (new_name, old_name))
        worse_rows_affected = cursor.rowcount
        emit_detail(f"Updated {worse_rows_affected} rows in '{TABLE_NAME}.worse_song'.")

        # Update votes: 'song_path'
        query_votes = f"UPDATE {VOTES_TABLE} SET song_path = ? WHERE song_path = ?"
        cursor.execute(query_votes, (new_name, old_name))
        votes_rows_affected = cursor.rowcount
        emit_detail(f"Updated {votes_rows_affected} rows in '{VOTES_TABLE}.song_path'.")

        # Update playback_history: 'song_path'
        query_history = f"UPDATE {PLAYBACK_HISTORY_TABLE} SET song_path = ? WHERE song_path = ?"
        cursor.execute(query_history, (new_name, old_name))
        history_rows_affected = cursor.rowcount
        emit_detail(f"Updated {history_rows_affected} rows in '{PLAYBACK_HISTORY_TABLE}.song_path'.")

        conn.commit()
        return {
            "better_song": better_rows_affected,
            "worse_song": worse_rows_affected,
            "votes": votes_rows_affected,
            "playback_history": history_rows_affected,
        }

    except sqlite3.Error as e:
        emit(f"SQLite error: {e}")
        if conn:
            conn.rollback()
            emit("Database changes rolled back.")
        return None
    finally:
        if conn:
            conn.close()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Rename an MP3 and update database, JSON metadata files."
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
    VERBOSE = args.verbose

    if not os.path.exists(DB_PATH):
        emit(
            f"Error: Database file '{DB_PATH}' not found. Please ensure it's in the same directory as the script."
        )
        sys.exit(1)

    if not os.path.isdir(MP3_FOLDER):
        emit(
            f"Error: MP3s folder './{MP3_FOLDER}/' not found. Please ensure it exists in the same directory as the script."
        )
        sys.exit(1)

    print_banner("[Rename MVA]", console)

    INTERACTIVE_MODE = True
    try:
        rename_count = run_rename_loop(
            build_rename_choice(MP3_FOLDER),
            lambda old_name, new_name: apply_rename(MP3_FOLDER, old_name, new_name),
        )
    finally:
        INTERACTIVE_MODE = False

    if rename_count == 0:
        emit("No changes made.")
        sys.exit(0)

    emit(f"{SUCCESS_MESSAGE} Applied {rename_count} rename(s).")

if __name__ == "__main__":
    main()
