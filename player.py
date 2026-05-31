import argparse
import json
import random
import os
import time
import math
import sys
import pygame
from tabulate import tabulate
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import torch
import threading
from datetime import datetime, timedelta

if sys.platform == "darwin":
    try:
        from Foundation import NSObject, NSDistributedNotificationCenter, NSRunLoop, NSDate
        from PyObjCTools import AppHelper

        MAC_LOCK_LISTENER_AVAILABLE = True
    except Exception:
        MAC_LOCK_LISTENER_AVAILABLE = False
else:
    MAC_LOCK_LISTENER_AVAILABLE = False

# --- Configuration ---
IS_SCREEN_LOCKED = threading.Event()
# --- Exit Policy ---
LOCK_EXIT_MINUTES = 30  # Exit if the screen stays locked this long (continuous)
EXIT_AT_LOCAL_HOUR = 1  # 1:30 AM local time
EXIT_AT_LOCAL_MINUTE = 30

EXIT_NOW = threading.Event()  # set from any thread to request process exit
LOCKED_SINCE_WALL: float | None = None  # wall-clock seconds since epoch when lock started
_OBSERVER = None  # keep strong ref to the NSNotification observer

MP3_FOLDER = "static/mp3"
TAGS_FILE = 'tags.json'
SAMPLE_MP3_FILENAME = "Deep Stone Crypt Theme.mp3"
MODEL_NAME = 'sentence-transformers/sentence-t5-base'

# --- Dynamically Set Device ---
if sys.platform == "darwin":
    DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'


# ---------------------

# --- Helper Functions (Volume Normalization) ---

def get_audio_loudness(full_filepath_with_extension):
    """
    Calculates the integrated loudness (approximated as dBFS) of an audio file.
    Expects the full path including the .mp3 extension.
    Note: pydub's dBFS is RMS-based; for true LUFS, pyloudnorm is preferred but not used here for consistency.
    """
    try:
        audio = AudioSegment.from_mp3(full_filepath_with_extension)
        if audio.duration_seconds > 0:
            loudness_dbfs = audio.dBFS  # Integrated loudness approximation
            if loudness_dbfs == -math.inf:
                return -100.0
            return loudness_dbfs
        else:
            return -100.0
    except CouldntDecodeError:
        print(f"Error: Could not decode file: {os.path.basename(full_filepath_with_extension)}. Skipping analysis.")
        return None
    except FileNotFoundError:
        print(f"Error: File not found during analysis: {full_filepath_with_extension}. Skipping.")
        return None
    except Exception as e:
        print(f"Error analyzing loudness for {os.path.basename(full_filepath_with_extension)}: {e}. Skipping analysis.")
        return None


def calculate_volume_scale(target_peak_dbfs, current_peak_dbfs):
    """Calculates the pygame volume scale factor (0.0 to 1.0)."""
    if target_peak_dbfs is None or current_peak_dbfs is None:
        return 0.5  # Default volume if analysis failed

    if target_peak_dbfs <= -100.0 or current_peak_dbfs <= -100.0:
        return 0.5

    db_difference = target_peak_dbfs - current_peak_dbfs
    scale_factor = 10 ** (db_difference / 20)  # Use 20 for peak normalization
    scaled_volume = max(0.0, min(1.0, 0.5 * scale_factor))
    return scaled_volume

def _next_local_time(hour: int, minute: int) -> datetime:
    """
    Return the next occurrence (today or tomorrow) at local hour:minute.
    """
    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def _maybe_exit_for_scheduled_time(next_exit_dt: datetime):
    """
    Exit immediately if we've reached the scheduled cutoff OR if the lock-exit flag is set.
    """
    if EXIT_NOW.is_set():
        print(f"\nScreen was locked for \u2265 {LOCK_EXIT_MINUTES} minutes (wall clock). Exiting.")
        try:
            pygame.mixer.stop()
        except Exception:
            pass
        raise SystemExit

    if datetime.now() >= next_exit_dt:
        print(f"\nReached scheduled exit time ({next_exit_dt.strftime('%Y-%m-%d %H:%M')} local). Exiting.")
        try:
            pygame.mixer.stop()
        except Exception:
            pass
        raise SystemExit


def _mac_is_locked_poll() -> bool:
    """Return True if the screen is locked (macOS), else False.
    Uses Quartz (CGSessionCopyCurrentDictionary) when available.
    Falls back to the IS_SCREEN_LOCKED event if Quartz or PyObjC isn't available.
    Side effect: clears IS_SCREEN_LOCKED if we detect an unlocked state via Quartz.
    """
    if sys.platform != "darwin":
        return IS_SCREEN_LOCKED.is_set()
    try:
        from Quartz import CGSessionCopyCurrentDictionary  # type: ignore
        sess = CGSessionCopyCurrentDictionary() or {}
        if "CGSSessionScreenIsLocked" in sess:
            locked = bool(sess.get("CGSSessionScreenIsLocked"))
        elif "CGSSessionOnConsoleKey" in sess:
            # OnConsoleKey==True -> user is at console -> not locked
            locked = not bool(sess.get("CGSSessionOnConsoleKey"))
        else:
            locked = IS_SCREEN_LOCKED.is_set()
        if not locked and IS_SCREEN_LOCKED.is_set():
            IS_SCREEN_LOCKED.clear()
        return locked
    except Exception:
        return _mac_is_locked_poll()

def _wait_while_locked_or_exit(lock_since_wall: float | None,
                               next_exit_dt: datetime,
                               tick_hz: int = 10) -> float | None:
    """
    Blocks while the screen is locked. If locked continuously for LOCK_EXIT_MINUTES (wall time), exit.
    Also exits if the scheduled cutoff hits while waiting.
    Returns updated lock_since_wall (None when unlocked).
    """
    clk = pygame.time.Clock()
    while True:
        _maybe_exit_for_scheduled_time(next_exit_dt)

        # Determine current lock-state. On macOS, poll Quartz as a fallback so we do not
        # get stuck if the NSDistributedNotification 'unlocked' event is missed.
        locked_now = _mac_is_locked_poll()
        if sys.platform == "darwin":
            try:
                from Quartz import CGSessionCopyCurrentDictionary  # type: ignore
                sess = CGSessionCopyCurrentDictionary() or {}
                locked_now = bool(sess.get("CGSSessionScreenIsLocked", locked_now))
            except Exception:
                pass

        if not locked_now:
            # Unlocked now; clear the timer
            return None

        # Track wall-clock duration across actual lock state
        if lock_since_wall is None:
            lock_since_wall = LOCKED_SINCE_WALL or time.time()
        else:
            if time.time() - lock_since_wall >= LOCK_EXIT_MINUTES * 60:
                EXIT_NOW.set()
                _maybe_exit_for_scheduled_time(next_exit_dt)

        clk.tick(tick_hz)

def _sleep_with_exit_checks(duration_sec: float,
                            next_exit_dt: datetime,
                            lock_since_wall: float | None) -> float | None:
    """
    Sleep in short increments while honoring the exit policies.
    Returns updated lock_since_wall (cleared if unlocked).
    """
    end = time.time() + duration_sec
    while time.time() < end:
        _maybe_exit_for_scheduled_time(next_exit_dt)

        if _mac_is_locked_poll():
            if lock_since_wall is None:
                lock_since_wall = LOCKED_SINCE_WALL or time.time()
            elif time.time() - lock_since_wall >= LOCK_EXIT_MINUTES * 60:
                EXIT_NOW.set()
                _maybe_exit_for_scheduled_time(next_exit_dt)
        else:
            lock_since_wall = None

        time.sleep(0.1)
    return lock_since_wall


# --- Main Combined Logic ---

def main(target_mood_str, top_n, mp3_folder_path, tags_file_path, sample_file_name, target_lufs=None, logging=False):
    # === Part 1: Find Top N Files Matching Mood ===
    print("--- Mood Matching Phase ---")
    if not os.path.exists(tags_file_path):
        print(f"Error: Tags file not found at '{tags_file_path}'")
        sys.exit(1)
    try:
        with open(tags_file_path) as f:
            tags_data = json.load(f)
            if not isinstance(tags_data, dict) or not tags_data:
                print(f"Error: Tags data file '{tags_file_path}' is empty or not a valid JSON object.")
                sys.exit(1)
            print(f"Loaded {len(tags_data)} tag data items from '{tags_file_path}'.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{tags_file_path}'.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading tags file '{tags_file_path}': {e}")
        sys.exit(1)

    print(f"Loading sentence transformer model '{MODEL_NAME}' onto device '{DEVICE}'...")
    try:
        model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading Sentence Transformer model: {e}")
        print("Please ensure sentence-transformers and its dependencies are installed.")
        sys.exit(1)

    strings_to_encode = []
    original_indices = {}
    filenames_in_tags = list(tags_data.keys())
    if logging:
        print("\nProcessing mood tags...")
    current_index = 0
    valid_filenames_for_embedding = []
    for filename_base in filenames_in_tags:
        potential_mp3_path = os.path.join(mp3_folder_path, filename_base + ".mp3")
        mood_list = tags_data.get(filename_base, [])
        if os.path.isfile(potential_mp3_path):
            if mood_list:
                shuffled_moods = mood_list[:]
                random.shuffle(shuffled_moods)
                combined_mood_string = ", ".join(shuffled_moods)
                strings_to_encode.append(combined_mood_string)
                original_indices[current_index] = filename_base
                valid_filenames_for_embedding.append(filename_base)
                current_index += 1
            elif logging:
                print(f"  Skipping '{filename_base}': No mood tags found (file exists).")
        elif logging:
            print(f"  Skipping '{filename_base}': Corresponding MP3 file not found at '{potential_mp3_path}'.")

    if not strings_to_encode:
        print("Error: No valid MP3 files with tags found to process.")
        sys.exit(1)

    strings_to_encode.append(target_mood_str)
    target_label = f"Target: {target_mood_str}"
    target_index = len(strings_to_encode) - 1
    original_indices[target_index] = target_label
    if logging:
        print(f"\nEncoding {len(strings_to_encode)} strings (target + {len(valid_filenames_for_embedding)} files)...")
    embeddings = model.encode(["music that is " + x for x in strings_to_encode], show_progress_bar=logging)
    print("Embeddings generated.")

    target_embedding = embeddings[target_index]
    other_embeddings = []
    other_labels = []
    for i in range(len(embeddings)):
        if i != target_index:
            other_embeddings.append(embeddings[i])
            other_labels.append(original_indices[i])
    if not other_labels:
        print("Error: No file embeddings generated to compare against target.")
        sys.exit(1)
    similarities = cosine_similarity([target_embedding], other_embeddings)[0]
    results = list(zip(other_labels, similarities))
    results_sorted = sorted(results, key=lambda item: item[1], reverse=True)
    top_results = results_sorted[:top_n]
    if not top_results:
        print(f"No files found matching the mood '{target_mood_str}'.")
        sys.exit(1)
    print("\n--- Top Mood Matches ---")
    headers = ["#", "File (Base Name)", "Similarity"]
    table_data = []
    top_files_full_paths_playback = []
    for rank, (base_filename, score) in enumerate(top_results):
        table_data.append([rank + 1, base_filename, f"{score:.4f}"])
        full_path = os.path.join(mp3_folder_path, base_filename + ".mp3")
        top_files_full_paths_playback.append(full_path)
    print(tabulate(table_data, headers=headers, tablefmt="rounded_grid"))

    # === Part 2: Volume Normalization and Playback ===
    print("\n--- Volume Normalization and Playback Phase ---")
    if not os.path.isdir(mp3_folder_path):
        print(f"Error: MP3 folder not found: '{mp3_folder_path}'")
        sys.exit(1)

    files_to_analyze = list(set(top_files_full_paths_playback))
    if target_lufs is None:
        sample_filepath_full = os.path.join(mp3_folder_path, sample_file_name)
        if not os.path.isfile(sample_filepath_full):
            print(f"Error: Sample MP3 for volume reference not found: '{sample_filepath_full}'")
            sys.exit(1)
        files_to_analyze.append(sample_filepath_full)

    print(f"\nAnalyzing {len(files_to_analyze)} audio files for loudness...")
    audio_data = {}
    for full_path in files_to_analyze:
        if not os.path.isfile(full_path):
            print(f"Warning: File path expected for analysis does not exist: '{full_path}'. Skipping.")
            continue
        filename_ext = os.path.basename(full_path)
        if logging:
            print(f"  Analyzing: {filename_ext}")
        loudness_dbfs = get_audio_loudness(full_path)
        if loudness_dbfs is not None:
            audio_data[full_path] = {'loudness_dbfs': loudness_dbfs, 'scale': 1.0}
    if not audio_data:
        print("Error: No audio files could be successfully analyzed.")
        pygame.quit()
        sys.exit(1)

    if target_lufs is None:
        sample_filepath_full = os.path.join(mp3_folder_path, sample_file_name)
        if sample_filepath_full not in audio_data:
            print(f"\nError: Sample file '{sample_file_name}' was found but could not be analyzed.")
            pygame.quit()
            sys.exit(1)
        target_loudness = audio_data[sample_filepath_full]['loudness_dbfs']
        print(f"\nReference Loudness (from {sample_file_name}): {target_loudness:.2f} LUFS")
    else:
        target_loudness = target_lufs
        print(f"\nTarget Loudness (from --vol): {target_loudness:.2f} LUFS")

    if target_loudness <= -100.0:
        print("Error: Invalid target loudness level. Cannot scale.")
        pygame.quit()
        sys.exit(1)

    print("\nCalculating volume scales for the top files:")
    for full_path in top_files_full_paths_playback:
        filename_ext = os.path.basename(full_path)
        if full_path in audio_data:
            data = audio_data[full_path]
            scale = calculate_volume_scale(target_loudness, data['loudness_dbfs'])
            audio_data[full_path]['scale'] = scale
            if logging:
                print(f"  - {filename_ext}:")
                print(f"      Loudness: {data['loudness_dbfs']:.2f} LUFS -> Scale: {scale:.3f}")
        else:
            print(
                f"  - Skipping scale calculation for {filename_ext} (analysis failed or file missing). Will use default scale 0.5.")
            audio_data[full_path] = {'loudness_dbfs': None, 'scale': 0.5}

    print("Initializing Pygame Mixer...")
    try:
        try:
            pygame.mixer.pre_init(44100, -16, 2, 2048)
            pygame.init()
            pygame.mixer.init()
        except pygame.error:
            print("Standard pygame init failed, trying frequency 22050...")
            pygame.mixer.pre_init(22050, -16, 2, 2048)
            pygame.init()
            pygame.mixer.init()
        print("Pygame initialized successfully.")
    except pygame.error as e:
        print(f"Error initializing Pygame: {e}")
        print("Ensure audio drivers are installed and configured.")
        sys.exit(1)

    # --- Exit policy state ---
    next_exit_dt = _next_local_time(EXIT_AT_LOCAL_HOUR, EXIT_AT_LOCAL_MINUTE)
    lock_since_wall = None
    print(f"\nExit policy: exit if locked for {LOCK_EXIT_MINUTES} minutes (wall clock, includes sleep), "
          f"or at {next_exit_dt.strftime('%Y-%m-%d %H:%M')} local (1:30 AM cutoff).")

    playback_list = top_files_full_paths_playback[:]
    print(f"\n--- Starting Looping Playback of Top {len(playback_list)} Mood Matches (Ctrl+C to stop) ---")
    try:
        while True:
            _maybe_exit_for_scheduled_time(next_exit_dt)
            random.shuffle(playback_list)
            for full_path in playback_list:
                _maybe_exit_for_scheduled_time(next_exit_dt)
                filename_ext = os.path.basename(full_path)
                if full_path not in audio_data:
                    print(f"\nSkipping {filename_ext}: Missing analysis data.")
                    continue
                volume_scale = audio_data[full_path].get('scale', 0.5)
                current_loudness = audio_data[full_path].get('loudness_dbfs', 'N/A')
                loudness_str = f"{current_loudness:.2f}" if isinstance(current_loudness, float) else current_loudness
                loudness_diff = -(target_loudness - current_loudness) if isinstance(current_loudness, float) else 'N/A'
                diff_str = f"{loudness_diff:.2f}" if isinstance(loudness_diff, float) else loudness_diff
                print(f"\nPlaying: {filename_ext}")
                print(f"  Target Loudness: {target_loudness:.2f} LUFS")
                print(f"  Current Song Loudness: {loudness_str} LUFS")
                print(f"  Loudness Diff: {diff_str}")
                print(f"  Adjusted volume: {volume_scale:.2f}")
                try:
                    sound = pygame.mixer.Sound(full_path)
                    sound.set_volume(volume_scale)

                    # If currently locked, wait — but honor exit policies
                    if _mac_is_locked_poll():
                        print("Screen is locked; pausing playback until unlock (or exit conditions)...")
                        lock_since_wall = _wait_while_locked_or_exit(lock_since_wall, next_exit_dt)
                        print("Unlocked — starting next MP3 in 10 seconds...")

                    sound.play()

                    # Start tracking for progress bar
                    __progress_start_ts = time.monotonic()
                    __total_duration_sec = 0.0
                    try:
                        __total_duration_sec = float(sound.get_length())
                    except Exception:
                        __total_duration_sec = 0.0

                    # While the sound is playing, break early if we lock
                    while pygame.mixer.get_busy():
                        _maybe_exit_for_scheduled_time(next_exit_dt)

                        # Update progress bar (15s ticks, single-line; no trailing newline)
                        if __total_duration_sec > 0.0:
                            __elapsed_sec = max(0.0, time.monotonic() - __progress_start_ts)
                            __elapsed_sec = min(__elapsed_sec, __total_duration_sec)

                            __segment_len = 10.0  # one tick per 15 seconds
                            __total_segments = max(1, int(math.ceil(__total_duration_sec / __segment_len)))
                            __filled_segments = min(__total_segments, int(__elapsed_sec // __segment_len))

                            __bar = '[' + ('-' * __filled_segments) + (
                                        ' ' * (__total_segments - __filled_segments)) + ']'

                            __mm = int(__elapsed_sec // 60)
                            __ss = int(__elapsed_sec % 60)
                            __tmm = int(__total_duration_sec // 60)
                            __tss = int(__total_duration_sec % 60)

                            # \r = return to line start; \x1b[K clears to end-of-line so no artifacts remain
                            sys.stdout.write(f'\r{__bar} {__mm}:{__ss:02d} / {__tmm}:{__tss:02d}\x1b[K')
                            sys.stdout.flush()

                        if _mac_is_locked_poll():
                            print("Screen locked detected — stopping playback.")
                            pygame.mixer.stop()
                            lock_since_wall = _wait_while_locked_or_exit(lock_since_wall, next_exit_dt)
                            print("Unlocked — starting next MP3 in 10 seconds...")
                            break

                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                print("\nQuit event detected. Stopping playback.")
                                pygame.mixer.stop()
                                raise SystemExit

                        pygame.time.Clock().tick(10)

                    # Optional pacing between tracks (only if not locked)
                    if not _mac_is_locked_poll():
                        lock_since_wall = _sleep_with_exit_checks(10, next_exit_dt, lock_since_wall)

                except pygame.error as e:
                    print(f"  Error playing {filename_ext}: {e}")
                    time.sleep(1)
                except Exception as e:
                    print(f"  An unexpected error occurred during playback of {filename_ext}: {e}")
                    time.sleep(1)
    except SystemExit:
        print("Playback stopped by user.")
    except KeyboardInterrupt:
        print("\nPlayback interrupted by user (Ctrl+C).")
    finally:
        print("\nPlayback finished or stopped. Cleaning up...")
        pygame.mixer.quit()
        pygame.quit()
        print("Done.")


# --- Script Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Find and play music files matching a specified mood with volume normalization.')
    parser.add_argument('-mood', '--mood', type=str, required=True,
                        help='Target mood description (e.g., "epic space battle", "calm ambient study").')
    parser.add_argument('-top', '--top', type=int, default=5,
                        help='Number of top matching files to find and play.')
    parser.add_argument('--vol', type=float, default=None,
                        help='Target loudness in LUFS (e.g., -17.50). If not provided, uses the sample file\'s loudness.')
    parser.add_argument('--log', action='store_true',
                        help='Enable detailed logging output.')
    parser.add_argument('--folder', type=str, default=MP3_FOLDER,
                        help=f'Path to the MP3 folder (default: {MP3_FOLDER}).')
    parser.add_argument('--tags', type=str, default=TAGS_FILE,
                        help=f'Path to the tags JSON file (default: {TAGS_FILE}).')
    parser.add_argument('--sample', type=str, default=SAMPLE_MP3_FILENAME,
                        help=f'Filename (with extension) of the reference volume MP3 inside the MP3 folder (default: {SAMPLE_MP3_FILENAME}).')

    args = parser.parse_args()

    try:
        import sentence_transformers
        import sklearn
        import pydub
        import pygame
        from tabulate import tabulate
    except ImportError as e:
        print(f"Error: Missing required Python package: {e.name}")
        print(
            "Please install required packages: pip install sentence-transformers scikit-learn pydub pygame tabulate torch")
        print("Note: pydub requires FFmpeg/libav. Pygame might have OS dependencies.")
        sys.exit(1)

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    main(args.mood, args.top,
         mp3_folder_path=args.folder,
         tags_file_path=args.tags,
         sample_file_name=args.sample,
         target_lufs=args.vol,
         logging=args.log)
