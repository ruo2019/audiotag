# todo list:
#   - move listens to the left side
#   - show table of mp3s picked
#   - ctrl+s to stop playing and wait for input

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
from typing import Dict, List, Tuple, Optional

# ========= curses TUI & helpers =========
import curses
import locale
locale.setlocale(locale.LC_ALL, "")

# ---------- mac lock hooks ----------
if sys.platform == "darwin":
    try:
        from Foundation import NSObject, NSDistributedNotificationCenter, NSRunLoop, NSDate
        from PyObjCTools import AppHelper
        MAC_LOCK_LISTENER_AVAILABLE = True
    except Exception:
        MAC_LOCK_LISTENER_AVAILABLE = False
else:
    MAC_LOCK_LISTENER_AVAILABLE = False

# --- Exit Policy / Globals ---
IS_SCREEN_LOCKED = threading.Event()
LOCK_EXIT_MINUTES = 30
EXIT_AT_LOCAL_HOUR = 1      # 01:30 local cutoff
EXIT_AT_LOCAL_MINUTE = 30
EXIT_NOW = threading.Event()
LOCKED_SINCE_WALL: float | None = None
_OBSERVER = None

# --- Paths / Files ---
MP3_FOLDER = "static/mp3"
TAGS_FILE = 'tags.json'
SAMPLE_MP3_FILENAME = "Deep Stone Crypt Theme.mp3"
MODEL_NAME = 'sentence-transformers/sentence-t5-base'
LISTEN_DB_FILE = "listen_counts.json"

# --- Device selection ---
if sys.platform == "darwin":
    DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

# ------------------ Loudness helpers ------------------

def get_audio_loudness(full_filepath_with_extension):
    try:
        audio = AudioSegment.from_mp3(full_filepath_with_extension)
        if audio.duration_seconds > 0:
            loudness_dbfs = audio.dBFS
            if loudness_dbfs == -math.inf:
                return -100.0
            return loudness_dbfs
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
    if target_peak_dbfs is None or current_peak_dbfs is None:
        return 0.5
    if target_peak_dbfs <= -100.0 or current_peak_dbfs <= -100.0:
        return 0.5
    db_difference = target_peak_dbfs - current_peak_dbfs
    scale_factor = 10 ** (db_difference / 20)
    scaled_volume = max(0.0, min(1.0, 0.5 * scale_factor))
    return scaled_volume

def _next_local_time(hour: int, minute: int) -> datetime:
    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target

def _maybe_exit_for_scheduled_time(next_exit_dt: datetime):
    if EXIT_NOW.is_set():
        try:
            pygame.mixer.stop()
        except Exception:
            pass
        raise SystemExit
    if datetime.now() >= next_exit_dt:
        try:
            pygame.mixer.stop()
        except Exception:
            pass
        raise SystemExit

def _mac_is_locked_poll() -> bool:
    if sys.platform != "darwin":
        return IS_SCREEN_LOCKED.is_set()
    try:
        from Quartz import CGSessionCopyCurrentDictionary  # type: ignore
        sess = CGSessionCopyCurrentDictionary() or {}
        if "CGSSessionScreenIsLocked" in sess:
            locked = bool(sess.get("CGSSessionScreenIsLocked"))
        elif "CGSSessionOnConsoleKey" in sess:
            locked = not bool(sess.get("CGSSessionOnConsoleKey"))
        else:
            locked = IS_SCREEN_LOCKED.is_set()
        if not locked and IS_SCREEN_LOCKED.is_set():
            IS_SCREEN_LOCKED.clear()
        return locked
    except Exception:
        return IS_SCREEN_LOCKED.is_set()

def _wait_while_locked_or_exit(lock_since_wall: float | None,
                               next_exit_dt: datetime,
                               tick_hz: int = 10) -> float | None:
    clk = pygame.time.Clock()
    while True:
        _maybe_exit_for_scheduled_time(next_exit_dt)
        locked_now = _mac_is_locked_poll()
        if not locked_now:
            return None
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

# -------- Listen counter persistence --------

def _db_path(base_dir: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), LISTEN_DB_FILE)

def load_listen_counts(mp3_dir: str) -> Dict[str, int]:
    path = _db_path(mp3_dir)
    data: Dict[str, int] = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    for fn in os.listdir(mp3_dir):
        if fn.lower().endswith(".mp3"):
            base = os.path.splitext(fn)[0]
            data.setdefault(base, 0)
    return data

def save_listen_counts(mp3_dir: str, counts: Dict[str, int]) -> None:
    path = _db_path(mp3_dir)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(counts, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def increment_count_for(full_path: str, counts: Dict[str, int]) -> None:
    base = os.path.splitext(os.path.basename(full_path))[0]
    counts[base] = counts.get(base, 0) + 1

def sorted_counts(counts: Dict[str, int]) -> List[Tuple[str, int]]:
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].lower()))

# ----------------- Curses TUI -----------------

class CursesTUI:
    """
    Header: Now Playing + progress
    Middle: Most Listened table (scrollable)
    Bottom: Input bar to type mood (hit Enter)
    """
    def __init__(self, enable: bool = True):
        self.enable = enable
        self.stdscr: Optional["curses._CursesWindow"] = None
        self.scroll = 0
        self.last_height = 0
        self.last_width = 0
        # theme attrs
        self.attr_header = 0
        self.attr_subtle = 0
        self.attr_table_header = 0
        self.attr_accent = 0
        # input
        self.input_buffer = ""
        self.status_msg = ""  # transient message (e.g., “queued mood ...”)
        self.skip_requested = False

    def __enter__(self):
        if not self.enable:
            return self
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.stdscr.nodelay(True)
        try:
            curses.curs_set(0)
        except Exception:
            pass
        # colors (follow terminal theme — you already added use_default_colors() in your copy;
        # keep it here too to be safe if you paste this whole file)
        try:
            if curses.has_colors():
                curses.start_color()
                curses.use_default_colors()
        except Exception:
            pass
        try:
            self.stdscr.keypad(True)
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.enable:
            return
        try:
            if self.stdscr is not None:
                self.stdscr.nodelay(False)
            curses.echo()
            curses.nocbreak()
            curses.endwin()
        except Exception:
            pass

    def _draw_line(self, y: int, x: int, text: str, attr=0):
        if not self.enable or self.stdscr is None:
            return
        h, w = self.stdscr.getmaxyx()
        if y >= h:
            return
        if len(text) > w - x:
            text = text[:max(0, w - x - 1)]
        try:
            self.stdscr.addstr(y, x, text, attr)
        except Exception:
            pass

    def _hline(self, y: int, ch="-"):
        if not self.enable or self.stdscr is None:
            return
        h, w = self.stdscr.getmaxyx()
        try:
            self.stdscr.hline(y, 0, ch, max(0, w-1))
        except Exception:
            pass

    def _handle_key(self) -> Tuple[Optional[str], int]:
        """
        Returns (submitted_text_or_None, keycode)
        """
        try:
            key = self.stdscr.getch()
        except Exception:
            key = -1
        submitted = None

        if key == -1:
            return (None, key)

        # Normalize common keys
        KEY_PGUP = getattr(curses, "KEY_PPAGE", 339)
        KEY_PGDN = getattr(curses, "KEY_NPAGE", 338)
        KEY_RESZ = getattr(curses, "KEY_RESIZE", -1)
        KEY_BS   = 127

        # Scrolling
        if key == curses.KEY_UP:
            self.scroll = max(0, self.scroll - 1)
        elif key == curses.KEY_DOWN:
            self.scroll = self.scroll + 1
        elif key == KEY_PGUP:
            self.scroll = max(0, self.scroll - max(1, self.last_height - 8))
        elif key == KEY_PGDN:
            self.scroll = self.scroll + max(1, self.last_height - 8)
        elif key == KEY_RESZ:
            pass
        elif key == curses.KEY_RIGHT:
            self.skip_requested = True
            self.status_msg = "Skipping to next in 10s..."
        # Input editing at bottom:
        elif key in (curses.KEY_BACKSPACE, KEY_BS, 8):
            if self.input_buffer:
                self.input_buffer = self.input_buffer[:-1]
        elif key in (10, 13):  # Enter
            text = self.input_buffer.strip()
            if text:
                submitted = text
                self.input_buffer = ""
        else:
            # printable?
            if 32 <= key <= 126:
                self.input_buffer += chr(key)

        return (submitted, key)

    def render(self,
               now_playing_name: str,
               target_lufs: Optional[float],
               current_lufs: Optional[float],
               loudness_diff: Optional[float],
               volume_scale: Optional[float],
               elapsed_sec: float,
               total_sec: float,
               counts: Dict[str, int],
               pending_mood: Optional[str]) -> Optional[str]:
        """
        Draw the screen and return submitted text if Enter was pressed.
        """
        if not self.enable or self.stdscr is None:
            return None

        submitted, key = self._handle_key()

        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()
        self.last_height, self.last_width = h, w

        # Header
        header = " Now Playing ".center(w, " ")
        self._draw_line(0, 0, header, self.attr_header)
        np_line = now_playing_name if now_playing_name else "(none)"
        # Show N/A when no stats available
        self._draw_line(1, 2, f"Track: {np_line}", self.attr_table_header)
        tlufs = "N/A" if target_lufs is None else f"{target_lufs:.2f}"
        clufs = "N/A" if current_lufs is None else f"{current_lufs:.2f}"
        ldiff = "N/A" if loudness_diff is None else f"{loudness_diff:.2f}"
        vscal = "N/A" if volume_scale is None else f"{volume_scale:.2f}"
        self._draw_line(2, 2, f"Target LUFS: {tlufs}", self.attr_subtle)
        self._draw_line(3, 2, f"Current LUFS: {clufs}", self.attr_subtle)
        self._draw_line(4, 2, f"Loudness Diff: {ldiff}", self.attr_subtle)
        self._draw_line(5, 2, f"Adjusted Volume: {vscal}", self.attr_subtle)

        # Progress
        bar_y = 6
        bar_width = max(10, w - 20)
        filled = 0
        if total_sec and total_sec > 0:
            filled = int(max(0.0, min(1.0, elapsed_sec / total_sec)) * bar_width)
        bar = "[" + ("-" * filled) + (" " * (bar_width - filled)) + "]"
        self._draw_line(bar_y, 2, bar)
        t1 = f" {int((elapsed_sec or 0)//60)}:{int((elapsed_sec or 0)%60):02d} / {int((total_sec or 0)//60)}:{int((total_sec or 0)%60):02d}"
        self._draw_line(bar_y, 4 + bar_width, t1, self.attr_accent)

        # Separator
        sep_y = bar_y + 1
        if sep_y < h:
            self._hline(sep_y, "-")

        # Table
        tbl_y = sep_y + 1
        self._draw_line(tbl_y, 0, " Most Listened ".center(w, " "), self.attr_header)
        self._draw_line(tbl_y + 1, 2, f"{'#':>3}  {'Track':<{max(10, w-20)}}  {'Plays':>5}", self.attr_table_header)
        entries = sorted_counts(counts)
        start_row = tbl_y + 2
        rows_available = max(0, h - start_row - 3)  # leave 2 lines for input + 1 footer line
        max_scroll = max(0, len(entries) - rows_available)
        self.scroll = min(self.scroll, max_scroll)
        for i in range(rows_available):
            idx = i + self.scroll
            if idx >= len(entries):
                break
            name, cnt = entries[idx]
            name_col_w = max(10, w - 20)
            line = f"{(idx+1):>3}  {name:<{name_col_w}}  {cnt:>5}"
            self._draw_line(start_row + i, 2, line)

        # Pending mood hint
        footer_y = start_row + rows_available
        if pending_mood:
            self._draw_line(footer_y, 2, f"Queued mood: “{pending_mood}” (applies after current track)", self.attr_subtle)
            footer_y += 1
        elif self.status_msg:
            self._draw_line(footer_y, 2, self.status_msg, self.attr_subtle)
            footer_y += 1

        # Input bar (pinned bottom-2 lines)
        prompt = "mood: "
        hint = "(type a mood and press Enter)"
        input_y = h - 2
        self._hline(input_y - 1, "-")
        if not self.input_buffer and not pending_mood and not now_playing_name:
            self._draw_line(input_y, 2, f"{prompt}", self.attr_table_header)
            self._draw_line(input_y, 2 + len(prompt), hint, self.attr_subtle)
        else:
            self._draw_line(input_y, 2, f"{prompt}{self.input_buffer}", self.attr_table_header)

        # Tiny footer
        self._draw_line(h - 1, 2, "↑/↓, PgUp/PgDn scroll • Enter to submit mood • → to switch to new mood", self.attr_subtle)

        try:
            self.stdscr.refresh()
        except Exception:
            pass

        # If user pressed Enter this frame:
        if submitted:
            # show a one-shot status for feedback
            self.status_msg = f"Queued mood: “{submitted}”"
            return submitted
        return None

# --------- Mood matching & analysis (refactored) ----------

def compute_top_for_mood(model: SentenceTransformer,
                         tags_data: Dict[str, List[str]],
                         mood: str,
                         mp3_folder: str,
                         top_n: int,
                         logging: bool = False) -> List[str]:
    """
    Return list of full paths for top-N files matching mood.
    """
    filenames_in_tags = list(tags_data.keys())
    strings_to_encode = []
    original_indices = {}
    valid_filenames_for_embedding = []
    current_index = 0

    for filename_base in filenames_in_tags:
        potential_mp3_path = os.path.join(mp3_folder, filename_base + ".mp3")
        mood_list = tags_data.get(filename_base, [])
        if os.path.isfile(potential_mp3_path) and mood_list:
            shuffled_moods = mood_list[:]
            random.shuffle(shuffled_moods)
            combined_mood_string = ", ".join(shuffled_moods)
            strings_to_encode.append(combined_mood_string)
            original_indices[current_index] = filename_base
            valid_filenames_for_embedding.append(filename_base)
            current_index += 1

    if not strings_to_encode:
        return []

    strings_to_encode.append(mood)
    target_index = len(strings_to_encode) - 1
    original_indices[target_index] = f"Target: {mood}"

    embeddings = model.encode(["music that is " + x for x in strings_to_encode], show_progress_bar=False)
    target_embedding = embeddings[target_index]
    other_embeddings, other_labels = [], []
    for i in range(len(embeddings)):
        if i != target_index:
            other_embeddings.append(embeddings[i])
            other_labels.append(original_indices[i])

    if not other_labels:
        return []

    similarities = cosine_similarity([target_embedding], other_embeddings)[0]
    results = list(zip(other_labels, similarities))
    results_sorted = sorted(results, key=lambda item: item[1], reverse=True)
    top_results = results_sorted[:top_n]
    return [os.path.join(mp3_folder, base + ".mp3") for (base, _score) in top_results]

def build_audio_data_for_playlist(playlist: List[str],
                                  target_lufs: Optional[float],
                                  mp3_folder: str,
                                  sample_file_name: str,
                                  loudness_cache: Dict[str, Optional[float]],
                                  logging: bool = False) -> Tuple[float, Dict[str, Dict[str, Optional[float]]]]:
    """
    Returns (resolved_target_loudness, audio_data dict mapping full_path -> {'loudness_dbfs', 'scale'})
    Uses a simple cache to avoid re-decoding mp3s repeatedly.
    """
    audio_data: Dict[str, Dict[str, Optional[float]]] = {}

    # Resolve target loudness
    if target_lufs is None:
        sample_filepath_full = os.path.join(mp3_folder, sample_file_name)
        if sample_filepath_full not in loudness_cache:
            loudness_cache[sample_filepath_full] = get_audio_loudness(sample_filepath_full)
        sample_loudness = loudness_cache.get(sample_filepath_full, None)
        if sample_loudness is None:
            raise RuntimeError(f"Could not analyze reference sample '{sample_file_name}'")
        resolved_target = sample_loudness
    else:
        resolved_target = target_lufs

    # Analyze each file
    for full_path in playlist:
        if full_path not in loudness_cache:
            loudness_cache[full_path] = get_audio_loudness(full_path)
        lufs = loudness_cache.get(full_path, None)
        if lufs is None:
            audio_data[full_path] = {'loudness_dbfs': None, 'scale': 0.5}
        else:
            scale = calculate_volume_scale(resolved_target, lufs)
            audio_data[full_path] = {'loudness_dbfs': lufs, 'scale': scale}

    return resolved_target, audio_data

# ----------------- Main with TUI input bar -----------------

def main(initial_mood: Optional[str],
         top_n: int,
         mp3_folder_path: str,
         tags_file_path: str,
         sample_file_name: str,
         target_lufs=None,
         logging=False,
         enable_tui=True):

    # --- Load tags & model once ---
    if not os.path.exists(tags_file_path):
        print(f"Error: Tags file not found at '{tags_file_path}'")
        sys.exit(1)
    try:
        with open(tags_file_path) as f:
            tags_data = json.load(f)
            if not isinstance(tags_data, dict) or not tags_data:
                print(f"Error: Tags data file '{tags_file_path}' is empty or invalid.")
                sys.exit(1)
            print(f"Loaded {len(tags_data)} tag data items from '{tags_file_path}'.")
    except Exception as e:
        print(f"Error loading tags: {e}")
        sys.exit(1)

    print(f"Loading sentence transformer model '{MODEL_NAME}' onto device '{DEVICE}'...")
    try:
        model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # --- init pygame ---
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
        sys.exit(1)

    # --- policy state ---
    next_exit_dt = _next_local_time(EXIT_AT_LOCAL_HOUR, EXIT_AT_LOCAL_MINUTE)
    lock_since_wall = None
    print(f"Exit policy: exit if locked for {LOCK_EXIT_MINUTES} minutes, or at {next_exit_dt.strftime('%Y-%m-%d %H:%M')} local.")

    # --- listen counters ---
    counters = load_listen_counts(mp3_folder_path)

    # --- dynamic playback state ---
    loudness_cache: Dict[str, Optional[float]] = {}
    current_playlist: Optional[List[str]] = None
    audio_data: Dict[str, Dict[str, Optional[float]]] = {}
    resolved_target_loudness: Optional[float] = None
    pending_mood: Optional[str] = None

    # Seed playlist if CLI mood provided
    if initial_mood:
        pl = compute_top_for_mood(model, tags_data, initial_mood, mp3_folder_path, top_n, logging=logging)
        if pl:
            resolved_target_loudness, audio_data = build_audio_data_for_playlist(
                pl, target_lufs, mp3_folder_path, sample_file_name, loudness_cache, logging=logging
            )
            current_playlist = pl
            print(f"Ready: top {len(pl)} files for mood “{initial_mood}”.")
        else:
            print(f"No files found for initial mood “{initial_mood}”. Waiting for input...")

    print(f"--- TUI ready. Type a mood in the bottom bar and press Enter. ---")

    with CursesTUI(enable=enable_tui) as tui:
        try:
            while True:
                _maybe_exit_for_scheduled_time(next_exit_dt)

                # If we have no playlist yet, just idle-render + collect input
                if not current_playlist:
                    # draw idle with N/A stats
                    submitted = tui.render(
                        now_playing_name="",
                        target_lufs=None,
                        current_lufs=None,
                        loudness_diff=None,
                        volume_scale=None,
                        elapsed_sec=0.0,
                        total_sec=0.0,
                        counts=counters,
                        pending_mood=pending_mood
                    )
                    if submitted:
                        # Build immediately since nothing is playing
                        try:
                            pl = compute_top_for_mood(model, tags_data, submitted, mp3_folder_path, top_n, logging=logging)
                            if pl:
                                resolved_target_loudness, audio_data = build_audio_data_for_playlist(
                                    pl, target_lufs, mp3_folder_path, sample_file_name, loudness_cache, logging=logging
                                )
                                current_playlist = pl
                                pending_mood = None
                                tui.status_msg = f"Loaded mood: “{submitted}” ({len(pl)} tracks)"
                            else:
                                tui.status_msg = f"No matches for mood: “{submitted}”"
                        except Exception as e:
                            tui.status_msg = f"Error building playlist: {e}"
                    time.sleep(0.05)
                    continue

                # We have a playlist: shuffle for the round
                round_list = current_playlist[:]
                random.shuffle(round_list)

                for full_path in round_list:
                    _maybe_exit_for_scheduled_time(next_exit_dt)

                    # Between tracks: if user queued a new mood, rebuild now
                    if pending_mood:
                        try:
                            pl = compute_top_for_mood(model, tags_data, pending_mood, mp3_folder_path, top_n, logging=logging)
                            if pl:
                                resolved_target_loudness, audio_data = build_audio_data_for_playlist(
                                    pl, target_lufs, mp3_folder_path, sample_file_name, loudness_cache, logging=logging
                                )
                                current_playlist = pl
                                tui.status_msg = f"Switched to mood: “{pending_mood}” ({len(pl)} tracks)"
                            else:
                                tui.status_msg = f"No matches for mood: “{pending_mood}”"
                            pending_mood = None
                        except Exception as e:
                            tui.status_msg = f"Error building playlist: {e}"
                            pending_mood = None

                    if full_path not in audio_data:
                        # Skip missing analysis
                        continue

                    filename_ext = os.path.basename(full_path)
                    volume_scale = audio_data[full_path].get('scale', 0.5)
                    current_loudness = audio_data[full_path].get('loudness_dbfs', None)
                    loudness_diff = (current_loudness - resolved_target_loudness) if (isinstance(current_loudness, float) and isinstance(resolved_target_loudness, float)) else None

                    try:
                        sound = pygame.mixer.Sound(full_path)
                        sound.set_volume(volume_scale)

                        # lock handling before play
                        if _mac_is_locked_poll():
                            lock_since_wall = _wait_while_locked_or_exit(lock_since_wall, next_exit_dt)
                            lock_since_wall = _sleep_with_exit_checks(10, next_exit_dt, lock_since_wall)

                        sound.play()

                        started_ts = time.monotonic()
                        try:
                            total_dur = float(sound.get_length())
                        except Exception:
                            total_dur = 0.0

                        reached_end = False
                        interrupted_by_lock = False
                        user_skip = False
                        clock = pygame.time.Clock()

                        while pygame.mixer.get_busy():
                            _maybe_exit_for_scheduled_time(next_exit_dt)

                            # collect input every frame; if Enter pressed, queue it (don’t switch yet)
                            submitted = None
                            if enable_tui:
                                elapsed = min(max(0.0, time.monotonic() - started_ts), total_dur if total_dur > 0 else 0.0)
                                submitted = tui.render(
                                    now_playing_name=filename_ext,
                                    target_lufs=resolved_target_loudness,
                                    current_lufs=current_loudness if isinstance(current_loudness, float) else None,
                                    loudness_diff=loudness_diff if isinstance(loudness_diff, float) else None,
                                    volume_scale=volume_scale,
                                    elapsed_sec=elapsed,
                                    total_sec=total_dur,
                                    counts=counters,
                                    pending_mood=pending_mood
                                )
                                if enable_tui and getattr(tui, "skip_requested", False):
                                    user_skip = True
                                    tui.skip_requested = False
                                    pygame.mixer.stop()  # stop immediately
                                    break
                            else:
                                # non-TUI fallback progress
                                if total_dur > 0.0:
                                    seg_len = 10.0
                                    total_segments = max(1, int(math.ceil(total_dur / seg_len)))
                                    elapsed = min(max(0.0, time.monotonic() - started_ts), total_dur)
                                    filled_segments = min(total_segments, int(elapsed // seg_len))
                                    bar = '[' + ('-' * filled_segments) + (' ' * (total_segments - filled_segments)) + ']'
                                    mm = int(elapsed // 60); ss = int(elapsed % 60)
                                    tmm = int(total_dur // 60); tss = int(total_dur % 60)
                                    sys.stdout.write(f'\rPlaying {filename_ext} {bar} {mm}:{ss:02d} / {tmm}:{tss:02d}\x1b[K')
                                    sys.stdout.flush()

                            if submitted:
                                pending_mood = submitted  # handled after this track ends

                            # lock mid-track?
                            if _mac_is_locked_poll():
                                interrupted_by_lock = True
                                pygame.mixer.stop()
                                lock_since_wall = _wait_while_locked_or_exit(lock_since_wall, next_exit_dt)
                                lock_since_wall = _sleep_with_exit_checks(10, next_exit_dt, lock_since_wall)
                                break

                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    pygame.mixer.stop()
                                    raise SystemExit

                            clock.tick(10)

                        # end-of-track decision
                        if not interrupted_by_lock and not user_skip and total_dur > 0:
                            elapsed_final = max(0.0, time.monotonic() - started_ts)
                            if elapsed_final >= max(0.0, total_dur - 0.75):
                                reached_end = True

                        # increment only on natural end
                        if reached_end:
                            increment_count_for(full_path, counters)
                            save_listen_counts(mp3_folder_path, counters)

                        # pacing between tracks:
                        if user_skip:
                            # honor the user-requested skip: wait 10s before next track
                            lock_since_wall = _sleep_with_exit_checks(10, next_exit_dt, lock_since_wall)
                        elif not _mac_is_locked_poll():
                            lock_since_wall = _sleep_with_exit_checks(10, next_exit_dt, lock_since_wall)

                    except pygame.error as e:
                        if enable_tui:
                            # surface the error as a transient status
                            tui.status_msg = f"Error playing {filename_ext}: {e}"
                        else:
                            print(f"Error playing {filename_ext}: {e}")
                        time.sleep(1)
                    except Exception as e:
                        if enable_tui:
                            tui.status_msg = f"Unexpected error: {e}"
                        else:
                            print(f"Unexpected error for {filename_ext}: {e}")
                        time.sleep(1)

        except SystemExit:
            pass
        except KeyboardInterrupt:
            pass

    try:
        pygame.mixer.quit()
    finally:
        pygame.quit()

# --- CLI entry point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Mood TUI player with volume normalization and listen counters.'
    )
    parser.add_argument('-mood', '--mood', type=str, required=False, default=None,
                        help='Optional initial mood (you can type a new one in the TUI bottom bar).')
    parser.add_argument('-top', '--top', type=int, default=5,
                        help='Number of top matching files to find and play.')
    parser.add_argument('--vol', type=float, default=None,
                        help='Target loudness in LUFS (e.g., -17.50). If not provided, uses the sample file loudness.')
    parser.add_argument('--log', action='store_true',
                        help='Enable detailed logging output.')
    parser.add_argument('--folder', type=str, default=MP3_FOLDER,
                        help=f'Path to the MP3 folder (default: {MP3_FOLDER}).')
    parser.add_argument('--tags', type=str, default=TAGS_FILE,
                        help=f'Path to the tags JSON file (default: {TAGS_FILE}).')
    parser.add_argument('--sample', type=str, default=SAMPLE_MP3_FILENAME,
                        help=f'Filename of the reference volume MP3 in the MP3 folder (default: {SAMPLE_MP3_FILENAME}).')
    parser.add_argument('--no-tui', action='store_true',
                        help='Disable curses TUI and use simple stdout printing.')

    args = parser.parse_args()

    try:
        import sentence_transformers
        import sklearn
        import pydub
        import pygame
        from tabulate import tabulate
    except ImportError as e:
        print(f"Error: Missing required package: {e.name}")
        print("Install: pip install sentence-transformers scikit-learn pydub pygame tabulate torch")
        sys.exit(1)

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    try:
        main(args.mood, args.top,
             mp3_folder_path=args.folder,
             tags_file_path=args.tags,
             sample_file_name=args.sample,
             target_lufs=args.vol,
             logging=args.log,
             enable_tui=(not args.no_tui))
    except SystemExit:
        print("\nPlayback stopped.")
    except KeyboardInterrupt:
        print("\nPlayback interrupted.")
