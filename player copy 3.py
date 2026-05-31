# todo list:
#   - move listens to the left side ✅
#   - show table of mp3s picked ✅
#   - ctrl+s to stop playing and wait for input ✅
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re  # <-- added for parsing inline directives
import sys
import threading
import time
from datetime import datetime, timedelta
from hashlib import sha256
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame
import torch
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate

try:
    from mutagen.mp3 import MP3  # optional, for fast duration metadata (no full decode)
except Exception:
    MP3 = None

# ========= curses TUI & helpers =========
import curses
import locale

locale.setlocale(locale.LC_ALL, "")

# ---------- mac lock hooks ----------
if sys.platform == "darwin":
    try:
        from Foundation import (
            NSDate,
            NSDistributedNotificationCenter,
            NSObject,
            NSRunLoop,
        )
        from PyObjCTools import AppHelper

        MAC_LOCK_LISTENER_AVAILABLE = True
    except Exception:
        MAC_LOCK_LISTENER_AVAILABLE = False
else:
    MAC_LOCK_LISTENER_AVAILABLE = False

# --- Exit Policy / Globals ---
IS_SCREEN_LOCKED = threading.Event()
LOCK_EXIT_MINUTES = 30
EXIT_AT_LOCAL_HOUR = 1  # 01:30 local cutoff
EXIT_AT_LOCAL_MINUTE = 30
EXIT_NOW = threading.Event()
LOCKED_SINCE_WALL: float | None = None
_OBSERVER = None

# --- Paths / Files ---
MP3_FOLDER = "static/mp3"
TAGS_FILE = "tags.json"
SAMPLE_MP3_FILENAME = "Deep Stone Crypt Theme.mp3"
MODEL_NAME = "sentence-transformers/sentence-t5-base"
LISTEN_DB_FILE = "listen_counts.json"

# --- Device selection ---
if sys.platform == "darwin":
    # Default to CPU on macOS for stability unless you explicitly opt in:
    #   PLAYER_DEVICE=mps   (if you want to try Metal)
    #   PLAYER_DEVICE=cuda  (if you have an eGPU that works)
    #   PLAYER_DEVICE=cpu   (default)
    _req = os.environ.get("PLAYER_DEVICE", "cpu").lower().strip()
    if _req == "mps" and torch.backends.mps.is_available():
        DEVICE = "mps"
    elif _req == "cuda" and torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
else:
    # Non-macOS: prefer CUDA if available, otherwise CPU. You can still override with PLAYER_DEVICE.
    _req = os.environ.get("PLAYER_DEVICE", "").lower().strip()
    if _req in ("cuda", "gpu") and torch.cuda.is_available():
        DEVICE = "cuda"
    elif _req == "cpu" or not torch.cuda.is_available():
        DEVICE = "cpu"
    else:
        DEVICE = "cuda"

# Tame thread counts to prevent “CPU storms” that freeze laptops
try:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    torch.set_num_threads(max(1, min(2, (os.cpu_count() or 2))))
except Exception:
    pass

# Also avoid tokenizer thread explosions
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
        print(
            f"Error: Could not decode file: {os.path.basename(full_filepath_with_extension)}. Skipping analysis."
        )
        return None
    except FileNotFoundError:
        print(
            f"Error: File not found during analysis: {full_filepath_with_extension}. Skipping."
        )
        return None
    except Exception as e:
        print(
            f"Error analyzing loudness for {os.path.basename(full_filepath_with_extension)}: {e}. Skipping analysis."
        )
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


def _wait_while_locked_or_exit(
    lock_since_wall: float | None, next_exit_dt: datetime, tick_hz: int = 10
) -> float | None:
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


def _sleep_with_exit_checks(
    duration_sec: float, next_exit_dt: datetime, lock_since_wall: float | None
) -> float | None:
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
    Middle: Toggleable table (Most Listened  <TAB>  Similar to Mood) with rounded-grid lines
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
        self.stop_requested = False
        # views
        self.view_mode: str = "most"  # "most" or "similar"

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
            text = text[: max(0, w - x - 1)]
        try:
            self.stdscr.addstr(y, x, text, attr)
        except Exception:
            pass

    def _hline(self, y: int, ch="-"):
        if not self.enable or self.stdscr is None:
            return
        h, w = self.stdscr.getmaxyx()
        try:
            ch_val = ch if isinstance(ch, int) else ord(ch)
            self.stdscr.hline(y, 0, ch_val, max(0, w - 1))
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
        KEY_BS = 127
        KEY_BTAB = getattr(curses, "KEY_BTAB", 353)

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
        # Toggle view on TAB / BACKTAB
        elif key in (9, KEY_BTAB):
            self.view_mode = "similar" if self.view_mode == "most" else "most"
            self.scroll = 0
            self.status_msg = f"View: {'Similar to Mood' if self.view_mode == 'similar' else 'Most Listened'}"
        elif key == curses.KEY_RIGHT:
            self.skip_requested = True
            self.status_msg = "Skipping to next in 10s..."
        elif key == 7:  # Ctrl+G
            self.stop_requested = True
            self.status_msg = "Stopped. Waiting for mood..."
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

    # ---------- Rounded-grid table renderer (like tabulate's rounded_grid) ----------
    def _render_grid_table(
        self,
        grid_top_y: int,
        headers: Tuple[str, str, str],
        rows: List[Tuple[str, str, str]],
        numeric_is_float: bool,
        listens_first: bool = False,
        numeric_position: Optional[str] = None,
    ) -> int:
        """
        Draws a rounded-grid table starting at grid_top_y (no title line here).
        Returns the next y after the table.

        Column order options (numeric = Listens/Sim/etc.):
          - "right"  (default / legacy when listens_first=False): [ # | Track  | Numeric ]
          - "left"   (legacy when listens_first=True):            [ Numeric | #     | Track  ]
          - "middle":                                             [ #     | Numeric | Track  ]
        """
        if not self.enable or self.stdscr is None:
            return grid_top_y

        # Resolve desired numeric column position (back-compat with listens_first)
        if numeric_position is None:
            numeric_position = "left" if listens_first else "right"
        else:
            numeric_position = str(numeric_position).lower().strip()
            if numeric_position not in ("left", "middle", "right"):
                numeric_position = "right"

        h, w = self.stdscr.getmaxyx()
        indent = 2
        max_grid_w = max(20, w - indent - 2)  # keep a right margin

        # Column widths
        total_rows = len(rows)
        idx_w = max(3, len(str(total_rows)))  # width for index column
        if numeric_is_float:
            num_w = 7  # e.g., "1.0000"
        else:
            max_num_len = 1
            for _i, _name, num in rows:
                max_num_len = max(max_num_len, len(num))
            num_w = max(5, max_num_len)

        # track width fills the rest
        track_w = max(
            10, max_grid_w - (idx_w + num_w + 8)
        )  # 8 = borders+seps and padding space
        if track_w < 5:
            track_w = 5

        # Determine how many rows fit vertically:
        lines_avail = h - grid_top_y - 3  # leave 2 lines for input and 1 footer
        if lines_avail < 4:
            return grid_top_y
        max_visible_rows = (lines_avail - 3) // 2
        max_visible_rows = max(1, max_visible_rows)
        max_scroll = max(0, total_rows - max_visible_rows)
        self.scroll = min(self.scroll, max_scroll)
        visible_rows = rows[self.scroll : self.scroll + max_visible_rows]

        # Box-drawing characters
        TL, TR, BL, BR = "╭", "╮", "╰", "╯"
        H, V = "─", "│"
        TJ, MJ, LJ, RJ, BJ = "┬", "┼", "├", "┤", "┴"

        # prebuild fragments
        h_idx = H * (idx_w + 2)
        h_track = H * (track_w + 2)
        h_num = H * (num_w + 2)

        # header labels (in semantic order)
        h_idx_lbl, h_track_lbl, h_num_lbl = headers

        if numeric_position == "right":
            # [ # | Track | Numeric ]
            self._draw_line(
                grid_top_y, indent, f"{TL}{h_idx}{TJ}{h_track}{TJ}{h_num}{TR}"
            )
            hdr = f"{V} {h_idx_lbl.center(idx_w)} {V} {h_track_lbl.ljust(track_w)} {V} {h_num_lbl.center(num_w)} {V}"
            self._draw_line(grid_top_y + 1, indent, hdr)
            self._draw_line(
                grid_top_y + 2, indent, f"{LJ}{h_idx}{MJ}{h_track}{MJ}{h_num}{RJ}"
            )

            y = grid_top_y + 3
            for i, (idx_str, name, num_str) in enumerate(visible_rows):
                row = f"{V} {idx_str.rjust(idx_w)} {V} {name.ljust(track_w)[:track_w]} {V} {num_str.rjust(num_w)} {V}"
                self._draw_line(y, indent, row)
                y += 1
                if i != len(visible_rows) - 1:
                    self._draw_line(
                        y, indent, f"{LJ}{h_idx}{MJ}{h_track}{MJ}{h_num}{RJ}"
                    )
                    y += 1
            self._draw_line(y, indent, f"{BL}{h_idx}{BJ}{h_track}{BJ}{h_num}{BR}")
            return y + 1

        if numeric_position == "left":
            # [ Numeric | # | Track ]
            self._draw_line(
                grid_top_y, indent, f"{TL}{h_num}{TJ}{h_idx}{TJ}{h_track}{TR}"
            )
            hdr = f"{V} {h_num_lbl.center(num_w)} {V} {h_idx_lbl.center(idx_w)} {V} {h_track_lbl.ljust(track_w)} {V}"
            self._draw_line(grid_top_y + 1, indent, hdr)
            self._draw_line(
                grid_top_y + 2, indent, f"{LJ}{h_num}{MJ}{h_idx}{MJ}{h_track}{RJ}"
            )

            y = grid_top_y + 3
            for i, (idx_str, name, num_str) in enumerate(visible_rows):
                row = f"{V} {num_str.rjust(num_w)} {V} {idx_str.rjust(idx_w)} {V} {name.ljust(track_w)[:track_w]} {V}"
                self._draw_line(y, indent, row)
                y += 1
                if i != len(visible_rows) - 1:
                    self._draw_line(
                        y, indent, f"{LJ}{h_num}{MJ}{h_idx}{MJ}{h_track}{RJ}"
                    )
                    y += 1
            self._draw_line(y, indent, f"{BL}{h_num}{BJ}{h_idx}{BJ}{h_track}{BR}")
            return y + 1

        # numeric_position == "middle"
        # [ # | Numeric | Track ]
        self._draw_line(grid_top_y, indent, f"{TL}{h_idx}{TJ}{h_num}{TJ}{h_track}{TR}")
        hdr = f"{V} {h_idx_lbl.center(idx_w)} {V} {h_num_lbl.center(num_w)} {V} {h_track_lbl.ljust(track_w)} {V}"
        self._draw_line(grid_top_y + 1, indent, hdr)
        self._draw_line(
            grid_top_y + 2, indent, f"{LJ}{h_idx}{MJ}{h_num}{MJ}{h_track}{RJ}"
        )

        y = grid_top_y + 3
        for i, (idx_str, name, num_str) in enumerate(visible_rows):
            row = f"{V} {idx_str.rjust(idx_w)} {V} {num_str.rjust(num_w)} {V} {name.ljust(track_w)[:track_w]} {V}"
            self._draw_line(y, indent, row)
            y += 1
            if i != len(visible_rows) - 1:
                self._draw_line(y, indent, f"{LJ}{h_idx}{MJ}{h_num}{MJ}{h_track}{RJ}")
                y += 1
        self._draw_line(y, indent, f"{BL}{h_idx}{BJ}{h_num}{BJ}{h_track}{BR}")
        return y + 1

    def _render_table_most(self, top_y: int, counts: Dict[str, int]) -> int:
        """Render the Most Listened table with rounded-grid lines. Returns the next y after the table."""
        if not self.enable or self.stdscr is None:
            return top_y
        h, w = self.stdscr.getmaxyx()
        # Title above the table
        self._draw_line(top_y, 0, " Most Listened ".center(w, " "), self.attr_header)

        # Prepare rows
        entries = sorted_counts(counts)
        rows: List[Tuple[str, str, str]] = []
        for idx, (name, cnt) in enumerate(entries, start=1):
            rows.append((str(idx), name, str(cnt)))

        # Draw table starting one line below title
        grid_top = top_y + 1
        next_y = self._render_grid_table(
            grid_top,
            headers=("#", "Track", "Plays"),
            rows=rows,
            numeric_is_float=False,
            numeric_position="middle",  # <— put numeric column in the middle
        )
        return next_y

    def _render_table_similar(
        self,
        top_y: int,
        similar_entries: Optional[List[Tuple[str, float]]],
        mood: Optional[str],
    ) -> int:
        """Render the Similar-to-Mood table with rounded-grid lines. Returns the next y after the table."""
        if not self.enable or self.stdscr is None:
            return top_y
        h, w = self.stdscr.getmaxyx()
        header_title = f" Similar to “{mood}” " if mood else " Similar (no mood yet) "
        self._draw_line(top_y, 0, header_title.center(w, " "), self.attr_header)

        entries = similar_entries or []
        rows: List[Tuple[str, str, str]] = []
        for idx, (base, score) in enumerate(entries, start=1):
            rows.append((str(idx), base, f"{score:.4f}"))

        grid_top = top_y + 1
        next_y = self._render_grid_table(
            grid_top, headers=("#", "Track", "Sim"), rows=rows, numeric_is_float=True
        )
        return next_y

    def render(
        self,
        now_playing_name: str,
        target_lufs: Optional[float],
        current_lufs: Optional[float],
        loudness_diff: Optional[float],
        volume_scale: Optional[float],
        elapsed_sec: float,
        total_sec: float,
        counts: Dict[str, int],
        similar_entries: Optional[List[Tuple[str, float]]],
        similar_mood: Optional[str],
        pending_mood: Optional[str],
    ) -> Optional[str]:
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
        t1 = f" {int((elapsed_sec or 0) // 60)}:{int((elapsed_sec or 0) % 60):02d} / {int((total_sec or 0) // 60)}:{int((total_sec or 0) % 60):02d}"
        self._draw_line(bar_y, 4 + bar_width, t1, self.attr_accent)

        # Separator
        sep_y = bar_y + 1
        if sep_y < h:
            self._hline(sep_y, "-")

        # Table area (toggle)
        tbl_y = sep_y + 1
        if self.view_mode == "similar":
            next_y = self._render_table_similar(tbl_y, similar_entries, similar_mood)
        else:
            next_y = self._render_table_most(tbl_y, counts)

        # Pending mood hint / status
        footer_y = next_y
        if pending_mood:
            self._draw_line(
                footer_y,
                2,
                f"Queued mood: “{pending_mood}” (applies after current track)",
                self.attr_subtle,
            )
            footer_y += 1
        elif self.status_msg:
            self._draw_line(footer_y, 2, self.status_msg, self.attr_subtle)
            footer_y += 1

        # Input bar (pinned bottom-2 lines)
        prompt = "mood: "
        hint = "(type a mood and press Enter; you can also add (top N))"  # <-- updated hint
        input_y = h - 2
        self._hline(input_y - 1, "-")
        if not self.input_buffer and not pending_mood and not now_playing_name:
            self._draw_line(input_y, 2, f"{prompt}", self.attr_table_header)
            self._draw_line(input_y, 2 + len(prompt), hint, self.attr_subtle)
        else:
            self._draw_line(
                input_y, 2, f"{prompt}{self.input_buffer}", self.attr_table_header
            )

        # Tiny footer
        self._draw_line(
            h - 1,
            2,
            "↑/↓ PgUp/PgDn scroll • Enter submit mood • TAB toggle views • → skip • Ctrl+G stop • (top N) change size",
            self.attr_subtle,
        )

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

# We cache per-track embeddings on disk so we don't re-encode the whole library every query.
EMB_CACHE_VERSION = "v2"


def _emb_cache_path(mp3_folder: str) -> str:
    # Hidden cache file inside the music folder (keeps caches per library)
    return os.path.join(mp3_folder, ".track_emb_cache.npz")


def _canonicalize_tags(tags):
    # deterministic, lowercase, deduped and sorted
    return ", ".join(
        sorted({str(t).strip().lower() for t in (tags or []) if str(t).strip()})
    )


def _tags_fingerprint(tags_data: dict, mp3_folder: str) -> str:
    # create a stable fingerprint so cache invalidates if tags or files change
    rows = []
    for base, tag_list in tags_data.items():
        full = os.path.join(mp3_folder, base + ".mp3")
        if os.path.isfile(full):
            try:
                mtime = os.path.getmtime(full)
            except Exception:
                mtime = 0.0
            rows.append(f"{base}|{_canonicalize_tags(tag_list)}|{mtime:.0f}")
    rows.sort()
    return sha256("\n".join(rows).encode("utf-8")).hexdigest()


# Module-level caches used by compute_top_for_mood
_EMB_NAMES: list[str] = []  # list of filename bases (no extension)
_EMB_MATRIX: np.ndarray | None = None  # shape (N, D), L2-normalized
_EMB_INDEX: dict[str, int] = {}  # base -> row index
_EMB_MODEL_NAME: str | None = None
_EMB_FINGERPRINT: str | None = None


def ensure_embeddings(model, tags_data: dict, mp3_folder: str, logging: bool = False):
    """
    Ensures the global embedding cache is built/loaded.
    Safe to call multiple times; it will no-op if the cache matches.
    """
    global _EMB_NAMES, _EMB_MATRIX, _EMB_INDEX, _EMB_MODEL_NAME, _EMB_FINGERPRINT

    fp_now = _tags_fingerprint(tags_data, mp3_folder)
    cache_file = _emb_cache_path(mp3_folder)

    # Try loading an existing cache
    if os.path.exists(cache_file) and _EMB_MATRIX is None:  # only load once per process
        try:
            blob = np.load(cache_file, allow_pickle=True)
            if (
                str(blob.get("version", "")) == EMB_CACHE_VERSION
                and str(blob.get("model_name", ""))
                == getattr(model, "model_card", None)
                or True
                and str(blob.get("fingerprint", "")) == fp_now
            ):
                names = list(blob["names"])
                embs = blob["embeddings"]
                # Basic sanity checks
                if (
                    isinstance(names, list)
                    and isinstance(embs, np.ndarray)
                    and embs.ndim == 2
                ):
                    _EMB_NAMES = names
                    _EMB_MATRIX = embs
                    _EMB_INDEX = {b: i for i, b in enumerate(_EMB_NAMES)}
                    _EMB_MODEL_NAME = str(blob.get("model_name", ""))
                    _EMB_FINGERPRINT = str(blob.get("fingerprint", ""))
                    if logging:
                        print(
                            f"[emb] loaded {_EMB_MATRIX.shape[0]} track embeddings from cache."
                        )
                    return
        except Exception as e:
            if logging:
                print(f"[emb] cache load failed, will rebuild: {e}")

    # Build inputs
    names, texts = [], []
    for base, tag_list in tags_data.items():
        full = os.path.join(mp3_folder, base + ".mp3")
        if os.path.isfile(full) and tag_list:
            names.append(base)
            texts.append("music that is " + _canonicalize_tags(tag_list))
    if not names:
        _EMB_NAMES, _EMB_MATRIX, _EMB_INDEX = [], None, {}
        return

    # Encode in small batches to keep memory safe on laptops
    if logging:
        print(f"[emb] encoding {len(texts)} tracks...")
    # NOTE: We L2-normalize so cosine similarity is a simple dot product
    emb = model.encode(
        texts, batch_size=8, show_progress_bar=logging, convert_to_numpy=True
    )
    emb = emb.astype(np.float32, copy=False)
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    emb = emb / norms

    _EMB_NAMES = names
    _EMB_MATRIX = emb
    _EMB_INDEX = {b: i for i, b in enumerate(_EMB_NAMES)}
    _EMB_MODEL_NAME = (
        getattr(model, "model_card", None)
        or getattr(model, "model_name", None)
        or "unknown"
    )
    _EMB_FINGERPRINT = fp_now

    # Save to disk
    try:
        np.savez(
            cache_file,
            version=EMB_CACHE_VERSION,
            model_name=_EMB_MODEL_NAME,
            fingerprint=_EMB_FINGERPRINT,
            names=np.array(_EMB_NAMES, dtype=object),
            embeddings=_EMB_MATRIX,
        )
        if logging:
            print(f"[emb] cache saved to {cache_file}")
    except Exception as e:
        if logging:
            print(f"[emb] failed to save cache: {e}")


def compute_top_for_mood(
    model: SentenceTransformer,
    tags_data: Dict[str, List[str]],
    mood: str,
    mp3_folder: str,
    top_n: int,
    logging: bool = False,
) -> Tuple[List[str], List[Tuple[str, float]]]:
    """
    FAST VERSION:
      - Uses precomputed per-track embeddings (ensure_embeddings must have run).
      - Encodes only the user's mood once, then does a single matrix-vector similarity.
      - Prefers tracks with more exact tag matches to any comma-separated tokens in the mood;
        within the same match-count group, sort by cosine similarity (desc).
    Returns:
      playlist_paths: [full_path, ...]
      similar_table:  [(filename_base, cosine_sim), ...]
    """
    if not mood or not isinstance(mood, str):
        return [], []

    # Make sure we have the cache ready (safe to call multiple times)
    ensure_embeddings(model, tags_data, mp3_folder, logging=logging)

    # If there are no embeddings (e.g., no valid files), bail early
    if _EMB_MATRIX is None or not len(_EMB_NAMES):
        return [], []

    # Tokenize mood on commas for exact-match counting (case-insensitive)
    tokens = [tok.strip().lower() for tok in mood.split(",") if tok.strip()]
    token_set = set(tokens)

    # Encode query once (L2 normalize)
    q = model.encode(
        ["music that is " + mood],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
    )[0]
    q = q.astype(np.float32, copy=False)
    q /= np.linalg.norm(q) + 1e-12

    # Cosine similarity by dot product (since rows are normalized)
    sims = _EMB_MATRIX @ q  # shape (N,)

    # Exact-match counts per track: how many mood tokens exactly equal a tag for this track
    match_counts = []
    for base in _EMB_NAMES:
        tags = [str(t).strip().lower() for t in (tags_data.get(base, []) or [])]
        tag_set = set(tags)
        if token_set:
            cnt = sum(1 for tok in token_set if tok in tag_set)
        else:
            cnt = 0  # no comma tokens -> no grouping; pure similarity
        match_counts.append(cnt)

    # Rank: more exact matches first, then cosine similarity desc, then by name for stability
    idxs = list(range(len(_EMB_NAMES)))
    idxs.sort(key=lambda i: (-match_counts[i], -float(sims[i]), _EMB_NAMES[i].lower()))

    # Clamp top-N
    top_n = max(1, min(int(top_n or 1), 50))
    idxs = idxs[:top_n]

    # Build outputs
    playlist = [os.path.join(mp3_folder, _EMB_NAMES[i] + ".mp3") for i in idxs]
    table = [(_EMB_NAMES[i], float(sims[i])) for i in idxs]

    if logging:
        if tokens:
            print(
                f"[emb] query tokens={tokens} -> {len(playlist)} results (top {top_n})"
            )
        else:
            print(f"[emb] query='{mood}' -> {len(playlist)} results (top {top_n})")
    return playlist, table


LOUDNESS_CACHE_VERSION = 1


def _loud_cache_path(mp3_folder: str) -> str:
    # hidden JSON file in the music folder
    return os.path.join(mp3_folder, ".loudness_cache.json")


def load_loudness_cache(mp3_folder: str) -> Dict[str, dict]:
    path = _loud_cache_path(mp3_folder)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or data.get("_v") != LOUDNESS_CACHE_VERSION:
            return {"_v": LOUDNESS_CACHE_VERSION}
        return data
    except Exception:
        return {"_v": LOUDNESS_CACHE_VERSION}


def save_loudness_cache(cache: Dict[str, dict], mp3_folder: str) -> None:
    cache = dict(cache or {})
    cache["_v"] = LOUDNESS_CACHE_VERSION
    path = _loud_cache_path(mp3_folder)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        os.replace(tmp, path)
    except Exception:
        # non-fatal
        pass


def _probe_duration_seconds(full_path: str) -> float:
    # Prefer mutagen if available; it reads headers and avoids full decode
    if MP3 is not None:
        try:
            return float(MP3(full_path).info.length)
        except Exception:
            pass
    # Fallback: 0.0; (we avoid pydub decode here to keep it fast—your existing loudness code
    # will decode when *needed* and can record the duration at that time.)
    return 0.0


def get_audio_stats(full_filepath_with_extension: str) -> dict:
    """
    Returns:
      {"loudness_dbfs": float|None, "duration": float, "mtime": float}
    Uses pydub to compute loudness (one-time per file when uncached).
    """
    try:
        audio = AudioSegment.from_mp3(full_filepath_with_extension)
        dur = (
            float(audio.duration_seconds)
            if audio.duration_seconds and audio.duration_seconds > 0
            else 0.0
        )
        loud = float(audio.dBFS)
        if loud == -math.inf:
            loud = -100.0
        return {
            "loudness_dbfs": loud,
            "duration": dur,
            "mtime": os.path.getmtime(full_filepath_with_extension),
        }
    except CouldntDecodeError:
        return {
            "loudness_dbfs": None,
            "duration": 0.0,
            "mtime": os.path.getmtime(full_filepath_with_extension),
        }
    except Exception:
        return {
            "loudness_dbfs": None,
            "duration": 0.0,
            "mtime": os.path.getmtime(full_filepath_with_extension),
        }


def build_audio_data_for_playlist(
    playlist: List[str],
    target_lufs: Optional[float],
    mp3_folder: str,
    sample_file_name: str,
    loudness_cache: Dict[str, dict],
    logging: bool = False,
) -> Tuple[float, Dict[str, Dict[str, Optional[float]]]]:
    """
    Returns (resolved_target_loudness, audio_data dict mapping full_path -> {'loudness_dbfs', 'scale', 'duration'})
    Uses a persistent cache to avoid re-decoding mp3s repeatedly.
    """
    audio_data: Dict[str, Dict[str, Optional[float]]] = {}

    # Resolve target loudness (use sample if not provided)
    if target_lufs is None:
        sample_filepath_full = os.path.join(mp3_folder, sample_file_name)
        # Cache key for sample
        try:
            mtime = os.path.getmtime(sample_filepath_full)
        except Exception:
            mtime = 0.0
        entry = loudness_cache.get(sample_filepath_full)
        if not entry or abs(entry.get("mtime", 0.0) - mtime) > 0.5:
            entry = get_audio_stats(sample_filepath_full)
            loudness_cache[sample_filepath_full] = entry
        sample_loudness = entry.get("loudness_dbfs", None)
        if sample_loudness is None:
            raise RuntimeError(
                f"Could not analyze reference sample '{sample_file_name}'"
            )
        resolved_target = float(sample_loudness)
    else:
        resolved_target = float(target_lufs)

    # Analyze each file in the playlist
    for full_path in playlist:
        try:
            mtime = os.path.getmtime(full_path)
        except Exception:
            mtime = 0.0

        entry = loudness_cache.get(full_path)
        # cache miss or stale -> recompute
        if not entry or abs(entry.get("mtime", 0.0) - mtime) > 0.5:
            entry = get_audio_stats(full_path)
            loudness_cache[full_path] = entry

        lufs = entry.get("loudness_dbfs", None)
        dur = float(entry.get("duration", 0.0) or 0.0)

        if lufs is None:
            audio_data[full_path] = {
                "loudness_dbfs": None,
                "scale": 0.5,
                "duration": dur,
            }
        else:
            scale = calculate_volume_scale(resolved_target, float(lufs))
            audio_data[full_path] = {
                "loudness_dbfs": float(lufs),
                "scale": float(scale),
                "duration": dur,
            }

    if logging:
        print(
            f"[loud] target={resolved_target:.2f} dBFS; analyzed {len(playlist)} files (cached={sum(1 for p in playlist if p in loudness_cache)})."
        )

    return resolved_target, audio_data


# ----------------- Inline command parsing -----------------

TOP_DIR_RE = re.compile(
    r"\(\s*top\s*[:=]?\s*(\d+)\s*\)|\btop\s*[:=]?\s*(\d+)\b", re.IGNORECASE
)
TOP_MIN, TOP_MAX = 1, 50  # clamp range; tweak as you like


def parse_mood_and_directives(raw: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Extracts an inline (top N) / top N / top=N / top: N directive (case-insensitive),
    returns (mood_text_or_None, top_n_or_None). Removes the directive from the mood.
    """
    if not raw:
        return None, None
    top_val = None
    for m in TOP_DIR_RE.finditer(raw):
        num = m.group(1) or m.group(2)
        if num is not None:
            try:
                top_val = int(num)
            except ValueError:
                pass
    # Remove all directive instances
    mood = TOP_DIR_RE.sub("", raw).strip()
    mood = mood.strip(" ,;:()[]{}-")
    if mood == "":
        mood = None
    return mood, top_val


# ----------------- Main with TUI input bar -----------------


def main(
    initial_mood: Optional[str],
    top_n: int,
    mp3_folder_path: str,
    tags_file_path: str,
    sample_file_name: str,
    target_lufs=None,
    logging=False,
    enable_tui=True,
):
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

    print(
        f"Loading sentence transformer model '{MODEL_NAME}' onto device '{DEVICE}'..."
    )
    try:
        model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    ensure_embeddings(model, tags_data, mp3_folder_path, logging=logging)

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
    print(
        f"Exit policy: exit if locked for {LOCK_EXIT_MINUTES} minutes, or at {next_exit_dt.strftime('%Y-%m-%d %H:%M')} local."
    )

    # --- listen counters ---
    counters = load_listen_counts(mp3_folder_path)

    # --- dynamic playback + UI state ---
    loudness_cache: Dict[str, dict] = load_loudness_cache(mp3_folder_path)
    current_playlist: Optional[List[str]] = None
    audio_data: Dict[str, Dict[str, Optional[float]]] = {}
    resolved_target_loudness: Optional[float] = None
    pending_mood: Optional[str] = None

    # Table state
    current_similar_entries: Optional[List[Tuple[str, float]]] = None
    current_similar_mood: Optional[str] = None
    start_in_similar_view = False

    # NEW: runtime-adjustable top
    current_top_n = top_n

    # Seed from CLI mood if provided
    if initial_mood:
        pl, sim_entries = compute_top_for_mood(
            model,
            tags_data,
            initial_mood,
            mp3_folder_path,
            current_top_n,
            logging=logging,
        )
        if pl:
            resolved_target_loudness, audio_data = build_audio_data_for_playlist(
                pl,
                target_lufs,
                mp3_folder_path,
                sample_file_name,
                loudness_cache,
                logging=logging,
            )
            current_playlist = pl
            current_similar_entries = sim_entries
            current_similar_mood = initial_mood
            start_in_similar_view = True
            if not enable_tui:
                # mimic player.py table print when not in TUI
                headers = ["#", "File (Base Name)", "Similarity"]
                table_data = [
                    [i + 1, base, f"{score:.4f}"]
                    for i, (base, score) in enumerate(sim_entries)
                ]
                print("\n--- Top Mood Matches ---")
                print(tabulate(table_data, headers=headers, tablefmt="rounded_grid"))
            print(f"Ready: top {len(pl)} files for mood “{initial_mood}”.")
        else:
            print(
                f"No files found for initial mood “{initial_mood}”. Waiting for input..."
            )

    print(f"--- TUI ready. Type a mood and press Enter. You can add (top N). ---")

    with CursesTUI(enable=enable_tui) as tui:
        # if seeded with a mood, start in the Similar view so the table is visible right away
        if enable_tui and start_in_similar_view:
            tui.view_mode = "similar"
            tui.status_msg = "View: Similar to Mood (TAB to toggle)"

        def _pause_silence(seconds: float) -> tuple[bool, float | None]:
            """
            Pause with *no audio* while keeping the TUI responsive.
            Returns (stopped, updated_lock_since_wall).
            """
            nonlocal lock_since_wall, pending_mood, current_top_n, current_playlist
            nonlocal current_similar_entries, current_similar_mood

            if seconds <= 0:
                return (False, lock_since_wall)

            if not enable_tui:
                lock_since_wall = _sleep_with_exit_checks(
                    seconds, next_exit_dt, lock_since_wall
                )
                return (False, lock_since_wall)

            start = time.monotonic()
            end = start + float(seconds)
            clock = pygame.time.Clock()

            while True:
                _maybe_exit_for_scheduled_time(next_exit_dt)

                now = time.monotonic()
                if now >= end:
                    break
                remaining = max(0.0, end - now)

                if _mac_is_locked_poll():
                    if lock_since_wall is None:
                        lock_since_wall = LOCKED_SINCE_WALL or time.time()
                    elif time.time() - lock_since_wall >= LOCK_EXIT_MINUTES * 60:
                        EXIT_NOW.set()
                        _maybe_exit_for_scheduled_time(next_exit_dt)
                else:
                    lock_since_wall = None

                submitted = tui.render(
                    now_playing_name="",
                    target_lufs=None,
                    current_lufs=None,
                    loudness_diff=None,
                    volume_scale=None,
                    elapsed_sec=max(0.0, float(seconds) - remaining),
                    total_sec=float(seconds),
                    counts=counters,
                    similar_entries=current_similar_entries,
                    similar_mood=current_similar_mood,
                    pending_mood=pending_mood,
                )

                if submitted:
                    mood_text, new_top = parse_mood_and_directives(submitted)
                    if new_top is not None:
                        current_top_n = max(TOP_MIN, min(TOP_MAX, new_top))
                        tui.status_msg = f"Top set to {current_top_n}"
                    if mood_text:
                        pending_mood = mood_text
                    elif new_top is not None and current_similar_mood:
                        pending_mood = current_similar_mood

                if tui.skip_requested:
                    tui.skip_requested = False
                    break

                if tui.stop_requested:
                    tui.stop_requested = False
                    try:
                        pygame.mixer.stop()
                    except Exception:
                        pass
                    current_playlist = None
                    current_similar_entries = None
                    current_similar_mood = None
                    pending_mood = None
                    tui.view_mode = "most"
                    tui.status_msg = "Stopped. Waiting for mood..."
                    return (True, lock_since_wall)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.mixer.stop()
                        raise SystemExit

                clock.tick(20)

            return (False, lock_since_wall)

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
                        similar_entries=current_similar_entries,
                        similar_mood=current_similar_mood,
                        pending_mood=pending_mood,
                    )
                    if submitted:
                        mood_text, new_top = parse_mood_and_directives(submitted)
                        if new_top is not None:
                            clamped = max(TOP_MIN, min(TOP_MAX, new_top))
                            current_top_n = clamped
                            tui.status_msg = f"Top set to {current_top_n}"
                        # Build immediately since nothing is playing
                        if mood_text:
                            try:
                                pl, sim_entries = compute_top_for_mood(
                                    model,
                                    tags_data,
                                    mood_text,
                                    mp3_folder_path,
                                    current_top_n,
                                    logging=logging,
                                )
                                if pl:
                                    resolved_target_loudness, audio_data = (
                                        build_audio_data_for_playlist(
                                            pl,
                                            target_lufs,
                                            mp3_folder_path,
                                            sample_file_name,
                                            loudness_cache,
                                            logging=logging,
                                        )
                                    )
                                    current_playlist = pl
                                    current_similar_entries = sim_entries
                                    current_similar_mood = mood_text
                                    pending_mood = None
                                    # show the Similar table immediately after entering mood
                                    tui.view_mode = "similar"
                                    tui.status_msg = f"Loaded mood: “{mood_text}” (top {current_top_n}) — TAB toggles views"
                                    if not enable_tui:
                                        headers = [
                                            "#",
                                            "File (Base Name)",
                                            "Similarity",
                                        ]
                                        table_data = [
                                            [i + 1, base, f"{score:.4f}"]
                                            for i, (base, score) in enumerate(
                                                sim_entries
                                            )
                                        ]
                                        print("\n--- Top Mood Matches ---")
                                        print(
                                            tabulate(
                                                table_data,
                                                headers=headers,
                                                tablefmt="rounded_grid",
                                            )
                                        )
                                else:
                                    tui.status_msg = (
                                        f"No matches for mood: “{mood_text}”"
                                    )
                            except Exception as e:
                                tui.status_msg = f"Error building playlist: {e}"
                        elif new_top is not None:
                            # only top changed; need a mood to build
                            pass
                    time.sleep(0.05)
                    continue

                # We have a playlist: shuffle for the round
                round_list = current_playlist[:]
                # random.shuffle(round_list)

                for full_path in round_list:
                    _maybe_exit_for_scheduled_time(next_exit_dt)

                    # Between tracks: if user queued a new mood, rebuild now
                    if pending_mood:
                        try:
                            pl, sim_entries = compute_top_for_mood(
                                model,
                                tags_data,
                                pending_mood,
                                mp3_folder_path,
                                current_top_n,
                                logging=logging,
                            )
                            if pl:
                                resolved_target_loudness, audio_data = (
                                    build_audio_data_for_playlist(
                                        pl,
                                        target_lufs,
                                        mp3_folder_path,
                                        sample_file_name,
                                        loudness_cache,
                                        logging=logging,
                                    )
                                )
                                current_playlist = pl
                                current_similar_entries = sim_entries
                                current_similar_mood = pending_mood
                                tui.view_mode = (
                                    "similar"  # jump to Similar to show fresh table
                                )
                                tui.status_msg = f"Switched to mood: “{pending_mood}” (top {current_top_n}) — TAB toggles views"
                            else:
                                tui.status_msg = (
                                    f"No matches for mood: “{pending_mood}”"
                                )
                            pending_mood = None
                        except Exception as e:
                            tui.status_msg = f"Error building playlist: {e}"
                            pending_mood = None

                    if full_path not in audio_data:
                        # Skip missing analysis
                        continue

                    filename_ext = os.path.basename(full_path)
                    volume_scale = audio_data[full_path].get("scale", 0.5)
                    current_loudness = audio_data[full_path].get("loudness_dbfs", None)
                    loudness_diff = (
                        (current_loudness - resolved_target_loudness)
                        if (
                            isinstance(current_loudness, float)
                            and isinstance(resolved_target_loudness, float)
                        )
                        else None
                    )

                    try:
                        sound = pygame.mixer.Sound(full_path)
                        sound.set_volume(volume_scale)

                        # lock handling before play
                        if _mac_is_locked_poll():
                            lock_since_wall = _wait_while_locked_or_exit(
                                lock_since_wall, next_exit_dt
                            )
                            stopped, lock_since_wall = _pause_silence(10)
                            if stopped:
                                break

                        sound.play()

                        started_ts = time.monotonic()
                        try:
                            total_dur = float(sound.get_length())
                        except Exception:
                            total_dur = 0.0

                        reached_end = False
                        interrupted_by_lock = False
                        user_skip = False
                        user_stop = False
                        clock = pygame.time.Clock()

                        while pygame.mixer.get_busy():
                            _maybe_exit_for_scheduled_time(next_exit_dt)

                            # collect input every frame; if Enter pressed, queue it (don’t switch yet)
                            submitted = None
                            if enable_tui:
                                elapsed = min(
                                    max(0.0, time.monotonic() - started_ts),
                                    total_dur if total_dur > 0 else 0.0,
                                )
                                submitted = tui.render(
                                    now_playing_name=filename_ext,
                                    target_lufs=resolved_target_loudness,
                                    current_lufs=current_loudness
                                    if isinstance(current_loudness, float)
                                    else None,
                                    loudness_diff=loudness_diff
                                    if isinstance(loudness_diff, float)
                                    else None,
                                    volume_scale=volume_scale,
                                    elapsed_sec=elapsed,
                                    total_sec=total_dur,
                                    counts=counters,
                                    similar_entries=current_similar_entries,
                                    similar_mood=current_similar_mood,
                                    pending_mood=pending_mood,
                                )
                                if submitted:
                                    mood_text, new_top = parse_mood_and_directives(
                                        submitted
                                    )
                                    if new_top is not None:
                                        clamped = max(TOP_MIN, min(TOP_MAX, new_top))
                                        current_top_n = clamped
                                        tui.status_msg = f"Top set to {current_top_n}"
                                    if mood_text:
                                        pending_mood = mood_text
                                    elif new_top is not None and current_similar_mood:
                                        # rebuild same mood with new top after this track
                                        pending_mood = current_similar_mood

                                if enable_tui and getattr(tui, "skip_requested", False):
                                    user_skip = True
                                    tui.skip_requested = False
                                    pygame.mixer.stop()  # stop immediately
                                    break

                                if enable_tui and getattr(tui, "stop_requested", False):
                                    user_stop = True
                                    tui.stop_requested = False
                                    pygame.mixer.stop()  # stop immediately
                                    break
                            else:
                                # non-TUI fallback progress
                                if total_dur > 0.0:
                                    seg_len = 10.0
                                    total_segments = max(
                                        1, int(math.ceil(total_dur / seg_len))
                                    )
                                    elapsed = min(
                                        max(0.0, time.monotonic() - started_ts),
                                        total_dur,
                                    )
                                    filled_segments = min(
                                        total_segments, int(elapsed // seg_len)
                                    )
                                    bar = (
                                        "["
                                        + ("-" * filled_segments)
                                        + (" " * (total_segments - filled_segments))
                                        + "]"
                                    )
                                    mm = int(elapsed // 60)
                                    ss = int(elapsed % 60)
                                    tmm = int(total_dur // 60)
                                    tss = int(total_dur % 60)
                                    sys.stdout.write(
                                        f"\rPlaying {filename_ext} {bar} {mm}:{ss:02d} / {tmm}:{tss:02d}\x1b[K"
                                    )
                                    sys.stdout.flush()

                            # lock mid-track?
                            if _mac_is_locked_poll():
                                interrupted_by_lock = True
                                pygame.mixer.stop()
                                lock_since_wall = _wait_while_locked_or_exit(
                                    lock_since_wall, next_exit_dt
                                )
                                stopped, lock_since_wall = _pause_silence(10)
                                if stopped:
                                    break
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

                        if user_stop:
                            current_playlist = None
                            current_similar_entries = None
                            current_similar_mood = None
                            pending_mood = None
                            if enable_tui:
                                tui.view_mode = "most"
                                tui.status_msg = "Stopped. Waiting for mood..."
                            # tiny settle so the mixer fully stops before the next render
                            lock_since_wall = _sleep_with_exit_checks(
                                0.2, next_exit_dt, lock_since_wall
                            )
                            break

                        # pacing between tracks:
                        if user_skip:
                            # honor the user-requested skip: wait 10s before next track
                            stopped, lock_since_wall = _pause_silence(10)
                            if stopped:
                                break
                        elif not _mac_is_locked_poll():
                            stopped, lock_since_wall = _pause_silence(10)
                            if stopped:
                                break

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
            save_loudness_cache(loudness_cache, mp3_folder_path)
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
        description="Mood TUI player with volume normalization, listen counters, and Similar/Most tabbed view."
    )
    parser.add_argument(
        "-mood",
        "--mood",
        type=str,
        required=False,
        default=None,
        help="Optional initial mood (you can type a new one in the TUI bottom bar).",
    )
    parser.add_argument(
        "-top",
        "--top",
        type=int,
        default=1,
        help="Number of top matching files to find and play.",
    )
    parser.add_argument(
        "--vol",
        type=float,
        default=None,
        help="Target loudness in LUFS (e.g., -17.50). If not provided, uses the sample file loudness.",
    )
    parser.add_argument(
        "--log", action="store_true", help="Enable detailed logging output."
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=MP3_FOLDER,
        help=f"Path to the MP3 folder (default: {MP3_FOLDER}).",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default=TAGS_FILE,
        help=f"Path to the tags JSON file (default: {TAGS_FILE}).",
    )
    parser.add_argument(
        "--sample",
        type=str,
        default=SAMPLE_MP3_FILENAME,
        help=f"Filename of the reference volume MP3 in the MP3 folder (default: {SAMPLE_MP3_FILENAME}).",
    )
    parser.add_argument(
        "--no-tui",
        action="store_true",
        help="Disable curses TUI and use simple stdout printing. Similar table will print to stdout.",
    )

    args = parser.parse_args()

    try:
        import pydub
        import pygame
        import sentence_transformers
        import sklearn
        from tabulate import tabulate
    except ImportError as e:
        print(f"Error: Missing required package: {e.name}")
        print(
            "Install: pip install sentence-transformers scikit-learn pydub pygame tabulate torch"
        )
        sys.exit(1)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        tag = args.tags
        if args.folder.split("/")[1] == "mid-mp3s":
            LISTEN_DB_FILE = "mid_listen_counts.json"
            tag = "mid_tags.json"
        main(
            args.mood,
            args.top,
            mp3_folder_path=args.folder,
            tags_file_path=tag,
            sample_file_name=args.sample,
            target_lufs=args.vol,
            logging=args.log,
            enable_tui=(not args.no_tui),
        )
    except SystemExit:
        print("\nPlayback stopped.")
    except KeyboardInterrupt:
        print("\nPlayback interrupted.")
