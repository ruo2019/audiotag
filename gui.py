from __future__ import annotations

import argparse
import curses
import json
import locale
import math
import os
import re
import sys
import threading
import time
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pygame
import torch
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from sentence_transformers import SentenceTransformer
from tabulate import tabulate

HEADPHONES_LOST = threading.Event()

# ============================== CONFIG ==============================

locale.setlocale(locale.LC_ALL, "")

MODEL_NAME = "sentence-transformers/sentence-t5-base"

DEFAULT_MP3_FOLDER = Path("static/mp3")
DEFAULT_TAGS_FILE = Path("tags.json")
DEFAULT_SAMPLE_MP3 = "Deep Stone Crypt Theme.mp3"

# Exit policy
LOCK_EXIT_MINUTES = 30
EXIT_AT_LOCAL_HOUR = 1
EXIT_AT_LOCAL_MINUTE = 30

# Mood directive clamp
TOP_MIN, TOP_MAX = 1, 50

# Between-track (and post-unlock) gap
GAP_SECONDS = 10.0

# Caches stored inside mp3 folder
EMB_CACHE_VERSION = "v2"
EMB_CACHE_NAME = ".track_emb_cache.npz"

LOUD_CACHE_VERSION = 1
LOUD_CACHE_NAME = ".loudness_cache.json"

# Listen count DB stored next to this script (your original behavior)
LISTEN_DB_FILE = "listen_counts.json"

# Globals for exit/lock state
EXIT_NOW = threading.Event()
IS_SCREEN_LOCKED = threading.Event()  # non-mac fallback stub


# ============================== DEVICE / ENV ==============================


def select_device() -> str:
    req = os.environ.get("PLAYER_DEVICE", "").lower().strip()
    if sys.platform == "darwin":
        if not req:
            req = "cpu"
        if req == "mps" and torch.backends.mps.is_available():
            return "mps"
        if req == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if req in {"cpu"}:
        return "cpu"
    if req in {"cuda", "gpu"} and torch.cuda.is_available():
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def tame_threads() -> None:
    try:
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("MKL_NUM_THREADS", "2")
        torch.set_num_threads(max(1, min(2, (os.cpu_count() or 2))))
    except Exception:
        pass
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


DEVICE = select_device()
tame_threads()


# ============================== EXIT / LOCK HELPERS ==============================


def next_local_time(hour: int, minute: int) -> datetime:
    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def maybe_exit(next_exit_dt: datetime) -> None:
    if EXIT_NOW.is_set() or datetime.now() >= next_exit_dt:
        try:
            pygame.mixer.stop()
        except Exception:
            pass
        raise SystemExit


def mac_is_locked_poll() -> bool:
    if sys.platform != "darwin":
        return IS_SCREEN_LOCKED.is_set()
    try:
        from Quartz import CGSessionCopyCurrentDictionary  # type: ignore

        sess = CGSessionCopyCurrentDictionary() or {}
        if "CGSSessionScreenIsLocked" in sess:
            return bool(sess.get("CGSSessionScreenIsLocked"))
        if "CGSSessionOnConsoleKey" in sess:
            return not bool(sess.get("CGSSessionOnConsoleKey"))
        return IS_SCREEN_LOCKED.is_set()
    except Exception:
        return IS_SCREEN_LOCKED.is_set()


def update_lock_or_exit(
    locked: bool,
    lock_since_wall: float | None,
    next_exit_dt: datetime,
    lock_exit_minutes: int = LOCK_EXIT_MINUTES,
) -> float | None:
    maybe_exit(next_exit_dt)
    if not locked:
        return None
    now = time.time()
    if lock_since_wall is None:
        return now
    if now - lock_since_wall >= lock_exit_minutes * 60:
        EXIT_NOW.set()
        maybe_exit(next_exit_dt)
    return lock_since_wall


def wait_while_locked_or_exit(
    lock_since_wall: float | None,
    next_exit_dt: datetime,
    tick_hz: int = 10,
) -> float | None:
    clk = pygame.time.Clock()
    while True:
        locked = mac_is_locked_poll()
        lock_since_wall = update_lock_or_exit(locked, lock_since_wall, next_exit_dt)
        if not locked:
            return None
        clk.tick(tick_hz)


def sleep_with_exit_checks(
    seconds: float,
    lock_since_wall: float | None,
    next_exit_dt: datetime,
) -> float | None:
    end = time.time() + float(seconds)
    while time.time() < end:
        locked = mac_is_locked_poll()
        lock_since_wall = update_lock_or_exit(locked, lock_since_wall, next_exit_dt)
        time.sleep(0.1)
    return lock_since_wall


# ============================== JSON UTILS ==============================


def atomic_write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def safe_read_json(path: Path, default: object) -> object:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


# ============================== LISTEN COUNTS ==============================


def listen_db_path(filename: str) -> Path:
    return Path(__file__).resolve().parent / filename


def load_listen_counts(mp3_dir: Path, db_filename: str) -> Dict[str, int]:
    raw = safe_read_json(listen_db_path(db_filename), {})
    counts: Dict[str, int] = raw if isinstance(raw, dict) else {}
    try:
        for p in mp3_dir.iterdir():
            if p.is_file() and p.suffix.lower() == ".mp3":
                counts.setdefault(p.stem, 0)
    except Exception:
        pass
    out: Dict[str, int] = {}
    for k, v in counts.items():
        try:
            out[str(k)] = int(v)
        except Exception:
            out[str(k)] = 0
    return out


def save_listen_counts(counts: Dict[str, int], db_filename: str) -> None:
    atomic_write_json(listen_db_path(db_filename), counts)


def increment_listen(track_path: Path, counts: Dict[str, int]) -> None:
    counts[track_path.stem] = counts.get(track_path.stem, 0) + 1


def sorted_counts(counts: Dict[str, int]) -> List[Tuple[str, int]]:
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].lower()))


# ============================== LOUDNESS CACHE ==============================


def loud_cache_path(mp3_folder: Path) -> Path:
    return mp3_folder / LOUD_CACHE_NAME


def load_loudness_cache(mp3_folder: Path) -> Dict[str, dict]:
    data = safe_read_json(loud_cache_path(mp3_folder), {"_v": LOUD_CACHE_VERSION})
    if not isinstance(data, dict) or data.get("_v") != LOUD_CACHE_VERSION:
        return {"_v": LOUD_CACHE_VERSION}
    return data


def save_loudness_cache(cache: Dict[str, dict], mp3_folder: Path) -> None:
    cache = dict(cache or {})
    cache["_v"] = LOUD_CACHE_VERSION
    atomic_write_json(loud_cache_path(mp3_folder), cache)


def calculate_volume_scale(
    target_peak_dbfs: float | None, current_peak_dbfs: float | None
) -> float:
    if target_peak_dbfs is None or current_peak_dbfs is None:
        return 0.5
    if target_peak_dbfs <= -100.0 or current_peak_dbfs <= -100.0:
        return 0.5
    db_difference = target_peak_dbfs - current_peak_dbfs
    scale_factor = 10 ** (db_difference / 20)
    return max(0.0, min(1.0, 0.5 * scale_factor))


def get_audio_stats(mp3_path: Path) -> dict:
    try:
        audio = AudioSegment.from_mp3(str(mp3_path))
        dur = float(audio.duration_seconds or 0.0)
        loud = float(audio.dBFS)
        if loud == -math.inf:
            loud = -100.0
        return {
            "loudness_dbfs": loud,
            "duration": dur,
            "mtime": mp3_path.stat().st_mtime,
        }
    except CouldntDecodeError:
        return {
            "loudness_dbfs": None,
            "duration": 0.0,
            "mtime": mp3_path.stat().st_mtime,
        }
    except Exception:
        try:
            mtime = mp3_path.stat().st_mtime
        except Exception:
            mtime = 0.0
        return {"loudness_dbfs": None, "duration": 0.0, "mtime": mtime}


def build_audio_data_for_playlist(
    playlist: List[Path],
    target_dBFS: Optional[float],
    mp3_folder: Path,
    sample_filename: str,
    cache: Dict[str, dict],
    logging: bool = False,
) -> Tuple[float, Dict[Path, Dict[str, float | None]]]:
    if target_dBFS is None:
        sample_path = mp3_folder / sample_filename
        sample_key = str(sample_path)
        sample_mtime = sample_path.stat().st_mtime if sample_path.exists() else 0.0
        entry = cache.get(sample_key)
        if not entry or abs(float(entry.get("mtime", 0.0)) - sample_mtime) > 0.5:
            entry = get_audio_stats(sample_path)
            cache[sample_key] = entry
        sample_loud = entry.get("loudness_dbfs")
        if sample_loud is None:
            raise RuntimeError(
                f"Could not analyze reference sample '{sample_filename}'"
            )
        resolved_target = float(sample_loud)
    else:
        resolved_target = float(target_dBFS)

    audio_data: Dict[Path, Dict[str, float | None]] = {}
    for p in playlist:
        key = str(p)
        try:
            mtime = p.stat().st_mtime
        except Exception:
            mtime = 0.0
        entry = cache.get(key)
        if not entry or abs(float(entry.get("mtime", 0.0)) - mtime) > 0.5:
            entry = get_audio_stats(p)
            cache[key] = entry

        loud = entry.get("loudness_dbfs")
        dur = float(entry.get("duration") or 0.0)
        if loud is None:
            audio_data[p] = {"loudness_dbfs": None, "scale": 0.5, "duration": dur}
        else:
            audio_data[p] = {
                "loudness_dbfs": float(loud),
                "scale": float(calculate_volume_scale(resolved_target, float(loud))),
                "duration": dur,
            }

    if logging:
        print(f"[loud] target={resolved_target:.2f} dBFS; tracks={len(playlist)}")

    return resolved_target, audio_data


# ============================== INLINE DIRECTIVES ==============================

TOP_DIR_RE = re.compile(
    r"\(\s*top\s*[:=]?\s*(\d+)\s*\)|\btop\s*[:=]?\s*(\d+)\b", re.IGNORECASE
)

VOL_DIR_RE = re.compile(
    r"\(\s*vol(?:ume)?\s*[:=]?\s*(-?\d+(?:\.\d+)?)\s*\)"
    r"|\bvol(?:ume)?\s*[:=]?\s*(-?\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)


def clamp_int(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(n)))


def parse_mood_and_directives(
    raw: str,
) -> Tuple[Optional[str], Optional[int], Optional[float]]:
    if not raw:
        return None, None, None
    top_val: Optional[int] = None
    vol_val: Optional[float] = None

    for m in TOP_DIR_RE.finditer(raw):
        num = m.group(1) or m.group(2)
        if num is not None:
            try:
                top_val = int(num)
            except ValueError:
                pass
    for m in VOL_DIR_RE.finditer(raw):
        num = m.group(1) or m.group(2)
        if num is not None:
            try:
                vol_val = float(num)
            except ValueError:
                pass

    mood = TOP_DIR_RE.sub("", raw)
    mood = VOL_DIR_RE.sub("", mood).strip()
    mood = mood.strip(" ,;:()[]{}-") or None
    return mood, top_val, vol_val


# ============================== EMBEDDING CACHE ==============================


def emb_cache_path(mp3_folder: Path) -> Path:
    return mp3_folder / EMB_CACHE_NAME


def canonicalize_tags(tags: Iterable[object]) -> str:
    return ", ".join(sorted({str(t).strip().lower() for t in tags if str(t).strip()}))


def tags_fingerprint(tags_data: dict, mp3_folder: Path) -> str:
    rows: List[str] = []
    for base, tag_list in (tags_data or {}).items():
        p = mp3_folder / f"{base}.mp3"
        if p.exists():
            try:
                mtime = p.stat().st_mtime
            except Exception:
                mtime = 0.0
            rows.append(f"{base}|{canonicalize_tags(tag_list or [])}|{mtime:.0f}")
    rows.sort()
    return sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _npz_str(npz, key: str, default: str = "") -> str:
    if key not in npz:
        return default
    v = npz[key]
    try:
        if isinstance(v, np.ndarray) and v.shape == ():
            return str(v.item())
        return str(v)
    except Exception:
        return default


class EmbeddingCache:
    def __init__(self):
        self.names: List[str] = []
        self.matrix: Optional[np.ndarray] = None  # L2-normalized rows
        self.fingerprint: Optional[str] = None

    def ensure(
        self,
        model: SentenceTransformer,
        tags_data: dict,
        mp3_folder: Path,
        logging: bool = False,
    ) -> None:
        fp_now = tags_fingerprint(tags_data, mp3_folder)
        cache_file = emb_cache_path(mp3_folder)

        # Try load once per process
        if self.matrix is None and cache_file.exists():
            try:
                blob = np.load(str(cache_file), allow_pickle=True)
                version = _npz_str(blob, "version", "")
                model_id = _npz_str(blob, "model_id", "")
                fp = _npz_str(blob, "fingerprint", "")
                # IMPORTANT: validate ALL fields (your old code effectively made this always true)
                if (
                    version == EMB_CACHE_VERSION
                    and model_id == MODEL_NAME
                    and fp == fp_now
                ):
                    names = list(blob["names"].tolist())
                    embs = blob["embeddings"]
                    if isinstance(embs, np.ndarray) and embs.ndim == 2 and names:
                        self.names = [str(n) for n in names]
                        self.matrix = embs.astype(np.float32, copy=False)
                        self.fingerprint = fp
                        if logging:
                            print(
                                f"[emb] loaded {self.matrix.shape[0]} embeddings from cache."
                            )
                        return
            except Exception as e:
                if logging:
                    print(f"[emb] cache load failed, will rebuild: {e}")

        # If already correct, no-op
        if self.matrix is not None and self.fingerprint == fp_now:
            return

        # Build
        names: List[str] = []
        texts: List[str] = []
        for base, tag_list in (tags_data or {}).items():
            p = mp3_folder / f"{base}.mp3"
            if p.exists() and tag_list:
                names.append(str(base))
                texts.append("music that is " + canonicalize_tags(tag_list))

        if not names:
            self.names, self.matrix, self.fingerprint = [], None, fp_now
            return

        if logging:
            print(f"[emb] encoding {len(texts)} tracks...")

        embs = model.encode(
            texts, batch_size=8, show_progress_bar=logging, convert_to_numpy=True
        )
        embs = embs.astype(np.float32, copy=False)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12

        self.names = names
        self.matrix = embs
        self.fingerprint = fp_now

        try:
            np.savez(
                str(cache_file),
                version=EMB_CACHE_VERSION,
                model_id=MODEL_NAME,
                fingerprint=fp_now,
                names=np.array(self.names, dtype=object),
                embeddings=self.matrix,
            )
            if logging:
                print(f"[emb] cache saved to {cache_file}")
        except Exception as e:
            if logging:
                print(f"[emb] failed to save cache: {e}")


EMB_CACHE = EmbeddingCache()


def compute_top_for_mood(
    model: SentenceTransformer,
    tags_data: Dict[str, List[str]],
    mood: str,
    mp3_folder: Path,
    top_n: int,
    logging: bool = False,
) -> Tuple[List[Path], List[Tuple[str, float]]]:
    if not mood:
        return [], []

    EMB_CACHE.ensure(model, tags_data, mp3_folder, logging=logging)
    if EMB_CACHE.matrix is None or not EMB_CACHE.names:
        return [], []

    tokens = [tok.strip().lower() for tok in mood.split(",") if tok.strip()]
    token_set = set(tokens)

    q = model.encode(
        ["music that is " + mood],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
    )[0]
    q = q.astype(np.float32, copy=False)
    q /= np.linalg.norm(q) + 1e-12

    sims = EMB_CACHE.matrix @ q

    match_counts: List[int] = []
    for base in EMB_CACHE.names:
        tag_set = {str(t).strip().lower() for t in (tags_data.get(base, []) or [])}
        match_counts.append(
            sum(1 for tok in token_set if tok in tag_set) if token_set else 0
        )

    idxs = list(range(len(EMB_CACHE.names)))
    idxs.sort(
        key=lambda i: (-match_counts[i], -float(sims[i]), EMB_CACHE.names[i].lower())
    )
    top_n = clamp_int(top_n, TOP_MIN, TOP_MAX)
    idxs = idxs[:top_n]

    playlist = [mp3_folder / f"{EMB_CACHE.names[i]}.mp3" for i in idxs]
    table = [(EMB_CACHE.names[i], float(sims[i])) for i in idxs]

    if logging:
        print(f"[emb] mood='{mood}' -> {len(playlist)} results (top {top_n})")

    return playlist, table


def stretch_rounded_grid_to_width(
    lines: list[str], target_width: int, track_col_index: int
) -> list[str]:
    """
    Stretches a tabulate(tablefmt="rounded_grid") table horizontally to target_width by
    widening the Track column (or last column if Track is last).

    track_col_index is the 0-based column index of the Track column:
      - Similar view: [#, Track, Sim]  -> track_col_index = 1
      - Most view:    [#, Plays, Track]-> track_col_index = 2
    """
    if not lines:
        return lines

    target_width = int(target_width)
    cur_width = max(len(l) for l in lines)

    if cur_width == target_width:
        return lines
    if cur_width > target_width:
        # truncate (best-effort; ideally also ellipsize track names separately)
        return [l[:target_width] for l in lines]

    delta = target_width - cur_width

    # Default: widen last column (insert before final border char)
    insert_at = cur_width - 1

    # If Track is not the last column, widen Track by inserting before the junction after it.
    # Top border line uses '┬' junctions between columns.
    top = lines[0]
    junctions = [i for i, ch in enumerate(top) if ch == "┬"]
    if 0 <= track_col_index < len(junctions):
        insert_at = junctions[track_col_index]

    out: list[str] = []
    for line in lines:
        if insert_at < 0 or insert_at > len(line):
            out.append(line.ljust(target_width))
            continue

        # Border lines: extend with '─'
        # Content/header lines: extend with spaces
        filler = "─" if line and line[0] in "╭├╰" else " "
        out.append(line[:insert_at] + (filler * delta) + line[insert_at:])

    return out


# ============================== CURSES TUI ==============================


class CursesTUI:
    """
    Minimal curses UI:
      - TAB toggles Most/Similar views
      - ↑/↓ PgUp/PgDn scroll table
      - → skip to next track (after a gap)
      - Ctrl+G (and Ctrl+S if it reaches the app) stops playback and returns to idle
      - Enter submits mood text (with optional top directive)

    NOTE: Ctrl+S may be swallowed by terminal flow control; Ctrl+G is reliable.
    """

    KEY_TAB = 9
    KEY_BS = 127
    CTRL_G = 7
    CTRL_S = 19  # may be eaten by terminal flow-control unless IXON disabled

    def __init__(self, enable: bool = True):
        self.enable = enable
        self.stdscr: Optional["curses._CursesWindow"] = None

        self.view_mode: str = "most"  # most | similar
        self.scroll = 0

        self.input_buffer = ""
        self.status_msg = ""

        self.skip_requested = False
        self.stop_requested = False

        self.last_h = 0
        self.last_w = 0

    def __enter__(self):
        if not self.enable:
            return self
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.stdscr.nodelay(True)
        try:
            self.stdscr.keypad(True)
        except Exception:
            pass
        try:
            curses.curs_set(0)
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.enable:
            return
        try:
            if self.stdscr:
                self.stdscr.nodelay(False)
            curses.echo()
            curses.nocbreak()
            curses.endwin()
        except Exception:
            pass

    def _draw(self, y: int, x: int, s: str) -> None:
        if not self.enable or not self.stdscr:
            return
        h, w = self.stdscr.getmaxyx()
        if y < 0 or y >= h or x >= w:
            return

        max_len = w - x
        if max_len <= 0:
            return
        if len(s) > max_len:
            s = s[:max_len]

        try:
            self.stdscr.addstr(y, x, s)
        except Exception:
            pass

    def _hline(self, y: int, ch: str = "-") -> None:
        if not self.enable or not self.stdscr:
            return
        h, w = self.stdscr.getmaxyx()
        if y < 0 or y >= h:
            return
        try:
            self.stdscr.hline(y, 0, ord(ch), max(0, w - 1))
        except Exception:
            pass

    def _handle_key(self) -> Optional[str]:
        if not self.enable or not self.stdscr:
            return None
        try:
            key = self.stdscr.getch()
        except Exception:
            key = -1
        if key == -1:
            return None

        KEY_PGUP = getattr(curses, "KEY_PPAGE", 339)
        KEY_PGDN = getattr(curses, "KEY_NPAGE", 338)
        KEY_BTAB = getattr(curses, "KEY_BTAB", 353)

        if key == curses.KEY_UP:
            self.scroll = max(0, self.scroll - 1)
        elif key == curses.KEY_DOWN:
            self.scroll += 1
        elif key == KEY_PGUP:
            self.scroll = max(0, self.scroll - max(1, self.last_h - 12))
        elif key == KEY_PGDN:
            self.scroll += max(1, self.last_h - 12)
        elif key in (self.KEY_TAB, KEY_BTAB):
            self.view_mode = "similar" if self.view_mode == "most" else "most"
            self.scroll = 0
        elif key == curses.KEY_RIGHT:
            self.skip_requested = True
            self.status_msg = "Skipping to next in 10s..."
        elif key in (self.CTRL_G, self.CTRL_S):
            self.stop_requested = True
            self.status_msg = "Stopped. Waiting for mood..."
        elif key in (curses.KEY_BACKSPACE, self.KEY_BS, 8):
            self.input_buffer = self.input_buffer[:-1]
        elif key in (10, 13):  # Enter
            text = self.input_buffer.strip()
            if text:
                self.input_buffer = ""
                return text
        else:
            if 32 <= key <= 126:
                self.input_buffer += chr(key)
        return None

    def render(
        self,
        now_playing: str,
        target_dBFS: Optional[float],
        current_dBFS: Optional[float],
        loudness_diff: Optional[float],
        volume_scale: Optional[float],
        elapsed_sec: float,
        total_sec: float,
        counts: Dict[str, int],
        similar_entries: Optional[List[Tuple[str, float]]],
        similar_mood: Optional[str],
        pending_mood: Optional[str],
    ) -> Optional[str]:
        if not self.enable or not self.stdscr:
            return None

        submitted = self._handle_key()

        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()
        self.last_h, self.last_w = h, w

        # Header block (fixed)
        self._draw(0, 0, " Now Playing ".center(w, " "))
        self._draw(1, 2, f"Track: {now_playing or '(none)'}")

        td = "N/A" if target_dBFS is None else f"{target_dBFS:.2f}"
        cd = "N/A" if current_dBFS is None else f"{current_dBFS:.2f}"
        ld = "N/A" if loudness_diff is None else f"{loudness_diff:.2f}"
        vs = "N/A" if volume_scale is None else f"{volume_scale:.2f}"

        self._draw(2, 2, f"Target dBFS: {td}")
        self._draw(3, 2, f"Current dBFS: {cd}")
        self._draw(4, 2, f"Loudness Diff: {ld}")
        self._draw(5, 2, f"Adjusted Volume: {vs}")

        bar_y = 6
        bar_w = max(10, w - 20)
        frac = (elapsed_sec / total_sec) if total_sec > 0 else 0.0
        filled = int(max(0.0, min(1.0, frac)) * bar_w)
        bar = "[" + ("-" * filled) + (" " * (bar_w - filled)) + "]"
        self._draw(bar_y, 2, bar)
        self._draw(
            bar_y,
            4 + bar_w,
            f" {int(elapsed_sec // 60)}:{int(elapsed_sec % 60):02d} / {int(total_sec // 60)}:{int(total_sec % 60):02d}",
        )

        self._hline(7, "-")

        # Layout: table region ends above input separator
        input_y = h - 2
        status_y = h - 4
        table_top = 8
        table_height = max(0, status_y - table_top)

        # Status line
        status = ""
        if pending_mood:
            status = f"Queued mood: “{pending_mood}” (applies after current track)"
        elif self.status_msg:
            status = self.status_msg
        if status:
            self._draw(status_y, 2, status)

        # Build table rows + scroll
        if self.view_mode == "similar":
            title = (
                f" Similar to “{similar_mood}” "
                if similar_mood
                else " Similar (no mood yet) "
            )
            entries = similar_entries or []
            all_rows = [
                [i + 1, base, f"{score:.4f}"] for i, (base, score) in enumerate(entries)
            ]
            headers = ["#", "Track", "Sim"]
        else:
            title = " Most Listened "
            entries = sorted_counts(counts)
            # numeric in the middle (idx, plays, track)
            all_rows = [[i + 1, cnt, name] for i, (name, cnt) in enumerate(entries)]
            headers = ["#", "Plays", "Track"]

        # rounded_grid is ~ 2 lines per row + ~4 framing lines
        max_rows = max(1, (table_height - 4) // 2) if table_height >= 6 else 1
        max_scroll = max(0, len(all_rows) - max_rows)
        self.scroll = min(self.scroll, max_scroll)
        visible = all_rows[self.scroll : self.scroll + max_rows]

        if table_height > 0:
            # build tabulate table as you already do:
            table_str = tabulate(
                visible, headers=headers, tablefmt="rounded_grid", disable_numparse=True
            )
            body_lines = table_str.splitlines()

            # choose which column is Track (0-based)
            track_col_index = 2 if self.view_mode == "most" else 1

            # stretch the table to the full terminal width
            body_lines = stretch_rounded_grid_to_width(body_lines, w, track_col_index)

            table_lines = [title.center(w, " ")] + body_lines
            for i, line in enumerate(table_lines[:table_height]):
                self._draw(table_top + i, 0, line)

        # Input bar + footer
        self._hline(input_y - 1, "-")
        prompt = "mood: "
        if not self.input_buffer and not pending_mood and not now_playing:
            self._draw(
                input_y, 2, prompt + "(type a mood + Enter; optional: (top N) / top=N)"
            )
        else:
            self._draw(input_y, 2, prompt + self.input_buffer)

        self._draw(
            h - 1,
            2,
            "↑/↓ PgUp/PgDn scroll • Enter submit • TAB toggle • → skip • Ctrl+G/Ctrl+S stop • (top N) sets list size",
        )

        try:
            self.stdscr.refresh()
        except Exception:
            pass

        if submitted:
            self.status_msg = f"Queued mood: “{submitted}”"
        return submitted


# --- SDL/pygame output device helpers ---


def list_pygame_output_devices() -> List[str]:
    # Return SDL2 playback device names as pygame sees them.
    # These names are what you pass to pygame.mixer.init(devicename=...).
    try:
        import pygame._sdl2.audio as sdl2_audio  # type: ignore
    except Exception:
        return []

    try:
        devs = sdl2_audio.get_audio_device_names(False)  # False = playback
        return [str(d) for d in devs]
    except Exception:
        return []

def start_device_presence_watchdog(device_name: Optional[str], logging: bool = False) -> None:
    """Background thread: if we pinned to a specific SDL output device and it disappears, exit.

    - If `device_name` is None/empty, this is a no-op (we didn't pin to a specific device).
    """
    if not device_name:
        return

    def _run() -> None:
        while not EXIT_NOW.is_set():
            devs = list_pygame_output_devices()

            # Only act if we can enumerate devices (empty list means "unknown").
            if devs and device_name not in devs:
                if logging:
                    print(f"[audio] Output device '{device_name}' disconnected. Stopping.")

                EXIT_NOW.set()
                try:
                    pygame.mixer.stop()
                except Exception:
                    pass
                return

            time.sleep(0.5)

    threading.Thread(target=_run, daemon=True).start()


# ============================== MAIN ==============================

def init_pygame(device_name: Optional[str] = None) -> None:
    """
    Initialize pygame mixer (audio only).

    IMPORTANT (macOS + Tkinter):
    Do NOT call pygame.init() here. pygame.init() initializes SDL's video/app layer,
    which can create an SDLApplication NSApplication instance. Tkinter may then crash with:
        '-[SDLApplication macOSVersion]: unrecognized selector'
    We only need audio, so initialize the mixer only.
    """
    try:
        # Newer pygame supports devicename=
        pygame.mixer.pre_init(44100, -16, 2, 2048, devicename=device_name)
        pygame.mixer.init(44100, -16, 2, 2048, devicename=device_name)
        return
    except TypeError:
        # Older pygame: no devicename= support.
        pygame.mixer.pre_init(44100, -16, 2, 2048)
        pygame.mixer.init()
        return
    except pygame.error:
        # Retry with a more compatible sample rate.
        try:
            pygame.mixer.pre_init(22050, -16, 2, 2048, devicename=device_name)
            pygame.mixer.init(22050, -16, 2, 2048, devicename=device_name)
            return
        except TypeError:
            pygame.mixer.pre_init(22050, -16, 2, 2048)
            pygame.mixer.init()
            return


def load_tags(tags_file: Path) -> Dict[str, List[str]]:
    if not tags_file.exists():
        raise FileNotFoundError(f"Tags file not found: {tags_file}")
    data = safe_read_json(tags_file, {})
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Tags data file '{tags_file}' is empty or invalid.")
    out: Dict[str, List[str]] = {}
    for k, v in data.items():
        out[str(k)] = [str(t) for t in v] if isinstance(v, list) else [str(v)]
    return out


def main_tui(
    initial_mood: Optional[str],
    top_n: int,
    mp3_folder: Path,
    tags_file: Path,
    sample_filename: str,
    target_dBFS: Optional[float],
    logging: bool,
    enable_tui: bool,
    listen_db_filename: str,
) -> None:
    tags_data = load_tags(tags_file)
    if logging:
        print(f"Loaded {len(tags_data)} tag items from '{tags_file}'.")

    if logging:
        print(f"Loading model '{MODEL_NAME}' on '{DEVICE}'...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    EMB_CACHE.ensure(model, tags_data, mp3_folder, logging=logging)

    FORCE_DEVICE: Optional[str] = None
    if sys.platform == "darwin":
        # Pin THIS program's audio to a specific output device (headphones),
        # without changing the macOS system output device.
        FORCE_DEVICE = "External Headphones"  # must match list_pygame_output_devices() exactly

        sdl_devs = list_pygame_output_devices()
        if logging and sdl_devs:
            print(f"[audio] SDL playback devices: {sdl_devs}")

        # Fail-closed: if we can enumerate devices and the requested one isn't present,
        # refuse to play rather than falling back to speakers.
        if sdl_devs and FORCE_DEVICE not in sdl_devs:
            raise RuntimeError(
                f"SDL/pygame output device '{FORCE_DEVICE}' not found.\n"
                f"Available SDL devices: {sdl_devs}\n"
                "Set FORCE_DEVICE to one of the exact names above."
            )

    # initialize pygame (pinned to FORCE_DEVICE on macOS)
    if logging:
        print("Initializing pygame mixer...")
    init_pygame(device_name=FORCE_DEVICE)
    start_device_presence_watchdog(device_name=FORCE_DEVICE, logging=logging)


    next_exit_dt = next_local_time(EXIT_AT_LOCAL_HOUR, EXIT_AT_LOCAL_MINUTE)
    lock_since_wall: float | None = None
    if logging:
        print(
            f"Exit policy: lock>{LOCK_EXIT_MINUTES}m or at {next_exit_dt:%Y-%m-%d %H:%M} local."
        )

    counts = load_listen_counts(mp3_folder, listen_db_filename)
    loud_cache = load_loudness_cache(mp3_folder)

    current_playlist: Optional[List[Path]] = None
    current_similar_entries: Optional[List[Tuple[str, float]]] = None
    current_similar_mood: Optional[str] = None
    pending_mood: Optional[str] = None

    audio_data: Dict[Path, Dict[str, float | None]] = {}
    resolved_target_loudness: Optional[float] = None

    current_top_n = clamp_int(top_n, TOP_MIN, TOP_MAX)
    runtime_target_dBFS: Optional[float] = target_dBFS

    def build_for_mood(mood_text: str) -> bool:
        nonlocal current_playlist, current_similar_entries, current_similar_mood
        nonlocal audio_data, resolved_target_loudness, pending_mood

        pl, sim = compute_top_for_mood(
            model=model,
            tags_data=tags_data,
            mood=mood_text,
            mp3_folder=mp3_folder,
            top_n=current_top_n,
            logging=logging,
        )
        if not pl:
            return False

        resolved, data = build_audio_data_for_playlist(
            playlist=pl,
            target_dBFS=runtime_target_dBFS,
            mp3_folder=mp3_folder,
            sample_filename=sample_filename,
            cache=loud_cache,
            logging=logging,
        )

        current_playlist = pl
        current_similar_entries = sim
        current_similar_mood = mood_text
        audio_data = data
        resolved_target_loudness = resolved
        pending_mood = None
        return True

    def apply_new_target_loudness(
        new_target: Optional[float],
        tui: Optional[CursesTUI],
        current_track: Optional[Path] = None,
        current_channel: Optional["pygame.mixer.Channel"] = None,
    ) -> None:
        nonlocal runtime_target_dBFS, resolved_target_loudness, audio_data

        runtime_target_dBFS = new_target

        # Resolve the "actual" target loudness:
        # - if user specified a number: use it
        # - if None: fall back to sample-based target (same logic as build_audio_data_for_playlist)
        if runtime_target_dBFS is None:
            sample_path = mp3_folder / sample_filename
            key = str(sample_path)
            sample_mtime = sample_path.stat().st_mtime if sample_path.exists() else 0.0
            entry = loud_cache.get(key)
            if not entry or abs(float(entry.get("mtime", 0.0)) - sample_mtime) > 0.5:
                entry = get_audio_stats(sample_path)
                loud_cache[key] = entry
            loud = entry.get("loudness_dbfs")
            if isinstance(loud, (int, float)):
                resolved_target_loudness = float(loud)
        else:
            resolved_target_loudness = float(runtime_target_dBFS)

        # Update cached per-track scales (fast: no re-decode)
        if isinstance(resolved_target_loudness, (int, float)):
            tgt = float(resolved_target_loudness)
            for p, d in audio_data.items():
                loud = d.get("loudness_dbfs")
                if isinstance(loud, (int, float)):
                    d["scale"] = float(calculate_volume_scale(tgt, float(loud)))
                else:
                    d["scale"] = 0.5

        # Apply immediately to currently playing track (no restart)
        if current_track and current_channel and current_track in audio_data:
            new_scale = float(audio_data[current_track].get("scale") or 0.5)
            try:
                current_channel.set_volume(new_scale)
            except Exception:
                pass

        if tui and tui.enable:
            if isinstance(resolved_target_loudness, (int, float)):
                src = "sample" if runtime_target_dBFS is None else "manual"
                tui.status_msg = (
                    f"Target volume set to {resolved_target_loudness:.2f} ({src})"
                )
            else:
                tui.status_msg = "Target volume updated"

    def apply_submission(
        text: str,
        tui: Optional[CursesTUI],
        current_track: Optional[Path] = None,
        current_channel: Optional["pygame.mixer.Channel"] = None,
    ) -> None:
        nonlocal pending_mood, current_top_n
        mood_text, new_top, new_vol = parse_mood_and_directives(text)

        if new_vol is not None:
            apply_new_target_loudness(new_vol, tui, current_track, current_channel)

        if new_top is not None:
            current_top_n = clamp_int(new_top, TOP_MIN, TOP_MAX)
            if tui and tui.enable:
                tui.status_msg = f"Top set to {current_top_n}"
        if mood_text:
            pending_mood = mood_text
        elif new_top is not None and current_similar_mood:
            pending_mood = current_similar_mood

    def reset_to_idle(tui: Optional[CursesTUI]) -> None:
        nonlocal \
            current_playlist, \
            current_similar_entries, \
            current_similar_mood, \
            pending_mood
        current_playlist = None
        current_similar_entries = None
        current_similar_mood = None
        pending_mood = None
        if tui and tui.enable:
            tui.view_mode = "most"
            tui.status_msg = "Stopped. Waiting for mood..."

    # Seed
    if initial_mood:
        if build_for_mood(initial_mood):
            if not enable_tui:
                headers = ["#", "File (Base Name)", "Similarity"]
                table_data = [
                    [i + 1, base, f"{score:.4f}"]
                    for i, (base, score) in enumerate(current_similar_entries or [])
                ]
                print("\n--- Top Mood Matches ---")
                print(tabulate(table_data, headers=headers, tablefmt="rounded_grid"))
        elif logging:
            print(f"No matches for initial mood '{initial_mood}'.")

    with CursesTUI(enable=enable_tui) as tui:
        if enable_tui and current_playlist:
            tui.view_mode = "similar"

        def pause_silence(seconds: float) -> tuple[bool, float | None]:
            """Responsive gap pause. Returns (stopped, updated_lock_since_wall)."""
            nonlocal lock_since_wall
            if seconds <= 0:
                return False, lock_since_wall

            if not enable_tui:
                lock_since_wall = sleep_with_exit_checks(
                    seconds, lock_since_wall, next_exit_dt
                )
                return False, lock_since_wall

            start = time.monotonic()
            end = start + float(seconds)
            clock = pygame.time.Clock()

            while True:
                maybe_exit(next_exit_dt)
                if time.monotonic() >= end:
                    break

                lock_since_wall = update_lock_or_exit(
                    mac_is_locked_poll(), lock_since_wall, next_exit_dt
                )

                submitted = tui.render(
                    now_playing="",
                    target_dBFS=None,
                    current_dBFS=None,
                    loudness_diff=None,
                    volume_scale=None,
                    elapsed_sec=max(0.0, seconds - max(0.0, end - time.monotonic())),
                    total_sec=seconds,
                    counts=counts,
                    similar_entries=current_similar_entries,
                    similar_mood=current_similar_mood,
                    pending_mood=pending_mood,
                )
                if submitted:
                    apply_submission(submitted, tui)

                if tui.skip_requested:
                    tui.skip_requested = False
                    break

                if tui.stop_requested:
                    tui.stop_requested = False
                    reset_to_idle(tui)
                    return True, lock_since_wall

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.mixer.stop()
                        raise SystemExit

                clock.tick(20)

            return False, lock_since_wall

        try:
            while True:
                maybe_exit(next_exit_dt)

                # Idle (only interactive in TUI mode)
                if not current_playlist:
                    if not enable_tui:
                        time.sleep(0.2)
                        continue
                    submitted = tui.render(
                        now_playing="",
                        target_dBFS=None,
                        current_dBFS=None,
                        loudness_diff=None,
                        volume_scale=None,
                        elapsed_sec=0.0,
                        total_sec=0.0,
                        counts=counts,
                        similar_entries=current_similar_entries,
                        similar_mood=current_similar_mood,
                        pending_mood=pending_mood,
                    )
                    if submitted:
                        mood_text, new_top, new_vol = parse_mood_and_directives(
                            submitted
                        )
                        if new_top is not None:
                            current_top_n = clamp_int(new_top, TOP_MIN, TOP_MAX)
                            tui.status_msg = f"Top set to {current_top_n}"
                        if new_vol is not None:
                            apply_new_target_loudness(new_vol, tui)
                        if mood_text:
                            try:
                                if build_for_mood(mood_text):
                                    tui.view_mode = "similar"
                                    tui.status_msg = f"Loaded mood: “{mood_text}” (top {current_top_n})"
                                else:
                                    tui.status_msg = (
                                        f"No matches for mood: “{mood_text}”"
                                    )
                            except Exception as e:
                                tui.status_msg = f"Error building playlist: {e}"
                    time.sleep(0.05)
                    continue

                # Playlist loop
                for track_path in list(current_playlist):
                    maybe_exit(next_exit_dt)

                    if pending_mood:
                        try:
                            if build_for_mood(pending_mood):
                                if enable_tui:
                                    tui.view_mode = "similar"
                                    tui.status_msg = f"Switched to mood: “{pending_mood}” (top {current_top_n})"
                            else:
                                if enable_tui:
                                    tui.status_msg = (
                                        f"No matches for mood: “{pending_mood}”"
                                    )
                            pending_mood = None
                        except Exception as e:
                            if enable_tui:
                                tui.status_msg = f"Error building playlist: {e}"
                            pending_mood = None

                    if track_path not in audio_data:
                        continue

                    filename_ext = track_path.name
                    vol_scale = float(audio_data[track_path].get("scale") or 0.5)
                    current_loudness = audio_data[track_path].get("loudness_dbfs")
                    loudness_diff = (
                        float(current_loudness) - float(resolved_target_loudness)
                        if isinstance(current_loudness, (int, float))
                        and isinstance(resolved_target_loudness, (int, float))
                        else None
                    )

                    try:
                        sound = pygame.mixer.Sound(str(track_path))
                        channel = sound.play()
                        if channel is None:
                            channel = pygame.mixer.find_channel(True)
                            channel.play(sound)
                        channel.set_volume(vol_scale)

                        # If locked before play
                        if mac_is_locked_poll():
                            lock_since_wall = wait_while_locked_or_exit(
                                lock_since_wall, next_exit_dt
                            )
                            stopped, lock_since_wall = pause_silence(GAP_SECONDS)
                            if stopped:
                                break

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

                        while channel.get_busy():
                            maybe_exit(next_exit_dt)

                            if enable_tui:
                                elapsed = min(
                                    max(0.0, time.monotonic() - started_ts),
                                    total_dur if total_dur > 0 else 0.0,
                                )
                                submitted = tui.render(
                                    now_playing=filename_ext,
                                    target_dBFS=resolved_target_loudness,
                                    current_dBFS=float(current_loudness)
                                    if isinstance(current_loudness, (int, float))
                                    else None,
                                    loudness_diff=float(loudness_diff)
                                    if isinstance(loudness_diff, (int, float))
                                    else None,
                                    volume_scale=vol_scale,
                                    elapsed_sec=elapsed,
                                    total_sec=total_dur,
                                    counts=counts,
                                    similar_entries=current_similar_entries,
                                    similar_mood=current_similar_mood,
                                    pending_mood=pending_mood,
                                )
                                if submitted:
                                    apply_submission(
                                        submitted, tui, track_path, channel
                                    )

                                    # refresh values for UI immediately after a vol change
                                    vol_scale = float(
                                        audio_data[track_path].get("scale") or vol_scale
                                    )
                                    if isinstance(
                                        current_loudness, (int, float)
                                    ) and isinstance(
                                        resolved_target_loudness, (int, float)
                                    ):
                                        loudness_diff = float(current_loudness) - float(
                                            resolved_target_loudness
                                        )
                                    else:
                                        loudness_diff = None

                                if tui.skip_requested:
                                    user_skip = True
                                    tui.skip_requested = False
                                    pygame.mixer.stop()
                                    break

                                if tui.stop_requested:
                                    user_stop = True
                                    tui.stop_requested = False
                                    pygame.mixer.stop()
                                    break

                            # Lock mid-track?
                            if mac_is_locked_poll():
                                interrupted_by_lock = True
                                pygame.mixer.stop()
                                lock_since_wall = wait_while_locked_or_exit(
                                    lock_since_wall, next_exit_dt
                                )
                                stopped, lock_since_wall = pause_silence(GAP_SECONDS)
                                if stopped:
                                    user_stop = True
                                break

                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    pygame.mixer.stop()
                                    raise SystemExit

                            clock.tick(10)

                        if not interrupted_by_lock and not user_skip and total_dur > 0:
                            elapsed_final = max(0.0, time.monotonic() - started_ts)
                            reached_end = elapsed_final >= max(0.0, total_dur - 0.75)

                        if reached_end:
                            increment_listen(track_path, counts)
                            save_listen_counts(counts, listen_db_filename)

                        if user_stop:
                            reset_to_idle(tui)
                            lock_since_wall = sleep_with_exit_checks(
                                0.2, lock_since_wall, next_exit_dt
                            )
                            break

                        # Between tracks
                        if user_skip or not mac_is_locked_poll():
                            stopped, lock_since_wall = pause_silence(GAP_SECONDS)
                            if stopped:
                                break

                    except pygame.error as e:
                        if enable_tui:
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

        except (SystemExit, KeyboardInterrupt):
            pass
        finally:
            try:
                save_loudness_cache(loud_cache, mp3_folder)
            except Exception:
                pass

    try:
        pygame.mixer.quit()
    finally:
        pygame.quit()



# ============================== QUEUE GUI ==============================

def main(
    initial_mood: Optional[str],
    top_n: int,
    mp3_folder: Path,
    tags_file: Path,
    sample_filename: str,
    target_dBFS: Optional[float],
    logging: bool,
    enable_tui: bool,
    listen_db_filename: str,
) -> None:
    """
    Queue-first GUI mode (Tkinter):

    - Add tracks by mood + Top-N (can be repeated; it appends to the queue)
    - Drag to reorder (except the currently playing item)
    - Toggle Loop/Once per item
    - Delete items
    - Loop items get re-appended to the end after they finish
    - TAB toggles the bottom panel between:
        * Most listened (your "full list" view)
        * Similar-to-last-mood (visibility like the old TUI)
    """
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as e:
        raise RuntimeError(
            "Tkinter could not be imported. Run with --tui to use the legacy terminal UI. "
            f"Original error: {e}"
        )

    # ---------------- data / model ----------------
    tags_data = load_tags(tags_file)
    if logging:
        print(f"Loaded {len(tags_data)} tag items from '{tags_file}'.")
        print(f"Loading model '{MODEL_NAME}' on '{DEVICE}'...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    EMB_CACHE.ensure(model, tags_data, mp3_folder, logging=logging)

    # ---------------- audio init ----------------
    FORCE_DEVICE: Optional[str] = None
    if sys.platform == "darwin":
        FORCE_DEVICE = "External Headphones"
        devs = list_pygame_output_devices()
        if logging and devs:
            print(f"[audio] SDL playback devices: {devs}")
        if devs and FORCE_DEVICE not in devs:
            raise RuntimeError(
                f"SDL/pygame output device '{FORCE_DEVICE}' not found.\n"
                f"Available SDL devices: {devs}\n"
                "Set FORCE_DEVICE to one of the exact names above."
            )

    if logging:
        print("Initializing pygame mixer...")
    init_pygame(device_name=FORCE_DEVICE)
    start_device_presence_watchdog(device_name=FORCE_DEVICE, logging=logging)

    next_exit_dt = next_local_time(EXIT_AT_LOCAL_HOUR, EXIT_AT_LOCAL_MINUTE)
    lock_since_wall: float | None = None

    counts = load_listen_counts(mp3_folder, listen_db_filename)
    loud_cache = load_loudness_cache(mp3_folder)

    audio_data: Dict[Path, Dict[str, float | None]] = {}
    runtime_target: Optional[float] = target_dBFS
    resolved_target: Optional[float] = None

    # Playback state
    state = "idle"  # idle | playing | gap | locked_wait
    gap_until = 0.0
    cur_iid: Optional[str] = None
    cur_track: Optional[Path] = None
    cur_ch: Optional["pygame.mixer.Channel"] = None
    cur_snd: Optional["pygame.mixer.Sound"] = None
    cur_start = 0.0
    cur_dur = 0.0

    def _resolve_target() -> Optional[float]:
        nonlocal resolved_target
        if runtime_target is not None:
            resolved_target = float(runtime_target)
            return resolved_target

        sample_path = mp3_folder / sample_filename
        key = str(sample_path)
        mtime = sample_path.stat().st_mtime if sample_path.exists() else 0.0
        entry = loud_cache.get(key)
        if not entry or abs(float(entry.get("mtime", 0.0)) - mtime) > 0.5:
            entry = get_audio_stats(sample_path)
            loud_cache[key] = entry
        loud = entry.get("loudness_dbfs")
        resolved_target = float(loud) if isinstance(loud, (int, float)) else None
        return resolved_target

    def _recompute_scales() -> None:
        tgt = _resolve_target()
        if not isinstance(tgt, (int, float)):
            return
        for p, d in audio_data.items():
            loud = d.get("loudness_dbfs")
            d["scale"] = (
                float(calculate_volume_scale(float(tgt), float(loud)))
                if isinstance(loud, (int, float))
                else 0.5
            )

    def _apply_target(new_target: Optional[float]) -> None:
        nonlocal runtime_target
        runtime_target = new_target
        _recompute_scales()
        if cur_ch and cur_track and cur_track in audio_data:
            try:
                cur_ch.set_volume(float(audio_data[cur_track].get("scale") or 0.5))
            except Exception:
                pass

    def _ensure_audio(tracks: List[Path]) -> None:
        nonlocal resolved_target
        missing = [p for p in tracks if p not in audio_data]
        if not missing:
            return
        resolved, data = build_audio_data_for_playlist(
            playlist=missing,
            target_dBFS=runtime_target,
            mp3_folder=mp3_folder,
            sample_filename=sample_filename,
            cache=loud_cache,
            logging=logging,
        )
        resolved_target = float(resolved)
        audio_data.update(data)
        _recompute_scales()

    # ---------------- UI ----------------
    root = tk.Tk()
    root.title("Headphones – Queue")

    def hide_ui():
        # Hide the window (does NOT minimize into the Dock)
        try:
            root.withdraw()
        except Exception:
            pass

    def show_ui():
        try:
            root.deiconify()
            root.update_idletasks()
            root.lift()
            root.focus_force()

            # topmost trick to bring forward
            root.attributes("-topmost", True)
            root.after(50, lambda: root.attributes("-topmost", False))
        except Exception:
            pass

        # --- add these lines ---
        try:
            _reset_pointer_state()
        except Exception:
            pass
        try:
            root.after(0, lambda: tree.focus_set())
        except Exception:
            pass

    # --- macOS: clicking the Dock icon should re-open the hidden window ---
    if sys.platform == "darwin":
        # expose a Tcl command that calls back into Python
        def _tcl_show_ui():
            show_ui()
            return ""  # Tcl expects a string result

        root.createcommand("PY_SHOW_UI", _tcl_show_ui)

        # Tk on macOS calls this proc when the Dock icon is clicked while the app is running
        root.tk.eval("""
            proc ::tk::mac::ReopenApplication {} {
                PY_SHOW_UI
            }
        """)

    mood_var = tk.StringVar(value=initial_mood or "")
    top_var = tk.IntVar(value=clamp_int(int(top_n), TOP_MIN, TOP_MAX))
    vol_var = tk.StringVar(value="" if target_dBFS is None else str(target_dBFS))
    status_var = tk.StringVar(value="")
    now_var = tk.StringVar(value="Now playing: —")
    list_title_var = tk.StringVar(value=" Most Listened ")
    progress_var = tk.DoubleVar(value=0.0)
    time_var = tk.StringVar(value="—")

    def status(msg: str) -> None:
        status_var.set(str(msg or ""))

    # Top controls
    top = ttk.Frame(root, padding=8)
    top.pack(side="top", fill="x")

    ttk.Label(top, text="Mood").pack(side="left")
    mood_entry = ttk.Entry(top, textvariable=mood_var, width=25)
    mood_entry.pack(side="left", padx=(6, 12))

    ttk.Label(top, text="Top").pack(side="left")
    ttk.Spinbox(top, from_=TOP_MIN, to=TOP_MAX, textvariable=top_var, width=5).pack(
        side="left", padx=(6, 12)
    )

    ttk.Label(top, text="Vol (dBFS)").pack(side="left")
    ttk.Entry(top, textvariable=vol_var, width=10).pack(side="left", padx=(6, 12))

    btn_col = ttk.Frame(top)
    btn_col.pack(side="right", padx=(0, 0))  # use side="right" if you want it at the far right

    btn_add = ttk.Button(btn_col, text="Add to Queue")
    btn_add.pack(fill="x", pady=(0, 0))

    btn_skip = ttk.Button(btn_col, text="Skip")
    btn_skip.pack(fill="x", pady=(0, 0))

    btn_clear = ttk.Button(btn_col, text="Clear")
    btn_clear.pack(fill="x")

    ttk.Label(root, textvariable=now_var, padding=(8, 0, 8, 6)).pack(
        side="top", fill="x"
    )

    # Progress row: bar + time
    prog_row = ttk.Frame(root)
    prog_row.pack(side="top", fill="x", padx=8, pady=(0, 8))

    pbar = ttk.Progressbar(
        prog_row,
        orient="horizontal",
        mode="determinate",
        maximum=100.0,
        variable=progress_var,
    )
    pbar.pack(side="left", fill="x", expand=True)

    ttk.Label(prog_row, textvariable=time_var, width=10, anchor="e").pack(
        side="left", padx=(10, 0)
    )

    # Split area: queue (top) + info list (bottom)
    paned = ttk.Panedwindow(root, orient="vertical")
    paned.pack(side="top", fill="both", expand=True, padx=8, pady=(0, 0))

    queue_frame = ttk.Frame(paned)
    list_frame = ttk.Frame(paned)
    paned.add(queue_frame, weight=3)
    paned.add(list_frame, weight=2)

    # ---- Queue tree ----
    tree = ttk.Treeview(
        queue_frame,
        columns=("track", "mode", "del"),
        show="headings",
        selectmode="browse",
    )
    tree.heading("track", text="Track")
    tree.heading("mode", text="Play")
    tree.heading("del", text="Delete")
    tree.column("track", width=600, stretch=True)
    tree.column("mode", width=70, stretch=False, anchor="center")
    tree.column("del", width=70, stretch=False, anchor="center")

    qsb = ttk.Scrollbar(queue_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=qsb.set)
    tree.pack(side="left", fill="both", expand=True)
    qsb.pack(side="left", fill="y")

    # ---- Bottom list (Most / Similar) ----
    ttk.Label(list_frame, textvariable=list_title_var, padding=(2, 0, 2, 6)).pack(
        side="top", fill="x"
    )
    list_tree = ttk.Treeview(
        list_frame,
        columns=("c1", "c2", "c3"),
        show="headings",
        selectmode="browse",
    )
    list_tree.column("c1", width=50, stretch=False, anchor="center")
    list_tree.column("c2", width=90, stretch=False, anchor="center")
    list_tree.column("c3", width=600, stretch=True, anchor="w")

    lsb = ttk.Scrollbar(list_frame, orient="vertical", command=list_tree.yview)
    list_tree.configure(yscrollcommand=lsb.set)
    list_tree.pack(side="left", fill="both", expand=True)
    lsb.pack(side="left", fill="y")

    ttk.Label(
        root,
        text="Queue: Drag to reorder • Click ‘Play’ to toggle Loop/Once • Click ‘Delete’ to remove",
        padding=(8, 6, 8, 0),
    ).pack(side="top", fill="x")
    ttk.Label(root, textvariable=status_var, padding=8).pack(side="bottom", fill="x")

    def refresh_list() -> None:
        """Redraw the bottom panel as the full list (most listened first)."""
        try:
            for iid in list_tree.get_children():
                list_tree.delete(iid)
        except Exception:
            pass

        # Fixed title + fixed columns
        try:
            list_title_var.set(" Most Listened ")
        except Exception:
            pass

        list_tree.heading("c1", text="#")
        list_tree.heading("c2", text="Plays")
        list_tree.heading("c3", text="Track")

        for i, (name, cnt) in enumerate(sorted_counts(counts), start=1):
            list_tree.insert("", "end", values=(i, cnt, name))

    # ---------------- Queue storage (iid -> {path, loop}) ----------------
    q: Dict[str, Dict[str, object]] = {}
    iid_counter = 0

    def _new_iid() -> str:
        nonlocal iid_counter
        iid_counter += 1
        return f"q{int(time.time()*1000)}_{iid_counter}"

    def _mode(loop: bool) -> str:
        return "Loop" if loop else "Once"

    def q_add(path: Path, loop: bool = True) -> str:
        iid = _new_iid()
        q[iid] = {"path": path, "loop": bool(loop)}
        tree.insert("", "end", iid=iid, values=(path.name, _mode(loop), "X"))
        return iid

    def q_del(iid: str) -> None:
        q.pop(iid, None)
        try:
            tree.delete(iid)
        except Exception:
            pass

    def q_toggle(iid: str) -> None:
        item = q.get(iid)
        if not item:
            return
        item["loop"] = not bool(item["loop"])
        try:
            tree.set(iid, "mode", _mode(bool(item["loop"])))
        except Exception:
            pass

    # Mouse: click mode/delete; drag reorder (mac trackpad reliable)
    press_iid: Optional[str] = None
    press_col: Optional[str] = None
    press_x = 0
    press_y = 0
    dragging = False
    drag_iid: Optional[str] = None

    # Trackpad taps sometimes "wiggle" a few pixels; keep drag threshold modest
    DRAG_THRESHOLD = 7

    def _tree_xy():
        """Return current pointer position in TREE-local coordinates (more reliable than event.x/y on mac)."""
        try:
            x_root = root.winfo_pointerx()
            y_root = root.winfo_pointery()
            return (x_root - tree.winfo_rootx(), y_root - tree.winfo_rooty())
        except Exception:
            return (0, 0)

    def _reset_pointer_state(*_):
        nonlocal press_iid, press_col, press_x, press_y, dragging, drag_iid
        press_iid = None
        press_col = None
        press_x = 0
        press_y = 0
        dragging = False
        drag_iid = None

    # When the window is hidden/shown (withdraw/deiconify or minimize/restore), clear any "stuck" gesture state
    def _on_map(_e=None):
        _reset_pointer_state()
        try:
            root.after(0, tree.focus_set)
        except Exception:
            pass

    def _on_unmap(_e=None):
        _reset_pointer_state()

    root.bind("<Map>", _on_map)
    root.bind("<Unmap>", _on_unmap)

    def on_press(event):
        nonlocal press_iid, press_col, press_x, press_y, dragging, drag_iid

        x, y = _tree_xy()
        press_x, press_y = x, y

        # Only consider real cell presses (ignore heading/separator)
        region = tree.identify_region(x, y)
        if region != "cell":
            _reset_pointer_state()
            return

        press_iid = tree.identify_row(y) or None
        press_col = tree.identify_column(x) or None

        dragging = False
        drag_iid = None

    def on_motion(event):
        nonlocal dragging, drag_iid

        if not press_iid:
            return
        if press_col != "#1":  # only drag by the Track column
            return
        if state == "playing" and press_iid == cur_iid:  # don't drag current playing item
            return

        x, y = _tree_xy()
        dx = abs(x - press_x)
        dy = abs(y - press_y)

        if not dragging and (dx + dy) < DRAG_THRESHOLD:
            return

        dragging = True
        drag_iid = press_iid

        target = tree.identify_row(y)
        if not target or target == drag_iid:
            return

        try:
            idx = tree.index(target)

            # Prevent dropping above currently playing row at index 0
            if state == "playing" and cur_iid and idx == 0 and drag_iid != cur_iid:
                idx = 1

            tree.move(drag_iid, "", idx)
        except Exception:
            pass

    def on_release(event):
        nonlocal press_iid, press_col, dragging, drag_iid

        # Use pointer-based coords again (event.x/y are the flaky part on mac trackpads)
        x, y = _tree_xy()

        if not dragging:
            region = tree.identify_region(x, y)
            if region == "cell":
                iid = tree.identify_row(y) or press_iid
                col = tree.identify_column(x) or press_col

                if iid and col == "#2":  # Play column
                    q_toggle(iid)

                elif iid and col == "#3":  # Delete column
                    if state == "playing" and iid == cur_iid:
                        try:
                            pygame.mixer.stop()
                        except Exception:
                            pass
                        finish(reached_end=False, force_drop=True)
                    else:
                        q_del(iid)

        _reset_pointer_state()

    tree.bind("<ButtonPress-1>", on_press)
    tree.bind("<B1-Motion>", on_motion)
    tree.bind("<ButtonRelease-1>", on_release)

    # ---------------- Mood search (background thread) ----------------
    search_thread: Optional[threading.Thread] = None

    def do_search(mood_text: str, top_val: int) -> None:
        mood_text = (mood_text or "").strip()
        if not mood_text:
            root.after(0, lambda: status("Enter a mood first."))
            return

        playlist, sim_table = compute_top_for_mood(
            model=model,
            tags_data=tags_data,
            mood=mood_text,
            mp3_folder=mp3_folder,
            top_n=top_val,
            logging=logging,
        )

        def apply():
            if not playlist:
                status(f"No matches for mood: “{mood_text}”")
                return

            _ensure_audio(playlist)
            for p in playlist:
                q_add(p, loop=True)

            status(f"Added {len(playlist)} track(s) for “{mood_text}” (top {top_val}).")

        root.after(0, apply)

    def on_add(event=None):
        nonlocal search_thread

        raw = (mood_var.get() or "").strip()
        mood_text, top_dir, vol_dir = parse_mood_and_directives(raw)

        try:
            top_val = int(top_var.get())
        except Exception:
            top_val = clamp_int(int(top_n), TOP_MIN, TOP_MAX)
        if isinstance(top_dir, int):
            top_val = clamp_int(top_dir, TOP_MIN, TOP_MAX)
            top_var.set(top_val)

        vol_raw = (vol_var.get() or "").strip()
        if isinstance(vol_dir, (int, float)):
            new_target = float(vol_dir)
            vol_var.set(str(new_target))
        elif vol_raw:
            try:
                new_target = float(vol_raw)
            except Exception:
                status("Vol must be a number (or blank for sample).")
                return
        else:
            new_target = None

        _apply_target(new_target)

        if search_thread and search_thread.is_alive():
            status("Search already running…")
            return
        if not mood_text:
            status("Enter a mood first.")
            return

        status("Searching…")
        search_thread = threading.Thread(
            target=do_search, args=(mood_text, top_val), daemon=True
        )
        search_thread.start()

    btn_add.configure(command=on_add)
    mood_entry.bind("<Return>", on_add)

    # ---------------- Playback helpers ----------------
    def mark(iid: str, playing: bool) -> None:
        item = q.get(iid)
        if not item:
            return
        name = str(item["path"].name)
        if playing:
            name = "▶ " + name
        try:
            tree.set(iid, "track", name)
        except Exception:
            pass

    def clear_current() -> None:
        nonlocal cur_iid, cur_track, cur_snd, cur_ch, cur_start, cur_dur
        if cur_iid:
            mark(cur_iid, False)
        cur_iid = None
        cur_track = None
        cur_snd = None
        cur_ch = None
        cur_start = 0.0
        cur_dur = 0.0
        now_var.set("Now playing: —")
        progress_var.set(0.0)
        time_var.set("—")

    def start_next() -> None:
        nonlocal state, cur_iid, cur_track, cur_snd, cur_ch, cur_start, cur_dur

        if state != "idle":
            return
        if EXIT_NOW.is_set():
            root.destroy()
            return
        if mac_is_locked_poll():
            state = "locked_wait"
            return

        kids = tree.get_children()
        if not kids:
            clear_current()
            return

        iid = kids[0]
        item = q.get(iid)
        if not item:
            q_del(iid)
            return

        p = item["path"]
        _ensure_audio([p])
        vol_scale = float(audio_data.get(p, {}).get("scale") or 0.5)

        try:
            snd = pygame.mixer.Sound(str(p))
            ch = snd.play()
            if ch is None:
                ch = pygame.mixer.find_channel(True)
                ch.play(snd)
            ch.set_volume(vol_scale)
        except Exception as e:
            status(f"Error playing {p.name}: {e}")
            q_del(iid)
            return

        cur_iid = iid
        cur_track = p
        cur_snd = snd
        cur_ch = ch
        cur_start = time.monotonic()
        try:
            cur_dur = float(snd.get_length())
        except Exception:
            cur_dur = float(audio_data.get(p, {}).get("duration") or 0.0)

        mark(iid, True)
        plays = counts.get(p.stem, 0)
        now_var.set(f"Now playing: {p.name}   •   plays: {plays}")
        state = "playing"

        progress_var.set(0.0)
        if cur_dur and cur_dur > 0:
            time_var.set(f"{_fmt_time(0)} / {_fmt_time(cur_dur)}")
        else:
            time_var.set("—")

    def finish(reached_end: bool, force_drop: bool = False) -> None:
        nonlocal state, gap_until

        if reached_end and cur_track is not None:
            increment_listen(cur_track, counts)
            save_listen_counts(counts, listen_db_filename)
            refresh_list()

        if not cur_iid:
            clear_current()
            state = "gap"
            gap_until = time.monotonic() + float(GAP_SECONDS)
            return

        item = q.get(cur_iid)
        loop = bool(item["loop"]) if item else False

        if item and (not force_drop) and loop:
            mark(cur_iid, False)
            try:
                kids = list(tree.get_children())
                if cur_iid in kids:
                    tree.move(cur_iid, "", len(kids) - 1)
            except Exception:
                pass
        else:
            q_del(cur_iid)

        clear_current()
        state = "gap"
        gap_until = time.monotonic() + float(GAP_SECONDS)

    def on_skip():
        nonlocal state
        if state == "playing":
            try:
                pygame.mixer.stop()
            except Exception:
                pass
            finish(reached_end=False, force_drop=False)

    def on_clear():
        nonlocal state
        try:
            pygame.mixer.stop()
        except Exception:
            pass
        for iid in list(tree.get_children()):
            q_del(iid)
        clear_current()
        state = "idle"
        status("Queue cleared.")

    btn_skip.configure(command=on_skip)
    btn_clear.configure(command=on_clear)

    def _fmt_time(seconds: float) -> str:
        try:
            s = int(max(0, seconds))
        except Exception:
            s = 0
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    # ---------------- Tick loop ----------------
    def tick():
        nonlocal state, gap_until, lock_since_wall

        try:
            maybe_exit(next_exit_dt)
        except SystemExit:
            root.destroy()
            return

        if EXIT_NOW.is_set():
            root.destroy()
            return

        locked = mac_is_locked_poll()
        try:
            lock_since_wall = update_lock_or_exit(locked, lock_since_wall, next_exit_dt)
        except SystemExit:
            root.destroy()
            return

        if state == "playing":
            if locked:
                try:
                    pygame.mixer.stop()
                except Exception:
                    pass
                finish(reached_end=False, force_drop=False)
                state = "locked_wait"
            else:
                # Update progress UI while audio is playing
                if cur_dur and cur_dur > 0:
                    elapsed = max(0.0, time.monotonic() - cur_start)
                    if elapsed > cur_dur:
                        elapsed = cur_dur
                    progress_var.set(100.0 * (elapsed / cur_dur))
                    time_var.set(f"{_fmt_time(elapsed)} / {_fmt_time(cur_dur)}")
                else:
                    progress_var.set(0.0)
                    time_var.set("—")

                if not (cur_ch and cur_ch.get_busy()):
                    reached_end = False
                    if cur_dur and cur_dur > 0:
                        elapsed = max(0.0, time.monotonic() - cur_start)
                        reached_end = elapsed >= max(0.0, cur_dur - 0.75)
                    finish(reached_end=reached_end, force_drop=False)

        if state == "locked_wait" and not locked:
            state = "gap"
            gap_until = time.monotonic() + float(GAP_SECONDS)

        if state == "gap" and time.monotonic() >= float(gap_until):
            state = "idle"

        if state == "idle":
            start_next()

        root.after(200, tick)

    def on_close():
        try:
            pygame.mixer.stop()
        except Exception:
            pass
        try:
            save_loudness_cache(loud_cache, mp3_folder)
        except Exception:
            pass
        try:
            pygame.mixer.quit()
        finally:
            pygame.quit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    _recompute_scales()
    refresh_list()

    if initial_mood:
        on_add()

    try:
        mood_entry.focus_set()
    except Exception:
        pass

    root.after(200, tick)
    # Start hidden so it doesn't pop up
    root.after(0, hide_ui)
    root.mainloop()


# ============================== CLI ==============================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Mood-based MP3 player with a queue GUI (drag reorder, play-once, delete) "
            "+ optional legacy terminal UI."
        )
    )
    p.add_argument("--mood", type=str, default=None, help="Optional initial mood.")
    p.add_argument("--top", type=int, default=1, help="Top-N matches to find/add.")
    p.add_argument(
        "--vol",
        type=float,
        default=None,
        help="Target loudness in dBFS (if omitted, uses sample).",
    )
    p.add_argument("--log", action="store_true", help="Enable detailed logging.")
    p.add_argument(
        "--folder", type=str, default=str(DEFAULT_MP3_FOLDER), help="MP3 folder path."
    )
    p.add_argument(
        "--tags", type=str, default=str(DEFAULT_TAGS_FILE), help="Tags JSON path."
    )
    p.add_argument(
        "--sample",
        type=str,
        default=DEFAULT_SAMPLE_MP3,
        help="Reference MP3 filename in the MP3 folder.",
    )

    # New: explicit legacy mode switch
    p.add_argument("--tui", action="store_true", help="Use the legacy curses UI (playlist loop).")

    # Kept for backwards compatibility (only relevant in --tui mode)
    p.add_argument("--no-tui", action="store_true", help="(Legacy) Disable curses UI.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    mp3_folder = Path(args.folder)
    tags_file = Path(args.tags)
    listen_db = LISTEN_DB_FILE

    # Keep your mid-mp3s hack, but avoid crashing on short paths
    parts = mp3_folder.parts
    if len(parts) > 1 and parts[1] == "mid-mp3s":
        listen_db = "mid_listen_counts.json"
        tags_file = Path("mid_tags.json")

    try:
        if args.tui:
            main_tui(
                initial_mood=args.mood,
                top_n=args.top,
                mp3_folder=mp3_folder,
                tags_file=tags_file,
                sample_filename=args.sample,
                target_dBFS=args.vol,
                logging=bool(args.log),
                enable_tui=(not args.no_tui),
                listen_db_filename=listen_db,
            )
        else:
            main(
                initial_mood=args.mood,
                top_n=args.top,
                mp3_folder=mp3_folder,
                tags_file=tags_file,
                sample_filename=args.sample,
                target_dBFS=args.vol,
                logging=bool(args.log),
                enable_tui=False,
                listen_db_filename=listen_db,
            )
    except SystemExit:
        print("\nPlayback stopped.")
    except KeyboardInterrupt:
        print("\nPlayback interrupted.")
