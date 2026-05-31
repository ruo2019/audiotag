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
PLAYLISTS_FILE = "queue_playlists.json"

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


def playlists_db_path(filename: str) -> Path:
    return Path(__file__).resolve().parent / filename


def load_playlists(db_filename: str) -> Dict[str, List[dict]]:
    raw = safe_read_json(playlists_db_path(db_filename), {})
    data: Dict[str, List[dict]] = raw if isinstance(raw, dict) else {}
    out: Dict[str, List[dict]] = {}
    for name, items in data.items():
        if not isinstance(items, list):
            continue
        clean: List[dict] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            base = item.get("base")
            if not base:
                continue
            clean.append(
                {
                    "base": str(base),
                    "play_once": bool(item.get("play_once", False)),
                }
            )
        if clean:
            out[str(name)] = clean
    return out


def save_playlists(playlists: Dict[str, List[dict]], db_filename: str) -> None:
    atomic_write_json(playlists_db_path(db_filename), playlists or {})


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
      - All panels visible (Queue, Similar, Most)
      - TAB cycles focus between panels
      - ↑/↓ PgUp/PgDn move within focused panel
      - → skip to next track (after a gap)
      - Ctrl+G stops playback and returns to idle
      - Enter submits mood text (with optional top directive)

    NOTE: Ctrl+S may be swallowed by terminal flow control; Ctrl+G is reliable.
    """

    KEY_TAB = 9
    KEY_BS = 127
    CTRL_G = 7
    CTRL_P = 16  # queue move up
    CTRL_N = 14  # queue move down
    CTRL_T = 20  # queue toggle play-once
    CTRL_X = 24  # queue delete
    CTRL_A = 1   # similar add
    SCROLL_DEBOUNCE_SEC = 0.10
    RENDER_MIN_INTERVAL = 1.0 / 30.0

    def __init__(self, enable: bool = True):
        self.enable = enable
        self.stdscr: Optional["curses._CursesWindow"] = None

        self.focus_panel: str = "queue"  # queue | similar | most
        self.queue_scroll = 0
        self.queue_selected = 0
        self.queue_len = 0
        self.similar_scroll = 0
        self.similar_selected = 0
        self.similar_len = 0
        self.most_scroll = 0

        self.input_buffer = ""
        self.status_msg = ""

        self.skip_requested = False
        self.stop_requested = False
        self.queue_move = 0  # -1 up, +1 down
        self.queue_toggle = False
        self.queue_delete = False
        self.similar_add = False
        self.show_help = False
        self._last_scroll_ts = 0.0
        self._last_render_ts = 0.0

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

        if self.show_help:
            if key == 27:  # Esc
                self.show_help = False
            return None

        KEY_PGUP = getattr(curses, "KEY_PPAGE", 339)
        KEY_PGDN = getattr(curses, "KEY_NPAGE", 338)
        KEY_BTAB = getattr(curses, "KEY_BTAB", 353)

        if key == curses.KEY_UP:
            now = time.monotonic()
            if now - self._last_scroll_ts < self.SCROLL_DEBOUNCE_SEC:
                return None
            self._last_scroll_ts = now
            if self.focus_panel == "queue":
                self.queue_selected = max(0, self.queue_selected - 1)
            elif self.focus_panel == "similar":
                self.similar_selected = max(0, self.similar_selected - 1)
            else:
                self.most_scroll = max(0, self.most_scroll - 1)
        elif key == curses.KEY_DOWN:
            now = time.monotonic()
            if now - self._last_scroll_ts < self.SCROLL_DEBOUNCE_SEC:
                return None
            self._last_scroll_ts = now
            if self.focus_panel == "queue":
                self.queue_selected = min(
                    max(0, self.queue_len - 1), self.queue_selected + 1
                )
            elif self.focus_panel == "similar":
                self.similar_selected = min(
                    max(0, self.similar_len - 1), self.similar_selected + 1
                )
            else:
                self.most_scroll += 1
        elif key == KEY_PGUP:
            now = time.monotonic()
            if now - self._last_scroll_ts < self.SCROLL_DEBOUNCE_SEC:
                return None
            self._last_scroll_ts = now
            page = max(1, self.last_h - 12)
            if self.focus_panel == "queue":
                self.queue_selected = max(0, self.queue_selected - page)
            elif self.focus_panel == "similar":
                self.similar_selected = max(0, self.similar_selected - page)
            else:
                self.most_scroll = max(0, self.most_scroll - page)
        elif key == KEY_PGDN:
            now = time.monotonic()
            if now - self._last_scroll_ts < self.SCROLL_DEBOUNCE_SEC:
                return None
            self._last_scroll_ts = now
            page = max(1, self.last_h - 12)
            if self.focus_panel == "queue":
                self.queue_selected = min(
                    max(0, self.queue_len - 1),
                    self.queue_selected + page,
                )
            elif self.focus_panel == "similar":
                self.similar_selected = min(
                    max(0, self.similar_len - 1),
                    self.similar_selected + page,
                )
            else:
                self.most_scroll += page
        elif key in (self.KEY_TAB, KEY_BTAB):
            order = ["queue", "similar", "most"]
            try:
                idx = order.index(self.focus_panel)
            except ValueError:
                idx = 0
            if key == KEY_BTAB:
                idx = (idx - 1) % len(order)
            else:
                idx = (idx + 1) % len(order)
            self.focus_panel = order[idx]
        elif key == curses.KEY_RIGHT:
            self.skip_requested = True
            self.status_msg = "Skipping to next in 10s..."
        elif key == self.CTRL_G:
            self.stop_requested = True
            self.status_msg = "Stopped. Waiting for mood..."
        elif key == self.CTRL_P and self.focus_panel == "queue":
            self.queue_move = -1
        elif key == self.CTRL_N and self.focus_panel == "queue":
            self.queue_move = 1
        elif key == self.CTRL_T and self.focus_panel == "queue":
            self.queue_toggle = True
        elif key == self.CTRL_X and self.focus_panel == "queue":
            self.queue_delete = True
        elif key == self.CTRL_A and self.focus_panel == "similar":
            self.similar_add = True
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
        queue_items: Optional[List[Dict[str, object]]],
    ) -> Optional[str]:
        if not self.enable or not self.stdscr:
            return None

        queue_items = queue_items or []
        self.queue_len = len(queue_items)
        self.similar_len = len(similar_entries or [])
        submitted = self._handle_key()

        now = time.monotonic()
        if not self.show_help and (now - self._last_render_ts) < self.RENDER_MIN_INTERVAL:
            return submitted
        self._last_render_ts = now

        self.stdscr.erase()
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
        if self.status_msg:
            status = self.status_msg
        if status:
            self._draw(status_y, 2, status)

        if table_height > 0:
            def render_pane(
                title: str, rows: List[str], x: int, y: int, width: int, height: int
            ) -> None:
                if height <= 0 or width <= 0:
                    return
                self._draw(y, x, title[:width].ljust(width))
                max_rows = max(0, height - 1)
                for i in range(max_rows):
                    line = rows[i] if i < len(rows) else ""
                    padded = (" " + line)[:width].ljust(width)
                    self._draw(y + 1 + i, x, padded)

            most_height = max(6, int(table_height * 0.6))
            middle_height = table_height - most_height - 1
            if middle_height < 4:
                middle_height = max(0, table_height - 4)
                most_height = max(0, table_height - middle_height - 1)

            left_w = max(24, (w - 1) // 2)
            right_w = max(0, w - left_w - 1)
            mid_y = table_top
            most_y = table_top + middle_height + 1

            # Queue pane (left)
            if self.queue_selected >= self.queue_len:
                self.queue_selected = max(0, self.queue_len - 1)
            q_content_h = max(0, middle_height - 1)
            q_max_scroll = max(0, self.queue_len - q_content_h)
            if self.queue_selected < self.queue_scroll:
                self.queue_scroll = self.queue_selected
            if self.queue_selected >= self.queue_scroll + q_content_h:
                self.queue_scroll = self.queue_selected - q_content_h + 1
            self.queue_scroll = min(self.queue_scroll, q_max_scroll)
            q_rows: List[str] = []
            q_name_w = max(1, left_w - (2 + 3 + 3 + 4 + 5))
            for i in range(self.queue_scroll, min(self.queue_len, self.queue_scroll + q_content_h)):
                item = queue_items[i]
                path = item.get("path")
                name = path.name if isinstance(path, Path) else "(unknown)"
                mode = "once" if item.get("play_once") else "loop"
                sel = ">" if i == self.queue_selected and self.focus_panel == "queue" else " "
                name = name[:q_name_w].ljust(q_name_w)
                q_rows.append(f"{sel} {i+1:>3} | {name}| {mode:>4}  ")

            # Similar pane (right)
            entries = similar_entries or []
            if self.similar_selected >= len(entries):
                self.similar_selected = max(0, len(entries) - 1)
            s_content_h = max(0, middle_height - 1)
            s_max_scroll = max(0, len(entries) - s_content_h)
            if self.similar_selected < self.similar_scroll:
                self.similar_scroll = self.similar_selected
            if self.similar_selected >= self.similar_scroll + s_content_h:
                self.similar_scroll = self.similar_selected - s_content_h + 1
            self.similar_scroll = min(self.similar_scroll, s_max_scroll)
            s_rows: List[str] = []
            s_name_w = max(1, right_w - (2 + 3 + 3 + 6 + 5))
            for i in range(self.similar_scroll, min(len(entries), self.similar_scroll + s_content_h)):
                base, score = entries[i]
                sel = ">" if i == self.similar_selected and self.focus_panel == "similar" else " "
                base = str(base)[:s_name_w].ljust(s_name_w)
                s_rows.append(f"{sel} {i+1:>3} | {base}| {score:>6.3f}  ")

            # Most pane (bottom full width)
            most_entries = sorted_counts(counts)
            m_content_h = max(0, most_height - 1)
            m_max_scroll = max(0, len(most_entries) - m_content_h)
            self.most_scroll = min(max(0, self.most_scroll), m_max_scroll)
            m_rows: List[str] = []
            m_name_w = max(1, w - (2 + 3 + 3 + 4 + 5))
            for i in range(self.most_scroll, min(len(most_entries), self.most_scroll + m_content_h)):
                name, cnt = most_entries[i]
                sel = ">" if self.focus_panel == "most" and i == self.most_scroll else " "
                name = str(name)[:m_name_w].ljust(m_name_w)
                m_rows.append(f"{sel} {i+1:>3} | {cnt:>4} | {name}")

            q_title = ("[Queue]" if self.focus_panel == "queue" else " Queue ").ljust(left_w)
            s_title = (
                "[Similar]" if self.focus_panel == "similar" else " Similar "
            ).ljust(right_w)
            if similar_mood:
                s_title = (s_title[: max(0, right_w - len(similar_mood) - 3)] + f" ({similar_mood})")[:right_w]
            m_title = "[Most]" if self.focus_panel == "most" else " Most "

            if middle_height > 0:
                render_pane(q_title, q_rows, 0, mid_y, left_w, middle_height)
                if right_w > 0:
                    render_pane(s_title, s_rows, left_w + 1, mid_y, right_w, middle_height)
                    for i in range(middle_height):
                        self._draw(mid_y + i, left_w, "│")

            if most_height > 0:
                self._hline(most_y - 1, "-")
                render_pane(m_title, m_rows, 0, most_y, w, most_height)

        # Input bar + footer
        self._hline(input_y - 1, "-")
        prompt = "mood: "
        if not self.input_buffer and not now_playing:
            self._draw(
                input_y, 2, prompt + "(type a mood + Enter; optional: (top N) / top=N)"
            )
        else:
            self._draw(input_y, 2, prompt + self.input_buffer)

        self._draw(
            h - 1,
            2,
            "Enter submit • TAB focus • Ctrl+A add • Ctrl+G stop • --help",
        )

        if self.show_help:
            help_lines = [
                "Shortcuts",
                "",
                "Enter: submit mood",
                "TAB: focus (Queue/Similar/Most)",
                "↑/↓ PgUp/PgDn: move in focused panel",
                "→: skip track",
                "Ctrl+G: stop",
                "Ctrl+A: add selected (Similar)",
                "Ctrl+P/Ctrl+N: move (Queue)",
                "Ctrl+T: toggle loop/once (Queue)",
                "Ctrl+X: delete (Queue)",
                "",
                "Commands:",
                ":list playlists",
                ":save NAME",
                ":load NAME",
                "",
                "Press Esc to close",
            ]
            max_line = max(len(line) for line in help_lines) if help_lines else 0
            box_w = min(w - 4, max(30, max_line + 4))
            box_h = min(h - 4, len(help_lines) + 4)
            box_x = max(0, (w - box_w) // 2)
            box_y = max(0, (h - box_h) // 2)
            top = "╭" + ("─" * (box_w - 2)) + "╮"
            mid = "│" + (" " * (box_w - 2)) + "│"
            bot = "╰" + ("─" * (box_w - 2)) + "╯"
            self._draw(box_y, box_x, top)
            for i in range(1, box_h - 1):
                self._draw(box_y + i, box_x, mid)
            self._draw(box_y + box_h - 1, box_x, bot)
            content_start = box_y + 2
            for i, line in enumerate(help_lines[: box_h - 4]):
                self._draw(content_start + i, box_x + 2, line[: box_w - 4])

        try:
            self.stdscr.noutrefresh()
            curses.doupdate()
        except Exception:
            pass

        if submitted:
            if submitted.strip() != "--help":
                self.status_msg = f"Submitted: “{submitted}”"
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

def start_device_presence_watchdog(device_name: str) -> None:
    def _run():
        while not EXIT_NOW.is_set():
            devs = list_pygame_output_devices()
            # If we can enumerate and it's gone -> stop and exit
            if devs and device_name not in devs:
                if logging:
                    print(f"[audio] Output device '{device_name}' disconnected. Stopping.")
                EXIT_NOW.set()
                try:
                    pygame.mixer.music.stop()
                except Exception:
                    pass
                return
            time.sleep(0.5)

    threading.Thread(target=_run, daemon=True).start()

# ============================== MAIN ==============================


def init_pygame(device_name: Optional[str] = None) -> None:
    # Initialize pygame mixer. If device_name is provided (pygame 2.x),
    # pin THIS program's audio output to that device.
    try:
        pygame.mixer.pre_init(44100, -16, 2, 2048, devicename=device_name)
        pygame.init()
        pygame.mixer.init(44100, -16, 2, 2048, devicename=device_name)
        return
    except TypeError:
        # Older pygame: no devicename= support.
        pygame.mixer.pre_init(44100, -16, 2, 2048)
        pygame.init()
        pygame.mixer.init()
        return
    except pygame.error:
        # Retry with a more compatible sample rate.
        try:
            pygame.mixer.pre_init(22050, -16, 2, 2048, devicename=device_name)
            pygame.init()
            pygame.mixer.init(22050, -16, 2, 2048, devicename=device_name)
            return
        except TypeError:
            pygame.mixer.pre_init(22050, -16, 2, 2048)
            pygame.init()
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
    start_device_presence_watchdog(device_name=FORCE_DEVICE)


    next_exit_dt = next_local_time(EXIT_AT_LOCAL_HOUR, EXIT_AT_LOCAL_MINUTE)
    lock_since_wall: float | None = None
    if logging:
        print(
            f"Exit policy: lock>{LOCK_EXIT_MINUTES}m or at {next_exit_dt:%Y-%m-%d %H:%M} local."
        )

    counts = load_listen_counts(mp3_folder, listen_db_filename)
    loud_cache = load_loudness_cache(mp3_folder)
    playlists_filename = f"queue_playlists_{mp3_folder.name or 'default'}.json"
    playlists_db = load_playlists(playlists_filename)

    current_similar_entries: Optional[List[Tuple[str, float]]] = None
    current_similar_mood: Optional[str] = None
    queue: List[Dict[str, object]] = []
    current_playing_item: Optional[Dict[str, object]] = None

    audio_data: Dict[Path, Dict[str, float | None]] = {}
    resolved_target_loudness: Optional[float] = None

    current_top_n = clamp_int(top_n, TOP_MIN, TOP_MAX)
    runtime_target_dBFS: Optional[float] = target_dBFS

    def handle_playlist_command(
        text: str,
        tui: Optional[CursesTUI],
    ) -> bool:
        nonlocal playlists_db, current_playing_item
        raw = (text or "").strip()
        if not raw.startswith(":"):
            return False
        parts = raw[1:].strip().split(None, 1)
        if not parts:
            return True
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in {"list", "ls"}:
            names = sorted(playlists_db.keys())
            if tui and tui.enable:
                tui.status_msg = (
                    "Playlists: " + ", ".join(names) if names else "No saved playlists."
                )
            return True

        if cmd == "save":
            if not arg:
                if tui and tui.enable:
                    tui.status_msg = "Usage: :save NAME"
                return True
            items: List[dict] = []
            if current_playing_item:
                path = current_playing_item.get("path")
                if isinstance(path, Path):
                    items.append(
                        {
                            "base": path.stem,
                            "play_once": bool(current_playing_item.get("play_once", False)),
                        }
                    )
            for item in queue:
                path = item.get("path")
                if isinstance(path, Path):
                    items.append(
                        {
                            "base": path.stem,
                            "play_once": bool(item.get("play_once", False)),
                        }
                    )
            playlists_db[arg] = items
            save_playlists(playlists_db, playlists_filename)
            if tui and tui.enable:
                tui.status_msg = f"Saved playlist: {arg} ({len(items)} tracks)"
            return True

        if cmd == "load":
            if not arg:
                if tui and tui.enable:
                    tui.status_msg = "Usage: :load NAME"
                return True
            items = playlists_db.get(arg)
            if not items:
                if tui and tui.enable:
                    tui.status_msg = f"Playlist not found: {arg}"
                return True
            paths: List[Path] = []
            for it in items:
                base = it.get("base")
                if not base:
                    continue
                p = mp3_folder / f"{base}.mp3"
                if p.exists():
                    queue.append(
                        {
                            "path": p,
                            "play_once": bool(it.get("play_once", False)),
                            "mood": f"playlist:{arg}",
                        }
                    )
                    paths.append(p)
            if paths:
                ensure_audio_data_for_tracks(paths)
            if tui and tui.enable:
                tui.status_msg = f"Loaded playlist: {arg} ({len(paths)} tracks)"
            return True

        if tui and tui.enable:
            tui.status_msg = "Commands: :list | :save NAME | :load NAME"
        return True

    def load_similar_for_mood(mood_text: str) -> Tuple[List[Path], List[Tuple[str, float]]]:
        nonlocal current_similar_entries, current_similar_mood

        pl, sim = compute_top_for_mood(
            model=model,
            tags_data=tags_data,
            mood=mood_text,
            mp3_folder=mp3_folder,
            top_n=current_top_n,
            logging=logging,
        )
        current_similar_entries = sim
        current_similar_mood = mood_text
        return pl, sim

    def enqueue_tracks(tracks: List[Path], mood_text: Optional[str]) -> None:
        for p in tracks:
            queue.append({"path": p, "play_once": False, "mood": mood_text})

    def ensure_audio_data_for_tracks(paths: List[Path]) -> None:
        nonlocal resolved_target_loudness
        missing = [p for p in paths if p not in audio_data]
        if not missing:
            return
        resolved, data = build_audio_data_for_playlist(
            playlist=missing,
            target_dBFS=runtime_target_dBFS,
            mp3_folder=mp3_folder,
            sample_filename=sample_filename,
            cache=loud_cache,
            logging=logging,
        )
        audio_data.update(data)
        if resolved_target_loudness is None:
            resolved_target_loudness = resolved

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
        nonlocal current_top_n
        if text.strip() == "--help":
            if tui and tui.enable:
                tui.show_help = True
            return
        if handle_playlist_command(text, tui):
            return
        mood_text, new_top, new_vol = parse_mood_and_directives(text)

        if new_vol is not None:
            apply_new_target_loudness(new_vol, tui, current_track, current_channel)

        if new_top is not None:
            current_top_n = clamp_int(new_top, TOP_MIN, TOP_MAX)
            if tui and tui.enable:
                tui.status_msg = f"Top set to {current_top_n}"
        if mood_text:
            pl, _ = load_similar_for_mood(mood_text)
            if pl:
                if tui and tui.enable:
                    tui.focus_panel = "similar"
                    tui.status_msg = (
                        f"Loaded mood: “{mood_text}” (top {current_top_n}) — CTRL+A to add"
                    )
                if not enable_tui:
                    enqueue_tracks(pl, mood_text)
            else:
                if tui and tui.enable:
                    tui.status_msg = f"No matches for mood: “{mood_text}”"

    def reset_to_idle(tui: Optional[CursesTUI]) -> None:
        nonlocal current_similar_entries, current_similar_mood, queue
        current_similar_entries = None
        current_similar_mood = None
        queue = []
        if tui and tui.enable:
            tui.focus_panel = "queue"
            tui.status_msg = "Stopped. Waiting for mood..."

    # Seed
    if initial_mood:
        pl, _ = load_similar_for_mood(initial_mood)
        if pl:
            if not enable_tui:
                enqueue_tracks(pl, initial_mood)
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
        if enable_tui and queue:
            tui.focus_panel = "queue"

        def apply_queue_actions() -> None:
            if not queue:
                tui.queue_selected = 0
                tui.queue_move = 0
                tui.queue_toggle = False
                tui.queue_delete = False
                return

            if tui.queue_move:
                idx = tui.queue_selected
                new_idx = idx + (1 if tui.queue_move > 0 else -1)
                if 0 <= idx < len(queue) and 0 <= new_idx < len(queue):
                    queue[idx], queue[new_idx] = queue[new_idx], queue[idx]
                    tui.queue_selected = new_idx
                    tui.status_msg = "Queue reordered."
                tui.queue_move = 0

            if tui.queue_toggle:
                idx = tui.queue_selected
                if 0 <= idx < len(queue):
                    cur = bool(queue[idx].get("play_once"))
                    queue[idx]["play_once"] = not cur
                    tui.status_msg = "Toggled play-once." if not cur else "Set to loop."
                tui.queue_toggle = False

            if tui.queue_delete:
                idx = tui.queue_selected
                if 0 <= idx < len(queue):
                    del queue[idx]
                    if idx >= len(queue):
                        tui.queue_selected = max(0, len(queue) - 1)
                    tui.status_msg = "Removed from queue."
                tui.queue_delete = False

        def apply_similar_actions() -> None:
            if not tui.similar_add:
                return
            tui.similar_add = False
            entries = current_similar_entries or []
            if not entries:
                tui.status_msg = "No similar tracks to add."
                return
            if tui.similar_selected >= len(entries):
                tui.similar_selected = max(0, len(entries) - 1)
            base = entries[tui.similar_selected][0]
            path = mp3_folder / f"{base}.mp3"
            if not path.exists():
                tui.status_msg = "Track not found on disk."
                return
            ensure_audio_data_for_tracks([path])
            queue.append({"path": path, "play_once": False, "mood": current_similar_mood})
            tui.status_msg = f"Added to queue: {base}"

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
                    queue_items=queue,
                )
                if submitted:
                    apply_submission(submitted, tui)
                apply_queue_actions()
                apply_similar_actions()

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

                clock.tick(60)

            return False, lock_since_wall

        try:
            while True:
                maybe_exit(next_exit_dt)

                # Idle (only interactive in TUI mode)
                if not queue:
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
                        queue_items=queue,
                    )
                    if submitted:
                        if submitted.strip() == "--help":
                            tui.show_help = True
                            time.sleep(0.01)
                            continue
                        if handle_playlist_command(submitted, tui):
                            time.sleep(0.01)
                            continue
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
                                pl, _ = load_similar_for_mood(mood_text)
                                if pl:
                                    tui.focus_panel = "similar"
                                    tui.status_msg = (
                                        f"Loaded mood: “{mood_text}” (top {current_top_n}) — press 'a' to add"
                                    )
                                else:
                                    tui.status_msg = f"No matches for mood: “{mood_text}”"
                            except Exception as e:
                                tui.status_msg = f"Error building queue: {e}"
                    apply_queue_actions()
                    apply_similar_actions()
                    time.sleep(0.01)
                    continue

                # Queue loop
                current_item = queue.pop(0)
                current_playing_item = current_item if isinstance(current_item, dict) else None
                track_path = current_item.get("path") if isinstance(current_item, dict) else None
                play_once = bool(current_item.get("play_once")) if isinstance(current_item, dict) else False

                if not isinstance(track_path, Path):
                    continue

                # refresh queue selection bounds
                if tui.queue_selected >= len(queue):
                    tui.queue_selected = max(0, len(queue) - 1)

                # Play single item
                while True:
                    maybe_exit(next_exit_dt)

                    if track_path not in audio_data:
                        break

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
                                    queue_items=queue,
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
                                apply_queue_actions()
                                apply_similar_actions()

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

                            clock.tick(60)

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
                            current_playing_item = None
                            break

                        # Between tracks
                        if user_skip or not mac_is_locked_poll():
                            stopped, lock_since_wall = pause_silence(GAP_SECONDS)
                            if stopped:
                                break

                        # Requeue if looping
                        if not play_once:
                            queue.append(current_item)
                        current_playing_item = None
                        break

                    except pygame.error as e:
                        if enable_tui:
                            tui.status_msg = f"Error playing {filename_ext}: {e}"
                        else:
                            print(f"Error playing {filename_ext}: {e}")
                        time.sleep(1)
                        if not play_once:
                            queue.append(current_item)
                        current_playing_item = None
                        break
                    except Exception as e:
                        if enable_tui:
                            tui.status_msg = f"Unexpected error: {e}"
                        else:
                            print(f"Unexpected error for {filename_ext}: {e}")
                        time.sleep(1)
                        if not play_once:
                            queue.append(current_item)
                        current_playing_item = None
                        break

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


# ============================== CLI ==============================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mood TUI player with volume normalization, listen counters, and Similar/Most tabbed view."
    )
    p.add_argument("--mood", type=str, default=None, help="Optional initial mood.")
    p.add_argument("--top", type=int, default=1, help="Top-N matches to find/play.")
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
    p.add_argument("--no-tui", action="store_true", help="Disable curses UI.")
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
        main(
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
    except SystemExit:
        print("\nPlayback stopped.")
    except KeyboardInterrupt:
        print("\nPlayback interrupted.")
