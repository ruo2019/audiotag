from __future__ import annotations

import argparse
import curses
import json
import locale
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pygame
import torch
from mutagen.mp3 import MP3
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from sentence_transformers import SentenceTransformer
from tabulate import tabulate

HEADPHONES_LOST = threading.Event()

# ============================== CONFIG ==============================

locale.setlocale(locale.LC_ALL, "")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

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
DEFAULT_TOP_N = 5

# Between-track (and post-unlock) gap
GAP_SECONDS = 5.0

# Caches stored inside mp3 folder
EMB_CACHE_VERSION = "v3"
EMB_CACHE_NAME = ".track_emb_cache.npz"

LOUD_CACHE_VERSION = 2
LOUD_CACHE_NAME = ".loudness_cache.json"
BASE_VOLUME_SCALE = 0.5
TRUE_PEAK_LIMIT_DBTP = -1.0

# Listen DBs stored next to this script
LISTEN_DB_FILE = "listen_counts.json"
LISTEN_TIMESTAMPS_FILE = "listen_timestamps.json"
MARKOV_SESSION_CUTOFF_SECONDS = 10 * 60

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


def listen_timestamps_filename_for_folder(mp3_folder: Path) -> str:
    folder_name = mp3_folder.name or "default"
    return f"listen_timestamps_{folder_name}.json"


def _listen_event_track_name(track_path: Path) -> str:
    return track_path.name


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
        out[str(name)] = clean
    return out


def save_playlists(playlists: Dict[str, List[dict]], db_filename: str) -> None:
    atomic_write_json(playlists_db_path(db_filename), playlists or {})


def save_listen_counts(counts: Dict[str, int], db_filename: str) -> None:
    atomic_write_json(listen_db_path(db_filename), counts)


def increment_listen(track_path: Path, counts: Dict[str, int]) -> None:
    counts[track_path.stem] = counts.get(track_path.stem, 0) + 1


def load_listen_timestamps(mp3_dir: Path, db_filename: str) -> List[dict]:
    raw = safe_read_json(listen_db_path(db_filename), [])
    events: List[dict] = []

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            ts = item.get("timestamp")
            track = item.get("track")
            if ts and track:
                events.append({"timestamp": str(ts), "track": str(track)})
        return events

    # Backward compatibility for the earlier {stem: [timestamp, ...]} format.
    if isinstance(raw, dict):
        known_tracks = {}
        try:
            for p in mp3_dir.iterdir():
                if p.is_file() and p.suffix.lower() == ".mp3":
                    known_tracks[p.stem] = p.name
        except Exception:
            pass

        for stem, timestamps in raw.items():
            if not isinstance(timestamps, list):
                continue
            track_name = known_tracks.get(str(stem), f"{stem}.mp3")
            for ts in timestamps:
                if ts:
                    events.append({"timestamp": str(ts), "track": track_name})

        events.sort(key=lambda item: item["timestamp"])

    return events


def save_listen_timestamps(history: List[dict], db_filename: str) -> None:
    atomic_write_json(listen_db_path(db_filename), history)


def record_listen_timestamp(
    track_path: Path,
    history: List[dict],
    listened_at: Optional[datetime] = None,
) -> dict:
    when = (listened_at or datetime.now().astimezone()).isoformat(timespec="seconds")
    event = {"timestamp": when, "track": _listen_event_track_name(track_path)}
    history.append(event)
    return event


def build_duration_by_track_name(mp3_dir: Path) -> Dict[str, float]:
    durations: Dict[str, float] = {}
    try:
        for path in mp3_dir.iterdir():
            if path.is_file() and path.suffix.lower() == ".mp3":
                durations[path.name] = get_audio_duration_seconds(path)
    except Exception:
        pass
    return durations


def build_markov_transition_counts(
    mp3_dir: Path,
    history: List[dict],
    duration_by_track_name: Dict[str, float],
    session_cutoff_seconds: float = MARKOV_SESSION_CUTOFF_SECONDS,
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    transitions: Dict[str, Dict[str, int]] = {}
    global_counts: Dict[str, int] = {}
    known_tracks: set[str] = set()

    try:
        for path in mp3_dir.iterdir():
            if path.is_file() and path.suffix.lower() == ".mp3":
                known_tracks.add(path.name)
    except Exception:
        pass

    events: List[Tuple[datetime, str]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        track = item.get("track")
        stamp = item.get("timestamp")
        if not track or not stamp:
            continue
        track_name = str(track)
        if known_tracks and track_name not in known_tracks:
            continue
        try:
            events.append((datetime.fromisoformat(str(stamp)), track_name))
        except Exception:
            continue

    events.sort(key=lambda item: item[0])

    for _, track_name in events:
        global_counts[track_name] = global_counts.get(track_name, 0) + 1

    for index in range(len(events) - 1):
        current_finish, current_track = events[index]
        next_finish, next_track = events[index + 1]
        next_duration = float(duration_by_track_name.get(next_track, 0.0) or 0.0)
        if next_duration <= 0.0:
            next_path = mp3_dir / next_track
            if next_path.exists():
                next_duration = get_audio_duration_seconds(next_path)
                duration_by_track_name[next_track] = next_duration
        next_start = next_finish - timedelta(seconds=max(0.0, next_duration))
        gap_seconds = (next_start - current_finish).total_seconds()
        if gap_seconds < 0.0:
            gap_seconds = 0.0
        if gap_seconds > float(session_cutoff_seconds):
            continue
        current_counts = transitions.setdefault(current_track, {})
        current_counts[next_track] = current_counts.get(next_track, 0) + 1

    return transitions, global_counts


def weighted_choice(weights: Dict[str, float]) -> Optional[str]:
    items = [(name, float(weight)) for name, weight in weights.items() if float(weight) > 0.0]
    if not items:
        return None
    population = [name for name, _ in items]
    probabilities = [weight for _, weight in items]
    return random.choices(population, weights=probabilities, k=1)[0]


def choose_auto_track_name(
    previous_track_name: Optional[str],
    transitions: Dict[str, Dict[str, int]],
    global_counts: Dict[str, int],
) -> Optional[str]:
    if previous_track_name:
        local_weights = transitions.get(previous_track_name) or {}
        picked = weighted_choice({name: float(weight) for name, weight in local_weights.items()})
        if picked:
            return picked

    fallback_weights: Dict[str, float] = {
        name: float(count) for name, count in global_counts.items() if int(count) > 0
    }
    if previous_track_name:
        fallback_weights[previous_track_name] = fallback_weights.get(previous_track_name, 0.0) + 3.0
    return weighted_choice(fallback_weights)


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
    target_lufs: float | None,
    current_lufs: float | None,
    current_true_peak_dbtp: float | None = None,
) -> float:
    if target_lufs is None or current_lufs is None:
        return BASE_VOLUME_SCALE
    if target_lufs <= -100.0 or current_lufs <= -100.0:
        return BASE_VOLUME_SCALE
    db_difference = target_lufs - current_lufs
    if (
        current_true_peak_dbtp is not None
        and current_true_peak_dbtp > -100.0
        and db_difference > 0.0
    ):
        base_gain_db = 20.0 * math.log10(BASE_VOLUME_SCALE)
        max_safe_gain = TRUE_PEAK_LIMIT_DBTP - current_true_peak_dbtp - base_gain_db
        db_difference = min(db_difference, max_safe_gain)
    scale_factor = 10 ** (db_difference / 20)
    return max(0.0, min(1.0, BASE_VOLUME_SCALE * scale_factor))


def analyze_loudness_lufs(mp3_path: Path) -> Tuple[float | None, float | None]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(mp3_path),
        "-af",
        f"loudnorm=I=-16:TP={TRUE_PEAK_LIMIT_DBTP}:LRA=11:print_format=json",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None, None
    matches = re.findall(r"(\{\s*\"input_i\".*?\})", proc.stderr, re.S)
    if not matches:
        return None, None
    try:
        stats = json.loads(matches[-1])
        lufs = float(stats["input_i"])
        true_peak = float(stats["input_tp"])
        return lufs, true_peak
    except Exception:
        return None, None


def get_audio_stats(mp3_path: Path) -> dict:
    try:
        audio = AudioSegment.from_mp3(str(mp3_path))
        dur = float(audio.duration_seconds or 0.0)
        loudness_lufs, true_peak_dbtp = analyze_loudness_lufs(mp3_path)
        if loudness_lufs == -math.inf:
            loudness_lufs = -100.0
        if true_peak_dbtp == -math.inf:
            true_peak_dbtp = -100.0
        return {
            "loudness_lufs": loudness_lufs,
            "true_peak_dbtp": true_peak_dbtp,
            "duration": dur,
            "mtime": mp3_path.stat().st_mtime,
        }
    except CouldntDecodeError:
        return {
            "loudness_lufs": None,
            "true_peak_dbtp": None,
            "duration": 0.0,
            "mtime": mp3_path.stat().st_mtime,
        }
    except Exception:
        try:
            mtime = mp3_path.stat().st_mtime
        except Exception:
            mtime = 0.0
        return {
            "loudness_lufs": None,
            "true_peak_dbtp": None,
            "duration": 0.0,
            "mtime": mtime,
        }


def get_audio_duration_seconds(mp3_path: Path) -> float:
    try:
        return float(MP3(str(mp3_path)).info.length or 0.0)
    except Exception:
        try:
            audio = AudioSegment.from_mp3(str(mp3_path))
            return float(audio.duration_seconds or 0.0)
        except Exception:
            return 0.0


def build_audio_data_for_playlist(
    playlist: List[Path],
    target_lufs: Optional[float],
    mp3_folder: Path,
    sample_filename: str,
    cache: Dict[str, dict],
    logging: bool = False,
) -> Tuple[float, Dict[Path, Dict[str, float | None]]]:
    if target_lufs is None:
        sample_path = mp3_folder / sample_filename
        sample_key = str(sample_path)
        sample_mtime = sample_path.stat().st_mtime if sample_path.exists() else 0.0
        entry = cache.get(sample_key)
        if not entry or abs(float(entry.get("mtime", 0.0)) - sample_mtime) > 0.5:
            entry = get_audio_stats(sample_path)
            cache[sample_key] = entry
        sample_loud = entry.get("loudness_lufs")
        if sample_loud is None:
            raise RuntimeError(
                f"Could not analyze reference sample '{sample_filename}'"
            )
        resolved_target = float(sample_loud)
    else:
        resolved_target = float(target_lufs)

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

        loud = entry.get("loudness_lufs")
        true_peak = entry.get("true_peak_dbtp")
        dur = float(entry.get("duration") or 0.0)
        if loud is None:
            audio_data[p] = {
                "loudness_lufs": None,
                "true_peak_dbtp": None,
                "scale": 0.5,
                "duration": dur,
            }
        else:
            audio_data[p] = {
                "loudness_lufs": float(loud),
                "true_peak_dbtp": (
                    float(true_peak) if isinstance(true_peak, (int, float)) else None
                ),
                "scale": float(
                    calculate_volume_scale(
                        resolved_target,
                        float(loud),
                        float(true_peak) if isinstance(true_peak, (int, float)) else None,
                    )
                ),
                "duration": dur,
            }

    if logging:
        print(f"[loud] target={resolved_target:.2f} LUFS; tracks={len(playlist)}")

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


def normalize_track_name_for_search(name: object) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"\.mp3$", "", text)
    text = re.sub(r"[_\-/.,;:()[\]{}]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tags_fingerprint(tags_data: dict, mp3_folder: Path) -> str:
    rows: List[str] = []
    try:
        mp3_paths = sorted(
            (
                p
                for p in mp3_folder.iterdir()
                if p.is_file() and p.suffix.lower() == ".mp3"
            ),
            key=lambda p: p.stem.lower(),
        )
    except Exception:
        mp3_paths = []

    for p in mp3_paths:
        base = p.stem
        tag_list = (tags_data or {}).get(base, [])
        try:
            mtime = p.stat().st_mtime
        except Exception:
            mtime = 0.0
        rows.append(
            f"{base}|{normalize_track_name_for_search(base)}|"
            f"{canonicalize_tags(tag_list or [])}|{mtime:.0f}"
        )
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
        try:
            mp3_paths = sorted(
                (
                    p
                    for p in mp3_folder.iterdir()
                    if p.is_file() and p.suffix.lower() == ".mp3"
                ),
                key=lambda p: p.stem.lower(),
            )
        except Exception:
            mp3_paths = []

        for p in mp3_paths:
            base = str(p.stem)
            tag_text = canonicalize_tags((tags_data or {}).get(base, []) or [])
            name_text = normalize_track_name_for_search(base)
            names.append(base)
            if tag_text:
                texts.append(f"music track named {name_text}; music that is {tag_text}")
            else:
                texts.append(f"music track named {name_text}")

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
    query_name_text = normalize_track_name_for_search(mood)
    query_words = set(re.findall(r"[a-z0-9]+", query_name_text))

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
    name_match_scores: List[int] = []
    for base in EMB_CACHE.names:
        tag_set = {str(t).strip().lower() for t in (tags_data.get(base, []) or [])}
        match_counts.append(
            sum(1 for tok in token_set if tok in tag_set) if token_set else 0
        )
        name_text = normalize_track_name_for_search(base)
        name_words = set(re.findall(r"[a-z0-9]+", name_text))
        name_score = 0
        if query_name_text:
            if query_name_text == name_text:
                name_score += 100
            elif query_name_text in name_text:
                name_score += 50
        if query_words:
            name_score += len(query_words & name_words)
        name_match_scores.append(name_score)

    idxs = list(range(len(EMB_CACHE.names)))
    idxs.sort(
        key=lambda i: (
            -match_counts[i],
            -name_match_scores[i],
            -float(sims[i]),
            EMB_CACHE.names[i].lower(),
        )
    )
    top_n = clamp_int(top_n, TOP_MIN, TOP_MAX)
    idxs = idxs[:top_n]

    playlist = [mp3_folder / f"{EMB_CACHE.names[i]}.mp3" for i in idxs]
    table = [(EMB_CACHE.names[i], float(sims[i])) for i in idxs]

    if logging:
        print(f"[emb] mood='{mood}' -> {len(playlist)} results (top {top_n})")

    return playlist, table


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
    CTRL_X = 24  # queue delete
    CTRL_S = 19  # playlist save (may be blocked by terminal)
    CTRL_W = 23  # playlist save (alternate)
    CTRL_D = 4   # playlist delete
    CTRL_T = 20  # Most sort toggle
    RENDER_MIN_INTERVAL = 1.0 / 30.0
    INPUT_DRAIN_LIMIT = 64
    MAX_WHEEL_STEPS_PER_RENDER = 1
    DOUBLE_CLICK_SECONDS = 0.35

    def __init__(self, enable: bool = True):
        self.enable = enable
        self.stdscr: Optional["curses._CursesWindow"] = None

        self.focus_panel: str = "queue"  # queue | similar | playlists | most | tags
        self.queue_scroll = 0
        self.queue_selected = 0
        self.queue_len = 0
        self.similar_scroll = 0
        self.similar_selected = 0
        self.similar_len = 0
        self.playlist_scroll = 0
        self.playlist_selected = 0
        self.playlist_len = 0
        self.playlist_track_scroll = 0
        self.playlist_track_selected = 0
        self.playlist_track_len = 0
        self.tag_scroll = 0
        self.tag_selected = 0
        self.tag_len = 0
        self.most_scroll = 0
        self.most_selected = 0

        self.input_buffer = ""
        self.status_msg = ""

        self.skip_requested = False
        self.stop_requested = False
        self.queue_move = 0  # used for drag reorder
        self.queue_delete = False
        self.similar_add = False
        self.show_help = False
        self.playlist_activate = False
        self.playlist_open_pending_idx: Optional[int] = None
        self.playlist_open_pending_ts = 0.0
        self.playlist_done_request = False
        self.playlist_load_request = False
        self.playlist_save_request = False
        self.playlist_delete_request = False
        self.playlist_editor_open = False
        self.playlist_add_current_request = False
        self.playlist_add_queue_request = False
        self.playlist_add_similar_request = False
        self.playlist_add_most_request = False
        self.playlist_item_remove_request: Optional[int] = None
        self.playlist_item_toggle_request: Optional[int] = None
        self.playlist_item_drag_start: Optional[int] = None
        self.playlist_item_drag_target: Optional[int] = None
        self.playlist_item_drag_commit_target: Optional[int] = None
        self.playlist_item_drag_commit_start: Optional[int] = None
        self.input_mode = "mood"  # mood | save_playlist | tag_add | tag_edit
        self.tag_panel_open = False
        self.tag_delete_request = False
        self.tag_edit_request = False
        self.queue_click_row: Optional[int] = None
        self.queue_click_x: Optional[int] = None
        self.queue_drag_start: Optional[int] = None
        self.queue_drag_target: Optional[int] = None
        self.queue_drag_commit_target: Optional[int] = None
        self.queue_drag_commit_start: Optional[int] = None
        self.queue_drag_started_selected = False
        self._last_click_ts = 0.0
        self._last_click_panel: Optional[str] = None
        self._last_click_row: Optional[int] = None
        self.most_add = False
        self.most_sort_mode = "count"  # count | time
        self.most_toggle_request = False
        self._last_render_ts = 0.0
        self._input_seen = False
        self._windows_size: Optional[Tuple[int, int]] = None
        self._header_win: Optional["curses._CursesWindow"] = None
        self._table_win: Optional["curses._CursesWindow"] = None
        self._status_win: Optional["curses._CursesWindow"] = None
        self._input_win: Optional["curses._CursesWindow"] = None
        self._draw_win: Optional["curses._CursesWindow"] = None
        self._draw_base_y = 0
        self._wheel_steps_this_render = 0

        self.last_h = 0
        self.last_w = 0
        self.queue_bounds = (0, 0, 0, 0)  # x, y, w, h
        self.queue_mode_col_start = 0
        self.queue_mode_col_end = 0
        self.queue_name_col_start = 0
        self.queue_name_col_end = 0
        self.queue_add_col_start = 0
        self.queue_add_col_end = 0
        self.similar_bounds = (0, 0, 0, 0)  # x, y, w, h
        self.similar_add_col_start = 0
        self.similar_add_col_end = 0
        self.playlist_bounds = (0, 0, 0, 0)  # x, y, w, h
        self.playlist_load_col_start = 0
        self.playlist_load_col_end = 0
        self.playlist_done_col_start = 0
        self.playlist_done_col_end = 0
        self.playlist_item_mode_col_start = 0
        self.playlist_item_mode_col_end = 0
        self.tag_bounds = (0, 0, 0, 0)  # x, y, w, h
        self.most_bounds = (0, 0, 0, 0)  # x, y, w, h
        self.most_add_col_start = 0
        self.most_add_col_end = 0
        self.tag_chip_col_start = 0
        self.tag_chip_col_end = 0
        self.current_add_col_start = 0
        self.current_add_col_end = 0
        self.tag_edit_col_start = 0
        self.tag_edit_col_end = 0
        self.tag_delete_col_start = 0
        self.tag_delete_col_end = 0
        self.tag_close_col_start = 0
        self.tag_close_col_end = 0
        self.most_len = 0
        self.most_sort_col_start = 0
        self.most_sort_col_end = 0
        self.playback_mode = "manual"
        self.playback_mode_toggle_request = False
        self.playback_mode_col_start = 0
        self.playback_mode_col_end = 0
        self.current_play_once_toggle_request = False
        self.current_mode_col_start = 0
        self.current_mode_col_end = 0

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
            curses.set_escdelay(25)
        except Exception:
            pass
        try:
            curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
            curses.mouseinterval(0)
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

    def _make_window(
        self, height: int, width: int, y: int, x: int
    ) -> Optional["curses._CursesWindow"]:
        if height <= 0 or width <= 0:
            return None
        try:
            win = curses.newwin(height, width, y, x)
            win.nodelay(True)
            try:
                win.keypad(True)
            except Exception:
                pass
            return win
        except Exception:
            return None

    def _ensure_windows(
        self,
        h: int,
        w: int,
        table_top: int,
        table_height: int,
        status_y: int,
        input_y: int,
    ) -> None:
        size = (h, w)
        if self._windows_size == size:
            return
        self._windows_size = size
        self._header_win = self._make_window(min(table_top, h), w, 0, 0)
        self._table_win = self._make_window(table_height, w, table_top, 0)
        self._status_win = self._make_window(1, w, status_y, 0)
        self._input_win = self._make_window(3, w, input_y - 1, 0)
        if self.stdscr:
            try:
                self.stdscr.erase()
            except Exception:
                pass

    def _set_draw_target(
        self, win: Optional["curses._CursesWindow"], base_y: int = 0
    ) -> None:
        self._draw_win = win
        self._draw_base_y = base_y

    def _draw(self, y: int, x: int, s: str) -> None:
        if not self.enable or not self.stdscr:
            return
        win = self._draw_win or self.stdscr
        local_y = y - self._draw_base_y
        h, w = win.getmaxyx()
        if local_y < 0 or local_y >= h or x >= w:
            return

        max_len = w - x
        if max_len <= 0:
            return
        if len(s) > max_len:
            s = s[:max_len]

        try:
            win.addstr(local_y, x, s)
        except Exception:
            pass

    def _hline(self, y: int, ch: str = "-") -> None:
        if not self.enable or not self.stdscr:
            return
        win = self._draw_win or self.stdscr
        local_y = y - self._draw_base_y
        h, w = win.getmaxyx()
        if local_y < 0 or local_y >= h:
            return
        try:
            win.hline(local_y, 0, ord(ch), max(0, w - 1))
        except Exception:
            pass

    def _render_help_overlay(self, h: int, w: int) -> None:
        help_lines = [
            "Shortcuts",
            "",
            "Enter: submit mood",
            "TAB: focus (Queue/Similar/Playlists/Most)",
            "↑/↓ PgUp/PgDn: move in focused panel",
            "→: skip track",
            "Ctrl+G: stop",
            "Enter: add selected (Similar)",
            "Drag in Queue: reorder",
            "Ctrl+X: delete (Queue)",
            "Click current/queue loop/once to toggle",
            "Click [mode:manual/auto] to arm auto takeover",
            "Click [tags] to open current-track tags",
            "Tags: click row, [edit], [delete], [close]",
            "Most: click [by:count/time] or Ctrl+T",
            "Playlists: click to edit, double-click to load; [add] adds tracks",
            "Playlist edit: click loop/once, drag reorder, double-click remove",
            "Ctrl+W: save queue (Playlists)",
            "Enter on Playlists: load selected",
            "Ctrl+D: delete selected playlist",
            "",
            "Press Esc to close",
        ]
        max_line = max(len(line) for line in help_lines) if help_lines else 0
        box_w = min(w - 4, max(30, max_line + 4))
        box_h = min(h - 4, len(help_lines) + 4)
        if box_w <= 0 or box_h <= 0:
            return
        box_x = max(0, (w - box_w) // 2)
        box_y = max(0, (h - box_h) // 2)
        try:
            help_win = curses.newwin(box_h, box_w, box_y, box_x)
            top = "╭" + ("─" * (box_w - 2)) + "╮"
            mid = "│" + (" " * (box_w - 2)) + "│"
            bot = "╰" + ("─" * (box_w - 2)) + "╯"
            help_win.addstr(0, 0, top)
            for i in range(1, box_h - 1):
                help_win.addstr(i, 0, mid)
            help_win.addstr(box_h - 1, 0, bot)
            content_start = 2
            for i, line in enumerate(help_lines[: box_h - 4]):
                help_win.addstr(content_start + i, 2, line[: box_w - 4])
            help_win.noutrefresh()
        except Exception:
            pass

    def _handle_playlist_row_click(self, row: int) -> None:
        idx = self.playlist_scroll + row
        if idx < 0 or idx >= self.playlist_len:
            return
        self.focus_panel = "playlists"
        self.playlist_selected = idx
        now = time.monotonic()
        if (
            self._last_click_panel == "playlists"
            and self._last_click_row == self.playlist_selected
            and (now - self._last_click_ts) <= self.DOUBLE_CLICK_SECONDS
        ):
            self.playlist_activate = True
            self.playlist_open_pending_idx = None
            self.playlist_open_pending_ts = 0.0
        else:
            self.playlist_open_pending_idx = idx
            self.playlist_open_pending_ts = now
        self._last_click_panel = "playlists"
        self._last_click_row = self.playlist_selected
        self._last_click_ts = now

    def _handle_playlist_press(self, mx: int, my: int) -> bool:
        px, py, pw, ph = self.playlist_bounds
        if not (px <= mx < px + pw and py <= my < py + ph):
            return False
        self.focus_panel = "playlists"
        if self.playlist_editor_open:
            if my == py:
                if self.playlist_load_col_start <= mx < self.playlist_load_col_end:
                    self.playlist_load_request = True
                elif self.playlist_done_col_start <= mx < self.playlist_done_col_end:
                    self.playlist_done_request = True
                return True
            row = my - (py + 1)
            idx = self.playlist_track_scroll + row
            if row >= 0 and 0 <= idx < self.playlist_track_len:
                self.playlist_track_selected = idx
                self.playlist_item_drag_start = idx
                self.playlist_item_drag_target = self.playlist_item_drag_start
                self.playlist_item_drag_commit_target = None
                self.playlist_item_drag_commit_start = self.playlist_item_drag_start
            return True
        if my >= py + 1:
            self._handle_playlist_row_click(my - (py + 1))
            return True
        return True

    def _handle_playlist_release(self, mx: int, my: int) -> bool:
        px, py, pw, ph = self.playlist_bounds
        if not self.playlist_editor_open:
            return False
        if px <= mx < px + pw and py + 1 <= my < py + ph:
            row = my - (py + 1)
            drop_idx = self.playlist_track_scroll + row
            if self.playlist_item_drag_start is not None and 0 <= drop_idx < self.playlist_track_len:
                self.playlist_item_drag_target = drop_idx
                self.playlist_track_selected = min(
                    max(0, self.playlist_track_len - 1), drop_idx
                )
                if drop_idx == self.playlist_item_drag_start:
                    if self.playlist_item_mode_col_start <= mx < self.playlist_item_mode_col_end:
                        self.playlist_item_toggle_request = drop_idx
                    self.playlist_item_drag_commit_target = None
                else:
                    self.playlist_item_drag_commit_target = drop_idx
        self.playlist_item_drag_start = None
        return px <= mx < px + pw and py <= my < py + ph

    def _handle_playlist_click(self, mx: int, my: int) -> bool:
        px, py, pw, ph = self.playlist_bounds
        if not (px <= mx < px + pw and py <= my < py + ph):
            return False
        self.focus_panel = "playlists"
        if self.playlist_editor_open:
            if my == py:
                if self.playlist_load_col_start <= mx < self.playlist_load_col_end:
                    self.playlist_load_request = True
                elif self.playlist_done_col_start <= mx < self.playlist_done_col_end:
                    self.playlist_done_request = True
                return True
            row = my - (py + 1)
            idx = self.playlist_track_scroll + row
            if row >= 0 and 0 <= idx < self.playlist_track_len:
                self.playlist_track_selected = idx
                if self.playlist_item_mode_col_start <= mx < self.playlist_item_mode_col_end:
                    self.playlist_item_toggle_request = idx
                    return True
                now = time.monotonic()
                if (
                    self._last_click_panel == "playlist_items"
                    and self._last_click_row == idx
                    and (now - self._last_click_ts) <= self.DOUBLE_CLICK_SECONDS
                ):
                    self.playlist_item_remove_request = idx
                self._last_click_panel = "playlist_items"
                self._last_click_row = idx
                self._last_click_ts = now
            return True
        if my >= py + 1:
            self._handle_playlist_row_click(my - (py + 1))
            return True
        return True

    def _handle_key(self) -> Optional[str]:
        if not self.enable or not self.stdscr:
            return None
        try:
            key = self.stdscr.getch()
        except Exception:
            key = -1
        if key == -1:
            return None
        self._input_seen = True

        if self.show_help:
            if key == 27:  # Esc
                self.show_help = False
            return None

        if key == curses.KEY_MOUSE:
            try:
                _, mx, my, _, bstate = curses.getmouse()
            except Exception:
                return None
            # Mouse wheel scrolling
            wheel_up = 0
            for name in ("BUTTON4_PRESSED", "BUTTON4_CLICKED", "BUTTON4_RELEASED"):
                wheel_up |= getattr(curses, name, 0)
            if wheel_up == 0:
                wheel_up = 0x00080000
            wheel_down = 0
            for name in ("BUTTON5_PRESSED", "BUTTON5_CLICKED", "BUTTON5_RELEASED"):
                wheel_down |= getattr(curses, name, 0)
            wheel_down |= 0x00200000 | 0x08000000
            if bstate & wheel_up or bstate & wheel_down:
                direction = -1 if bstate & wheel_up else 1
                if self._wheel_steps_this_render >= self.MAX_WHEEL_STEPS_PER_RENDER:
                    return None
                self._wheel_steps_this_render += 1
                # determine which panel is under the cursor
                x, y, w, h = self.queue_bounds
                if x <= mx < x + w and y + 1 <= my < y + h:
                    self.focus_panel = "queue"
                    self.queue_selected = max(0, min(self.queue_len - 1, self.queue_selected + direction))
                else:
                    sx, sy, sw, sh = self.similar_bounds
                    if sx <= mx < sx + sw and sy + 1 <= my < sy + sh:
                        self.focus_panel = "similar"
                        self.similar_selected = max(0, min(self.similar_len - 1, self.similar_selected + direction))
                    else:
                        px, py, pw, ph = self.playlist_bounds
                        if px <= mx < px + pw and py + 1 <= my < py + ph:
                            self.focus_panel = "playlists"
                            if self.playlist_editor_open:
                                self.playlist_track_selected = max(
                                    0,
                                    min(
                                        self.playlist_track_len - 1,
                                        self.playlist_track_selected + direction,
                                    ),
                                )
                            else:
                                self.playlist_selected = max(0, min(self.playlist_len - 1, self.playlist_selected + direction))
                        else:
                            tx, ty, tw, th = self.tag_bounds
                            if self.tag_panel_open and tx <= mx < tx + tw and ty + 2 <= my < ty + th:
                                self.focus_panel = "tags"
                                self.tag_selected = max(0, min(self.tag_len - 1, self.tag_selected + direction))
                            else:
                                mx0, my0, mw, mh = self.most_bounds
                                if mx0 <= mx < mx0 + mw and my0 <= my < my0 + mh:
                                    self.focus_panel = "most"
                                    if self.most_len:
                                        self.most_selected = max(0, min(self.most_len - 1, self.most_selected + direction))
                return None
            # Live drag tracking (when terminal reports motion)
            if self.queue_drag_start is not None:
                move_mask = (
                    curses.REPORT_MOUSE_POSITION
                    | getattr(curses, "BUTTON1_MOVED", 0)
                    | curses.BUTTON1_PRESSED
                )
                if bstate & move_mask:
                    x, y, w, h = self.queue_bounds
                    if x <= mx < x + w and y + 1 <= my < y + h:
                        row = my - (y + 1)
                        self.queue_drag_target = self.queue_scroll + row
            if self.playlist_item_drag_start is not None:
                move_mask = (
                    curses.REPORT_MOUSE_POSITION
                    | getattr(curses, "BUTTON1_MOVED", 0)
                    | curses.BUTTON1_PRESSED
                )
                if bstate & move_mask:
                    px, py, pw, ph = self.playlist_bounds
                    if px <= mx < px + pw and py + 1 <= my < py + ph:
                        row = my - (py + 1)
                        self.playlist_item_drag_target = self.playlist_track_scroll + row
            if bstate & curses.BUTTON1_PRESSED:
                x, y, w, h = self.queue_bounds
                if x <= mx < x + w and y + 1 <= my < y + h:
                    row = my - (y + 1)
                    idx = self.queue_scroll + row
                    self.focus_panel = "queue"
                    self.queue_drag_started_selected = idx == self.queue_selected
                    self.queue_selected = min(max(0, self.queue_len - 1), idx)
                    self.queue_drag_start = idx
                    self.queue_drag_target = self.queue_drag_start
                    self.queue_drag_commit_target = None
                    self.queue_drag_commit_start = self.queue_drag_start
                else:
                    sx, sy, sw, sh = self.similar_bounds
                    if sx <= mx < sx + sw and sy + 1 <= my < sy + sh:
                        row = my - (sy + 1)
                        idx = self.similar_scroll + row
                        was_selected = idx == self.similar_selected
                        self.focus_panel = "similar"
                        self.similar_selected = min(max(0, self.similar_len - 1), idx)
                        if (
                            was_selected
                            and self.similar_add_col_start <= mx < self.similar_add_col_end
                        ):
                            self.playlist_add_similar_request = True
                            return None
                        now = time.monotonic()
                        if (
                            self._last_click_panel == "similar"
                            and self._last_click_row == self.similar_selected
                            and (now - self._last_click_ts) <= self.DOUBLE_CLICK_SECONDS
                        ):
                            self.similar_add = True
                        self._last_click_panel = "similar"
                        self._last_click_row = self.similar_selected
                        self._last_click_ts = now
                    else:
                        tx, ty, tw, th = self.tag_bounds
                        if self.tag_panel_open and tx <= mx < tx + tw and ty <= my < ty + th:
                            self.focus_panel = "tags"
                            if my == ty:
                                if self.tag_edit_col_start <= mx < self.tag_edit_col_end:
                                    self.tag_edit_request = True
                                elif self.tag_delete_col_start <= mx < self.tag_delete_col_end:
                                    self.tag_delete_request = True
                                elif self.tag_close_col_start <= mx < self.tag_close_col_end:
                                    self.tag_panel_open = False
                                    self.input_mode = "mood"
                                    self.input_buffer = ""
                                    self.focus_panel = "queue"
                                return None
                            row = my - (ty + 2)
                            if row >= 0:
                                self.tag_selected = min(
                                    max(0, self.tag_len - 1),
                                    self.tag_scroll + row,
                                )
                        else:
                            px, py, pw, ph = self.playlist_bounds
                            if px <= mx < px + pw and py <= my < py + ph:
                                self._handle_playlist_press(mx, my)
                            else:
                                mx0, my0, mw, mh = self.most_bounds
                                if mx0 <= mx < mx0 + mw and my0 <= my < my0 + mh:
                                    self.focus_panel = "most"
                                    if my == my0:
                                        return None
                                    row = my - (my0 + 1)
                                    most_idx = self.most_scroll + row
                                    was_selected = most_idx == self.most_selected
                                    if row >= 0:
                                        self.most_selected = min(
                                            max(0, most_idx),
                                            max(0, self.most_len - 1),
                                        )
                                    if (
                                        was_selected
                                        and self.most_add_col_start <= mx < self.most_add_col_end
                                    ):
                                        self.playlist_add_most_request = True
                                        return None
                                    most_idx = self.most_selected
                                    now = time.monotonic()
                                    if (
                                        self._last_click_panel == "most"
                                        and self._last_click_row == most_idx
                                        and (now - self._last_click_ts) <= self.DOUBLE_CLICK_SECONDS
                                    ):
                                        self.most_add = True
                                    self._last_click_panel = "most"
                                    self._last_click_row = most_idx
                                    self._last_click_ts = now
            elif bstate & curses.BUTTON1_RELEASED:
                x, y, w, h = self.queue_bounds
                if x <= mx < x + w and y + 1 <= my < y + h:
                    row = my - (y + 1)
                    drop_idx = self.queue_scroll + row
                    if self.queue_drag_start is not None:
                        self.queue_drag_target = drop_idx
                        self.queue_selected = min(max(0, self.queue_len - 1), drop_idx)
                        if drop_idx == self.queue_drag_start:
                            # treat as click; allow toggle if click in mode column
                            if (
                                self.queue_drag_started_selected
                                and self.queue_add_col_start <= mx < self.queue_add_col_end
                            ):
                                self.playlist_add_queue_request = True
                                self.queue_click_row = None
                                self.queue_click_x = None
                            else:
                                self.queue_click_row = drop_idx
                                self.queue_click_x = mx
                            self.queue_drag_commit_target = None
                        else:
                            self.queue_drag_commit_target = drop_idx
                self.queue_drag_start = None
                self.queue_drag_started_selected = False
                self._handle_playlist_release(mx, my)
            elif bstate & curses.BUTTON1_CLICKED:
                if (
                    my == 0
                    and self.playback_mode_col_start <= mx < self.playback_mode_col_end
                ):
                    self.playback_mode_toggle_request = True
                    return None
                if (
                    my == 1
                    and self.tag_chip_col_start <= mx < self.tag_chip_col_end
                ):
                    self.tag_panel_open = not self.tag_panel_open
                    self.input_mode = "tag_add" if self.tag_panel_open else "mood"
                    self.focus_panel = "tags" if self.tag_panel_open else "queue"
                    self.input_buffer = ""
                    self.status_msg = "Tags opened." if self.tag_panel_open else "Tags closed."
                    return None
                if (
                    my == 1
                    and self.current_mode_col_start <= mx < self.current_mode_col_end
                ):
                    self.current_play_once_toggle_request = True
                    return None
                if (
                    my == 1
                    and self.current_add_col_start <= mx < self.current_add_col_end
                ):
                    self.playlist_add_current_request = True
                    return None
                x, y, w, h = self.queue_bounds
                if x <= mx < x + w and y + 1 <= my < y + h:
                    row = my - (y + 1)
                    idx = self.queue_scroll + row
                    was_selected = idx == self.queue_selected
                    self.focus_panel = "queue"
                    self.queue_selected = min(max(0, self.queue_len - 1), idx)
                    self.queue_click_row = idx
                    self.queue_click_x = mx
                    if (
                        was_selected
                        and self.queue_add_col_start <= mx < self.queue_add_col_end
                    ):
                        self.playlist_add_queue_request = True
                        self.queue_click_row = None
                        self.queue_click_x = None
                        return None
                    # double click to remove from queue
                    now = time.monotonic()
                    if (
                        self._last_click_panel == "queue"
                        and self._last_click_row == self.queue_click_row
                        and (now - self._last_click_ts) <= self.DOUBLE_CLICK_SECONDS
                    ):
                        # only delete when double-clicking on the name column
                        if self.queue_name_col_start <= mx < self.queue_name_col_end:
                            self.queue_delete = True
                        else:
                            self.queue_delete = False
                    self._last_click_panel = "queue"
                    self._last_click_row = self.queue_click_row
                    self._last_click_ts = now
                else:
                    sx, sy, sw, sh = self.similar_bounds
                    if sx <= mx < sx + sw and sy + 1 <= my < sy + sh:
                        row = my - (sy + 1)
                        idx = self.similar_scroll + row
                        was_selected = idx == self.similar_selected
                        self.focus_panel = "similar"
                        self.similar_selected = min(max(0, self.similar_len - 1), idx)
                        if (
                            was_selected
                            and self.similar_add_col_start <= mx < self.similar_add_col_end
                        ):
                            self.playlist_add_similar_request = True
                            return None
                        now = time.monotonic()
                        if (
                            self._last_click_panel == "similar"
                            and self._last_click_row == self.similar_selected
                            and (now - self._last_click_ts) <= self.DOUBLE_CLICK_SECONDS
                        ):
                            self.similar_add = True
                        self._last_click_panel = "similar"
                        self._last_click_row = self.similar_selected
                        self._last_click_ts = now
                    else:
                        tx, ty, tw, th = self.tag_bounds
                        if self.tag_panel_open and tx <= mx < tx + tw and ty <= my < ty + th:
                            self.focus_panel = "tags"
                            if my == ty:
                                if self.tag_edit_col_start <= mx < self.tag_edit_col_end:
                                    self.tag_edit_request = True
                                elif self.tag_delete_col_start <= mx < self.tag_delete_col_end:
                                    self.tag_delete_request = True
                                elif self.tag_close_col_start <= mx < self.tag_close_col_end:
                                    self.tag_panel_open = False
                                    self.input_mode = "mood"
                                    self.input_buffer = ""
                                    self.focus_panel = "queue"
                                return None
                            row = my - (ty + 2)
                            if row >= 0:
                                self.tag_selected = min(
                                    max(0, self.tag_len - 1),
                                    self.tag_scroll + row,
                                )
                                now = time.monotonic()
                                if (
                                    self._last_click_panel == "tags"
                                    and self._last_click_row == self.tag_selected
                                    and (now - self._last_click_ts) <= self.DOUBLE_CLICK_SECONDS
                                ):
                                    self.tag_edit_request = True
                                self._last_click_panel = "tags"
                                self._last_click_row = self.tag_selected
                                self._last_click_ts = now
                        else:
                            px, py, pw, ph = self.playlist_bounds
                            if px <= mx < px + pw and py <= my < py + ph:
                                self._handle_playlist_click(mx, my)
                            else:
                                mx0, my0, mw, mh = self.most_bounds
                                if mx0 <= mx < mx0 + mw and my0 <= my < my0 + mh:
                                    self.focus_panel = "most"
                                    if (
                                        my == my0
                                        and self.most_sort_col_start <= mx < self.most_sort_col_end
                                    ):
                                        self.most_toggle_request = True
                                        return None
                                    row = my - (my0 + 1)
                                    most_idx = self.most_scroll + row
                                    was_selected = most_idx == self.most_selected
                                    if row >= 0:
                                        self.most_selected = min(
                                            max(0, most_idx),
                                            max(0, self.most_len - 1),
                                        )
                                    if (
                                        was_selected
                                        and self.most_add_col_start <= mx < self.most_add_col_end
                                    ):
                                        self.playlist_add_most_request = True
                                        return None
                                    # double click in Most to add to queue
                                    most_idx = self.most_selected
                                    now = time.monotonic()
                                    if (
                                        self._last_click_panel == "most"
                                        and self._last_click_row == most_idx
                                        and (now - self._last_click_ts) <= self.DOUBLE_CLICK_SECONDS
                                    ):
                                        self.most_add = True
                                    self._last_click_panel = "most"
                                    self._last_click_row = most_idx
                                    self._last_click_ts = now
            return None

        KEY_PGUP = getattr(curses, "KEY_PPAGE", 339)
        KEY_PGDN = getattr(curses, "KEY_NPAGE", 338)
        KEY_BTAB = getattr(curses, "KEY_BTAB", 353)

        if key == curses.KEY_UP:
            if self.focus_panel == "queue":
                self.queue_selected = max(0, self.queue_selected - 1)
            elif self.focus_panel == "similar":
                self.similar_selected = max(0, self.similar_selected - 1)
            elif self.focus_panel == "playlists":
                if self.playlist_editor_open:
                    self.playlist_track_selected = max(0, self.playlist_track_selected - 1)
                else:
                    self.playlist_selected = max(0, self.playlist_selected - 1)
            elif self.focus_panel == "tags":
                self.tag_selected = max(0, self.tag_selected - 1)
            else:
                self.most_selected = max(0, self.most_selected - 1)
                if self.most_len:
                    self.most_selected = min(self.most_selected, self.most_len - 1)
        elif key == curses.KEY_DOWN:
            if self.focus_panel == "queue":
                self.queue_selected = min(
                    max(0, self.queue_len - 1), self.queue_selected + 1
                )
            elif self.focus_panel == "similar":
                self.similar_selected = min(
                    max(0, self.similar_len - 1), self.similar_selected + 1
                )
            elif self.focus_panel == "playlists":
                if self.playlist_editor_open:
                    self.playlist_track_selected = min(
                        max(0, self.playlist_track_len - 1),
                        self.playlist_track_selected + 1,
                    )
                else:
                    self.playlist_selected = min(
                        max(0, self.playlist_len - 1), self.playlist_selected + 1
                    )
            elif self.focus_panel == "tags":
                self.tag_selected = min(
                    max(0, self.tag_len - 1), self.tag_selected + 1
                )
            else:
                self.most_selected += 1
                if self.most_len:
                    self.most_selected = min(self.most_selected, self.most_len - 1)
        elif key == KEY_PGUP:
            page = max(1, self.last_h - 12)
            if self.focus_panel == "queue":
                self.queue_selected = max(0, self.queue_selected - page)
            elif self.focus_panel == "similar":
                self.similar_selected = max(0, self.similar_selected - page)
            elif self.focus_panel == "playlists":
                if self.playlist_editor_open:
                    self.playlist_track_selected = max(0, self.playlist_track_selected - page)
                else:
                    self.playlist_selected = max(0, self.playlist_selected - page)
            elif self.focus_panel == "tags":
                self.tag_selected = max(0, self.tag_selected - page)
            else:
                self.most_selected = max(0, self.most_selected - page)
                if self.most_len:
                    self.most_selected = min(self.most_selected, self.most_len - 1)
        elif key == KEY_PGDN:
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
            elif self.focus_panel == "playlists":
                if self.playlist_editor_open:
                    self.playlist_track_selected = min(
                        max(0, self.playlist_track_len - 1),
                        self.playlist_track_selected + page,
                    )
                else:
                    self.playlist_selected = min(
                        max(0, self.playlist_len - 1),
                        self.playlist_selected + page,
                    )
            elif self.focus_panel == "tags":
                self.tag_selected = min(
                    max(0, self.tag_len - 1),
                    self.tag_selected + page,
                )
            else:
                self.most_selected += page
                if self.most_len:
                    self.most_selected = min(self.most_selected, self.most_len - 1)
        elif key in (self.KEY_TAB, KEY_BTAB):
            order = ["queue", "similar"]
            order.append("tags" if self.tag_panel_open else "playlists")
            order.append("most")
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
            self.status_msg = f"Skipping to next in {int(GAP_SECONDS)}s..."
        elif key == self.CTRL_G:
            self.stop_requested = True
            self.status_msg = "Stopped. Waiting for mood..."
        elif key == self.CTRL_X and self.focus_panel == "queue":
            self.queue_delete = True
        elif key == self.CTRL_X and self.focus_panel == "playlists" and self.playlist_editor_open:
            self.playlist_item_remove_request = self.playlist_track_selected
        elif key == self.CTRL_X and self.focus_panel == "tags":
            self.tag_delete_request = True
        elif key in (self.CTRL_S, self.CTRL_W) and self.focus_panel == "playlists":
            self.playlist_save_request = True
        elif key == self.CTRL_D and self.focus_panel == "playlists":
            self.playlist_delete_request = True
        elif key == self.CTRL_T and self.focus_panel == "most":
            self.most_toggle_request = True
        elif key in (curses.KEY_BACKSPACE, self.KEY_BS, 8):
            self.input_buffer = self.input_buffer[:-1]
        elif key in (10, 13):  # Enter
            if not self.input_buffer and self.focus_panel == "playlists":
                self.playlist_activate = True
                return None
            if not self.input_buffer and self.focus_panel == "similar":
                self.similar_add = True
                return None
            if not self.input_buffer and self.focus_panel == "tags":
                self.tag_edit_request = True
                return None
            text = self.input_buffer.strip()
            if text:
                self.input_buffer = ""
                return text
        else:
            if 32 <= key <= 126:
                self.input_buffer += chr(key)
        return None

    def _handle_pending_keys(self) -> Tuple[Optional[str], bool]:
        submitted: Optional[str] = None
        saw_input = False
        self._wheel_steps_this_render = 0
        for _ in range(self.INPUT_DRAIN_LIMIT):
            self._input_seen = False
            text = self._handle_key()
            if not self._input_seen:
                break
            saw_input = True
            if text:
                submitted = text
                break
        return submitted, saw_input

    def render(
        self,
        now_playing: str,
        target_lufs: Optional[float],
        current_lufs: Optional[float],
        loudness_diff: Optional[float],
        volume_scale: Optional[float],
        elapsed_sec: float,
        total_sec: float,
        playback_mode: str,
        current_play_once: Optional[bool],
        counts: Dict[str, int],
        listen_hours_for_stem: Optional[Callable[[str, int], float]],
        total_listens: int,
        total_listen_hours: float,
        similar_entries: Optional[List[Tuple[str, float]]],
        similar_mood: Optional[str],
        queue_items: Optional[List[Dict[str, object]]],
        playlists: Optional[List[Tuple[str, int]]],
        current_tags: Optional[List[str]] = None,
        pending_tags: Optional[List[str]] = None,
        active_playlist_name: Optional[str] = None,
        active_playlist_items: Optional[List[dict]] = None,
    ) -> Optional[str]:
        if not self.enable or not self.stdscr:
            return None

        queue_items = queue_items or []
        self.queue_len = len(queue_items)
        self.similar_len = len(similar_entries or [])
        playlists = playlists or []
        self.playlist_len = len(playlists)
        submitted, saw_input = self._handle_pending_keys()

        now = time.monotonic()
        if (
            not saw_input
            and not self.show_help
            and (now - self._last_render_ts) < self.RENDER_MIN_INTERVAL
        ):
            return submitted
        self._last_render_ts = now

        h, w = self.stdscr.getmaxyx()
        self.last_h, self.last_w = h, w
        self.playback_mode = playback_mode
        input_y = h - 2
        status_y = h - 4
        table_top = 8
        table_height = max(0, status_y - table_top)
        self._ensure_windows(h, w, table_top, table_height, status_y, input_y)
        for win in (
            self._header_win,
            self._table_win,
            self._status_win,
            self._input_win,
        ):
            if win:
                try:
                    win.erase()
                except Exception:
                    pass

        # Header block (fixed)
        self._set_draw_target(self._header_win, 0)
        header = " Now Playing ".center(w, " ")
        mode_chip = f"[mode:{playback_mode}]"
        chip_x = max(0, w - len(mode_chip) - 2)
        self.playback_mode_col_start = chip_x
        self.playback_mode_col_end = chip_x + len(mode_chip)
        self._draw(0, 0, header)
        self._draw(0, chip_x, mode_chip)
        current_chip = ""
        tag_chip = ""
        add_chip = ""
        if now_playing and current_play_once is not None:
            current_chip = f"[{'once' if current_play_once else 'loop'}]"
            current_chip_x = max(0, w - len(current_chip) - 2)
            self.current_mode_col_start = current_chip_x
            self.current_mode_col_end = current_chip_x + len(current_chip)
            tag_chip = "[tags]" if not self.tag_panel_open else "[tags*]"
            tag_chip_x = max(0, current_chip_x - len(tag_chip) - 1)
            self.tag_chip_col_start = tag_chip_x
            self.tag_chip_col_end = tag_chip_x + len(tag_chip)
            if active_playlist_name:
                add_chip = "[add]"
                add_chip_x = max(0, tag_chip_x - len(add_chip) - 1)
                self.current_add_col_start = add_chip_x
                self.current_add_col_end = add_chip_x + len(add_chip)
            else:
                add_chip_x = tag_chip_x
                self.current_add_col_start = 0
                self.current_add_col_end = 0
        else:
            current_chip_x = 0
            tag_chip_x = 0
            add_chip_x = 0
            self.current_mode_col_start = 0
            self.current_mode_col_end = 0
            self.tag_chip_col_start = 0
            self.tag_chip_col_end = 0
            self.current_add_col_start = 0
            self.current_add_col_end = 0

        track_line = f"Track: {now_playing or '(none)'}"
        if current_chip or tag_chip or add_chip:
            first_chip_x = add_chip_x if add_chip else (tag_chip_x or current_chip_x)
            max_track_len = max(0, first_chip_x - 4)
            track_line = track_line[:max_track_len]
        self._draw(1, 2, track_line)
        if add_chip:
            self._draw(1, add_chip_x, add_chip)
        if tag_chip:
            self._draw(1, tag_chip_x, tag_chip)
        if current_chip:
            self._draw(1, current_chip_x, current_chip)

        td = "N/A" if target_lufs is None else f"{target_lufs:.2f}"
        cd = "N/A" if current_lufs is None else f"{current_lufs:.2f}"
        ld = "N/A" if loudness_diff is None else f"{loudness_diff:.2f}"
        vs = "N/A" if volume_scale is None else f"{volume_scale:.2f}"

        self._draw(2, 2, f"Target LUFS: {td}")
        self._draw(3, 2, f"Current LUFS: {cd}")
        self._draw(4, 2, f"LUFS Diff: {ld}")
        volume_line = f"Adjusted Volume: {vs}"
        self._draw(5, 2, volume_line)
        listen_word = "listen" if total_listens == 1 else "listens"
        stats_line = f"Total: {total_listens:n} {listen_word}, {total_listen_hours:.1f} hours"
        stats_x = max(len(volume_line) + 6, w - len(stats_line) - 2)
        if stats_x + len(stats_line) < w:
            self._draw(5, stats_x, stats_line)

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

        # Status line
        self._set_draw_target(self._status_win, status_y)
        status = ""
        if self.status_msg:
            status = self.status_msg
        if status:
            right_x = max(0, w - 2 - len(status))
            self._draw(status_y, right_x, status)

        if table_height > 0:
            self._set_draw_target(self._table_win, table_top)

            def display_play_once(item: Dict[str, object]) -> bool:
                play_once = bool(item.get("play_once"))
                if item.get("play_once_overridden"):
                    return play_once
                source = str(item.get("source") or "manual")
                return play_once or (playback_mode == "auto" and source == "manual")

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

            left_w = max(24, int(w * 0.48))
            right_w = max(0, w - left_w - 1)
            left_x = 0
            right_x = left_w + 1

            sep_h = 1  # separator rows between right panes
            avail = max(0, table_height - 2 * sep_h)
            top_h = max(3, avail // 3)
            mid_h = max(3, avail // 3)
            bot_h = max(0, avail - top_h - mid_h)
            if bot_h < 3:
                bot_h = max(0, avail - top_h - 3)
                mid_h = max(3, avail - top_h - bot_h)

            top_y = table_top
            sep1_y = top_y + top_h
            mid_y = sep1_y + sep_h
            sep2_y = mid_y + mid_h
            bot_y = sep2_y + sep_h

            # Most (left)
            most_time_metric: Dict[str, float] = {}
            if self.most_sort_mode == "time" and listen_hours_for_stem is not None:
                for name, cnt in counts.items():
                    most_time_metric[name] = float(listen_hours_for_stem(name, cnt))
                most_entries = sorted(
                    counts.items(),
                    key=lambda kv: (-most_time_metric.get(kv[0], 0.0), -kv[1], kv[0].lower()),
                )
            else:
                most_entries = sorted_counts(counts)
            self.most_len = len(most_entries)
            m_content_h = max(0, table_height - 1)
            m_max_scroll = max(0, len(most_entries) - m_content_h)
            self.most_selected = max(0, min(self.most_selected, max(0, len(most_entries) - 1)))
            if self.most_selected < self.most_scroll:
                self.most_scroll = self.most_selected
            if self.most_selected >= self.most_scroll + m_content_h:
                self.most_scroll = self.most_selected - m_content_h + 1
            self.most_scroll = min(max(0, self.most_scroll), m_max_scroll)
            m_rows: List[str] = []
            m_visible_entries = most_entries[
                self.most_scroll : min(
                    len(most_entries), self.most_scroll + m_content_h
                )
            ]
            most_hours_metric: Dict[str, float] = {}
            if listen_hours_for_stem is not None:
                for name, cnt in m_visible_entries:
                    if self.most_sort_mode == "time":
                        most_hours_metric[name] = most_time_metric.get(name, 0.0)
                    else:
                        most_hours_metric[name] = float(
                            listen_hours_for_stem(name, cnt)
                        )
            m_count_num_w = max(
                1, max((len(str(cnt)) for _, cnt in most_entries), default=1)
            )
            m_time_tok_w = max(
                1,
                max(
                    (
                        len(f"({most_hours_metric.get(name, 0.0):.1f}h)")
                        for name, _ in m_visible_entries
                    ),
                    default=len("(0.0h)"),
                ),
            )
            playlist_add_chip = "[add]" if active_playlist_name else ""
            m_count_w = m_count_num_w + 1 + m_time_tok_w
            m_add_space = len(playlist_add_chip) + 1 if playlist_add_chip else 0
            m_name_w = max(1, left_w - (2 + 3 + 3 + m_count_w + 5 + m_add_space))
            self.most_add_col_start = (
                left_x + left_w - len(playlist_add_chip) - 1
                if playlist_add_chip
                else 0
            )
            self.most_add_col_end = (
                self.most_add_col_start + len(playlist_add_chip)
                if playlist_add_chip
                else 0
            )
            for offset, (name, cnt) in enumerate(m_visible_entries):
                i = self.most_scroll + offset
                sel = ">" if self.focus_panel == "most" and i == self.most_selected else " "
                hours = (
                    most_time_metric.get(name, 0.0)
                    if self.most_sort_mode == "time"
                    else most_hours_metric.get(name, 0.0)
                )
                time_col = f"({hours:.1f}h)"
                count_col = f"{cnt:>{m_count_num_w}} {time_col:>{m_time_tok_w}}"
                name = str(name)[:m_name_w].ljust(m_name_w)
                row = f"{sel} {i+1:>3} | {count_col} | {name}"
                if playlist_add_chip:
                    row_add_chip = playlist_add_chip if i == self.most_selected else ""
                    row = row[: max(0, left_w - len(playlist_add_chip) - 1)].ljust(
                        max(0, left_w - len(playlist_add_chip) - 1)
                    ) + row_add_chip
                m_rows.append(row)
            m_focus = "[Most]" if self.focus_panel == "most" else " Most "
            m_sort_chip = f"[by:{self.most_sort_mode}]"
            m_title = f"{m_focus} {m_sort_chip}"
            chip_idx = m_title.find(m_sort_chip)
            if chip_idx >= 0:
                self.most_sort_col_start = left_x + chip_idx
                self.most_sort_col_end = self.most_sort_col_start + len(m_sort_chip)
            else:
                self.most_sort_col_start = 0
                self.most_sort_col_end = 0
            render_pane(m_title.ljust(left_w), m_rows, left_x, table_top, left_w, table_height)
            self.most_bounds = (left_x, table_top, left_w, table_height)

            # Similar (right top)
            entries = similar_entries or []
            if self.similar_selected >= len(entries):
                self.similar_selected = max(0, len(entries) - 1)
            s_content_h = max(0, top_h - 1)
            s_max_scroll = max(0, len(entries) - s_content_h)
            if self.similar_selected < self.similar_scroll:
                self.similar_scroll = self.similar_selected
            if self.similar_selected >= self.similar_scroll + s_content_h:
                self.similar_scroll = self.similar_selected - s_content_h + 1
            self.similar_scroll = min(self.similar_scroll, s_max_scroll)
            s_rows: List[str] = []
            s_name_w = max(1, right_w - (2 + 3 + 3 + 6 + 5))
            self.similar_add_col_start = (
                right_x + 8 + s_name_w - len(playlist_add_chip)
                if playlist_add_chip
                else 0
            )
            self.similar_add_col_end = (
                self.similar_add_col_start + len(playlist_add_chip)
                if playlist_add_chip
                else 0
            )
            for i in range(self.similar_scroll, min(len(entries), self.similar_scroll + s_content_h)):
                base, score = entries[i]
                sel = ">" if i == self.similar_selected and self.focus_panel == "similar" else " "
                if playlist_add_chip and i == self.similar_selected:
                    base_w = max(1, s_name_w - len(playlist_add_chip) - 1)
                    base = str(base)[:base_w].ljust(base_w) + " " + playlist_add_chip
                else:
                    base = str(base)[:s_name_w].ljust(s_name_w)
                row = f"{sel} {i+1:>3} | {base}| {score:>6.3f}  "
                s_rows.append(row)
            s_title = "[Similar]" if self.focus_panel == "similar" else " Similar "
            if similar_mood:
                s_title = (s_title[: max(0, right_w - len(similar_mood) - 3)] + f" ({similar_mood})")[:right_w]
            render_pane(s_title.ljust(right_w), s_rows, right_x, top_y, right_w, top_h)
            self.similar_bounds = (right_x, top_y, right_w, top_h)

            # Queue (right middle)
            if self.queue_selected >= self.queue_len:
                self.queue_selected = max(0, self.queue_len - 1)
            q_content_h = max(0, mid_h - 1)
            q_max_scroll = max(0, self.queue_len - q_content_h)
            if self.queue_selected < self.queue_scroll:
                self.queue_scroll = self.queue_selected
            if self.queue_selected >= self.queue_scroll + q_content_h:
                self.queue_scroll = self.queue_selected - q_content_h + 1
            self.queue_scroll = min(self.queue_scroll, q_max_scroll)
            q_rows: List[str] = []
            q_name_w = max(1, right_w - (2 + 3 + 3 + 4 + 5))
            self.queue_name_col_start = right_x + 8
            self.queue_name_col_end = self.queue_name_col_start + q_name_w
            self.queue_mode_col_start = self.queue_name_col_end + 2
            self.queue_mode_col_end = self.queue_mode_col_start + 4
            self.queue_add_col_start = (
                right_x + 8 + q_name_w - len(playlist_add_chip)
                if playlist_add_chip
                else 0
            )
            self.queue_add_col_end = (
                self.queue_add_col_start + len(playlist_add_chip)
                if playlist_add_chip
                else 0
            )
            order = list(range(self.queue_len))
            if self.queue_drag_start is not None and self.queue_drag_target is not None:
                start = max(0, min(self.queue_len - 1, self.queue_drag_start))
                target = max(0, min(self.queue_len - 1, self.queue_drag_target))
                if start != target:
                    item_idx = order.pop(start)
                    order.insert(target, item_idx)
            for i in range(self.queue_scroll, min(self.queue_len, self.queue_scroll + q_content_h)):
                item = queue_items[order[i]]
                path = item.get("path")
                name = path.name if isinstance(path, Path) else "(unknown)"
                mode = "once" if display_play_once(item) else "loop"
                if self.queue_drag_start is not None and order[i] == self.queue_drag_start:
                    sel = "D"
                else:
                    sel = ">" if i == self.queue_selected and self.focus_panel == "queue" else " "
                if playlist_add_chip and i == self.queue_selected:
                    name_w = max(1, q_name_w - len(playlist_add_chip) - 1)
                    name = name[:name_w].ljust(name_w) + " " + playlist_add_chip
                else:
                    name = name[:q_name_w].ljust(q_name_w)
                row = f"{sel} {i+1:>3} | {name}| {mode:>4}  "
                q_rows.append(row)
            q_title = "[Queue]" if self.focus_panel == "queue" else " Queue "
            render_pane(q_title.ljust(right_w), q_rows, right_x, mid_y, right_w, mid_h)
            self.queue_bounds = (right_x, mid_y, right_w, mid_h)

            # Playlists / Tags (right bottom)
            if self.tag_panel_open:
                tag_rows_data = [("saved", str(tag)) for tag in (current_tags or [])]
                tag_rows_data.extend(("pending", str(tag)) for tag in (pending_tags or []))
                self.tag_len = len(tag_rows_data)
                if self.tag_selected >= self.tag_len:
                    self.tag_selected = max(0, self.tag_len - 1)
                t_content_h = max(0, bot_h - 2)
                t_max_scroll = max(0, self.tag_len - t_content_h)
                if self.tag_selected < self.tag_scroll:
                    self.tag_scroll = self.tag_selected
                if self.tag_selected >= self.tag_scroll + t_content_h:
                    self.tag_scroll = self.tag_selected - t_content_h + 1
                self.tag_scroll = min(self.tag_scroll, t_max_scroll)

                mode_label = "edit" if self.input_mode == "tag_edit" else "add"
                t_rows = [f"{mode_label}: {self.input_buffer}"[:right_w]]
                t_name_w = max(1, right_w - 8)
                for i in range(self.tag_scroll, min(self.tag_len, self.tag_scroll + t_content_h)):
                    kind, tag = tag_rows_data[i]
                    sel = ">" if i == self.tag_selected and self.focus_panel == "tags" else " "
                    mark = "+" if kind == "pending" else " "
                    t_rows.append(f"{sel} {mark} {tag[:t_name_w]}")
                if not tag_rows_data:
                    t_rows.append("  (no tags)")

                t_focus = "[Tags]" if self.focus_panel == "tags" else " Tags "
                edit_chip = "[edit]"
                delete_chip = "[delete]"
                close_chip = "[close]"
                t_title = f"{t_focus} {edit_chip} {delete_chip} {close_chip}"
                edit_idx = t_title.find(edit_chip)
                delete_idx = t_title.find(delete_chip)
                close_idx = t_title.find(close_chip)
                self.tag_edit_col_start = right_x + edit_idx if edit_idx >= 0 else 0
                self.tag_edit_col_end = self.tag_edit_col_start + len(edit_chip)
                self.tag_delete_col_start = right_x + delete_idx if delete_idx >= 0 else 0
                self.tag_delete_col_end = self.tag_delete_col_start + len(delete_chip)
                self.tag_close_col_start = right_x + close_idx if close_idx >= 0 else 0
                self.tag_close_col_end = self.tag_close_col_start + len(close_chip)
                render_pane(t_title.ljust(right_w), t_rows, right_x, bot_y, right_w, bot_h)
                self.tag_bounds = (right_x, bot_y, right_w, bot_h)
                self.playlist_bounds = (0, 0, 0, 0)
                self.playlist_item_mode_col_start = 0
                self.playlist_item_mode_col_end = 0
            else:
                self.tag_len = 0
                self.tag_bounds = (0, 0, 0, 0)
                if active_playlist_name:
                    playlist_items = active_playlist_items or []
                    self.playlist_track_len = len(playlist_items)
                    if self.playlist_track_selected >= self.playlist_track_len:
                        self.playlist_track_selected = max(0, self.playlist_track_len - 1)
                    p_content_h = max(0, bot_h - 1)
                    p_max_scroll = max(0, self.playlist_track_len - p_content_h)
                    if self.playlist_track_selected < self.playlist_track_scroll:
                        self.playlist_track_scroll = self.playlist_track_selected
                    if self.playlist_track_selected >= self.playlist_track_scroll + p_content_h:
                        self.playlist_track_scroll = self.playlist_track_selected - p_content_h + 1
                    self.playlist_track_scroll = min(self.playlist_track_scroll, p_max_scroll)

                    p_rows = []
                    p_name_w = max(1, right_w - (2 + 3 + 3 + 4 + 5))
                    self.playlist_item_mode_col_start = right_x + 8 + p_name_w + 2
                    self.playlist_item_mode_col_end = self.playlist_item_mode_col_start + 4
                    order = list(range(self.playlist_track_len))
                    if (
                        self.playlist_item_drag_start is not None
                        and self.playlist_item_drag_target is not None
                    ):
                        start = max(0, min(self.playlist_track_len - 1, self.playlist_item_drag_start))
                        target = max(0, min(self.playlist_track_len - 1, self.playlist_item_drag_target))
                        if start != target and order:
                            item_idx = order.pop(start)
                            order.insert(target, item_idx)
                    for i in range(
                        self.playlist_track_scroll,
                        min(self.playlist_track_len, self.playlist_track_scroll + p_content_h),
                    ):
                        item = playlist_items[order[i]]
                        base = str(item.get("base") or "(unknown)")
                        mode = "once" if bool(item.get("play_once")) else "loop"
                        if (
                            self.playlist_item_drag_start is not None
                            and order[i] == self.playlist_item_drag_start
                        ):
                            sel = "D"
                        else:
                            sel = ">" if i == self.playlist_track_selected and self.focus_panel == "playlists" else " "
                        base = base[:p_name_w].ljust(p_name_w)
                        row = f"{sel} {i+1:>3} | {base}| {mode:>4}  "
                        p_rows.append(row)
                    if not playlist_items:
                        p_rows.append("  (empty; click [add] on a track)")

                    p_focused = self.focus_panel == "playlists"
                    p_focus = "[Playlist:" if p_focused else " Playlist:"
                    p_suffix = "]" if p_focused else ""
                    load_chip = "[load]"
                    done_chip = "[done]"
                    title_name_w = max(
                        1,
                        right_w
                        - len(p_focus)
                        - len(p_suffix)
                        - len(load_chip)
                        - len(done_chip)
                        - 4,
                    )
                    p_title = (
                        f"{p_focus}{active_playlist_name[:title_name_w]}{p_suffix} "
                        f"{load_chip} {done_chip}"
                    )
                    load_idx = p_title.find(load_chip)
                    done_idx = p_title.find(done_chip)
                    self.playlist_load_col_start = right_x + load_idx if load_idx >= 0 else 0
                    self.playlist_load_col_end = self.playlist_load_col_start + len(load_chip)
                    self.playlist_done_col_start = right_x + done_idx if done_idx >= 0 else 0
                    self.playlist_done_col_end = self.playlist_done_col_start + len(done_chip)
                    render_pane(p_title.ljust(right_w), p_rows, right_x, bot_y, right_w, bot_h)
                    self.playlist_bounds = (right_x, bot_y, right_w, bot_h)
                else:
                    self.playlist_track_len = 0
                    self.playlist_load_col_start = 0
                    self.playlist_load_col_end = 0
                    self.playlist_done_col_start = 0
                    self.playlist_done_col_end = 0
                    self.playlist_item_mode_col_start = 0
                    self.playlist_item_mode_col_end = 0
                    if self.playlist_selected >= len(playlists):
                        self.playlist_selected = max(0, len(playlists) - 1)
                    p_content_h = max(0, bot_h - 1)
                    p_max_scroll = max(0, len(playlists) - p_content_h)
                    if self.playlist_selected < self.playlist_scroll:
                        self.playlist_scroll = self.playlist_selected
                    if self.playlist_selected >= self.playlist_scroll + p_content_h:
                        self.playlist_scroll = self.playlist_selected - p_content_h + 1
                    self.playlist_scroll = min(self.playlist_scroll, p_max_scroll)
                    p_rows: List[str] = []
                    p_name_w = max(1, right_w - (2 + 3 + 3 + 4 + 5))
                    for i in range(self.playlist_scroll, min(len(playlists), self.playlist_scroll + p_content_h)):
                        name, count = playlists[i]
                        sel = ">" if i == self.playlist_selected and self.focus_panel == "playlists" else " "
                        name = str(name)[:p_name_w].ljust(p_name_w)
                        p_rows.append(f"{sel} {i+1:>3} | {name}| {count:>4}  ")
                    p_title = "[Playlists]" if self.focus_panel == "playlists" else " Playlists "
                    render_pane(p_title.ljust(right_w), p_rows, right_x, bot_y, right_w, bot_h)
                    self.playlist_bounds = (right_x, bot_y, right_w, bot_h)

            # Separator between left/right
            for i in range(table_height):
                self._draw(table_top + i, left_w, "│")
            # Right-side horizontal separators (do not overwrite content)
            if right_w > 0 and sep_h and top_h > 0 and mid_h > 0:
                self._draw(sep1_y, right_x, "─" * right_w)
            if right_w > 0 and sep_h and mid_h > 0 and bot_h > 0:
                self._draw(sep2_y, right_x, "─" * right_w)

        # Input bar + footer
        self._set_draw_target(self._input_win, input_y - 1)
        if self.input_mode == "save_playlist":
            self._hline(input_y - 1, "-")
            prompt = "playlist name: "
        elif self.input_mode in ("tag_add", "tag_edit"):
            self._hline(input_y - 1, "-")
            prompt = ""
        else:
            self._hline(input_y - 1, "-")
            prompt = "mood: "
        if not self.input_buffer and not now_playing and self.input_mode == "mood":
            self._draw(
                input_y, 2, prompt + "(type a mood + Enter; optional: (top N) / top=N)"
            )
        elif self.input_mode in ("tag_add", "tag_edit"):
            self._draw(input_y, 2, "Tags panel: type in panel, click edit/delete/close")
        else:
            self._draw(input_y, 2, prompt + self.input_buffer)

        self._draw(
            h - 1,
            2,
            "Click playlist to edit • Double-click playlist to load • [add] adds to playlist • Ctrl+G stop • --help",
        )

        try:
            for win in (
                self._header_win,
                self._table_win,
                self._status_win,
                self._input_win,
            ):
                if win:
                    win.noutrefresh()
            if self.show_help:
                self._render_help_overlay(h, w)
            curses.doupdate()
        except Exception:
            pass
        finally:
            self._set_draw_target(None, 0)

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
        return {}
    data = safe_read_json(tags_file, {})
    if not isinstance(data, dict):
        return {}
    out: Dict[str, List[str]] = {}
    for k, v in data.items():
        out[str(k)] = [str(t) for t in v] if isinstance(v, list) else [str(v)]
    return out


def save_tags(tags_file: Path, tags_data: Dict[str, List[str]]) -> None:
    atomic_write_json(tags_file, tags_data)


def add_tags_for_track(
    tags_data: Dict[str, List[str]], track_stem: str, new_tags: Iterable[str]
) -> int:
    existing = [str(t).strip() for t in tags_data.get(track_stem, []) if str(t).strip()]
    seen = {t.casefold() for t in existing}
    added = 0
    for raw in new_tags:
        tag = str(raw).strip()
        if not tag:
            continue
        key = tag.casefold()
        if key in seen:
            continue
        existing.append(tag)
        seen.add(key)
        added += 1
    tags_data[track_stem] = existing
    return added


class TrackTagSession:
    def __init__(self, tags_file: Path, tags_data: Dict[str, List[str]]):
        self.tags_file = tags_file
        self.tags_data = tags_data
        self.track_path: Optional[Path] = None
        self.pending: List[str] = []
        self.editing_ref: Optional[Tuple[str, int]] = None

    def start(self, track_path: Path) -> None:
        self.track_path = track_path
        self.pending = []

    def abandon(self) -> None:
        self.track_path = None
        self.pending = []
        self.editing_ref = None

    def existing_tags(self) -> List[str]:
        if self.track_path is None:
            return []
        return list(self.tags_data.get(self.track_path.stem, []) or [])

    def pending_tags(self) -> List[str]:
        return list(self.pending)

    def submit(self, raw_tag: str) -> str:
        if self.editing_ref is not None:
            return self.apply_edit(raw_tag)
        return self.add(raw_tag)

    def add(self, raw_tag: str) -> str:
        tag = raw_tag.strip()
        if not tag:
            return ""
        if self.track_path is None:
            return "No current track to tag."
        if self._has_tag(tag) or self._has_pending(tag):
            return f"Tag already exists: {tag}"
        self.pending.append(tag)
        return f"Queued tag for {self.track_path.stem}: {tag}"

    def begin_edit(self, row_index: int) -> Tuple[str, str]:
        ref = self._ref_for_row(row_index)
        if ref is None:
            return "", "No tag selected."
        kind, idx = ref
        self.editing_ref = ref
        tags = self.existing_tags() if kind == "saved" else self.pending
        return tags[idx], "Editing tag."

    def apply_edit(self, raw_tag: str) -> str:
        tag = raw_tag.strip()
        ref = self.editing_ref
        self.editing_ref = None
        if not tag:
            return "Edit cancelled."
        if self.track_path is None or ref is None:
            return "No current track to tag."
        if self._has_tag(tag, exclude=ref) or self._has_pending(tag, exclude=ref):
            return f"Tag already exists: {tag}"
        kind, idx = ref
        if kind == "saved":
            existing = self.existing_tags()
            if not 0 <= idx < len(existing):
                return "Tag no longer exists."
            existing[idx] = tag
            self.tags_data[self.track_path.stem] = existing
            save_tags(self.tags_file, self.tags_data)
            return f"Updated tag: {tag}"
        if not 0 <= idx < len(self.pending):
            return "Tag no longer exists."
        self.pending[idx] = tag
        return f"Updated queued tag: {tag}"

    def delete(self, row_index: int) -> str:
        ref = self._ref_for_row(row_index)
        if self.track_path is None or ref is None:
            return "No tag selected."
        if self.editing_ref == ref:
            self.editing_ref = None
        kind, idx = ref
        if kind == "saved":
            existing = self.existing_tags()
            if not 0 <= idx < len(existing):
                return "Tag no longer exists."
            removed = existing.pop(idx)
            self.tags_data[self.track_path.stem] = existing
            save_tags(self.tags_file, self.tags_data)
            return f"Deleted tag: {removed}"
        if not 0 <= idx < len(self.pending):
            return "Tag no longer exists."
        removed = self.pending.pop(idx)
        return f"Deleted queued tag: {removed}"

    def finish(self, trailing_tag: str = "") -> int:
        if trailing_tag.strip():
            self.submit(trailing_tag)
        if self.track_path is None or not self.pending:
            self.abandon()
            return 0
        added = add_tags_for_track(self.tags_data, self.track_path.stem, self.pending)
        if added:
            save_tags(self.tags_file, self.tags_data)
        self.abandon()
        return added

    def _has_tag(
        self, tag: str, exclude: Optional[Tuple[str, int]] = None
    ) -> bool:
        key = tag.casefold()
        for idx, existing in enumerate(self.existing_tags()):
            if exclude == ("saved", idx):
                continue
            if existing.casefold() == key:
                return True
        return False

    def _has_pending(
        self, tag: str, exclude: Optional[Tuple[str, int]] = None
    ) -> bool:
        key = tag.casefold()
        for idx, pending in enumerate(self.pending):
            if exclude == ("pending", idx):
                continue
            if pending.casefold() == key:
                return True
        return False

    def _append_unique_pending(self, raw_tag: str) -> None:
        tag = raw_tag.strip()
        if tag and not self._has_tag(tag) and not self._has_pending(tag):
            self.pending.append(tag)

    def _ref_for_row(self, row_index: int) -> Optional[Tuple[str, int]]:
        existing_len = len(self.existing_tags())
        if 0 <= row_index < existing_len:
            return ("saved", row_index)
        pending_idx = row_index - existing_len
        if 0 <= pending_idx < len(self.pending):
            return ("pending", pending_idx)
        return None


def main(
    initial_mood: Optional[str],
    top_n: int,
    mp3_folder: Path,
    tags_file: Path,
    sample_filename: str,
    target_lufs: Optional[float],
    logging: bool,
    enable_tui: bool,
    listen_db_filename: str,
    playback_mode: str,
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

    if sys.platform == "darwin":
        sdl_devs = list_pygame_output_devices()
        if logging and sdl_devs:
            print(f"[audio] SDL playback devices: {sdl_devs}")
        if sdl_devs and FORCE_DEVICE not in sdl_devs:
            raise RuntimeError(
                f"SDL/pygame output device '{FORCE_DEVICE}' not found.\n"
                f"Available SDL devices: {sdl_devs}\n"
                "Set FORCE_DEVICE to one of the exact names above."
            )

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
    listen_timestamps_filename = listen_timestamps_filename_for_folder(mp3_folder)
    listen_timestamps = load_listen_timestamps(
        mp3_folder, listen_timestamps_filename
    )
    history_duration_by_track_name = build_duration_by_track_name(mp3_folder)
    markov_transitions, markov_global_counts = build_markov_transition_counts(
        mp3_folder,
        listen_timestamps,
        history_duration_by_track_name,
    )
    loud_cache = load_loudness_cache(mp3_folder)
    playlists_filename = f"queue_playlists_{mp3_folder.name or 'default'}.json"
    playlists_db = load_playlists(playlists_filename)
    tag_session = TrackTagSession(tags_file, tags_data)

    current_similar_entries: Optional[List[Tuple[str, float]]] = None
    current_similar_mood: Optional[str] = None
    queue: List[Dict[str, object]] = []
    current_playing_item: Optional[Dict[str, object]] = None
    active_playlist_name: Optional[str] = None
    skip_requeue_item: Optional[Dict[str, object]] = None
    last_completed_track_name: Optional[str] = None
    playback_mode = str(playback_mode)

    def item_effective_play_once(item: Dict[str, object]) -> bool:
        play_once = bool(item.get("play_once"))
        if item.get("play_once_overridden"):
            return play_once
        source = str(item.get("source") or "manual")
        return play_once or (playback_mode == "auto" and source == "manual")

    def set_item_effective_play_once(
        item: Dict[str, object], play_once: bool
    ) -> None:
        item["play_once"] = bool(play_once)
        item["play_once_overridden"] = True

    audio_data: Dict[Path, Dict[str, float | None]] = {}
    resolved_target_loudness: Optional[float] = None
    duration_by_stem: Dict[str, float] = {}
    for key, entry in loud_cache.items():
        if key == "_v" or not isinstance(entry, dict):
            continue
        stem = Path(key).stem
        duration_by_stem[stem] = float(entry.get("duration") or 0.0)
    for track_name, duration in history_duration_by_track_name.items():
        duration_by_stem.setdefault(Path(track_name).stem, float(duration or 0.0))

    current_top_n = clamp_int(top_n, TOP_MIN, TOP_MAX)
    runtime_target_lufs: Optional[float] = target_lufs

    def get_playlists_list() -> List[Tuple[str, int]]:
        return sorted(
            [(name, len(items or [])) for name, items in playlists_db.items()],
            key=lambda x: x[0].lower(),
        )

    def get_active_playlist_items(tui: Optional[CursesTUI]) -> Optional[List[dict]]:
        if not tui or not tui.enable or not tui.playlist_editor_open:
            return None
        if not active_playlist_name:
            return None
        return playlists_db.get(active_playlist_name, [])

    def open_playlist_editor(name: str, tui: Optional[CursesTUI]) -> None:
        nonlocal active_playlist_name
        if name not in playlists_db:
            if tui and tui.enable:
                tui.status_msg = f"Playlist not found: {name}"
            return
        active_playlist_name = name
        if tui and tui.enable:
            tui.playlist_editor_open = True
            tui.focus_panel = "playlists"
            tui.playlist_track_selected = 0
            tui.playlist_track_scroll = 0
            tui.status_msg = f"Editing playlist: {name}"

    def add_base_to_active_playlist(
        base: str,
        play_once: bool,
        tui: Optional[CursesTUI],
    ) -> None:
        if not active_playlist_name or active_playlist_name not in playlists_db:
            if tui and tui.enable:
                tui.status_msg = "Click a playlist first."
            return
        items = playlists_db.setdefault(active_playlist_name, [])
        if any(str(item.get("base") or "") == base for item in items):
            if tui and tui.enable:
                tui.status_msg = f"Already in {active_playlist_name}: {base}"
            return
        items.append({"base": base, "play_once": bool(play_once)})
        save_playlists(playlists_db, playlists_filename)
        if tui and tui.enable:
            tui.status_msg = f"Added to {active_playlist_name}: {base}"

    def add_path_to_active_playlist(
        path: Path,
        play_once: bool,
        tui: Optional[CursesTUI],
    ) -> None:
        add_base_to_active_playlist(path.stem, play_once, tui)

    def save_playlist(name: str, tui: Optional[CursesTUI]) -> None:
        nonlocal playlists_db, current_playing_item
        items: List[dict] = []
        if current_playing_item:
            path = current_playing_item.get("path")
            if isinstance(path, Path):
                items.append(
                    {
                        "base": path.stem,
                        "play_once": item_effective_play_once(current_playing_item),
                    }
                )
        for item in queue:
            path = item.get("path")
            if isinstance(path, Path):
                items.append(
                    {
                        "base": path.stem,
                        "play_once": item_effective_play_once(item),
                    }
                )
        playlists_db[name] = items
        save_playlists(playlists_db, playlists_filename)
        if tui and tui.enable:
            tui.status_msg = f"Saved playlist: {name} ({len(items)} tracks)"

    def load_playlist(name: str, tui: Optional[CursesTUI], replace_queue: bool = True) -> None:
        nonlocal current_playing_item, skip_requeue_item
        items = playlists_db.get(name)
        if items is None:
            if tui and tui.enable:
                tui.status_msg = f"Playlist not found: {name}"
            return
        if not items:
            if tui and tui.enable:
                tui.status_msg = f"Playlist is empty: {name}"
            return
        if replace_queue:
            queue.clear()
            skip_requeue_item = current_playing_item
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
                        "play_once_overridden": True,
                        "mood": f"playlist:{name}",
                        "source": "manual",
                    }
                )
                paths.append(p)
        if paths:
            ensure_audio_data_for_tracks(paths)
        if tui and tui.enable:
            mode = "replaced" if replace_queue else "appended"
            if replace_queue and current_playing_item:
                tui.status_msg = (
                    f"Loaded playlist: {name} ({len(paths)} tracks, {mode} after current)"
                )
            else:
                tui.status_msg = f"Loaded playlist: {name} ({len(paths)} tracks, {mode})"

    def delete_playlist(name: str, tui: Optional[CursesTUI]) -> None:
        nonlocal active_playlist_name
        if name not in playlists_db:
            if tui and tui.enable:
                tui.status_msg = f"Playlist not found: {name}"
            return
        del playlists_db[name]
        save_playlists(playlists_db, playlists_filename)
        if active_playlist_name == name:
            active_playlist_name = None
            if tui and tui.enable:
                tui.playlist_editor_open = False
        if tui and tui.enable:
            tui.status_msg = f"Deleted playlist: {name}"

    def load_similar_for_mood(
        model: "SentenceTransformer", mood_text: str
    ) -> Tuple[List[Path], List[Tuple[str, float]]]:
        nonlocal current_similar_entries, current_similar_mood

        pl, sim = compute_top_for_mood(
            model=model,
            tags_data=tags_data or {},
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
            queue.append(
                {
                    "path": p,
                    "play_once": False,
                    "play_once_overridden": False,
                    "mood": mood_text,
                    "source": "manual",
                }
            )

    def ensure_audio_data_for_tracks(paths: List[Path]) -> None:
        nonlocal resolved_target_loudness
        missing = [p for p in paths if p not in audio_data]
        if not missing:
            return
        resolved, data = build_audio_data_for_playlist(
            playlist=missing,
            target_lufs=runtime_target_lufs,
            mp3_folder=mp3_folder,
            sample_filename=sample_filename,
            cache=loud_cache,
            logging=logging,
        )
        audio_data.update(data)
        for p, d in data.items():
            duration_by_stem[p.stem] = float(d.get("duration") or 0.0)
        if resolved_target_loudness is None:
            resolved_target_loudness = resolved

    def enqueue_auto_track(tui: Optional[CursesTUI]) -> bool:
        track_name = choose_auto_track_name(
            previous_track_name=last_completed_track_name,
            transitions=markov_transitions,
            global_counts=markov_global_counts,
        )
        if not track_name:
            if tui and tui.enable:
                tui.status_msg = "Auto mode has no Markov data yet."
            return False
        track_path = mp3_folder / track_name
        if not track_path.exists():
            if tui and tui.enable:
                tui.status_msg = f"Auto track missing on disk: {track_name}"
            return False
        ensure_audio_data_for_tracks([track_path])
        queue.append(
            {
                "path": track_path,
                "play_once": True,
                "play_once_overridden": False,
                "mood": "auto",
                "source": "auto",
            }
        )
        if tui and tui.enable:
            tui.status_msg = f"Auto-picked: {track_path.stem}"
        return True

    def toggle_playback_mode(tui: Optional[CursesTUI]) -> None:
        nonlocal playback_mode
        playback_mode = "auto" if playback_mode == "manual" else "manual"
        if tui and tui.enable:
            if playback_mode == "auto":
                tui.status_msg = "Auto mode armed. Manual current/queued tracks will finish first."
            else:
                tui.status_msg = "Manual mode active."

    def listen_hours_for_stem(stem: str, listens: int) -> float:
        dur = float(duration_by_stem.get(stem, 0.0))
        if dur <= 0.0:
            track_path = mp3_folder / f"{stem}.mp3"
            if track_path.exists():
                dur = get_audio_duration_seconds(track_path)
                duration_by_stem[stem] = dur
        return (max(0, int(listens)) * dur) / 3600.0

    def total_listen_stats() -> Tuple[int, float]:
        total_listens = 0
        total_hours = 0.0
        for stem, listens in counts.items():
            try:
                listen_count = max(0, int(listens))
            except Exception:
                listen_count = 0
            total_listens += listen_count
            total_hours += listen_hours_for_stem(stem, listen_count)
        return total_listens, total_hours

    def apply_new_target_loudness(
        new_target: Optional[float],
        tui: Optional[CursesTUI],
        current_track: Optional[Path] = None,
        current_channel: Optional["pygame.mixer.Channel"] = None,
    ) -> None:
        nonlocal runtime_target_lufs, resolved_target_loudness, audio_data

        runtime_target_lufs = new_target

        # Resolve the "actual" target loudness:
        # - if user specified a number: use it
        # - if None: fall back to sample-based target (same logic as build_audio_data_for_playlist)
        if runtime_target_lufs is None:
            sample_path = mp3_folder / sample_filename
            key = str(sample_path)
            sample_mtime = sample_path.stat().st_mtime if sample_path.exists() else 0.0
            entry = loud_cache.get(key)
            if not entry or abs(float(entry.get("mtime", 0.0)) - sample_mtime) > 0.5:
                entry = get_audio_stats(sample_path)
                loud_cache[key] = entry
            loud = entry.get("loudness_lufs")
            if isinstance(loud, (int, float)):
                resolved_target_loudness = float(loud)
        else:
            resolved_target_loudness = float(runtime_target_lufs)

        # Update cached per-track scales (fast: no re-decode)
        if isinstance(resolved_target_loudness, (int, float)):
            tgt = float(resolved_target_loudness)
            for p, d in audio_data.items():
                loud = d.get("loudness_lufs")
                true_peak = d.get("true_peak_dbtp")
                if isinstance(loud, (int, float)):
                    d["scale"] = float(
                        calculate_volume_scale(
                            tgt,
                            float(loud),
                            float(true_peak)
                            if isinstance(true_peak, (int, float))
                            else None,
                        )
                    )
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
                src = "sample" if runtime_target_lufs is None else "manual"
                tui.status_msg = (
                    f"Target loudness set to {resolved_target_loudness:.2f} LUFS ({src})"
                )
            else:
                tui.status_msg = "Target loudness updated"

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
        mood_text, new_top, new_vol = parse_mood_and_directives(text)

        if new_vol is not None:
            apply_new_target_loudness(new_vol, tui, current_track, current_channel)

        if new_top is not None:
            current_top_n = clamp_int(new_top, TOP_MIN, TOP_MAX)
            if tui and tui.enable:
                tui.status_msg = f"Top set to {current_top_n}"
        if mood_text:
            pl, _ = load_similar_for_mood(model, mood_text)
            if pl:
                if tui and tui.enable:
                    tui.focus_panel = "similar"
                    tui.status_msg = (
                        f"Loaded mood: “{mood_text}” (top {current_top_n}) — double click to add"
                    )
                if not enable_tui:
                    enqueue_tracks(pl, mood_text)
            else:
                if tui and tui.enable:
                    tui.status_msg = f"No matches for mood: “{mood_text}”"

    def reset_to_idle(tui: Optional[CursesTUI]) -> None:
        nonlocal current_similar_entries, current_similar_mood, queue
        queue = []
        tag_session.abandon()
        if tui and tui.enable:
            tui.focus_panel = "queue"
            tui.input_mode = "mood"
            tui.status_msg = "Stopped. Waiting for mood..."

    def handle_tag_submission(submitted: str, tui: CursesTUI) -> bool:
        if tui.input_mode not in ("tag_add", "tag_edit"):
            return False
        if tui.input_mode == "tag_add":
            tag_session.editing_ref = None
        msg = tag_session.submit(submitted)
        if msg:
            tui.status_msg = msg
        tui.input_mode = "tag_add"
        return True

    def apply_tag_actions() -> None:
        if tui.tag_delete_request:
            tui.tag_delete_request = False
            tui.status_msg = tag_session.delete(tui.tag_selected)
            tui.input_mode = "tag_add"
            tui.input_buffer = ""
            if tui.tag_selected >= tui.tag_len - 1:
                tui.tag_selected = max(0, tui.tag_len - 2)
        if tui.tag_edit_request:
            tui.tag_edit_request = False
            text, msg = tag_session.begin_edit(tui.tag_selected)
            tui.status_msg = msg
            if text:
                tui.tag_panel_open = True
                tui.focus_panel = "tags"
                tui.input_mode = "tag_edit"
                tui.input_buffer = text

    # Seed
    if initial_mood:
        pl, _ = load_similar_for_mood(model, initial_mood)
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
        # Audio already initialized before TUI

        def get_most_entries() -> List[Tuple[str, int]]:
            if tui.most_sort_mode == "time":
                return sorted(
                    counts.items(),
                    key=lambda kv: (
                        -listen_hours_for_stem(kv[0], kv[1]),
                        -kv[1],
                        kv[0].lower(),
                    ),
                )
            return sorted_counts(counts)

        def apply_queue_actions() -> None:
            if not queue:
                tui.queue_selected = 0
                tui.queue_move = 0
                tui.queue_delete = False
                tui.queue_click_row = None
                tui.queue_click_x = None
                tui.queue_drag_target = None
                tui.queue_drag_commit_target = None
                tui.queue_drag_commit_start = None
                tui.queue_drag_started_selected = False
                return

            if tui.queue_click_row is not None:
                idx = tui.queue_click_row
                click_x = tui.queue_click_x or 0
                if 0 <= idx < len(queue):
                    if tui.queue_mode_col_start <= click_x < tui.queue_mode_col_end:
                        cur = item_effective_play_once(queue[idx])
                        set_item_effective_play_once(queue[idx], not cur)
                        tui.status_msg = (
                            "Toggled play-once." if not cur else "Set to loop."
                        )
                tui.queue_click_row = None
                tui.queue_click_x = None

            if tui.queue_drag_commit_target is not None:
                start_idx = (
                    tui.queue_drag_commit_start
                    if tui.queue_drag_commit_start is not None
                    else tui.queue_selected
                )
                target_idx = tui.queue_drag_commit_target
                if 0 <= start_idx < len(queue):
                    target_idx = max(0, min(len(queue) - 1, target_idx))
                    if target_idx != start_idx:
                        item = queue.pop(start_idx)
                        queue.insert(target_idx, item)
                        tui.queue_selected = target_idx
                        tui.status_msg = "Queue reordered."
                tui.queue_drag_commit_target = None
                tui.queue_drag_commit_start = None

            if tui.queue_move:
                idx = tui.queue_selected
                new_idx = idx + tui.queue_move
                if 0 <= idx < len(queue):
                    new_idx = max(0, min(len(queue) - 1, new_idx))
                    if new_idx != idx:
                        item = queue.pop(idx)
                        queue.insert(new_idx, item)
                        tui.queue_selected = new_idx
                        tui.status_msg = "Queue reordered."
                tui.queue_move = 0

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
            queue.append(
                {
                    "path": path,
                    "play_once": False,
                    "play_once_overridden": False,
                    "mood": current_similar_mood,
                }
            )
            queue[-1]["source"] = "manual"
            tui.status_msg = f"Added to queue: {base}"

        def apply_most_actions() -> None:
            if not tui.most_add:
                return
            tui.most_add = False
            entries = get_most_entries()
            if not entries:
                tui.status_msg = "No tracks in Most."
                return
            idx = min(max(0, tui.most_selected), len(entries) - 1)
            name, _ = entries[idx]
            path = mp3_folder / f"{name}.mp3"
            if not path.exists():
                tui.status_msg = "Track not found on disk."
                return
            ensure_audio_data_for_tracks([path])
            queue.append(
                {
                    "path": path,
                    "play_once": False,
                    "play_once_overridden": False,
                    "mood": "most",
                    "source": "manual",
                }
            )
            tui.status_msg = f"Added to queue: {name}"

        def apply_playlist_editor_actions() -> None:
            nonlocal active_playlist_name

            if (
                tui.playlist_open_pending_idx is not None
                and not tui.playlist_editor_open
                and (
                    time.monotonic() - tui.playlist_open_pending_ts
                ) > tui.DOUBLE_CLICK_SECONDS
            ):
                plist = get_playlists_list()
                idx = tui.playlist_open_pending_idx
                tui.playlist_open_pending_idx = None
                tui.playlist_open_pending_ts = 0.0
                if 0 <= idx < len(plist):
                    tui.playlist_selected = idx
                    open_playlist_editor(plist[idx][0], tui)

            if tui.playlist_done_request:
                tui.playlist_done_request = False
                tui.playlist_editor_open = False
                active_playlist_name = None
                tui.status_msg = "Closed playlist."

            if tui.playlist_load_request:
                tui.playlist_load_request = False
                if active_playlist_name:
                    load_playlist(active_playlist_name, tui, replace_queue=True)
                else:
                    tui.status_msg = "No playlist is open."

            if tui.playlist_add_current_request:
                tui.playlist_add_current_request = False
                if current_playing_item:
                    path = current_playing_item.get("path")
                    if isinstance(path, Path):
                        add_path_to_active_playlist(
                            path,
                            item_effective_play_once(current_playing_item),
                            tui,
                        )
                    else:
                        tui.status_msg = "No current track to add."
                else:
                    tui.status_msg = "No current track to add."

            if tui.playlist_add_queue_request:
                tui.playlist_add_queue_request = False
                if queue:
                    idx = min(max(0, tui.queue_selected), len(queue) - 1)
                    item = queue[idx]
                    path = item.get("path")
                    if isinstance(path, Path):
                        add_path_to_active_playlist(
                            path,
                            item_effective_play_once(item),
                            tui,
                        )
                    else:
                        tui.status_msg = "Track not found on disk."
                else:
                    tui.status_msg = "No queued track to add."

            if tui.playlist_add_similar_request:
                tui.playlist_add_similar_request = False
                entries = current_similar_entries or []
                if entries:
                    idx = min(max(0, tui.similar_selected), len(entries) - 1)
                    base = entries[idx][0]
                    path = mp3_folder / f"{base}.mp3"
                    if path.exists():
                        add_path_to_active_playlist(path, False, tui)
                    else:
                        tui.status_msg = "Track not found on disk."
                else:
                    tui.status_msg = "No similar track to add."

            if tui.playlist_add_most_request:
                tui.playlist_add_most_request = False
                entries = get_most_entries()
                if entries:
                    idx = min(max(0, tui.most_selected), len(entries) - 1)
                    name, _ = entries[idx]
                    path = mp3_folder / f"{name}.mp3"
                    if path.exists():
                        add_path_to_active_playlist(path, False, tui)
                    else:
                        tui.status_msg = "Track not found on disk."
                else:
                    tui.status_msg = "No track to add."

            items = playlists_db.get(active_playlist_name or "", [])
            if tui.playlist_item_toggle_request is not None:
                idx = tui.playlist_item_toggle_request
                tui.playlist_item_toggle_request = None
                if 0 <= idx < len(items):
                    item = items[idx]
                    play_once = not bool(item.get("play_once", False))
                    item["play_once"] = play_once
                    save_playlists(playlists_db, playlists_filename)
                    mode = "once" if play_once else "loop"
                    tui.status_msg = f"Set playlist item to {mode}."

            if tui.playlist_item_drag_commit_target is not None:
                start_idx = (
                    tui.playlist_item_drag_commit_start
                    if tui.playlist_item_drag_commit_start is not None
                    else tui.playlist_track_selected
                )
                target_idx = tui.playlist_item_drag_commit_target
                if 0 <= start_idx < len(items):
                    target_idx = max(0, min(len(items) - 1, target_idx))
                    if target_idx != start_idx:
                        item = items.pop(start_idx)
                        items.insert(target_idx, item)
                        save_playlists(playlists_db, playlists_filename)
                        tui.playlist_track_selected = target_idx
                        tui.status_msg = f"Reordered {active_playlist_name}."
                tui.playlist_item_drag_commit_target = None
                tui.playlist_item_drag_commit_start = None

            if tui.playlist_item_remove_request is not None:
                idx = tui.playlist_item_remove_request
                tui.playlist_item_remove_request = None
                if 0 <= idx < len(items):
                    base = str(items[idx].get("base") or "track")
                    del items[idx]
                    save_playlists(playlists_db, playlists_filename)
                    if idx >= len(items):
                        tui.playlist_track_selected = max(0, len(items) - 1)
                    tui.status_msg = f"Removed from {active_playlist_name}: {base}"

        def apply_most_sort_toggle() -> None:
            if not tui.most_toggle_request:
                return
            tui.most_toggle_request = False
            tui.most_sort_mode = "time" if tui.most_sort_mode == "count" else "count"
            tui.most_selected = 0
            tui.most_scroll = 0
            tui.status_msg = f"Most sort: {tui.most_sort_mode}"

        def apply_current_play_once_toggle() -> None:
            if not tui.current_play_once_toggle_request:
                return
            tui.current_play_once_toggle_request = False
            if not current_playing_item:
                tui.status_msg = "No current track to toggle."
                return
            cur = item_effective_play_once(current_playing_item)
            set_item_effective_play_once(current_playing_item, not cur)
            tui.status_msg = "Toggled play-once." if not cur else "Set to loop."

        def handle_save_playlist_input(submitted: str) -> bool:
            if tui.input_mode != "save_playlist":
                return False
            name = submitted.strip()
            if name:
                save_playlist(name, tui)
            tui.input_mode = "mood"
            tui.input_buffer = ""
            return True

        def handle_playlist_requests() -> None:
            if tui.playlist_save_request:
                tui.playlist_save_request = False
                tui.input_mode = "save_playlist"
                tui.input_buffer = ""
                tui.status_msg = "Enter playlist name to save."
            if tui.playlist_activate:
                tui.playlist_activate = False
                tui.playlist_open_pending_idx = None
                tui.playlist_open_pending_ts = 0.0
                if tui.playlist_editor_open and active_playlist_name:
                    load_playlist(active_playlist_name, tui, replace_queue=True)
                else:
                    plist = get_playlists_list()
                    if plist:
                        idx = min(tui.playlist_selected, len(plist) - 1)
                        load_playlist(plist[idx][0], tui, replace_queue=True)
                    else:
                        tui.status_msg = "No playlists to load."
            if tui.playlist_delete_request:
                tui.playlist_delete_request = False
                if tui.playlist_editor_open and active_playlist_name:
                    delete_playlist(active_playlist_name, tui)
                else:
                    plist = get_playlists_list()
                    if plist:
                        idx = min(tui.playlist_selected, len(plist) - 1)
                        delete_playlist(plist[idx][0], tui)
                    else:
                        tui.status_msg = "No playlists to delete."

        def handle_common_actions() -> None:
            if tui.playback_mode_toggle_request:
                tui.playback_mode_toggle_request = False
                toggle_playback_mode(tui)
            apply_current_play_once_toggle()
            apply_most_sort_toggle()
            apply_queue_actions()
            apply_similar_actions()
            apply_most_actions()
            apply_playlist_editor_actions()
            apply_tag_actions()
            handle_playlist_requests()

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

                total_listens, total_hours = total_listen_stats()
                submitted = tui.render(
                    now_playing="",
                    target_lufs=None,
                    current_lufs=None,
                    loudness_diff=None,
                    volume_scale=None,
                    elapsed_sec=max(0.0, seconds - max(0.0, end - time.monotonic())),
                    total_sec=seconds,
                    playback_mode=playback_mode,
                    current_play_once=None,
                    counts=counts,
                    listen_hours_for_stem=listen_hours_for_stem,
                    total_listens=total_listens,
                    total_listen_hours=total_hours,
                    similar_entries=current_similar_entries,
                    similar_mood=current_similar_mood,
                    queue_items=queue,
                    playlists=get_playlists_list(),
                    active_playlist_name=active_playlist_name
                    if tui.playlist_editor_open
                    else None,
                    active_playlist_items=get_active_playlist_items(tui),
                )
                if submitted:
                    if handle_save_playlist_input(submitted):
                        continue
                    if handle_tag_submission(submitted, tui):
                        continue
                    apply_submission(submitted, tui)
                handle_common_actions()

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
                lock_since_wall = update_lock_or_exit(
                    mac_is_locked_poll(), lock_since_wall, next_exit_dt
                )

                # Idle (only interactive in TUI mode)
                if not queue:
                    if playback_mode == "auto":
                        if enqueue_auto_track(tui):
                            continue
                    if not enable_tui:
                        time.sleep(0.2)
                        continue
                    total_listens, total_hours = total_listen_stats()
                    submitted = tui.render(
                        now_playing="",
                        target_lufs=None,
                        current_lufs=None,
                        loudness_diff=None,
                        volume_scale=None,
                        elapsed_sec=0.0,
                        total_sec=0.0,
                        playback_mode=playback_mode,
                        current_play_once=None,
                        counts=counts,
                        listen_hours_for_stem=listen_hours_for_stem,
                        total_listens=total_listens,
                        total_listen_hours=total_hours,
                        similar_entries=current_similar_entries,
                        similar_mood=current_similar_mood,
                        queue_items=queue,
                        playlists=get_playlists_list(),
                        active_playlist_name=active_playlist_name
                        if tui.playlist_editor_open
                        else None,
                        active_playlist_items=get_active_playlist_items(tui),
                    )
                    if submitted:
                        if submitted.strip() == "--help":
                            tui.show_help = True
                            time.sleep(0.01)
                            continue
                        if handle_save_playlist_input(submitted):
                            time.sleep(0.01)
                            continue
                        if handle_tag_submission(submitted, tui):
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
                                pl, _ = load_similar_for_mood(model, mood_text)
                                if pl:
                                    tui.focus_panel = "similar"
                                    tui.status_msg = (
                                        f"Loaded mood: “{mood_text}” (top {current_top_n}) — double click to add"
                                    )
                                else:
                                    tui.status_msg = f"No matches for mood: “{mood_text}”"
                            except Exception as e:
                                tui.status_msg = f"Error building queue: {e}"
                    handle_common_actions()
                    time.sleep(0.01)
                    continue

                # Queue loop
                current_item = queue.pop(0)
                current_playing_item = current_item if isinstance(current_item, dict) else None
                track_path = current_item.get("path") if isinstance(current_item, dict) else None
                track_source = (
                    str(current_item.get("source"))
                    if isinstance(current_item, dict) and current_item.get("source")
                    else "manual"
                )

                if not isinstance(track_path, Path):
                    current_playing_item = None
                    continue
                tag_session.start(track_path)
                if enable_tui and tui.input_mode in ("tag_add", "tag_edit"):
                    tui.input_buffer = ""

                # refresh queue selection bounds
                if tui.queue_selected >= len(queue):
                    tui.queue_selected = max(0, len(queue) - 1)

                # Play single item
                while True:
                    maybe_exit(next_exit_dt)

                    if track_path not in audio_data:
                        tag_session.abandon()
                        current_playing_item = None
                        break

                    filename_ext = track_path.name
                    vol_scale = float(audio_data[track_path].get("scale") or 0.5)
                    current_loudness = audio_data[track_path].get("loudness_lufs")
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
                        if total_dur <= 0.0:
                            total_dur = float(audio_data[track_path].get("duration") or 0.0)

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
                                total_listens, total_hours = total_listen_stats()
                                submitted = tui.render(
                                    now_playing=filename_ext,
                                    target_lufs=resolved_target_loudness,
                                    current_lufs=float(current_loudness)
                                    if isinstance(current_loudness, (int, float))
                                    else None,
                                    loudness_diff=float(loudness_diff)
                                    if isinstance(loudness_diff, (int, float))
                                    else None,
                                    volume_scale=vol_scale,
                                    elapsed_sec=elapsed,
                                    total_sec=total_dur,
                                    playback_mode=playback_mode,
                                    current_play_once=item_effective_play_once(current_item),
                                    counts=counts,
                                    listen_hours_for_stem=listen_hours_for_stem,
                                    total_listens=total_listens,
                                    total_listen_hours=total_hours,
                                    similar_entries=current_similar_entries,
                                    similar_mood=current_similar_mood,
                                    queue_items=queue,
                                    playlists=get_playlists_list(),
                                    current_tags=tag_session.existing_tags(),
                                    pending_tags=tag_session.pending_tags(),
                                    active_playlist_name=active_playlist_name
                                    if tui.playlist_editor_open
                                    else None,
                                    active_playlist_items=get_active_playlist_items(tui),
                                )
                                if submitted:
                                    if handle_save_playlist_input(submitted):
                                        pass
                                    elif handle_tag_submission(submitted, tui):
                                        pass
                                    else:
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
                                handle_common_actions()

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

                        if not interrupted_by_lock and not user_skip and not user_stop:
                            # The loop above exits normally when pygame reports the
                            # channel is no longer busy. Treat that as completion;
                            # elapsed duration is only a fallback sanity check.
                            reached_end = True
                            if total_dur > 0:
                                elapsed_final = max(0.0, time.monotonic() - started_ts)
                                reached_end = (
                                    not channel.get_busy()
                                    or elapsed_final >= max(0.0, total_dur - 0.75)
                                )

                        if reached_end:
                            last_completed_track_name = track_path.name
                            trailing_tag = (
                                tui.input_buffer.strip()
                                if enable_tui and tui.input_mode in ("tag_add", "tag_edit")
                                else ""
                            )
                            saved_tags = tag_session.finish(trailing_tag)
                            if trailing_tag and enable_tui and tui.input_mode in ("tag_add", "tag_edit"):
                                tui.input_buffer = ""
                            if track_source == "manual":
                                increment_listen(track_path, counts)
                                save_listen_counts(counts, listen_db_filename)
                                record_listen_timestamp(track_path, listen_timestamps)
                                save_listen_timestamps(
                                    listen_timestamps, listen_timestamps_filename
                                )
                                if tui and tui.enable:
                                    listen_msg = (
                                        f"Recorded listen: {track_path.stem} "
                                        f"({counts.get(track_path.stem, 0)})"
                                    )
                                    if saved_tags:
                                        listen_msg += f"; saved {saved_tags} tag(s)"
                                    tui.status_msg = listen_msg
                        else:
                            tag_session.abandon()

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
                        if (
                            not item_effective_play_once(current_item)
                            and current_item is not skip_requeue_item
                        ):
                            queue.append(current_item)
                        if current_item is skip_requeue_item:
                            skip_requeue_item = None
                        current_playing_item = None
                        break

                    except pygame.error as e:
                        tag_session.abandon()
                        if enable_tui:
                            tui.status_msg = f"Error playing {filename_ext}: {e}"
                        else:
                            print(f"Error playing {filename_ext}: {e}")
                        time.sleep(1)
                        if (
                            not item_effective_play_once(current_item)
                            and current_item is not skip_requeue_item
                        ):
                            queue.append(current_item)
                        if current_item is skip_requeue_item:
                            skip_requeue_item = None
                        current_playing_item = None
                        break
                    except Exception as e:
                        tag_session.abandon()
                        if enable_tui:
                            tui.status_msg = f"Unexpected error: {e}"
                        else:
                            print(f"Unexpected error for {filename_ext}: {e}")
                        time.sleep(1)
                        if (
                            not item_effective_play_once(current_item)
                            and current_item is not skip_requeue_item
                        ):
                            queue.append(current_item)
                        if current_item is skip_requeue_item:
                            skip_requeue_item = None
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
    p.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP_N,
        help="Top-N matches to find/play.",
    )
    p.add_argument(
        "--vol",
        type=float,
        default=None,
        help="Target loudness in integrated LUFS (if omitted, uses sample).",
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
    p.add_argument(
        "--mode",
        choices=("manual", "auto"),
        default="manual",
        help="Manual mode records listens; auto mode uses the Markov chain without recording.",
    )
    p.add_argument("--no-tui", action="store_true", help="Disable curses UI.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    mp3_folder = Path(args.folder)
    tags_file = Path(args.tags)
    listen_db = LISTEN_DB_FILE

    if mp3_folder.name == "mid-mp3s":
        listen_db = "mid_listen_counts.json"
        tags_file = Path("mid_tags.json")

    try:
        main(
            initial_mood=args.mood,
            top_n=args.top,
            mp3_folder=mp3_folder,
            tags_file=tags_file,
            sample_filename=args.sample,
            target_lufs=args.vol,
            logging=bool(args.log),
            enable_tui=(not args.no_tui),
            listen_db_filename=listen_db,
            playback_mode=str(args.mode),
        )
    except SystemExit:
        print("\nPlayback stopped.")
    except KeyboardInterrupt:
        print("\nPlayback interrupted.")
