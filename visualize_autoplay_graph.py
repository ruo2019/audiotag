#!/usr/bin/env python3
"""Build an interactive autoplay-transition graph from an MP3 folder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import webbrowser
from collections import Counter, defaultdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    from mutagen.mp3 import MP3
except ModuleNotFoundError:  # pragma: no cover - duration data is optional
    MP3 = None

try:
    from cli_helpers import print_banner, print_message, require_console
except ModuleNotFoundError:  # pragma: no cover - standalone fallback
    def print_banner(message: str, _console=None) -> None:
        print(message)

    def print_message(message: str, _console=None) -> None:
        print(message)

    def require_console():
        return None

try:
    from player.constants import (
        AUTOPLAY_FALLBACK_PROBABILITY,
        AUTOPLAY_WINDOW_SECONDS,
        CONFIG_KEY,
        META_FILENAME,
        PLAY_HISTORY_CONFIG_KEY,
    )
except ModuleNotFoundError:  # pragma: no cover - standalone fallback
    AUTOPLAY_FALLBACK_PROBABILITY = 0.05
    AUTOPLAY_WINDOW_SECONDS = 10 * 60
    CONFIG_KEY = "__config__"
    META_FILENAME = ".mp3meta.json"
    PLAY_HISTORY_CONFIG_KEY = "play_history"

DEFAULT_WINDOW_MINUTES = AUTOPLAY_WINDOW_SECONDS / 60.0
DEFAULT_FALLBACK_PROBABILITY = 0.0
DEFAULT_OUTPUT_TEMPLATE = "autoplay_graph_{folder}.html"
DEFAULT_PAIRING_MODE = "consecutive"
TRENDING_HALF_LIFE_SECONDS = 2 * 24 * 60 * 60
TRENDING_BASELINE_DURATION_SECONDS = 3 * 60 + 30
NODE_MIN_SIZE = 7
NODE_SIZE_RANGE = 63
NODE_SIZE_POWER = 1.25
VIS_NETWORK_CDN = "https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"
LISTEN_COUNTS_FILE = "listen_counts.json"
MID_LISTEN_COUNTS_FILE = "mid_listen_counts.json"
ARTISTS_FILE = "artists.json"
MID_ARTISTS_FILE = "mid_artists.json"
ALL_FOLDERS = (Path("static/mp3"), Path("static/mid-mp3s"))
GRAPH_VIEW_OPTIONS = (
    ("mp3", "MP3", Path("static/mp3"), False),
    ("mid-mp3s", "Mid MP3s", Path("static/mid-mp3s"), False),
    ("all", "All", Path("all"), True),
)
GRAPH_VIEW_KEYS = tuple(key for key, _label, _path, _use_all_folders in GRAPH_VIEW_OPTIONS)

console = None
audio_duration_cache = {}


def emit(message: str) -> None:
    print_message(message, console)
    sys.stdout.flush()
    sys.stderr.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an interactive HTML visualization of autoplay-style song "
            "transitions inferred from play history."
        )
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default="mp3",
        choices=GRAPH_VIEW_KEYS,
        help=(
            "Dataset to visualize: mp3, mid-mp3s, or all (default: mp3)."
        ),
    )
    parser.add_argument(
        "--window-minutes",
        type=float,
        default=DEFAULT_WINDOW_MINUTES,
        help=(
            "Maximum idle time after the source track duration for linking plays "
            "(default: 10)"
        ),
    )
    parser.add_argument(
        "--pairing",
        choices=("window", "consecutive"),
        default=DEFAULT_PAIRING_MODE,
        help=(
            "How to turn the global play timeline into transitions: "
            "'window' links every later song within the time window; "
            "'consecutive' only links adjacent plays. Defaults to "
            "'consecutive' because that matches next-track autoplay behavior."
        ),
    )
    parser.add_argument(
        "--fallback-probability",
        type=float,
        default=DEFAULT_FALLBACK_PROBABILITY,
        help=(
            "Reserved probability mass for unseen outgoing edges. "
            "Defaults to 0 so observed transitions sum to 100%%."
        ),
    )
    parser.add_argument(
        "--include-self-loops",
        action="store_true",
        help="Include transitions from a song to itself",
    )
    parser.add_argument(
        "--output",
        help=(
            "Output HTML path (default: autoplay_graph_<folder>.html in the repo root)"
        ),
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Write the HTML but do not open it in a browser",
    )
    parser.add_argument(
        "--live",
        "--watch",
        dest="live",
        action="store_true",
        help=(
            "Start a local live-updating graph server. The browser checks for "
            "new listen data and updates without rerunning this script."
        ),
    )
    parser.add_argument(
        "--refresh-seconds",
        type=float,
        default=2.0,
        help="How often --live checks for new listen data (default: 2)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for --live mode (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port for --live mode; 0 chooses an available port (default: 0)",
    )
    return parser.parse_args()


def decode_play_history(encoded_history):
    if not isinstance(encoded_history, list):
        return []

    decoded_history = []
    current_timestamp = None
    for index, value in enumerate(encoded_history):
        if not isinstance(value, (int, float)):
            continue
        if index == 0 or current_timestamp is None:
            current_timestamp = int(value)
        else:
            current_timestamp += int(value)
        decoded_history.append(current_timestamp)
    return decoded_history


def listen_timestamps_filename_for_folder(folder: Path) -> str:
    folder_name = folder.name or "default"
    return f"listen_timestamps_{folder_name}.json"


def listen_counts_filename_for_folder(folder: Path) -> str:
    if folder.name == "mid-mp3s":
        return MID_LISTEN_COUNTS_FILE
    return LISTEN_COUNTS_FILE


def artists_filename_for_folder(folder: Path) -> str:
    if folder.name == "mid-mp3s":
        return MID_ARTISTS_FILE
    return ARTISTS_FILE


def parse_repo_timestamp(value):
    try:
        return int(datetime.fromisoformat(str(value)).timestamp())
    except Exception:
        return None


def get_audio_duration_seconds(mp3_path: Path) -> float:
    if MP3 is None:
        return 0.0
    try:
        stat_result = mp3_path.stat()
    except OSError:
        return 0.0

    cache_key = str(mp3_path.resolve())
    cache_signature = (stat_result.st_mtime_ns, stat_result.st_size)
    cached_duration = audio_duration_cache.get(cache_key)
    if cached_duration and cached_duration[0] == cache_signature:
        return cached_duration[1]

    try:
        duration_seconds = max(0.0, float(MP3(str(mp3_path)).info.length or 0.0))
    except Exception:
        duration_seconds = 0.0

    audio_duration_cache[cache_key] = (cache_signature, duration_seconds)
    return duration_seconds


def load_folder_durations(folder: Path):
    durations = {}
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() == ".mp3":
            durations[path.name] = get_audio_duration_seconds(path)
    return durations


def load_repo_listen_counts(folder: Path, repo_root: Path):
    counts_path = repo_root / listen_counts_filename_for_folder(folder)
    raw = json.loads(counts_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid listen counts file: {counts_path}")

    song_counts = {}
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() == ".mp3":
            song_counts[path.name] = int(raw.get(path.stem, 0) or 0)
    return song_counts, counts_path


def load_repo_history_mapping(folder: Path, repo_root: Path, song_counts):
    timestamps_path = repo_root / listen_timestamps_filename_for_folder(folder)
    if not timestamps_path.is_file():
        history_mapping = {song_name: [] for song_name in song_counts}
        return history_mapping, timestamps_path

    raw = json.loads(timestamps_path.read_text(encoding="utf-8"))
    history_mapping = {song_name: [] for song_name in song_counts}

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            song_name = item.get("track")
            timestamp = parse_repo_timestamp(item.get("timestamp"))
            if song_name in history_mapping and timestamp is not None:
                history_mapping[song_name].append(timestamp)
    elif isinstance(raw, dict):
        for stem, timestamps in raw.items():
            song_name = f"{stem}.mp3"
            if song_name not in history_mapping or not isinstance(timestamps, list):
                continue
            decoded = []
            for value in timestamps:
                parsed = parse_repo_timestamp(value)
                if parsed is not None:
                    decoded.append(parsed)
            history_mapping[song_name].extend(decoded)
    else:
        raise ValueError(f"Invalid timestamp history file: {timestamps_path}")

    for song_name in history_mapping:
        history_mapping[song_name].sort()
    return history_mapping, timestamps_path


def normalize_artist_names(value):
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = []
    seen = set()
    artists = []
    for item in raw_items:
        name = " ".join(str(item).strip().split())
        key = name.casefold()
        if name and key not in seen:
            seen.add(key)
            artists.append(name)
    return artists


def load_repo_artists(folder: Path, repo_root: Path, song_counts):
    artists_path = repo_root / artists_filename_for_folder(folder)
    raw = {}
    if artists_path.is_file():
        data = json.loads(artists_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            raw = data

    song_artists = {}
    for song_name in song_counts:
        stem = Path(song_name).stem
        artists = normalize_artist_names(raw.get(stem))
        if artists:
            song_artists[song_name] = artists
    return song_artists, artists_path


def load_folder_data(folder: Path):
    repo_root = Path(__file__).resolve().parent
    durations = load_folder_durations(folder)
    meta_path = folder / META_FILENAME
    if meta_path.is_file():
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        song_counts = {
            name: count
            for name, count in data.items()
            if not str(name).startswith("__") and isinstance(count, int)
        }
        config = data.get(CONFIG_KEY, {})
        raw_history_mapping = config.get(PLAY_HISTORY_CONFIG_KEY, {})

        history_mapping = {}
        for song_name in song_counts:
            decoded_history = decode_play_history(raw_history_mapping.get(song_name, []))
            history_mapping[song_name] = decoded_history

        song_artists, artists_path = load_repo_artists(folder, repo_root, song_counts)
        return song_counts, history_mapping, durations, song_artists, Path(
            f"{meta_path.name} + {artists_path.name}"
        )

    song_counts, counts_path = load_repo_listen_counts(folder, repo_root)
    history_mapping, timestamps_path = load_repo_history_mapping(
        folder, repo_root, song_counts
    )
    song_artists, artists_path = load_repo_artists(folder, repo_root, song_counts)
    source_description = Path(
        f"{counts_path.name} + {timestamps_path.name} + {artists_path.name}"
    )
    return song_counts, history_mapping, durations, song_artists, source_description


def prefix_mapping_keys(mapping, prefix: str):
    return {f"{prefix}/{key}": value for key, value in mapping.items()}


def load_all_folder_data(repo_root: Path):
    combined_counts = {}
    combined_history = {}
    combined_durations = {}
    combined_artists = {}
    sources = []

    for relative_folder in ALL_FOLDERS:
        folder = repo_root / relative_folder
        if not folder.is_dir():
            raise FileNotFoundError(f"folder '{folder}' does not exist")

        song_counts, counts_path = load_repo_listen_counts(folder, repo_root)
        history_mapping, timestamps_path = load_repo_history_mapping(
            folder, repo_root, song_counts
        )
        song_artists, artists_path = load_repo_artists(folder, repo_root, song_counts)
        prefix = folder.name
        combined_counts.update(prefix_mapping_keys(song_counts, prefix))
        combined_history.update(prefix_mapping_keys(history_mapping, prefix))
        combined_durations.update(prefix_mapping_keys(load_folder_durations(folder), prefix))
        combined_artists.update(prefix_mapping_keys(song_artists, prefix))
        sources.extend([counts_path.name, timestamps_path.name, artists_path.name])

    return combined_counts, combined_history, combined_durations, combined_artists, Path(" + ".join(sources))


def build_global_events(history_mapping):
    events = []
    for song_name, timestamps in history_mapping.items():
        for timestamp in timestamps:
            if isinstance(timestamp, (int, float)):
                events.append((int(timestamp), song_name))
    events.sort(key=lambda item: (item[0], item[1]))
    return events


def build_transition_counts(
    events,
    window_seconds: int,
    pairing_mode: str,
    include_self_loops: bool,
    durations=None,
):
    edge_counts = Counter()
    edge_gaps = defaultdict(list)
    durations = durations or {}

    def max_start_gap_seconds(song_name: str) -> float:
        duration_seconds = max(0.0, float(durations.get(song_name, 0.0) or 0.0))
        return float(window_seconds) + duration_seconds

    if pairing_mode == "consecutive":
        for current_event, next_event in zip(events, events[1:]):
            current_timestamp, current_song = current_event
            next_timestamp, next_song = next_event
            gap_seconds = next_timestamp - current_timestamp
            if gap_seconds < 0 or gap_seconds > max_start_gap_seconds(current_song):
                continue
            if not include_self_loops and current_song == next_song:
                continue
            edge_counts[(current_song, next_song)] += 1
            edge_gaps[(current_song, next_song)].append(gap_seconds)
        return edge_counts, edge_gaps

    for start_index, (current_timestamp, current_song) in enumerate(events):
        next_index = start_index + 1
        max_gap_seconds = max_start_gap_seconds(current_song)
        while next_index < len(events):
            next_timestamp, next_song = events[next_index]
            gap_seconds = next_timestamp - current_timestamp
            if gap_seconds > max_gap_seconds:
                break
            if gap_seconds >= 0 and (include_self_loops or current_song != next_song):
                edge_counts[(current_song, next_song)] += 1
                edge_gaps[(current_song, next_song)].append(gap_seconds)
            next_index += 1

    return edge_counts, edge_gaps


def clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, value))


def scale_node_size(value: float, max_value: float) -> float:
    if max_value <= 0 or value <= 0:
        return float(NODE_MIN_SIZE)
    normalized = max(0.0, min(1.0, value / max_value))
    return NODE_MIN_SIZE + NODE_SIZE_RANGE * (normalized ** NODE_SIZE_POWER)


def encode_timestamps(timestamps):
    encoded = []
    previous = None
    for timestamp in sorted(int(timestamp) for timestamp in timestamps):
        if previous is None:
            encoded.append(timestamp)
        else:
            encoded.append(timestamp - previous)
        previous = timestamp
    return encoded


def trending_weight(duration_seconds: float | None) -> float:
    if not isinstance(duration_seconds, (int, float)) or duration_seconds <= 0:
        return 1.0
    return max(0.0, float(duration_seconds) / TRENDING_BASELINE_DURATION_SECONDS)


def build_trending_payload(
    folder: Path,
    song_counts,
    history_mapping,
    durations,
    events,
    generated_from: str,
):
    rows = []
    historyless_play_count = 0
    mismatched_history_count = 0
    for song_name in sorted(song_counts, key=str.casefold):
        play_count = max(0, int(song_counts.get(song_name, 0) or 0))
        timestamps = [
            int(timestamp)
            for timestamp in history_mapping.get(song_name, [])
            if isinstance(timestamp, (int, float))
        ]
        timestamps.sort()
        history_count = len(timestamps)
        if play_count and not timestamps:
            historyless_play_count += 1
        if play_count != history_count:
            mismatched_history_count += 1
        duration_seconds = max(0.0, float(durations.get(song_name, 0.0) or 0.0))
        rows.append(
            {
                "name": song_name,
                "playCount": play_count,
                "historyCount": history_count,
                "history": encode_timestamps(timestamps),
                "durationSeconds": round(duration_seconds, 3),
                "trendingWeight": round(trending_weight(duration_seconds), 6),
            }
        )

    timeline = [timestamp for timestamp, _song_name in events]
    return {
        "summary": {
            "folder": str(folder),
            "songCount": len(rows),
            "metadataPlayCount": sum(
                max(0, int(count or 0)) for count in song_counts.values()
            ),
            "historyEventCount": len(events),
            "timelinePointCount": len(timeline),
            "firstTimestamp": timeline[0] if timeline else None,
            "lastTimestamp": timeline[-1] if timeline else None,
            "generatedTimestamp": int(time.time()),
            "generatedFrom": generated_from,
            "trendingHalfLifeSeconds": TRENDING_HALF_LIFE_SECONDS,
            "trendingBaselineDurationSeconds": TRENDING_BASELINE_DURATION_SECONDS,
            "historylessPlayCount": historyless_play_count,
            "mismatchedHistoryCount": mismatched_history_count,
        },
        "timeline": encode_timestamps(sorted(set(timeline))),
        "songs": rows,
    }


def build_artist_payload(
    folder: Path,
    song_counts,
    history_mapping,
    durations,
    song_artists,
    events,
    window_minutes: float,
    pairing_mode: str,
    include_self_loops: bool,
    generated_from: str,
):
    artist_song_sets = defaultdict(set)
    artist_play_counts = Counter()
    artist_history_counts = Counter()
    artist_listen_seconds = defaultdict(float)
    artist_timestamps = defaultdict(list)
    artist_song_play_counts = defaultdict(Counter)
    artist_song_details = defaultdict(dict)
    artist_before_context = defaultdict(Counter)
    artist_after_context = defaultdict(Counter)
    song_artist_rows = []
    tagged_song_count = 0
    tagged_play_count = 0
    unassigned_play_count = 0
    generated_timestamp = int(time.time())

    for song_name, play_count_raw in song_counts.items():
        play_count = max(0, int(play_count_raw or 0))
        artists = list(song_artists.get(song_name, []) or [])
        if not artists:
            unassigned_play_count += play_count
            continue
        tagged_song_count += 1
        tagged_play_count += play_count
        duration_seconds = max(0.0, float(durations.get(song_name, 0.0) or 0.0))
        history = [
            int(timestamp)
            for timestamp in history_mapping.get(song_name, [])
            if isinstance(timestamp, (int, float))
        ]
        for artist in artists:
            artist_song_sets[artist].add(song_name)
            artist_play_counts[artist] += play_count
            artist_history_counts[artist] += len(history)
            artist_listen_seconds[artist] += duration_seconds * play_count
            artist_timestamps[artist].extend(history)
            artist_song_play_counts[artist][song_name] += play_count
            artist_song_details[artist][song_name] = {
                "name": song_name,
                "playCount": play_count,
                "historyCount": len(history),
                "durationSeconds": round(duration_seconds, 2),
                "listenTimeSeconds": round(duration_seconds * play_count, 2),
                "history": encode_timestamps(sorted(history)),
            }
        song_artist_rows.append(
            {
                "song": song_name,
                "artists": artists,
                "playCount": play_count,
                "historyCount": len(history),
                "durationSeconds": round(duration_seconds, 2),
            }
        )

    artist_names = sorted(artist_play_counts, key=str.casefold)
    max_artist_plays = max(artist_play_counts.values(), default=1)
    max_artist_listen_seconds = max(artist_listen_seconds.values(), default=1.0)
    artist_nodes = []
    for artist in artist_names:
        play_count = int(artist_play_counts[artist])
        listen_seconds = float(artist_listen_seconds[artist])
        timestamps = sorted(artist_timestamps[artist])
        first_listen = timestamps[0] if timestamps else None
        last_listen = timestamps[-1] if timestamps else None
        active_span_days = (
            round((last_listen - first_listen) / 86400, 1)
            if first_listen is not None and last_listen is not None
            else 0.0
        )
        day_counts = Counter(
            datetime.fromtimestamp(timestamp).date().isoformat()
            for timestamp in timestamps
        )
        peak_day, peak_day_count = (None, 0)
        if day_counts:
            peak_day, peak_day_count = sorted(
                day_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[0]
        gaps = [
            (right - left) / 86400
            for left, right in zip(timestamps, timestamps[1:])
            if right >= left
        ]
        average_gap_days = round(sum(gaps) / len(gaps), 2) if gaps else None
        longest_gap_days = round(max(gaps), 2) if gaps else None
        recent_cutoff = generated_timestamp - 30 * 86400
        recent_30_play_count = sum(1 for timestamp in timestamps if timestamp >= recent_cutoff)
        song_count = len(artist_song_sets[artist])
        plays_per_song = round(play_count / song_count, 2) if song_count else 0.0
        top_song, top_song_plays = (None, 0)
        if artist_song_play_counts[artist]:
            top_song, top_song_plays = sorted(
                artist_song_play_counts[artist].items(),
                key=lambda item: (-item[1], item[0].casefold()),
            )[0]
        top_song_share = round(top_song_plays / play_count, 4) if play_count else 0.0
        play_size = scale_node_size(play_count, max_artist_plays)
        listen_size = scale_node_size(listen_seconds, max_artist_listen_seconds)
        artist_nodes.append(
            {
                "id": artist,
                "label": artist,
                "playCount": play_count,
                "songCount": song_count,
                "historyCount": int(artist_history_counts[artist]),
                "listenTimeSeconds": round(listen_seconds, 2),
                "playsPerSong": plays_per_song,
                "topSong": top_song,
                "topSongPlays": int(top_song_plays),
                "topSongShare": top_song_share,
                "firstListen": first_listen,
                "lastListen": last_listen,
                "activeSpanDays": active_span_days,
                "activeDayCount": len(day_counts),
                "peakDay": peak_day,
                "peakDayCount": int(peak_day_count),
                "recent30PlayCount": int(recent_30_play_count),
                "averageGapDays": average_gap_days,
                "longestGapDays": longest_gap_days,
                "playSize": round(play_size, 2),
                "listenTimeSize": round(listen_size, 2),
                "songs": sorted(artist_song_sets[artist], key=str.casefold),
                "songDetails": sorted(
                    artist_song_details[artist].values(),
                    key=lambda row: (-int(row["playCount"]), row["name"].casefold()),
                ),
                "context": {
                    "before": [
                        {"name": name, "count": int(count)}
                        for name, count in artist_before_context[artist].most_common(12)
                    ],
                    "after": [
                        {"name": name, "count": int(count)}
                        for name, count in artist_after_context[artist].most_common(12)
                    ],
                },
                "history": encode_timestamps(timestamps),
                "title": (
                    f"<b>{artist}</b><br>"
                    f"Plays: {play_count}<br>"
                    f"Songs: {song_count}<br>"
                    f"Listen time: {listen_seconds / 3600:.2f} hours"
                ),
            }
        )

    artist_edge_counts = Counter()
    artist_edge_gaps = defaultdict(list)
    event_artists = [
        (timestamp, song_name, list(song_artists.get(song_name, []) or []))
        for timestamp, song_name in events
    ]
    for index, (_timestamp, _song_name, artists) in enumerate(event_artists):
        if not artists:
            continue
        before_song = event_artists[index - 1][1] if index > 0 else None
        after_song = event_artists[index + 1][1] if index + 1 < len(event_artists) else None
        for artist in artists:
            if before_song:
                artist_before_context[artist][before_song] += 1
            if after_song:
                artist_after_context[artist][after_song] += 1
    duration_by_song = {
        song_name: max(0.0, float(durations.get(song_name, 0.0) or 0.0))
        for song_name in song_counts
    }

    if pairing_mode == "consecutive":
        event_pairs = zip(event_artists, event_artists[1:])
        for current_event, next_event in event_pairs:
            current_ts, current_song, current_artists = current_event
            next_ts, next_song, next_artists = next_event
            if not current_artists or not next_artists:
                continue
            gap_seconds = next_ts - current_ts
            max_gap = int(round(window_minutes * 60)) + duration_by_song.get(current_song, 0.0)
            if gap_seconds < 0 or gap_seconds > max_gap:
                continue
            for source_artist in current_artists:
                for target_artist in next_artists:
                    if not include_self_loops and source_artist == target_artist:
                        continue
                    artist_edge_counts[(source_artist, target_artist)] += 1
                    artist_edge_gaps[(source_artist, target_artist)].append(gap_seconds)
    else:
        window_seconds = int(round(window_minutes * 60))
        for start_index, (current_ts, current_song, current_artists) in enumerate(event_artists):
            if not current_artists:
                continue
            max_gap = window_seconds + duration_by_song.get(current_song, 0.0)
            for next_ts, _next_song, next_artists in event_artists[start_index + 1:]:
                gap_seconds = next_ts - current_ts
                if gap_seconds > max_gap:
                    break
                if not next_artists or gap_seconds < 0:
                    continue
                for source_artist in current_artists:
                    for target_artist in next_artists:
                        if not include_self_loops and source_artist == target_artist:
                            continue
                        artist_edge_counts[(source_artist, target_artist)] += 1
                        artist_edge_gaps[(source_artist, target_artist)].append(gap_seconds)

    outgoing_counts = Counter()
    for (source_artist, _target_artist), count in artist_edge_counts.items():
        outgoing_counts[source_artist] += count
    max_artist_edge_count = max(artist_edge_counts.values(), default=1)
    artist_edges = []
    for (source_artist, target_artist), count in artist_edge_counts.items():
        probability = count / outgoing_counts[source_artist] if outgoing_counts[source_artist] else 0.0
        gaps = artist_edge_gaps[(source_artist, target_artist)]
        average_gap = sum(gaps) / len(gaps) if gaps else 0.0
        artist_edges.append(
            {
                "id": f"{source_artist}→{target_artist}",
                "from": source_artist,
                "to": target_artist,
                "count": int(count),
                "probability": round(probability, 6),
                "averageGapSeconds": round(average_gap, 2),
                "value": int(count),
                "width": round(1.0 + 10.0 * probability + 3.0 * math.sqrt(count / max_artist_edge_count), 2),
                "label": f"{probability * 100:.1f}%",
                "title": (
                    f"<b>{source_artist} → {target_artist}</b><br>"
                    f"Transition count: {count}<br>"
                    f"Observed probability: {probability * 100:.2f}%<br>"
                    f"Average gap: {average_gap / 60:.2f} minutes"
                ),
            }
        )
    artist_edges.sort(key=lambda edge: (-edge["count"], edge["from"], edge["to"]))

    max_song_plays = max((row["playCount"] for row in song_artist_rows), default=1)
    song_artist_nodes = []
    for node in artist_nodes:
        song_artist_nodes.append(
            {
                "id": f"artist:{node['id']}",
                "label": node["label"],
                "kind": "artist",
                "playCount": node["playCount"],
                "songCount": node["songCount"],
                "historyCount": node["historyCount"],
                "listenTimeSeconds": node["listenTimeSeconds"],
                "playsPerSong": node["playsPerSong"],
                "topSong": node["topSong"],
                "topSongPlays": node["topSongPlays"],
                "topSongShare": node["topSongShare"],
                "firstListen": node["firstListen"],
                "lastListen": node["lastListen"],
                "activeSpanDays": node["activeSpanDays"],
                "activeDayCount": node["activeDayCount"],
                "peakDay": node["peakDay"],
                "peakDayCount": node["peakDayCount"],
                "recent30PlayCount": node["recent30PlayCount"],
                "averageGapDays": node["averageGapDays"],
                "longestGapDays": node["longestGapDays"],
                "songs": node["songs"],
                "songDetails": node["songDetails"],
                "context": node["context"],
                "size": node["playSize"],
                "title": node["title"],
            }
        )
    for row in song_artist_rows:
        display_label = row["song"].split("/", 1)[1] if "/" in row["song"] else row["song"]
        song_artist_nodes.append(
            {
                "id": f"song:{row['song']}",
                "label": display_label,
                "kind": "song",
                "playCount": row["playCount"],
                "historyCount": row["historyCount"],
                "durationSeconds": row["durationSeconds"],
                "size": round(scale_node_size(row["playCount"], max_song_plays) * 0.62, 2),
                "title": (
                    f"<b>{display_label}</b><br>"
                    f"Artists: {', '.join(row['artists'])}<br>"
                    f"Plays: {row['playCount']}"
                ),
            }
        )
    song_artist_edges = []
    for row in song_artist_rows:
        for artist in row["artists"]:
            song_artist_edges.append(
                {
                    "id": f"song:{row['song']}→artist:{artist}",
                    "from": f"song:{row['song']}",
                    "to": f"artist:{artist}",
                    "count": max(1, row["playCount"]),
                    "value": max(1, row["playCount"]),
                    "width": round(1.0 + 4.0 * math.sqrt(max(1, row["playCount"]) / max_song_plays), 2),
                }
            )

    all_artist_timestamps = [
        timestamp for timestamps in artist_timestamps.values() for timestamp in timestamps
    ]
    all_artist_timestamps.sort()
    artist_trending_rows = []
    for node in artist_nodes:
        artist_trending_rows.append(
            {
                "name": node["label"],
                "playCount": node["playCount"],
                "historyCount": node["historyCount"],
                "songCount": node["songCount"],
                "listenTimeSeconds": node["listenTimeSeconds"],
                "history": node["history"],
            }
        )

    top_artists = sorted(
        artist_nodes,
        key=lambda node: (-node["playCount"], -node["listenTimeSeconds"], node["label"].casefold()),
    )[:25]
    eligible_nodes = [node for node in artist_nodes if node["playCount"] > 0]

    def highlight(label, node, value, detail):
        if not node:
            return None
        return {
            "label": label,
            "artist": node["label"],
            "value": value,
            "detail": detail,
        }

    most_played = max(eligible_nodes, key=lambda node: node["playCount"], default=None)
    widest_catalog = max(eligible_nodes, key=lambda node: node["songCount"], default=None)
    hottest_recent = max(eligible_nodes, key=lambda node: node["recent30PlayCount"], default=None)
    most_focused = max(
        [node for node in eligible_nodes if node["playCount"] >= 20],
        key=lambda node: node["topSongShare"],
        default=None,
    )
    longest_era = max(eligible_nodes, key=lambda node: node["activeSpanDays"], default=None)
    longest_gap = max(
        [node for node in eligible_nodes if node["longestGapDays"] is not None],
        key=lambda node: node["longestGapDays"],
        default=None,
    )
    highlights = [
        highlight(
            "Most plays",
            most_played,
            f"{most_played['playCount']} plays" if most_played else "",
            "largest artist by total plays",
        ),
        highlight(
            "Deepest catalog",
            widest_catalog,
            f"{widest_catalog['songCount']} songs" if widest_catalog else "",
            f"{widest_catalog['playCount']} plays" if widest_catalog else "",
        ),
        highlight(
            "Hottest lately",
            hottest_recent,
            f"{hottest_recent['recent30PlayCount']} timestamped plays" if hottest_recent else "",
            "last 30 days",
        ),
        highlight(
            "Most single-song dominated",
            most_focused,
            f"{most_focused['topSongShare'] * 100:.0f}% one song" if most_focused else "",
            most_focused["topSong"] if most_focused else "",
        ),
        highlight(
            "Longest era",
            longest_era,
            f"{longest_era['activeSpanDays']:.0f} days" if longest_era else "",
            "between first and last timestamp",
        ),
        highlight(
            "Longest gap",
            longest_gap,
            f"{longest_gap['longestGapDays']:.0f} days" if longest_gap else "",
            "between timestamped plays",
        ),
    ]
    highlights = [item for item in highlights if item is not None]
    return {
        "summary": {
            "folder": str(folder),
            "artistCount": len(artist_nodes),
            "taggedSongCount": tagged_song_count,
            "unassignedSongCount": max(0, len(song_counts) - tagged_song_count),
            "taggedPlayCount": tagged_play_count,
            "unassignedPlayCount": unassigned_play_count,
            "artistEdgeCount": len(artist_edges),
            "songArtistEdgeCount": len(song_artist_edges),
            "historyEventCount": len(all_artist_timestamps),
            "timelinePointCount": len(set(all_artist_timestamps)),
            "firstTimestamp": all_artist_timestamps[0] if all_artist_timestamps else None,
            "lastTimestamp": all_artist_timestamps[-1] if all_artist_timestamps else None,
            "generatedTimestamp": generated_timestamp,
            "generatedFrom": generated_from,
            "trendingHalfLifeSeconds": TRENDING_HALF_LIFE_SECONDS,
        },
        "topArtists": top_artists,
        "highlights": highlights,
        "artistGraph": {"nodes": artist_nodes, "edges": artist_edges},
        "songArtistGraph": {"nodes": song_artist_nodes, "edges": song_artist_edges},
        "trending": {"artists": artist_trending_rows},
    }


def build_graph_payload(
    folder: Path,
    song_counts,
    history_mapping,
    durations,
    events,
    edge_counts,
    edge_gaps,
    window_minutes: float,
    pairing_mode: str,
    fallback_probability: float,
    generated_from: str,
):
    song_names = sorted(song_counts)
    total_song_count = len(song_names)
    fallback_probability = 0.0

    outgoing_counts = Counter()
    incoming_counts = Counter()
    outgoing_neighbors = defaultdict(set)
    incoming_neighbors = defaultdict(set)

    for (source_song, target_song), count in edge_counts.items():
        outgoing_counts[source_song] += count
        incoming_counts[target_song] += count
        outgoing_neighbors[source_song].add(target_song)
        incoming_neighbors[target_song].add(source_song)

    max_play_count = max(song_counts.values(), default=1)
    duration_by_song = {
        song_name: max(0.0, float(durations.get(song_name, 0.0) or 0.0))
        for song_name in song_names
    }
    listen_time_by_song = {
        song_name: max(0, int(song_counts[song_name])) * duration_by_song.get(song_name, 0.0)
        for song_name in song_names
    }
    max_listen_time_seconds = max(listen_time_by_song.values(), default=0.0)
    max_edge_count = max(edge_counts.values(), default=1)

    nodes = []
    for song_name in song_names:
        source_group = None
        display_label = song_name
        if "/" in song_name:
            source_group, display_label = song_name.split("/", 1)

        play_count = song_counts[song_name]
        history_count = len(history_mapping.get(song_name, []))
        observed_neighbor_count = len(outgoing_neighbors.get(song_name, set()) - {song_name})
        missing_neighbor_count = max(0, total_song_count - 1 - observed_neighbor_count)
        observed_outgoing_count = outgoing_counts.get(song_name, 0)

        fallback_total_probability = 0.0
        fallback_per_song_probability = 0.0

        duration_seconds = duration_by_song.get(song_name, 0.0)
        listen_time_seconds = listen_time_by_song.get(song_name, 0.0)
        play_node_size = scale_node_size(play_count, max_play_count)
        listen_time_node_size = scale_node_size(
            listen_time_seconds, max_listen_time_seconds
        )
        nodes.append(
            {
                "id": song_name,
                "label": display_label,
                "sourceGroup": source_group,
                "size": round(play_node_size, 2),
                "playSize": round(play_node_size, 2),
                "listenTimeSize": round(listen_time_node_size, 2),
                "playCount": play_count,
                "durationSeconds": round(duration_seconds, 2),
                "listenTimeSeconds": round(listen_time_seconds, 2),
                "historyCount": history_count,
                "outgoingObservedCount": observed_outgoing_count,
                "incomingObservedCount": incoming_counts.get(song_name, 0),
                "outgoingNeighborCount": observed_neighbor_count,
                "incomingNeighborCount": len(incoming_neighbors.get(song_name, set()) - {song_name}),
                "missingNeighborCount": missing_neighbor_count,
                "fallbackTotalProbability": fallback_total_probability,
                "fallbackPerSongProbability": fallback_per_song_probability,
                "title": (
                    f"<b>{display_label}</b><br>"
                    f"{f'Folder: {source_group}<br>' if source_group else ''}"
                    f"Plays: {play_count}<br>"
                    f"Duration: {duration_seconds / 60:.2f} minutes<br>"
                    f"Total listen time: {listen_time_seconds / 3600:.2f} hours<br>"
                    f"History entries: {history_count}<br>"
                    f"Observed outgoing transitions: {observed_outgoing_count}<br>"
                    f"Observed incoming transitions: {incoming_counts.get(song_name, 0)}"
                ),
            }
        )

    edges = []
    for (source_song, target_song), count in edge_counts.items():
        observed_outgoing_count = outgoing_counts[source_song]
        probability = (count / observed_outgoing_count) if observed_outgoing_count > 0 else 0.0
        average_gap_seconds = sum(edge_gaps[(source_song, target_song)]) / len(edge_gaps[(source_song, target_song)])
        edge_width = 1.0 + 12.0 * probability + 3.0 * math.sqrt(count / max_edge_count)

        edges.append(
            {
                "id": f"{source_song}→{target_song}",
                "from": source_song,
                "to": target_song,
                "count": count,
                "probability": round(probability, 6),
                "averageGapSeconds": round(average_gap_seconds, 2),
                "value": count,
                "width": round(edge_width, 2),
                "label": f"{probability * 100:.1f}%",
                "title": (
                    f"<b>{source_song} → {target_song}</b><br>"
                    f"Transition count: {count}<br>"
                    f"Observed probability mass: {probability * 100:.2f}%<br>"
                    f"Average gap: {average_gap_seconds / 60:.2f} minutes"
                ),
            }
        )

    edges.sort(key=lambda edge: (-edge["count"], edge["from"], edge["to"]))

    payload = {
        "summary": {
            "folder": str(folder),
            "songCount": total_song_count,
            "eventCount": len(events),
            "edgeCount": len(edges),
            "windowMinutes": window_minutes,
            "pairingMode": pairing_mode,
            "generatedFrom": generated_from,
            "maxEdgeCount": max_edge_count,
        },
        "nodes": nodes,
        "edges": edges,
    }
    return payload


def build_graph_payload_from_options(
    args: argparse.Namespace,
    repo_root: Path,
    folder: Path,
    use_all_folders: bool,
    fallback_probability: float,
    view_key: str | None = None,
    view_label: str | None = None,
):
    if use_all_folders:
        song_counts, history_mapping, durations, song_artists, meta_path = load_all_folder_data(repo_root)
    else:
        song_counts, history_mapping, durations, song_artists, meta_path = load_folder_data(folder)

    events = build_global_events(history_mapping)
    window_seconds = int(round(args.window_minutes * 60))
    edge_counts, edge_gaps = build_transition_counts(
        events=events,
        window_seconds=window_seconds,
        pairing_mode=args.pairing,
        include_self_loops=args.include_self_loops,
        durations=durations,
    )

    graph_payload = build_graph_payload(
        folder=folder,
        song_counts=song_counts,
        history_mapping=history_mapping,
        durations=durations,
        events=events,
        edge_counts=edge_counts,
        edge_gaps=edge_gaps,
        window_minutes=args.window_minutes,
        pairing_mode=args.pairing,
        fallback_probability=fallback_probability,
        generated_from=str(meta_path),
    )
    if view_key is not None:
        graph_payload["summary"]["viewKey"] = view_key
    if view_label is not None:
        graph_payload["summary"]["viewLabel"] = view_label
    return graph_payload, meta_path


def build_combined_payload_from_options(
    args: argparse.Namespace,
    repo_root: Path,
    folder: Path,
    use_all_folders: bool,
    fallback_probability: float,
    view_key: str | None = None,
    view_label: str | None = None,
):
    if use_all_folders:
        song_counts, history_mapping, durations, song_artists, meta_path = load_all_folder_data(repo_root)
    else:
        song_counts, history_mapping, durations, song_artists, meta_path = load_folder_data(folder)

    events = build_global_events(history_mapping)
    window_seconds = int(round(args.window_minutes * 60))
    edge_counts, edge_gaps = build_transition_counts(
        events=events,
        window_seconds=window_seconds,
        pairing_mode=args.pairing,
        include_self_loops=args.include_self_loops,
        durations=durations,
    )

    graph_payload = build_graph_payload(
        folder=folder,
        song_counts=song_counts,
        history_mapping=history_mapping,
        durations=durations,
        events=events,
        edge_counts=edge_counts,
        edge_gaps=edge_gaps,
        window_minutes=args.window_minutes,
        pairing_mode=args.pairing,
        fallback_probability=fallback_probability,
        generated_from=str(meta_path),
    )
    trending_payload = build_trending_payload(
        folder=folder,
        song_counts=song_counts,
        history_mapping=history_mapping,
        durations=durations,
        events=events,
        generated_from=str(meta_path),
    )
    artist_payload = build_artist_payload(
        folder=folder,
        song_counts=song_counts,
        history_mapping=history_mapping,
        durations=durations,
        song_artists=song_artists,
        events=events,
        window_minutes=args.window_minutes,
        pairing_mode=args.pairing,
        include_self_loops=args.include_self_loops,
        generated_from=str(meta_path),
    )
    if view_key is not None:
        graph_payload["summary"]["viewKey"] = view_key
        trending_payload["summary"]["viewKey"] = view_key
        artist_payload["summary"]["viewKey"] = view_key
    if view_label is not None:
        graph_payload["summary"]["viewLabel"] = view_label
        trending_payload["summary"]["viewLabel"] = view_label
        artist_payload["summary"]["viewLabel"] = view_label
    return {
        "graph": graph_payload,
        "trending": trending_payload,
        "artists": artist_payload,
    }, meta_path


def graph_view_definitions():
    return [
        {
            "key": key,
            "label": label,
        }
        for key, label, _path, _use_all_folders in GRAPH_VIEW_OPTIONS
    ]


def graph_view_keys():
    return set(GRAPH_VIEW_KEYS)


def graph_view_for_key(repo_root: Path, view_key: str):
    for key, label, path, use_all_folders in GRAPH_VIEW_OPTIONS:
        if key == view_key:
            folder = path if use_all_folders else (repo_root / path)
            return {
                "key": key,
                "label": label,
                "folder": folder,
                "use_all_folders": use_all_folders,
            }
    raise KeyError(view_key)


def resolve_cli_folder(repo_root: Path, folder_arg: str) -> tuple[Path, bool]:
    view_definition = graph_view_for_key(repo_root, folder_arg)
    folder = view_definition["folder"]
    if view_definition["use_all_folders"]:
        return folder, True
    return folder.resolve(), False


def initial_graph_view_key(folder: Path, use_all_folders: bool) -> str:
    if use_all_folders:
        return "all"
    if folder.name == "mid-mp3s":
        return "mid-mp3s"
    if folder.name == "mp3":
        return "mp3"
    return "all"


def hash_graph_payload(graph_payload) -> str:
    encoded_payload = json.dumps(
        graph_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded_payload).hexdigest()


def source_paths_for_folder(repo_root: Path, folder: Path, use_all_folders: bool):
    if use_all_folders:
        source_paths = []
        for relative_folder in ALL_FOLDERS:
            source_folder = repo_root / relative_folder
            source_paths.extend(
                [
                    source_folder,
                    repo_root / listen_counts_filename_for_folder(source_folder),
                    repo_root / listen_timestamps_filename_for_folder(source_folder),
                    repo_root / artists_filename_for_folder(source_folder),
                    source_folder / ".loudness_cache.json",
                ]
            )
        return source_paths

    meta_path = folder / META_FILENAME
    if meta_path.is_file():
        return [meta_path]

    return [
        folder,
        repo_root / listen_counts_filename_for_folder(folder),
        repo_root / listen_timestamps_filename_for_folder(folder),
        repo_root / artists_filename_for_folder(folder),
        folder / ".loudness_cache.json",
    ]


def source_signature_for_paths(source_paths):
    signature = []
    for source_path in source_paths:
        try:
            stat_result = source_path.stat()
            signature.append(
                (
                    str(source_path),
                    stat_result.st_mtime_ns,
                    stat_result.st_size,
                )
            )
        except OSError:
            signature.append((str(source_path), None, None))
    return tuple(signature)


def build_html(graph_payload, trending_payload=None, artist_payload=None, live_config=None):
    graph_json = json.dumps(graph_payload, ensure_ascii=False)
    trending_json = json.dumps(
        trending_payload
        or {
            "summary": {
                "folder": graph_payload.get("summary", {}).get("folder", ""),
                "songCount": 0,
                "historyEventCount": 0,
                "timelinePointCount": 0,
                "trendingHalfLifeSeconds": TRENDING_HALF_LIFE_SECONDS,
                "trendingBaselineDurationSeconds": TRENDING_BASELINE_DURATION_SECONDS,
            },
            "timeline": [],
            "songs": [],
        },
        ensure_ascii=False,
    )
    artist_json = json.dumps(
        artist_payload
        or {
            "summary": {},
            "topArtists": [],
            "artistGraph": {"nodes": [], "edges": []},
            "songArtistGraph": {"nodes": [], "edges": []},
            "trending": {"artists": []},
        },
        ensure_ascii=False,
    )
    live_config_json = json.dumps(live_config or {"enabled": False})
    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Autoplay Graph</title>
  <script src=\"{VIS_NETWORK_CDN}\"></script>
  <style>
    :root {{
      color-scheme: light;
      --bg: #edf4fb;
      --panel: #f8fbff;
      --panel-2: #ffffff;
      --border: #c8d8e8;
      --text: #142130;
      --muted: #5d7288;
      --accent: #2f80ed;
      --accent-2: #eb5757;
      --trend-accent: #64748b;
      --trend-accent-2: #8aa6a0;
      --trend-track: #e8eef5;
      --trend-track-border: #cbd7e5;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{
      margin: 0;
      height: 100%;
    }}
    body {{
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      overflow: hidden;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: end;
      padding: 14px 16px;
      border-bottom: 1px solid var(--border);
      background: rgba(248, 251, 255, 0.94);
      z-index: 10;
      box-shadow: 0 6px 20px rgba(20, 33, 48, 0.06);
    }}
    .toolbar label {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      font-size: 13px;
      color: var(--muted);
      min-width: 150px;
    }}
    .toolbar input, .toolbar select, .toolbar button {{
      font: inherit;
      color: var(--text);
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 10px;
    }}
    .graph-tabs {{
      display: none;
      align-items: center;
      gap: 6px;
      padding-right: 4px;
    }}
    .graph-tabs.visible {{ display: flex; }}
    .graph-tabs button {{
      min-width: 76px;
      cursor: pointer;
    }}
    .graph-tabs button.active {{
      background: var(--accent);
      border-color: var(--accent);
      color: #ffffff;
    }}
    .app-tabs {{
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 4px;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: rgba(255, 255, 255, 0.72);
    }}
    .app-tabs button {{
      min-width: 92px;
      cursor: pointer;
    }}
    .app-tabs button.active {{
      background: var(--accent);
      border-color: var(--accent);
      color: #ffffff;
    }}
    body.trending-mode .layout {{
      display: none;
    }}
    body.artist-mode .layout {{
      display: none;
    }}
    body.stats-mode .layout {{
      display: none;
    }}
    body.graph-mode .trending-layout {{
      display: none;
    }}
    body.artist-mode .trending-layout, body.stats-mode .trending-layout {{
      display: none;
    }}
    body.graph-mode .artist-layout, body.trending-mode .artist-layout, body.stats-mode .artist-layout {{
      display: none;
    }}
    body.graph-mode .artist-stats-layout, body.trending-mode .artist-stats-layout, body.artist-mode .artist-stats-layout {{
      display: none;
    }}
    body.trending-mode {{
      overflow: auto;
      padding-bottom: 132px;
    }}
    body.artist-mode {{
      overflow: auto;
    }}
    body.stats-mode {{
      overflow: auto;
    }}
    body.graph-mode .trending-timeline-dock, body.artist-mode .trending-timeline-dock, body.stats-mode .trending-timeline-dock {{
      display: none;
    }}
    .trending-layout {{
      padding: 12px 16px 30px;
      max-width: 1220px;
      width: 100%;
      margin: 0 auto;
      overflow: visible;
      --trending-row-height: 58px;
      --trending-row-gap: 7px;
    }}
    .trending-hero {{
      margin-bottom: 12px;
    }}
    .trending-intro {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
      align-items: center;
      justify-content: space-between;
      padding: 8px 2px;
      color: var(--text);
    }}
    .trending-ranking {{
      background: rgba(255, 255, 255, 0.9);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      box-shadow: 0 8px 26px rgba(20, 33, 48, 0.05);
    }}
    .trending-intro h2, .trending-ranking h3 {{
      margin: 0 0 8px;
    }}
    .trending-intro h2 {{
      margin: 0;
      font-size: 20px;
    }}
    .trending-subtitle {{
      min-width: min(100%, 320px);
      color: var(--muted);
      font-size: 13px;
    }}
    .trending-metrics {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      align-items: center;
      justify-content: flex-end;
    }}
    .trending-card {{
      display: inline-flex;
      gap: 5px;
      align-items: baseline;
      border: 1px solid rgba(200, 216, 232, 0.9);
      border-radius: 999px;
      padding: 4px 8px;
      background: rgba(255, 255, 255, 0.62);
    }}
    .trending-card .label {{
      color: var(--muted);
      font-size: 12px;
    }}
    .trending-card .value {{
      font-size: 12px;
      font-weight: 750;
    }}
    .trending-controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: end;
      margin-bottom: 14px;
    }}
    .trending-controls label {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      min-width: 160px;
      color: var(--muted);
      font-size: 13px;
    }}
    .trending-controls input, .trending-timeline-label select {{
      font: inherit;
      color: var(--text);
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 10px;
    }}
    .trending-ranking {{
      overflow: visible;
      min-height: 440px;
    }}
    .trending-ranking-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
      margin-bottom: 12px;
    }}
    .trending-stage {{
      position: relative;
      min-height: var(--trending-row-height);
      transition: height 420ms cubic-bezier(0.2, 0.8, 0.2, 1);
      isolation: isolate;
    }}
    .trending-empty {{
      padding: 18px;
      border: 1px dashed var(--border);
      border-radius: 10px;
      color: var(--muted);
      text-align: center;
      background: rgba(245, 249, 253, 0.86);
    }}
    .trending-row {{
      --song-trend-a: var(--trend-accent);
      --song-trend-b: var(--trend-accent-2);
      position: absolute;
      left: 0;
      right: 0;
      display: grid;
      grid-template-columns: 58px 34px minmax(0, 1fr) 118px 112px;
      gap: 12px;
      align-items: center;
      height: var(--trending-row-height);
      padding: 9px 16px;
      border-radius: 0;
      background: transparent;
      opacity: 1;
      transform: translateY(0);
      transition:
        transform 520ms cubic-bezier(0.2, 0.8, 0.2, 1),
        opacity 260ms ease,
        background 260ms ease;
      will-change: transform;
    }}
    .trending-row.is-leader {{
      background: linear-gradient(90deg, color-mix(in srgb, var(--song-trend-a) 16%, transparent), rgba(255, 255, 255, 0));
    }}
    .trending-row:hover {{
      background: linear-gradient(90deg, color-mix(in srgb, var(--song-trend-a) 13%, transparent), rgba(255, 255, 255, 0.5));
    }}
    .trending-row.is-exiting {{
      opacity: 0;
    }}
    .trending-rank {{
      color: var(--song-trend-a);
      font-variant-numeric: tabular-nums;
      font-weight: 800;
    }}
    .trending-color {{
      width: 28px;
      height: 28px;
      padding: 0;
      border: 1px solid var(--trend-track-border);
      border-radius: 999px;
      background: transparent;
      cursor: pointer;
      overflow: hidden;
    }}
    .trending-color::-webkit-color-swatch-wrapper {{
      padding: 0;
    }}
    .trending-color::-webkit-color-swatch {{
      border: 0;
      border-radius: 999px;
    }}
    .trending-color::-moz-color-swatch {{
      border: 0;
      border-radius: 999px;
    }}
    .trending-track {{
      display: grid;
      gap: 7px;
      min-width: 0;
    }}
    .trending-name {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-weight: 720;
    }}
    .trending-bar {{
      height: 9px;
      border-radius: 999px;
      overflow: hidden;
      border: 1px solid var(--trend-track-border);
      background: var(--trend-track);
    }}
    .trending-fill {{
      width: 0%;
      height: 100%;
      background: linear-gradient(90deg, var(--song-trend-a), var(--song-trend-b));
      border-radius: inherit;
      transition: width 520ms cubic-bezier(0.2, 0.8, 0.2, 1);
    }}
    .trending-score, .trending-plays {{
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    .trending-score strong, .trending-plays strong {{
      display: block;
    }}
    .trending-score span, .trending-plays span {{
      display: block;
      margin-top: 3px;
      color: var(--muted);
      font-size: 12px;
    }}
    .trending-timeline-dock {{
      position: fixed;
      left: 0;
      right: 0;
      bottom: 0;
      z-index: 3000;
      display: grid;
      gap: 10px;
      padding: 14px 16px 16px;
      border-top: 1px solid var(--border);
      background: rgba(248, 251, 255, 0.94);
      backdrop-filter: blur(12px);
      box-shadow: 0 -10px 30px rgba(20, 33, 48, 0.08);
    }}
    .trending-timeline-top {{
      display: grid;
      grid-template-columns: auto minmax(0, 1fr) auto;
      gap: 12px;
      align-items: center;
    }}
    .trending-timeline-meta {{
      display: grid;
      gap: 2px;
      min-width: 0;
    }}
    .trending-timeline-date {{
      font-weight: 750;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .trending-timeline-count {{
      color: var(--muted);
      font-size: 12px;
    }}
    .trending-play-button {{
      width: 42px;
      height: 42px;
      border: 1px solid rgba(100, 116, 139, 0.45);
      border-radius: 999px;
      color: #334155;
      background: rgba(100, 116, 139, 0.1);
      cursor: pointer;
      font-weight: 800;
    }}
    .trending-play-button:hover {{
      background: rgba(100, 116, 139, 0.16);
    }}
    .trending-timeline-label {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      min-width: 150px;
      font-size: 13px;
      color: var(--muted);
    }}
    #trending-timeline {{
      width: 100%;
      accent-color: var(--trend-accent);
    }}
    .toolbar input[type=range] {{ padding: 0; }}
    .toolbar .checkbox {{
      min-width: unset;
      flex-direction: row;
      align-items: center;
      gap: 8px;
      padding-top: 22px;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      min-height: 0;
      height: 100%;
      overflow: hidden;
    }}
    .network-panel {{
      position: relative;
      min-width: 0;
      min-height: 0;
      overflow: hidden;
      background:
        radial-gradient(circle at top, rgba(47, 128, 237, 0.08), transparent 35%),
        linear-gradient(180deg, #f9fcff 0%, #eef5fb 100%);
    }}
    #network {{
      width: 100%;
      height: 100%;
      min-height: 0;
    }}
    .network-status {{
      position: absolute;
      inset: 18px;
      display: none;
      align-items: center;
      justify-content: center;
      text-align: center;
      padding: 24px;
      border: 1px solid var(--border);
      border-radius: 14px;
      background: rgba(248, 251, 255, 0.96);
      color: var(--muted);
      font-size: 15px;
      line-height: 1.5;
    }}
    .network-status.visible {{ display: flex; }}
    .artist-layout {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      min-height: 0;
      height: 100%;
      overflow: hidden;
    }}
    .artist-main {{
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      min-height: 0;
      overflow: hidden;
    }}
    .artist-header {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 14px;
      align-items: center;
      justify-content: space-between;
      padding: 14px 16px;
      border-bottom: 1px solid var(--border);
      background: rgba(248, 251, 255, 0.88);
    }}
    .artist-header h2 {{ margin: 0; font-size: 20px; }}
    .artist-controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }}
    .artist-controls button {{
      cursor: pointer;
    }}
    .artist-controls button.active {{
      color: #ffffff;
      background: var(--accent);
      border-color: var(--accent);
    }}
    .artist-network-panel {{
      position: relative;
      min-height: 0;
      background:
        radial-gradient(circle at top left, rgba(235, 87, 87, 0.08), transparent 32%),
        linear-gradient(180deg, #fbfdff 0%, #eef5fb 100%);
    }}
    #artist-network {{
      width: 100%;
      height: 100%;
      min-height: 480px;
    }}
    .artist-sidebar {{
      border-left: 1px solid var(--border);
      background: rgba(248, 251, 255, 0.94);
      padding: 18px;
      overflow-y: auto;
      min-height: 0;
    }}
    .artist-bars {{
      display: grid;
      gap: 8px;
      margin-top: 10px;
    }}
    .artist-bar-row {{
      display: grid;
      grid-template-columns: 34px minmax(0, 1fr) 64px;
      gap: 8px;
      align-items: center;
      font-size: 13px;
    }}
    .artist-bar-name {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-weight: 700;
    }}
    .artist-bar-track {{
      grid-column: 2 / 4;
      height: 8px;
      overflow: hidden;
      border-radius: 999px;
      background: var(--trend-track);
      border: 1px solid var(--trend-track-border);
    }}
    .artist-bar-fill {{
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #2f80ed, #eb5757);
      border-radius: inherit;
    }}
    .artist-timeline {{
      width: 100%;
      accent-color: var(--accent);
    }}
    .artist-stats-layout {{
      padding: 14px 16px 32px;
      max-width: 1320px;
      width: 100%;
      margin: 0 auto;
      overflow: visible;
    }}
    .stats-header {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: end;
      gap: 12px;
      margin-bottom: 14px;
    }}
    .stats-header h2 {{
      margin: 0;
      font-size: 22px;
    }}
    .stats-controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }}
    .stats-controls button, .stats-controls select {{
      font: inherit;
      color: var(--text);
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 10px;
    }}
    .stats-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(320px, 0.65fr);
      gap: 14px;
      align-items: start;
    }}
    .stats-section-heading {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin: 18px 2px 10px;
    }}
    .stats-section-heading h3 {{
      margin: 0;
      font-size: 16px;
    }}
    .stats-wide {{
      grid-column: 1 / -1;
    }}
    .moving-stage {{
      position: relative;
      width: 100%;
      min-height: 360px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: linear-gradient(180deg, #fbfdff 0%, #eef5fb 100%);
      overflow: hidden;
    }}
    .moving-point {{
      position: absolute;
      display: grid;
      place-items: center;
      min-width: 34px;
      height: 34px;
      padding: 0 9px;
      border-radius: 999px;
      border: 2px solid rgba(255, 255, 255, 0.95);
      color: #142130;
      font-size: 11px;
      font-weight: 800;
      box-shadow: 0 8px 20px rgba(20, 33, 48, 0.14);
      transform: translate(-50%, -50%);
      transition:
        left 520ms cubic-bezier(0.2, 0.8, 0.2, 1),
        top 520ms cubic-bezier(0.2, 0.8, 0.2, 1),
        width 420ms ease,
        height 420ms ease,
        opacity 240ms ease;
      cursor: pointer;
    }}
    .moving-label {{
      position: absolute;
      max-width: 150px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 11px;
      font-weight: 750;
      color: #142130;
      transform: translate(12px, -50%);
      transition: left 520ms cubic-bezier(0.2, 0.8, 0.2, 1), top 520ms cubic-bezier(0.2, 0.8, 0.2, 1), opacity 240ms ease;
    }}
    .stats-lane {{
      position: relative;
      height: 38px;
      border-bottom: 1px solid rgba(200, 216, 232, 0.65);
    }}
    .stats-lane:last-child {{ border-bottom: 0; }}
    .stats-lane-label {{
      position: absolute;
      left: 10px;
      top: 50%;
      width: 138px;
      transform: translateY(-50%);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 12px;
      font-weight: 750;
      z-index: 2;
    }}
    .stats-lane-fill {{
      position: absolute;
      left: 158px;
      top: 9px;
      height: 20px;
      border-radius: 999px;
      transition: width 520ms cubic-bezier(0.2, 0.8, 0.2, 1), opacity 240ms ease;
    }}
    .stats-chip {{
      position: absolute;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      min-width: 150px;
      height: 34px;
      padding: 0 10px;
      border: 1px solid var(--border);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.86);
      box-shadow: 0 8px 18px rgba(20, 33, 48, 0.09);
      transform: translate(-50%, -50%);
      transition: left 520ms cubic-bezier(0.2, 0.8, 0.2, 1), top 520ms cubic-bezier(0.2, 0.8, 0.2, 1), opacity 240ms ease;
      font-size: 12px;
    }}
    .stats-chip strong {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .phase-map {{
      display: grid;
      gap: 6px;
      padding: 12px;
    }}
    .phase-row {{
      display: grid;
      grid-template-columns: 138px minmax(0, 1fr);
      gap: 10px;
      align-items: center;
      min-height: 24px;
    }}
    .phase-name {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 12px;
      font-weight: 750;
    }}
    .phase-track {{
      position: relative;
      height: 16px;
      border-radius: 999px;
      background: rgba(200, 216, 232, 0.45);
      overflow: hidden;
    }}
    .phase-trail {{
      position: absolute;
      inset: 0 auto 0 0;
      border-radius: inherit;
      opacity: 0.38;
      transition: width 520ms cubic-bezier(0.2, 0.8, 0.2, 1);
    }}
    .phase-spark {{
      position: absolute;
      top: 50%;
      width: 12px;
      height: 12px;
      border-radius: 999px;
      border: 2px solid #ffffff;
      transform: translate(-50%, -50%);
      box-shadow: 0 4px 12px rgba(20, 33, 48, 0.16);
      transition: left 520ms cubic-bezier(0.2, 0.8, 0.2, 1), opacity 240ms ease;
    }}
    .river-band {{
      position: absolute;
      left: 8%;
      height: 24px;
      border-radius: 999px;
      box-shadow: 0 8px 18px rgba(20, 33, 48, 0.1);
      transition: top 520ms cubic-bezier(0.2, 0.8, 0.2, 1), width 520ms cubic-bezier(0.2, 0.8, 0.2, 1), opacity 240ms ease;
    }}
    .river-label {{
      position: absolute;
      left: 12px;
      top: 50%;
      max-width: 190px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      transform: translateY(-50%);
      font-size: 12px;
      font-weight: 800;
      color: #142130;
    }}
    .weather-cloud {{
      position: absolute;
      display: grid;
      place-items: center;
      min-width: 58px;
      height: 36px;
      padding: 0 13px;
      border: 1px solid rgba(200, 216, 232, 0.95);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.9);
      box-shadow: 0 10px 24px rgba(20, 33, 48, 0.12);
      transform: translate(-50%, -50%);
      transition: left 620ms cubic-bezier(0.2, 0.8, 0.2, 1), top 620ms cubic-bezier(0.2, 0.8, 0.2, 1), width 520ms ease, opacity 240ms ease;
      font-size: 12px;
      font-weight: 800;
      cursor: pointer;
    }}
    .event-burst {{
      position: absolute;
      min-width: 128px;
      height: 34px;
      padding: 0 10px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: rgba(255, 255, 255, 0.9);
      box-shadow: 0 8px 18px rgba(20, 33, 48, 0.09);
      transform: translate(-50%, -50%);
      transition: left 520ms cubic-bezier(0.2, 0.8, 0.2, 1), top 520ms cubic-bezier(0.2, 0.8, 0.2, 1), opacity 240ms ease;
      font-size: 12px;
      cursor: pointer;
    }}
    .event-burst strong {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .artist-overview-grid {{
      display: grid;
      grid-template-columns: minmax(0, 0.75fr) minmax(0, 1.25fr);
      gap: 14px;
      align-items: start;
    }}
    .artist-inspector-controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: end;
      margin-bottom: 12px;
    }}
    .artist-inspector-controls label {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      min-width: min(100%, 320px);
      color: var(--muted);
      font-size: 13px;
    }}
    .artist-inspector-controls select {{
      font: inherit;
      color: var(--text);
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 10px;
    }}
    .artist-inspector-title {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: baseline;
      justify-content: space-between;
      margin-bottom: 10px;
    }}
    .artist-inspector-title strong {{
      font-size: 18px;
    }}
    .song-breakdown-row {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 72px;
      gap: 8px;
      align-items: center;
      font-size: 13px;
    }}
    .song-breakdown-name {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-weight: 750;
    }}
    .song-breakdown-meta {{
      text-align: right;
      color: var(--muted);
      font-size: 12px;
    }}
    .song-breakdown-track {{
      grid-column: 1 / -1;
      height: 9px;
      overflow: hidden;
      border-radius: 999px;
      background: var(--trend-track);
      border: 1px solid var(--trend-track-border);
    }}
    .song-breakdown-fill {{
      height: 100%;
      width: 0%;
      border-radius: inherit;
      background: linear-gradient(90deg, #56cc9d, #f2994a);
      transition: width 520ms cubic-bezier(0.2, 0.8, 0.2, 1);
    }}
    .artist-visual-grid {{
      display: grid;
      grid-template-columns: minmax(0, 0.85fr) minmax(0, 1.15fr);
      gap: 14px;
      align-items: start;
    }}
    .mini-stage {{
      position: relative;
      min-height: 230px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: linear-gradient(180deg, #fbfdff 0%, #eef5fb 100%);
      overflow: hidden;
    }}
    .hour-bars {{
      display: grid;
      grid-template-columns: repeat(24, minmax(4px, 1fr));
      align-items: end;
      gap: 3px;
      height: 150px;
      padding: 42px 12px 24px;
    }}
    .hour-bar {{
      min-height: 3px;
      border-radius: 999px 999px 0 0;
      background: linear-gradient(180deg, #2f80ed, #56cc9d);
      transition: height 520ms cubic-bezier(0.2, 0.8, 0.2, 1), opacity 240ms ease;
    }}
    #artist-coverage-stage {{
      min-height: 320px;
      max-height: 520px;
      overflow: auto;
    }}
    .coverage-map {{
      display: grid;
      grid-template-columns: minmax(118px, 1.15fr) minmax(260px, 3fr) 84px 76px;
      gap: 7px 10px;
      align-items: center;
      min-width: 660px;
      padding: 12px;
      font-size: 12px;
    }}
    .coverage-map-head {{
      color: var(--muted);
      font-size: 11px;
      font-weight: 750;
      text-transform: uppercase;
    }}
    .coverage-name {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-weight: 750;
    }}
    .coverage-cells {{
      display: grid;
      gap: 2px;
      height: 22px;
      align-items: stretch;
    }}
    .coverage-cell {{
      min-width: 5px;
      border-radius: 3px;
      background: #e7eef4;
      border: 1px solid rgba(38, 56, 74, 0.06);
      transition: transform 160ms ease, opacity 180ms ease;
    }}
    .coverage-cell.is-active {{
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.25);
    }}
    .coverage-cell.is-missing {{
      background: repeating-linear-gradient(
        135deg,
        #eef2f6,
        #eef2f6 5px,
        #dbe4ec 5px,
        #dbe4ec 10px
      );
      opacity: 0.85;
    }}
    .coverage-cell:hover {{
      transform: scaleY(1.55);
    }}
    .coverage-axis {{
      grid-column: 2 / 3;
      display: flex;
      justify-content: space-between;
      color: var(--muted);
      font-size: 11px;
      overflow: hidden;
      white-space: nowrap;
    }}
    @media (max-width: 920px) {{
      .artist-overview-grid, .artist-visual-grid {{
        grid-template-columns: 1fr;
      }}
    }}
    .stats-timeline {{
      width: min(520px, 100%);
      accent-color: var(--accent);
    }}
    .sidebar {{
      border-left: 1px solid var(--border);
      background: rgba(248, 251, 255, 0.92);
      padding: 18px;
      overflow-y: auto;
      min-height: 0;
    }}
    .card {{
      background: rgba(255, 255, 255, 0.9);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      margin-bottom: 14px;
      box-shadow: 0 8px 26px rgba(20, 33, 48, 0.05);
    }}
    .card h2, .card h3 {{ margin: 0 0 10px; }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .metric {{
      background: rgba(242, 247, 252, 0.95);
      border-radius: 10px;
      padding: 10px;
      border: 1px solid rgba(200, 216, 232, 0.95);
    }}
    .metric .label {{ font-size: 12px; color: var(--muted); }}
    .metric .value {{ font-size: 20px; font-weight: 700; margin-top: 4px; }}
    .list {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-top: 10px;
    }}
    .item {{
      border: 1px solid rgba(200, 216, 232, 0.95);
      border-radius: 10px;
      padding: 10px;
      background: rgba(245, 249, 253, 0.98);
    }}
    .item strong {{ display: block; margin-bottom: 4px; }}
    .muted {{ color: var(--muted); }}
    .pill {{
      display: inline-block;
      margin-right: 6px;
      padding: 3px 8px;
      border-radius: 999px;
      background: rgba(47, 128, 237, 0.12);
      color: #255ea8;
      font-size: 12px;
    }}
    @media (max-width: 1100px) {{
      body {{
        grid-template-rows: auto auto;
        overflow: auto;
      }}
      .layout {{
        grid-template-columns: 1fr;
        height: auto;
        overflow: visible;
      }}
      .network-panel {{
        height: 70vh;
        min-height: 420px;
      }}
      .artist-layout {{
        grid-template-columns: 1fr;
        height: auto;
        overflow: visible;
      }}
      #artist-network {{
        height: 70vh;
      }}
      .artist-sidebar {{ border-left: none; border-top: 1px solid var(--border); }}
      .sidebar {{ border-left: none; border-top: 1px solid var(--border); }}
    }}
    @media (max-width: 680px) {{
      body.trending-mode {{
        padding-bottom: 156px;
      }}
      .trending-row {{
        grid-template-columns: 42px 32px minmax(0, 1fr);
      }}
      .trending-score, .trending-plays {{
        grid-column: 3;
        text-align: left;
      }}
      .trending-name {{
        white-space: normal;
      }}
      .trending-timeline-top {{
        grid-template-columns: 1fr;
      }}
      .trending-timeline-label {{
        min-width: 0;
      }}
    }}
  </style>
</head>
<body class=\"graph-mode\">
  <div class=\"toolbar\">
    <div class=\"app-tabs\" aria-label=\"Dashboard mode\">
      <button id=\"graph-mode-button\" class=\"active\" type=\"button\">Graph</button>
      <button id=\"trending-mode-button\" type=\"button\">Trending</button>
      <button id=\"artist-mode-button\" type=\"button\">Artists</button>
      <button id=\"stats-mode-button\" type=\"button\">Artist Stats</button>
    </div>
    <div id=\"graph-tabs\" class=\"graph-tabs\" aria-label=\"Graph views\"></div>
    <label>
      Search song
      <input id=\"song-search\" type=\"text\" placeholder=\"Type part of a song name\">
    </label>
    <button id=\"focus-button\">Focus</button>
    <label>
      Minimum edge count
      <input id=\"count-filter\" type=\"range\" min=\"1\" max=\"{max(1, graph_payload['summary']['maxEdgeCount'])}\" value=\"1\" step=\"1\">
      <span id=\"count-filter-value\">1</span>
    </label>
    <label>
      Minimum edge probability
      <input id=\"probability-filter\" type=\"range\" min=\"0\" max=\"100\" value=\"0\" step=\"0.5\">
      <span id=\"probability-filter-value\">0.0%</span>
    </label>
    <label class=\"checkbox\">
      <input id=\"labels-toggle\" type=\"checkbox\" checked>
      Show edge labels
    </label>
    <label class=\"checkbox\">
      <input id=\"neighborhood-toggle\" type=\"checkbox\">
      Selected-node neighborhood only
    </label>
    <label class=\"checkbox\">
      <input id=\"listen-time-size-toggle\" type=\"checkbox\">
      Size by listen hours
    </label>
  </div>
  <div class=\"layout\">
    <div class=\"network-panel\">
      <div id=\"network\"></div>
      <div id=\"network-status\" class=\"network-status\"></div>
    </div>
    <aside class=\"sidebar\">
      <section class=\"card\">
        <h2>Autoplay Graph</h2>
        <div class=\"muted\">Built from global play history in <code id=\"summary-folder\"></code>.</div>
        <div id=\"live-status\" class=\"muted\" style=\"display: none; margin-top: 6px;\"></div>
        <div class=\"metric-grid\" style=\"margin-top: 12px;\">
          <div class=\"metric\"><div class=\"label\">Songs</div><div class=\"value\" id=\"summary-song-count\"></div></div>
          <div class=\"metric\"><div class=\"label\">Play Events</div><div class=\"value\" id=\"summary-event-count\"></div></div>
          <div class=\"metric\"><div class=\"label\">Observed Edges</div><div class=\"value\" id=\"summary-edge-count\"></div></div>
          <div class=\"metric\"><div class=\"label\">Window</div><div class=\"value\" id=\"summary-window\"></div></div>
        </div>
        <div style=\"margin-top: 12px;\">
          <span class=\"pill\" id=\"summary-pairing\"></span>
          <span class=\"pill\" id=\"summary-sizing\"></span>
        </div>
      </section>
      <section class=\"card\">
        <h3>Selection</h3>
        <div id=\"selection-summary\" class=\"muted\">Select a node or edge to inspect it.</div>
        <div id=\"selection-content\"></div>
      </section>
    </aside>
  </div>
  <main class=\"trending-layout\">
    <section class=\"trending-hero\">
      <div class=\"trending-intro\">
        <div>
          <h2>Trending MP3s</h2>
          <div class=\"trending-subtitle\" id=\"trending-subtitle\"></div>
        </div>
        <div class=\"trending-metrics\">
          <span class=\"trending-card\"><span class=\"label\">Songs</span><span class=\"value\" id=\"trending-song-count\"></span></span>
          <span class=\"trending-card\"><span class=\"label\">Events</span><span class=\"value\" id=\"trending-event-count\"></span></span>
          <span class=\"trending-card\"><span class=\"label\">Half-life</span><span class=\"value\">2d</span></span>
          <span class=\"trending-card\"><span class=\"label\">Baseline</span><span class=\"value\">3:30</span></span>
        </div>
      </div>
    </section>
    <section class=\"trending-controls\">
      <label>
        Top songs
        <input id=\"trending-top-limit\" type=\"number\" min=\"1\" max=\"200\" step=\"1\" value=\"40\">
      </label>
    </section>
    <section class=\"trending-ranking\">
      <div class=\"trending-ranking-head\">
        <div>
          <h3>Trending Bar Race</h3>
          <div class=\"muted\" id=\"trending-through\"></div>
        </div>
        <div class=\"muted\" id=\"trending-rank-caption\"></div>
      </div>
      <div class=\"trending-empty\" id=\"trending-empty\" hidden>No timestamped plays at this point in the timeline.</div>
      <div class=\"trending-stage\" id=\"trending-stage\"></div>
    </section>
  </main>
  <main class=\"artist-layout\">
    <section class=\"artist-main\">
      <div class=\"artist-header\">
        <div>
          <h2>Artists</h2>
          <div class=\"muted\" id=\"artist-subtitle\"></div>
        </div>
        <div class=\"artist-controls\">
          <button id=\"artist-transition-button\" class=\"active\" type=\"button\">Artist Transitions</button>
          <button id=\"artist-song-button\" type=\"button\">Song Links</button>
          <label>
            Search artist
            <input id=\"artist-search\" type=\"text\" placeholder=\"Type artist name\">
          </label>
          <button id=\"artist-focus-button\" type=\"button\">Focus</button>
        </div>
      </div>
      <div class=\"artist-network-panel\">
        <div id=\"artist-network\"></div>
      </div>
    </section>
    <aside class=\"artist-sidebar\">
      <section class=\"card\">
        <h3>Artist Summary</h3>
        <div class=\"metric-grid\">
          <div class=\"metric\"><div class=\"label\">Artists</div><div class=\"value\" id=\"artist-count\"></div></div>
          <div class=\"metric\"><div class=\"label\">Tagged Songs</div><div class=\"value\" id=\"artist-tagged-songs\"></div></div>
          <div class=\"metric\"><div class=\"label\">Unassigned Songs</div><div class=\"value\" id=\"artist-unassigned-songs\"></div></div>
          <div class=\"metric\"><div class=\"label\">Artist Edges</div><div class=\"value\" id=\"artist-edge-count\"></div></div>
        </div>
      </section>
      <section class=\"card\">
        <h3>Top Artists</h3>
        <div id=\"artist-bars\" class=\"artist-bars\"></div>
      </section>
      <section class=\"card\">
        <h3>Artist Timeline</h3>
        <div class=\"muted\" id=\"artist-timeline-date\"></div>
        <input id=\"artist-timeline\" class=\"artist-timeline\" type=\"range\" min=\"0\" max=\"1000\" value=\"1000\" step=\"1\">
        <div id=\"artist-timeline-bars\" class=\"artist-bars\"></div>
      </section>
      <section class=\"card\">
        <h3>Artist Selection</h3>
        <div id=\"artist-selection-summary\" class=\"muted\">Select an artist, song, or edge.</div>
        <div id=\"artist-selection-content\"></div>
      </section>
    </aside>
  </main>
  <main class=\"artist-stats-layout\">
    <div class=\"stats-header\">
      <div>
        <h2>Artist Stats</h2>
        <div class=\"muted\" id=\"artist-stats-subtitle\"></div>
      </div>
      <div class=\"stats-controls\">
        <button id=\"artist-stats-play\" type=\"button\" title=\"Play artist timeline\">▶</button>
        <input id=\"artist-stats-timeline\" class=\"stats-timeline\" type=\"range\" min=\"0\" max=\"1000\" value=\"1000\" step=\"1\">
        <select id=\"artist-stats-speed\" title=\"Timeline playback speed\">
          <option value=\"0.75\">0.75s/day</option>
          <option value=\"0.5\" selected>0.5s/day</option>
          <option value=\"0.25\">0.25s/day</option>
        </select>
      </div>
    </div>
    <section class=\"card stats-wide\">
      <div class=\"metric-grid\">
        <div class=\"metric\"><div class=\"label\">Artists</div><div class=\"value\" id=\"stats-artist-count\"></div></div>
        <div class=\"metric\"><div class=\"label\">Tagged plays</div><div class=\"value\" id=\"stats-tagged-plays\"></div></div>
        <div class=\"metric\"><div class=\"label\">Unassigned plays</div><div class=\"value\" id=\"stats-unassigned-plays\"></div></div>
        <div class=\"metric\"><div class=\"label\">Timeline events</div><div class=\"value\" id=\"stats-history-events\"></div></div>
      </div>
    </section>
    <div class=\"stats-section-heading\">
      <h3>Top Movement</h3>
      <div class=\"muted\">Animated with the shared timeline controls.</div>
    </div>
    <section class=\"card stats-wide\">
      <h3>Moving Artist Race</h3>
      <div class=\"muted\" id=\"artist-stats-date\"></div>
      <div id=\"artist-stats-bars\" class=\"artist-bars\"></div>
    </section>
    <div class=\"stats-section-heading\">
      <h3>Artist Details</h3>
      <div class=\"muted\">Pick one artist to inspect their songs.</div>
    </div>
    <section class=\"card stats-wide\">
      <h3>Selected Artist</h3>
      <div class=\"artist-inspector-controls\">
        <label>
          Select artist
          <select id=\"artist-inspector-select\"></select>
        </label>
      </div>
      <div id=\"artist-stats-selection\" class=\"muted\">Select an artist.</div>
      <div class=\"artist-overview-grid\">
        <div>
          <div id=\"artist-inspector-metrics\" class=\"metric-grid\" style=\"margin-top: 10px;\"></div>
        </div>
        <div>
          <div class=\"muted\">Song breakdown</div>
          <div id=\"artist-inspector-songs\" class=\"artist-bars\"></div>
        </div>
      </div>
    </section>
    <div class=\"stats-section-heading\">
      <h3>Selected Artist Visuals</h3>
      <div class=\"muted\">Coverage and gravity update from the artist dropdown.</div>
    </div>
    <div class=\"artist-visual-grid\">
      <section class=\"card\">
        <h3>Song Lifespans</h3>
        <div class=\"muted\" id=\"artist-coverage-caption\">When each song was active across your history.</div>
        <div id=\"artist-coverage-stage\" class=\"mini-stage\"></div>
      </section>
      <section class=\"card\">
        <h3>Song Gravity</h3>
        <div class=\"muted\" id=\"artist-song-gravity-caption\">Songs move closer when they are active.</div>
        <div id=\"artist-song-gravity-stage\" class=\"moving-stage\"></div>
      </section>
    </div>
    <div class=\"stats-section-heading\">
      <h3>Library Movement</h3>
      <div class=\"muted\">Uses the shared Artist Stats timeline above.</div>
    </div>
    <div class=\"artist-visual-grid\">
      <section class=\"card\">
        <h3>Artist Takeover Timeline</h3>
        <div class=\"muted\">Current share of timestamped artist momentum.</div>
        <div id=\"artist-takeover-stage\" class=\"moving-stage\"></div>
      </section>
      <section class=\"card\">
        <h3>Artist Orbit Plot</h3>
        <div class=\"muted\" id=\"artist-orbit-caption\">Artists orbit closer when they are active in the current timeline window.</div>
        <div id=\"artist-orbit-stage\" class=\"moving-stage\"></div>
      </section>
    </div>
  </main>
  <section class=\"trending-timeline-dock\">
    <div class=\"trending-timeline-top\">
      <button id=\"trending-play-toggle\" class=\"trending-play-button\" type=\"button\" title=\"Play timeline\">▶</button>
      <div class=\"trending-timeline-meta\">
        <div class=\"trending-timeline-date\" id=\"trending-timeline-date\"></div>
        <div class=\"trending-timeline-count\" id=\"trending-timeline-count\"></div>
      </div>
      <label class=\"trending-timeline-label\">
        Playback speed
        <select id=\"trending-play-speed\">
          <option value=\"0.5\">0.5s/day</option>
          <option value=\"0.38\" selected>0.38s/day</option>
          <option value=\"0.25\">0.25s/day</option>
        </select>
      </label>
    </div>
    <input id=\"trending-timeline\" type=\"range\" min=\"0\" max=\"1000\" step=\"1\" value=\"1000\">
  </section>
  <script>
    let graphData = {graph_json};
    let trendingData = {trending_json};
    let artistData = {artist_json};
    const liveConfig = {live_config_json};
    let activeViewKey = liveConfig.initialViewKey || graphData.summary.viewKey || 'all';
    const graphHashesByView = new Map([[activeViewKey, liveConfig.initialHash || null]]);
    let currentGraphHash = graphHashesByView.get(activeViewKey) || null;
    let allNodes = graphData.nodes;
    let allEdges = graphData.edges;
    let nodeLookup = new Map(allNodes.map((node) => [node.id, node]));
    let edgeLookup = new Map(allEdges.map((edge) => [edge.id, edge]));
    let selectedNodeId = null;
    let trendingSongs = [];
    let trendingFirstTimestamp = 0;
    let trendingLastTimestamp = 0;
    let trendingRows = new Map();
    let trendingIsPlaying = false;
    let trendingPlayAnimationFrame = null;
    let trendingLastPlaybackFrame = null;
    const trendingColorStorageKey = 'audiotagTrendingSongColors:v1';
    let trendingColorOverrides = loadTrendingColorOverrides();

    const container = document.getElementById('network');
    const networkStatus = document.getElementById('network-status');
    const listenTimeSizeToggle = document.getElementById('listen-time-size-toggle');
    const trendingTopLimit = document.getElementById('trending-top-limit');
    const trendingTimeline = document.getElementById('trending-timeline');
    const trendingStage = document.getElementById('trending-stage');
    const trendingEmpty = document.getElementById('trending-empty');
    const trendingPlayToggle = document.getElementById('trending-play-toggle');
    const trendingPlaySpeed = document.getElementById('trending-play-speed');
    const secondsPerDay = 86400;
    const artistContainer = document.getElementById('artist-network');
    const artistTimeline = document.getElementById('artist-timeline');
    const artistStatsTimeline = document.getElementById('artist-stats-timeline');
    const artistStatsPlay = document.getElementById('artist-stats-play');
    const artistStatsSpeed = document.getElementById('artist-stats-speed');
    const artistInspectorSelect = document.getElementById('artist-inspector-select');
    const artistTakeoverStage = document.getElementById('artist-takeover-stage');
    const artistOrbitStage = document.getElementById('artist-orbit-stage');
    const artistPhaseStage = document.getElementById('artist-phase-stage');
    const artistWeatherStage = document.getElementById('artist-weather-stage');
    const artistRiverStage = document.getElementById('artist-river-stage');
    const artistRediscoveryStage = document.getElementById('artist-rediscovery-stage');
    const artistObsessionStage = document.getElementById('artist-obsession-stage');
    const artistSongGravityStage = document.getElementById('artist-song-gravity-stage');
    const artistCoverageStage = document.getElementById('artist-coverage-stage');
    let artistNetwork = null;
    let artistMode = 'transitions';
    let artistSelectedId = null;
    let artistTimelineRows = [];
    let artistFirstTimestamp = 0;
    let artistLastTimestamp = 0;
    let artistStatsIsPlaying = false;
    let artistStatsAnimationFrame = null;
    let artistStatsLastFrame = null;
    let artistWeeklyData = null;
    let selectedStatsArtist = null;

    const artistNetworkOptions = {{
      autoResize: true,
      height: '100%',
      width: '100%',
      nodes: {{
        shape: 'dot',
        borderWidth: 1.5,
        font: {{ color: '#142130', size: 16, face: 'Inter' }},
      }},
      edges: {{
        arrows: {{ to: {{ enabled: true, scaleFactor: 0.55 }} }},
        smooth: {{ type: 'dynamic' }},
        color: {{ color: 'rgba(94, 122, 147, 0.42)', highlight: '#eb5757', hover: '#ff8a65' }},
        font: {{ color: '#4f6478', strokeWidth: 2, strokeColor: '#f8fbff', size: 10, align: 'top' }},
      }},
      interaction: {{ hover: true, tooltipDelay: 120, navigationButtons: true, keyboard: true }},
      physics: {{
        enabled: true,
        stabilization: {{ iterations: 180, updateInterval: 25 }},
        timestep: 0.3,
        minVelocity: 1.0,
        barnesHut: {{ gravitationalConstant: -2600, springLength: 130, springConstant: 0.012, damping: 0.78, avoidOverlap: 0.08 }}
      }}
    }};

    function getNodeColor(node) {{
      if (node.sourceGroup === 'mid-mp3s') {{
        return {{
          background: '#ff9f5a',
          border: '#d66d1f',
          highlight: {{ background: '#ff8a65', border: '#eb5757' }}
        }};
      }}
      if (node.sourceGroup === 'mp3') {{
        return {{
          background: '#5aa9ff',
          border: '#1f6fd6',
          highlight: {{ background: '#ff8a65', border: '#eb5757' }}
        }};
      }}
      return {{
        background: '#5aa9ff',
        border: '#1f6fd6',
        highlight: {{ background: '#ff8a65', border: '#eb5757' }}
      }};
    }}

    function showNetworkStatus(message) {{
      if (!networkStatus) {{
        return;
      }}
      networkStatus.textContent = message;
      networkStatus.classList.add('visible');
    }}

    if (typeof vis === 'undefined') {{
      showNetworkStatus('Could not load the graph library. If you opened this HTML offline, rerun the script with internet access or open it while online.');
      throw new Error('vis-network failed to load');
    }}

    function getNodeSize(node) {{
      return listenTimeSizeToggle.checked ? node.listenTimeSize : node.playSize;
    }}

    function buildVisibleNode(node) {{
      const rememberedPosition = nodePositions.get(node.id);
      return {{
        ...node,
        value: undefined,
        size: getNodeSize(node),
        font: {{ color: '#142130', size: 18, face: 'Inter' }},
        color: node.id === selectedNodeId
          ? {{
              background: '#ff8a65',
              border: '#eb5757',
              highlight: {{ background: '#ff8a65', border: '#eb5757' }}
            }}
          : getNodeColor(node),
        ...(rememberedPosition ? {{ x: rememberedPosition.x, y: rememberedPosition.y }} : {{}}),
        fixed: {{ x: false, y: false }}
      }};
    }}

    function describeSizingMode() {{
      if (!listenTimeSizeToggle.checked) {{
        return 'Sizing: play count';
      }}
      return 'Sizing: listen hours';
    }}

    function refreshSizingSummary() {{
      document.getElementById('summary-sizing').textContent = describeSizingMode();
    }}

    let nodeDataSet = new vis.DataSet([]);
    let edgeDataSet = new vis.DataSet([]);
    let network = null;
    let graphWasManuallyMoved = false;
    const nodePositions = new Map();

    const networkOptions = {{
      autoResize: true,
      height: '100%',
      width: '100%',
      nodes: {{
        shape: 'dot',
        scaling: {{ min: 10, max: 42 }},
        borderWidth: 1.5,
      }},
      edges: {{
        arrows: {{ to: {{ enabled: true, scaleFactor: 0.65 }} }},
        smooth: {{ type: 'dynamic' }},
        color: {{ color: 'rgba(94, 122, 147, 0.48)', highlight: '#eb5757', hover: '#ff8a65' }},
        font: {{ color: '#4f6478', strokeWidth: 2, strokeColor: '#f8fbff', size: 11, align: 'top' }},
      }},
      interaction: {{ hover: true, tooltipDelay: 120, navigationButtons: true, keyboard: true }},
      physics: {{
        enabled: true,
        stabilization: {{ iterations: 180, updateInterval: 25 }},
        timestep: 0.3,
        minVelocity: 1.05,
        barnesHut: {{ gravitationalConstant: -2300, springLength: 135, springConstant: 0.011, damping: 0.78, avoidOverlap: 0.05 }}
      }}
    }};

    function formatPercent(probability) {{
      return `${{(probability * 100).toFixed(2)}}%`;
    }}

    function formatMinutes(seconds) {{
      return `${{(seconds / 60).toFixed(2)}} min`;
    }}

    function formatDuration(seconds) {{
      if (!Number.isFinite(seconds) || seconds <= 0) {{
        return 'Unknown';
      }}
      const minutes = Math.floor(seconds / 60);
      const remainingSeconds = Math.round(seconds % 60).toString().padStart(2, '0');
      return `${{minutes}}:${{remainingSeconds}}`;
    }}

    function formatListenTime(seconds) {{
      if (!Number.isFinite(seconds) || seconds <= 0) {{
        return '0m';
      }}
      if (seconds < 3600) {{
        return `${{Math.round(seconds / 60)}}m`;
      }}
      return `${{(seconds / 3600).toFixed(1)}}h`;
    }}

    function formatScore(value) {{
      const score = Number(value || 0);
      if (score >= 100) {{
        return score.toFixed(0);
      }}
      if (score >= 10) {{
        return score.toFixed(1);
      }}
      return score.toFixed(2);
    }}

    function formatDateTime(timestamp) {{
      if (!Number.isFinite(timestamp) || timestamp <= 0) {{
        return '-';
      }}
      return new Intl.DateTimeFormat(undefined, {{
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
      }}).format(new Date(timestamp * 1000));
    }}

    function formatNumber(value) {{
      return new Intl.NumberFormat().format(value || 0);
    }}

    function escapeHtml(value) {{
      return String(value).replace(/[&<>\"']/g, (ch) => ({{
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '\"': '&quot;',
        "'": '&#39;',
      }}[ch]));
    }}

    function hashString(value) {{
      let hash = 2166136261;
      for (let index = 0; index < value.length; index += 1) {{
        hash ^= value.charCodeAt(index);
        hash = Math.imul(hash, 16777619);
      }}
      return hash >>> 0;
    }}

    function loadTrendingColorOverrides() {{
      try {{
        const parsed = JSON.parse(localStorage.getItem(trendingColorStorageKey) || '{{}}');
        return parsed && typeof parsed === 'object' ? parsed : {{}};
      }} catch (_error) {{
        return {{}};
      }}
    }}

    function saveTrendingColorOverrides() {{
      try {{
        localStorage.setItem(trendingColorStorageKey, JSON.stringify(trendingColorOverrides));
      }} catch (_error) {{
        // Ignore private-mode or quota errors; the picker still works for this session.
      }}
    }}

    function hexToRgb(hex) {{
      const match = /^#?([a-f\\d]{{2}})([a-f\\d]{{2}})([a-f\\d]{{2}})$/i.exec(hex || '');
      if (!match) {{
        return null;
      }}
      return {{
        r: Number.parseInt(match[1], 16),
        g: Number.parseInt(match[2], 16),
        b: Number.parseInt(match[3], 16),
      }};
    }}

    function rgbToHex({{ r, g, b }}) {{
      return `#${{[r, g, b].map((value) => Math.max(0, Math.min(255, Math.round(value))).toString(16).padStart(2, '0')).join('')}}`;
    }}

    function mixHex(leftHex, rightHex, amount) {{
      const left = hexToRgb(leftHex);
      const right = hexToRgb(rightHex);
      if (!left || !right) {{
        return leftHex;
      }}
      return rgbToHex({{
        r: left.r + (right.r - left.r) * amount,
        g: left.g + (right.g - left.g) * amount,
        b: left.b + (right.b - left.b) * amount,
      }});
    }}

    function trendingColorKey(songName) {{
      if (songName.includes('/')) {{
        return songName;
      }}
      const viewKey = trendingData.summary.viewKey || activeViewKey;
      if (viewKey === 'mp3' || viewKey === 'mid-mp3s') {{
        return `${{viewKey}}/${{songName}}`;
      }}
      return songName;
    }}

    function fallbackColorKeys(songName) {{
      const keys = [trendingColorKey(songName), songName];
      if (songName.includes('/')) {{
        keys.push(songName.split('/').slice(1).join('/'));
      }}
      return Array.from(new Set(keys));
    }}

    function mutedSongColors(songName) {{
      const override = fallbackColorKeys(songName)
        .map((key) => trendingColorOverrides[key])
        .find((value) => /^#[0-9a-f]{{6}}$/i.test(value || ''));
      if (/^#[0-9a-f]{{6}}$/i.test(override || '')) {{
        return {{ a: override, b: mixHex(override, '#ffffff', 0.42) }};
      }}
      const palette = [
        ['#5f7fa3', '#92a9bd'],
        ['#6f9588', '#9db5ad'],
        ['#92789a', '#b2a1b8'],
        ['#9b8062', '#bba992'],
        ['#7680aa', '#a2a9c4'],
        ['#a07069', '#bd9d98'],
        ['#5f8b72', '#95ad9d'],
        ['#7b8795', '#a6afb9'],
        ['#8b7aa2', '#ada2bd'],
        ['#5f8e9e', '#95b1ba'],
      ];
      const index = hashString(songName) % palette.length;
      return {{ a: palette[index][0], b: palette[index][1] }};
    }}

    function decodeTrendingTimestamps(encoded) {{
      const out = [];
      let current = null;
      for (let index = 0; index < (encoded || []).length; index += 1) {{
        const value = Number(encoded[index]);
        if (!Number.isFinite(value)) {{
          continue;
        }}
        current = current === null || index === 0 ? value : current + value;
        out.push(current);
      }}
      return out;
    }}

    function timestampForTrendingSlider() {{
      const ratio = Math.max(0, Math.min(1, Number(trendingTimeline.value || 0) / Number(trendingTimeline.max || 1)));
      return trendingFirstTimestamp + ((trendingLastTimestamp - trendingFirstTimestamp) * ratio);
    }}

    function scoreTrendingSong(song, timestamp) {{
      let score = 0;
      let count = 0;
      let lastListen = null;
      const halfLifeSeconds = Math.max(1, Number(trendingData.summary.trendingHalfLifeSeconds || 172800));
      const baselineSeconds = Math.max(1, Number(trendingData.summary.trendingBaselineDurationSeconds || 210));
      const duration = Math.max(0, Number(song.durationSeconds || 0));
      const weight = duration > 0 ? duration / baselineSeconds : 1;
      for (const playedAt of song.timestamps) {{
        if (playedAt > timestamp) {{
          break;
        }}
        count += 1;
        lastListen = playedAt;
        const age = Math.max(0, timestamp - playedAt);
        score += Math.pow(2, -age / halfLifeSeconds) * weight;
      }}
      return {{ score, count, lastListen, weight }};
    }}

    function applyTrendingData(nextTrendingData) {{
      const keepAtEnd = Number(trendingTimeline.value || 0) >= Number(trendingTimeline.max || 1);
      trendingData = nextTrendingData || trendingData;
      const summary = trendingData.summary || {{}};
      trendingSongs = (trendingData.songs || []).map((song) => ({{
        ...song,
        timestamps: decodeTrendingTimestamps(song.history),
      }}));
      trendingFirstTimestamp = Number(summary.firstTimestamp || summary.generatedTimestamp || 0);
      trendingLastTimestamp = Math.max(
        Number(summary.lastTimestamp || 0),
        Number(summary.generatedTimestamp || 0),
        trendingFirstTimestamp,
      );
      document.getElementById('trending-subtitle').textContent =
        `${{summary.folder || '-'}} · duration-weighted 2d decay`;
      document.getElementById('trending-song-count').textContent = formatNumber(summary.songCount);
      document.getElementById('trending-event-count').textContent = formatNumber(summary.historyEventCount);
      trendingTimeline.disabled = !summary.timelinePointCount;
      if (keepAtEnd) {{
        trendingTimeline.value = trendingTimeline.max;
      }}
      if (trendingTimeline.disabled) {{
        stopTrendingPlayback();
      }}
      renderTrending();
    }}

    function renderTrending() {{
      const timestamp = timestampForTrendingSlider();
      const limit = Math.max(1, Math.min(200, Number(trendingTopLimit.value || 1)));
      trendingTopLimit.value = limit;
      const ranked = [];
      for (const song of trendingSongs) {{
        const scored = scoreTrendingSong(song, timestamp);
        if (scored.score <= 0) {{
          continue;
        }}
        ranked.push({{ ...song, ...scored }});
      }}
      ranked.sort((left, right) => right.score - left.score || left.name.localeCompare(right.name));
      const selected = ranked.slice(0, limit);
      const maxScore = Math.max(1, ...selected.map((song) => song.score));
      document.getElementById('trending-through').textContent =
        `Through ${{formatDateTime(timestamp)}}`;
      document.getElementById('trending-rank-caption').textContent =
        `${{formatNumber(selected.length)}} of ${{formatNumber(ranked.length)}} active`;
      const selectedScore = selected.reduce((sum, song) => sum + song.score, 0);
      const totalScore = ranked.reduce((sum, song) => sum + song.score, 0);
      document.getElementById('trending-timeline-date').textContent =
        `Through ${{formatDateTime(timestamp)}}`;
      document.getElementById('trending-timeline-count').textContent =
        selected[0]
          ? `Leader: ${{selected[0].name}} · +${{formatScore(totalScore)}} weighted score · ${{formatNumber(ranked.length)}} active`
          : `+${{formatScore(totalScore)}} weighted score · ${{formatNumber(ranked.length)}} active`;

      if (!selected.length) {{
        trendingEmpty.hidden = false;
      }} else {{
        trendingEmpty.hidden = true;
      }}

      const selectedNames = new Set(selected.map((song) => song.name));
      const rowHeight = Number.parseFloat(getComputedStyle(document.querySelector('.trending-layout')).getPropertyValue('--trending-row-height')) || 58;
      const rowGap = Number.parseFloat(getComputedStyle(document.querySelector('.trending-layout')).getPropertyValue('--trending-row-gap')) || 7;
      const totalHeight = selected.length
        ? selected.length * rowHeight + Math.max(0, selected.length - 1) * rowGap
        : rowHeight;
      trendingStage.style.height = `${{totalHeight}}px`;

      selected.forEach((song, index) => {{
        const yPosition = index * (rowHeight + rowGap);
        let row = trendingRows.get(song.name);
        if (!row) {{
          row = createTrendingRow(song.name);
          trendingRows.set(song.name, row);
          row.style.transform = `translateY(${{yPosition + 10}}px)`;
          requestAnimationFrame(() => updateTrendingRow(row, song, index + 1, maxScore, yPosition));
        }} else {{
          updateTrendingRow(row, song, index + 1, maxScore, yPosition);
        }}
      }});

      for (const [songName, row] of trendingRows.entries()) {{
        if (selectedNames.has(songName)) {{
          continue;
        }}
        row.classList.add('is-exiting');
        row.style.opacity = '0';
        window.setTimeout(() => {{
          if (!trendingRows.has(songName)) {{
            return;
          }}
          const currentRow = trendingRows.get(songName);
          if (currentRow === row && row.classList.contains('is-exiting')) {{
            row.remove();
            trendingRows.delete(songName);
          }}
        }}, 280);
      }}
    }}

    function createTrendingRow(songName) {{
      const row = document.createElement('article');
      row.className = 'trending-row';
      row.dataset.songName = songName;
      row.innerHTML = `
        <div class=\"trending-rank\"></div>
        <input class=\"trending-color\" type=\"color\" title=\"Set track color\" aria-label=\"Set track color\">
        <div class=\"trending-track\">
          <div class=\"trending-name\"></div>
          <div class=\"trending-bar\"><div class=\"trending-fill\"></div></div>
          <div class=\"muted trending-meta\"></div>
        </div>
        <div class=\"trending-score\"><strong></strong><span></span></div>
        <div class=\"trending-plays\"><strong></strong><span></span></div>
      `;
      row.style.opacity = '0';
      row.querySelector('.trending-color').addEventListener('input', (event) => {{
        const nextColor = String(event.target.value || '').toLowerCase();
        if (/^#[0-9a-f]{{6}}$/i.test(nextColor)) {{
          trendingColorOverrides[trendingColorKey(row.dataset.songName)] = nextColor;
          saveTrendingColorOverrides();
          const colors = mutedSongColors(row.dataset.songName);
          row.style.setProperty('--song-trend-a', colors.a);
          row.style.setProperty('--song-trend-b', colors.b);
        }}
      }});
      trendingStage.appendChild(row);
      return row;
    }}

    function updateTrendingRow(row, song, rank, maxScore, yPosition) {{
      const width = Math.max(1, 100 * song.score / maxScore).toFixed(2);
      const colors = mutedSongColors(song.name);
      row.style.setProperty('--song-trend-a', colors.a);
      row.style.setProperty('--song-trend-b', colors.b);
      row.querySelector('.trending-color').value = colors.a;
      row.classList.toggle('is-leader', rank === 1);
      row.classList.remove('is-exiting');
      row.style.transform = `translateY(${{yPosition}}px)`;
      row.style.zIndex = String(1000 - rank);
      row.style.opacity = '1';
      row.querySelector('.trending-rank').textContent = `#${{rank}}`;
      const nameEl = row.querySelector('.trending-name');
      nameEl.textContent = song.name;
      nameEl.title = song.name;
      row.querySelector('.trending-meta').textContent =
        `last ${{formatDateTime(song.lastListen)}} · duration ${{formatDuration(song.durationSeconds)}} · weight ${{song.weight.toFixed(2)}}x`;
      row.querySelector('.trending-fill').style.width = `${{width}}%`;
      row.querySelector('.trending-score strong').textContent = `+${{formatScore(song.score)}}`;
      row.querySelector('.trending-score span').textContent = 'weighted decay';
      row.querySelector('.trending-plays strong').textContent = formatNumber(song.count);
      row.querySelector('.trending-plays span').textContent = 'timestamped';
    }}

    function stopTrendingPlayback() {{
      trendingIsPlaying = false;
      trendingPlayToggle.textContent = '▶';
      trendingPlayToggle.title = 'Play timeline';
      trendingLastPlaybackFrame = null;
      if (trendingPlayAnimationFrame !== null) {{
        cancelAnimationFrame(trendingPlayAnimationFrame);
        trendingPlayAnimationFrame = null;
      }}
    }}

    function stepTrendingPlayback(frameTimestamp) {{
      if (!trendingIsPlaying || trendingTimeline.disabled) {{
        return;
      }}
      if (trendingLastPlaybackFrame === null) {{
        trendingLastPlaybackFrame = frameTimestamp;
      }}
      const elapsedMs = Math.max(0, frameTimestamp - trendingLastPlaybackFrame);
      trendingLastPlaybackFrame = frameTimestamp;
      const current = Number(trendingTimeline.value) || 0;
      const sliderMax = Number(trendingTimeline.max) || 1000;
      if (current >= sliderMax) {{
        stopTrendingPlayback();
        return;
      }}
      const secondsPerTimelineDay = Math.max(0.1, Number(trendingPlaySpeed.value) || 5);
      const timelineSpanSeconds = Math.max(1, trendingLastTimestamp - trendingFirstTimestamp);
      const advancedTimelineSeconds = elapsedMs * secondsPerDay / (secondsPerTimelineDay * 1000);
      const sliderDelta = sliderMax * advancedTimelineSeconds / timelineSpanSeconds;
      trendingTimeline.value = Math.min(sliderMax, current + sliderDelta);
      renderTrending();
      trendingPlayAnimationFrame = requestAnimationFrame(stepTrendingPlayback);
    }}

    function toggleTrendingPlayback() {{
      if (trendingIsPlaying) {{
        stopTrendingPlayback();
        return;
      }}
      if (trendingTimeline.disabled) {{
        return;
      }}
      const sliderMax = Number(trendingTimeline.max) || 1000;
      if (Number(trendingTimeline.value) >= sliderMax) {{
        trendingTimeline.value = 0;
      }}
      trendingIsPlaying = true;
      trendingLastPlaybackFrame = null;
      trendingPlayToggle.textContent = '❚❚';
      trendingPlayToggle.title = 'Pause timeline';
      trendingPlayAnimationFrame = requestAnimationFrame(stepTrendingPlayback);
    }}

    function decodeArtistHistory(encoded) {{
      return decodeTrendingTimestamps(encoded);
    }}

    function artistColor(node) {{
      if (node.kind === 'song') {{
        return {{
          background: '#a7c7e7',
          border: '#5f8fbf',
          highlight: {{ background: '#ffb088', border: '#eb5757' }}
        }};
      }}
      return {{
        background: '#ffb36b',
        border: '#d8792b',
        highlight: {{ background: '#ff8a65', border: '#eb5757' }}
      }};
    }}

    function applyArtistData(nextArtistData) {{
      artistData = nextArtistData || artistData;
      const summary = artistData.summary || {{}};
      document.getElementById('artist-subtitle').textContent =
        `${{summary.folder || '-'}} · unassigned one-offs excluded from artist graph`;
      document.getElementById('artist-count').textContent = formatNumber(summary.artistCount);
      document.getElementById('artist-tagged-songs').textContent = formatNumber(summary.taggedSongCount);
      document.getElementById('artist-unassigned-songs').textContent = formatNumber(summary.unassignedSongCount);
      document.getElementById('artist-edge-count').textContent = formatNumber(summary.artistEdgeCount);
      document.getElementById('artist-stats-subtitle').textContent =
        `${{summary.folder || '-'}} · moving artist plots use timestamped plays only`;
      document.getElementById('stats-artist-count').textContent = formatNumber(summary.artistCount);
      document.getElementById('stats-tagged-plays').textContent = formatNumber(summary.taggedPlayCount);
      document.getElementById('stats-unassigned-plays').textContent = formatNumber(summary.unassignedPlayCount);
      document.getElementById('stats-history-events').textContent = formatNumber(summary.historyEventCount);
      artistTimelineRows = ((artistData.trending || {{}}).artists || []).map((artist) => ({{
        ...artist,
        timestamps: decodeArtistHistory(artist.history),
      }}));
      artistWeeklyData = null;
      artistFirstTimestamp = Number(summary.firstTimestamp || summary.generatedTimestamp || 0);
      artistLastTimestamp = Math.max(Number(summary.lastTimestamp || 0), Number(summary.generatedTimestamp || 0), artistFirstTimestamp);
      artistTimeline.disabled = !summary.timelinePointCount;
      if (!artistTimeline.disabled) {{
        artistTimeline.value = artistTimeline.max;
      }}
      artistStatsTimeline.disabled = !summary.timelinePointCount;
      if (!artistStatsTimeline.disabled) {{
        artistStatsTimeline.value = artistStatsTimeline.max;
      }}
      populateArtistInspectorSelect();
      renderArtistBars();
      renderArtistHighlights();
      renderArtistTimeline();
      renderArtistStatsDashboard();
      if (selectedStatsArtist && findArtistNodeByName(selectedStatsArtist)) {{
        showArtistInspector(selectedStatsArtist);
      }} else {{
        const firstArtist = (artistData.topArtists || [])[0];
        showArtistInspector(firstArtist ? firstArtist.label : null);
      }}
      createArtistNetwork();
    }}

    function getArtistGraphData() {{
      return artistMode === 'songs'
        ? (artistData.songArtistGraph || {{ nodes: [], edges: [] }})
        : (artistData.artistGraph || {{ nodes: [], edges: [] }});
    }}

    function buildVisibleArtistNode(node) {{
      return {{
        ...node,
        size: Math.max(8, Number(node.size || node.playSize || 12)),
        color: artistColor(node),
        font: {{ color: '#142130', size: node.kind === 'song' ? 13 : 17, face: 'Inter' }},
      }};
    }}

    function createArtistNetwork() {{
      if (!artistContainer) {{
        return;
      }}
      if (artistNetwork) {{
        artistNetwork.destroy();
      }}
      const graph = getArtistGraphData();
      const nodes = new vis.DataSet((graph.nodes || []).map(buildVisibleArtistNode));
      const edges = new vis.DataSet((graph.edges || []).map((edge) => ({{
        ...edge,
        label: artistMode === 'transitions' ? edge.label : '',
      }})));
      artistNetwork = new vis.Network(artistContainer, {{ nodes, edges }}, artistNetworkOptions);
      artistNetwork.on('selectNode', (params) => {{
        artistSelectedId = params.nodes[0] || null;
        showArtistNodeDetails(artistSelectedId);
      }});
      artistNetwork.on('selectEdge', (params) => {{
        showArtistEdgeDetails(params.edges[0]);
      }});
      artistNetwork.once('stabilized', () => {{
        artistNetwork.fit({{ animation: {{ duration: 350, easingFunction: 'easeInOutQuad' }} }});
      }});
    }}

    function renderArtistBars() {{
      const container = document.getElementById('artist-bars');
      const artists = artistData.topArtists || [];
      const maxPlays = Math.max(1, ...artists.map((artist) => Number(artist.playCount || 0)));
      container.innerHTML = artists.slice(0, 18).map((artist, index) => `
        <div class=\"artist-bar-row\" title=\"${{escapeHtml(artist.label)}}\">
          <div class=\"muted\">#${{index + 1}}</div>
          <div class=\"artist-bar-name\">${{escapeHtml(artist.label)}}</div>
          <div>${{formatNumber(artist.playCount)}}</div>
          <div class=\"artist-bar-track\"><div class=\"artist-bar-fill\" style=\"width:${{Math.max(2, 100 * Number(artist.playCount || 0) / maxPlays).toFixed(1)}}%\"></div></div>
        </div>
      `).join('') || '<div class=\"muted\">No tagged artists yet.</div>';
    }}

    function renderArtistHighlights() {{
      const container = document.getElementById('artist-highlights');
      if (!container) {{
        return;
      }}
      const highlights = artistData.highlights || [];
      container.innerHTML = highlights.map((item) => `
        <div class=\"item\">
          <strong>${{escapeHtml(item.label)}}: ${{escapeHtml(item.artist)}}</strong>
          <div class=\"muted\">${{escapeHtml(item.value || '')}}${{item.detail ? ` · ${{escapeHtml(item.detail)}}` : ''}}</div>
        </div>
      `).join('') || '<div class=\"muted\">No artist facts yet.</div>';
    }}

    function artistTimelineTimestamp() {{
      const ratio = Math.max(0, Math.min(1, Number(artistTimeline.value || 0) / Number(artistTimeline.max || 1)));
      return artistFirstTimestamp + ((artistLastTimestamp - artistFirstTimestamp) * ratio);
    }}

    function scoreArtist(artist, timestamp) {{
      let score = 0;
      let count = 0;
      const halfLifeSeconds = Math.max(1, Number((artistData.summary || {{}}).trendingHalfLifeSeconds || 172800));
      for (const playedAt of artist.timestamps || []) {{
        if (playedAt > timestamp) {{
          break;
        }}
        count += 1;
        score += Math.pow(2, -Math.max(0, timestamp - playedAt) / halfLifeSeconds);
      }}
      return {{ score, count }};
    }}

    function renderArtistTimeline() {{
      const container = document.getElementById('artist-timeline-bars');
      const timestamp = artistTimelineTimestamp();
      document.getElementById('artist-timeline-date').textContent = artistTimeline.disabled
        ? 'No timestamped plays for tagged artists.'
        : `Through ${{formatDateTime(timestamp)}}`;
      const ranked = [];
      for (const artist of artistTimelineRows) {{
        const scored = scoreArtist(artist, timestamp);
        if (scored.score > 0) {{
          ranked.push({{ ...artist, ...scored }});
        }}
      }}
      ranked.sort((left, right) => right.score - left.score || left.name.localeCompare(right.name));
      const selected = ranked.slice(0, 12);
      const maxScore = Math.max(1, ...selected.map((artist) => artist.score));
      container.innerHTML = selected.map((artist, index) => `
        <div class=\"artist-bar-row\" title=\"${{escapeHtml(artist.name)}}\">
          <div class=\"muted\">#${{index + 1}}</div>
          <div class=\"artist-bar-name\">${{escapeHtml(artist.name)}}</div>
          <div>+${{formatScore(artist.score)}}</div>
          <div class=\"artist-bar-track\"><div class=\"artist-bar-fill\" style=\"width:${{Math.max(2, 100 * artist.score / maxScore).toFixed(1)}}%\"></div></div>
        </div>
      `).join('') || '<div class=\"muted\">No active artists at this point.</div>';
    }}

    function artistStatsTimestamp() {{
      const ratio = Math.max(0, Math.min(1, Number(artistStatsTimeline.value || 0) / Number(artistStatsTimeline.max || 1)));
      return artistFirstTimestamp + ((artistLastTimestamp - artistFirstTimestamp) * ratio);
    }}

    function artistProgressAt(artist, timestamp) {{
      const timestamps = artist.timestamps || [];
      if (!timestamps.length) {{
        return {{ count: 0, progress: 0 }};
      }}
      let count = 0;
      for (const playedAt of timestamps) {{
        if (playedAt > timestamp) {{
          break;
        }}
        count += 1;
      }}
      return {{ count, progress: count / timestamps.length }};
    }}

    function countInWindow(timestamps, startTimestamp, endTimestamp) {{
      let count = 0;
      for (const timestamp of timestamps || []) {{
        if (timestamp > endTimestamp) {{
          break;
        }}
        if (timestamp >= startTimestamp) {{
          count += 1;
        }}
      }}
      return count;
    }}

    function artistRecentCount(artist, timestamp, days) {{
      return countInWindow(artist.timestamps || [], timestamp - days * secondsPerDay, timestamp);
    }}

    function artistWeeklyBaseline(artist) {{
      const timestamps = artist.timestamps || [];
      if (!timestamps.length) {{
        return 0;
      }}
      const spanWeeks = Math.max(1, (timestamps[timestamps.length - 1] - timestamps[0]) / (7 * secondsPerDay));
      return timestamps.length / spanWeeks;
    }}

    function artistMapColor(name) {{
      const colors = mutedSongColors(name);
      return colors.a;
    }}

    function indexSafeLabel(name) {{
      const words = String(name || '').trim().split(/\s+/).filter(Boolean);
      if (!words.length) {{
        return '?';
      }}
      if (words.length === 1) {{
        return words[0].slice(0, 2).toUpperCase();
      }}
      return words.slice(0, 2).map((word) => word[0]).join('').toUpperCase();
    }}

    function weekStartTimestamp(timestamp) {{
      const date = new Date(timestamp * 1000);
      const utcDay = date.getUTCDay();
      const diffToMonday = (utcDay + 6) % 7;
      const monday = Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate() - diffToMonday);
      return Math.floor(monday / 1000);
    }}

    function localDayKey(timestamp) {{
      const date = new Date(timestamp * 1000);
      const year = date.getFullYear();
      const month = String(date.getMonth() + 1).padStart(2, '0');
      const day = String(date.getDate()).padStart(2, '0');
      return `${{year}}-${{month}}-${{day}}`;
    }}

    function shortDateLabel(dayKey) {{
      const [year, month, day] = String(dayKey || '').split('-').map(Number);
      if (!year || !month || !day) {{
        return dayKey || '-';
      }}
      return new Intl.DateTimeFormat(undefined, {{ month: 'short', day: 'numeric' }}).format(new Date(year, month - 1, day));
    }}

    function renderSimpleCharts() {{
      renderSimpleCalendarHeatmap();
      renderSimpleWeekdayHeatmap();
      renderSimpleSongScatter();
      renderSimpleNewSongsOverTime();
      renderSimpleRepeatRate();
    }}

    function renderSimpleCalendarHeatmap() {{
      const container = document.getElementById('simple-calendar-heatmap');
      if (!container) {{
        return;
      }}
      const dayCounts = new Map();
      for (const song of trendingSongs) {{
        for (const timestamp of song.timestamps || []) {{
          const key = localDayKey(timestamp);
          dayCounts.set(key, (dayCounts.get(key) || 0) + 1);
        }}
      }}
      const days = Array.from(dayCounts.keys()).sort();
      const selected = days.slice(-98);
      const maxCount = Math.max(1, ...selected.map((day) => dayCounts.get(day) || 0));
      const caption = document.getElementById('simple-calendar-caption');
      if (caption) {{
        caption.textContent = selected.length
          ? `${{shortDateLabel(selected[0])}} to ${{shortDateLabel(selected[selected.length - 1])}}`
          : 'No timestamped plays yet.';
      }}
      container.innerHTML = selected.map((day) => {{
        const count = dayCounts.get(day) || 0;
        const alpha = 0.12 + 0.78 * count / maxCount;
        return `<div class=\"heatmap-cell\" style=\"background:rgba(47, 128, 237, ${{alpha.toFixed(2)}});\" title=\"${{day}} · ${{formatNumber(count)}} plays\"></div>`;
      }}).join('') || '<div class=\"muted\">No calendar data.</div>';
    }}

    function renderSimpleWeekdayHeatmap() {{
      const container = document.getElementById('simple-weekday-heatmap');
      if (!container) {{
        return;
      }}
      const counts = Array.from({{ length: 7 }}, () => Array.from({{ length: 24 }}, () => 0));
      for (const song of trendingSongs) {{
        for (const timestamp of song.timestamps || []) {{
          const date = new Date(timestamp * 1000);
          counts[date.getDay()][date.getHours()] += 1;
        }}
      }}
      const labels = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
      const maxCount = Math.max(1, ...counts.flat());
      container.innerHTML = counts.map((row, dayIndex) => `
        <div class=\"weekday-label\">${{labels[dayIndex]}}</div>
        ${{row.map((count, hour) => {{
          const alpha = count ? 0.12 + 0.78 * count / maxCount : 0.06;
          return `<div class=\"heatmap-cell\" style=\"background:rgba(86, 204, 157, ${{alpha.toFixed(2)}});\" title=\"${{labels[dayIndex]}} ${{hour}}:00 · ${{formatNumber(count)}} plays\"></div>`;
        }}).join('')}}
      `).join('');
    }}

    function renderSimpleSongScatter() {{
      const container = document.getElementById('simple-song-scatter');
      if (!container) {{
        return;
      }}
      const rows = trendingSongs.map((song) => {{
        const timestamps = song.timestamps || [];
        return {{
          ...song,
          firstListen: timestamps[0] || null,
          recentCount: countInWindow(timestamps, trendingLastTimestamp - 30 * secondsPerDay, trendingLastTimestamp),
        }};
      }}).filter((song) => song.firstListen && Number(song.playCount || 0) > 0)
        .sort((left, right) => Number(right.playCount || 0) - Number(left.playCount || 0))
        .slice(0, 90);
      if (!rows.length) {{
        container.innerHTML = '<div class=\"muted\" style=\"padding:18px;\">No song scatter data.</div>';
        return;
      }}
      const minFirst = Math.min(...rows.map((song) => song.firstListen));
      const maxFirst = Math.max(...rows.map((song) => song.firstListen));
      const maxPlays = Math.max(1, ...rows.map((song) => Number(song.playCount || 0)));
      const maxRecent = Math.max(1, ...rows.map((song) => Number(song.recentCount || 0)));
      const span = Math.max(1, maxFirst - minFirst);
      container.innerHTML = rows.map((song) => {{
        const left = 6 + 88 * ((song.firstListen - minFirst) / span);
        const top = 92 - 84 * Math.log1p(Number(song.playCount || 0)) / Math.log1p(maxPlays);
        const size = 10 + 24 * Math.sqrt(Number(song.recentCount || 0) / maxRecent);
        const label = songDisplayName(song.name);
        return `<div class=\"scatter-point\" style=\"left:${{left.toFixed(2)}}%; top:${{top.toFixed(2)}}%; width:${{size.toFixed(1)}}px; height:${{size.toFixed(1)}}px; background:${{artistMapColor(label)}};\" title=\"${{escapeHtml(label)}} · first ${{formatDateTime(song.firstListen)}} · ${{formatNumber(song.playCount)}} plays · ${{formatNumber(song.recentCount)}} recent\">${{size > 24 ? indexSafeLabel(label) : ''}}</div>`;
      }}).join('');
    }}

    function renderSimpleNewSongsOverTime() {{
      const container = document.getElementById('simple-new-songs');
      if (!container) {{
        return;
      }}
      const weekCounts = new Map();
      for (const song of trendingSongs) {{
        const first = (song.timestamps || [])[0];
        if (!first) {{
          continue;
        }}
        const week = weekStartTimestamp(first);
        weekCounts.set(week, (weekCounts.get(week) || 0) + 1);
      }}
      const weeks = Array.from(weekCounts.keys()).sort((left, right) => left - right).slice(-18);
      const maxCount = Math.max(1, ...weeks.map((week) => weekCounts.get(week) || 0));
      container.innerHTML = weeks.map((week) => {{
        const count = weekCounts.get(week) || 0;
        const width = Math.max(2, 100 * count / maxCount);
        return `
          <div class=\"simple-bar-row\">
            <div class=\"muted\">${{shortDateLabel(localDayKey(week))}}</div>
            <div class=\"simple-bar-track\"><div class=\"simple-bar-fill\" style=\"width:${{width.toFixed(1)}}%;\"></div></div>
            <div>${{formatNumber(count)}}</div>
          </div>
        `;
      }}).join('') || '<div class=\"muted\">No new-song data.</div>';
    }}

    function renderSimpleRepeatRate() {{
      const container = document.getElementById('simple-repeat-rate');
      if (!container) {{
        return;
      }}
      const events = [];
      for (const song of trendingSongs) {{
        for (const timestamp of song.timestamps || []) {{
          events.push({{ timestamp, name: song.name }});
        }}
      }}
      events.sort((left, right) => left.timestamp - right.timestamp || left.name.localeCompare(right.name));
      const seenBySong = new Map();
      const weeks = new Map();
      for (const event of events) {{
        const week = weekStartTimestamp(event.timestamp);
        if (!weeks.has(week)) {{
          weeks.set(week, {{ total: 0, repeat: 0 }});
        }}
        const row = weeks.get(week);
        const previous = seenBySong.get(event.name);
        row.total += 1;
        if (previous && event.timestamp - previous <= 30 * secondsPerDay) {{
          row.repeat += 1;
        }}
        seenBySong.set(event.name, event.timestamp);
      }}
      const selectedWeeks = Array.from(weeks.keys()).sort((left, right) => left - right).slice(-18);
      container.innerHTML = selectedWeeks.map((week) => {{
        const row = weeks.get(week) || {{ total: 0, repeat: 0 }};
        const rate = row.total ? row.repeat / row.total : 0;
        return `
          <div class=\"simple-bar-row\">
            <div class=\"muted\">${{shortDateLabel(localDayKey(week))}}</div>
            <div class=\"simple-bar-track\"><div class=\"simple-bar-fill\" style=\"width:${{(rate * 100).toFixed(1)}}%; background:linear-gradient(90deg, #f2994a, #eb5757);\"></div></div>
            <div>${{(rate * 100).toFixed(0)}}%</div>
          </div>
        `;
      }}).join('') || '<div class=\"muted\">No repeat-rate data.</div>';
    }}

    function buildArtistWeeklyData() {{
      const weekSet = new Set();
      const rows = artistTimelineRows.map((artist) => {{
        const weekCounts = new Map();
        for (const timestamp of artist.timestamps || []) {{
          const week = weekStartTimestamp(timestamp);
          weekSet.add(week);
          weekCounts.set(week, (weekCounts.get(week) || 0) + 1);
        }}
        return {{ ...artist, weekCounts }};
      }});
      const weeks = Array.from(weekSet).sort((left, right) => left - right);
      const totals = new Map(weeks.map((week) => [week, 0]));
      for (const row of rows) {{
        for (const [week, count] of row.weekCounts.entries()) {{
          totals.set(week, (totals.get(week) || 0) + count);
        }}
      }}
      return {{ weeks, rows, totals }};
    }}

    function getArtistWeeklyData() {{
      if (!artistWeeklyData) {{
        artistWeeklyData = buildArtistWeeklyData();
      }}
      return artistWeeklyData;
    }}

    function findArtistNodeByName(name) {{
      if (!name) {{
        return null;
      }}
      const normalized = String(name).toLowerCase();
      return (((artistData.artistGraph || {{}}).nodes || []).find((node) =>
        String(node.label || node.id || '').toLowerCase() === normalized
      )) || null;
    }}

    function getSortedArtistNodes() {{
      return ((artistData.artistGraph || {{}}).nodes || [])
        .filter((node) => node && node.label)
        .slice()
        .sort((left, right) => Number(right.playCount || 0) - Number(left.playCount || 0) || String(left.label || '').localeCompare(String(right.label || '')));
    }}

    function populateArtistInspectorSelect() {{
      if (!artistInspectorSelect) {{
        return;
      }}
      const artists = getSortedArtistNodes();
      artistInspectorSelect.innerHTML = artists.map((artist) => `
        <option value=\"${{escapeHtml(artist.label)}}\">${{escapeHtml(artist.label)}} · ${{formatNumber(artist.playCount)}} plays</option>
      `).join('');
    }}

    function syncArtistInspectorSelect() {{
      if (artistInspectorSelect && selectedStatsArtist) {{
        artistInspectorSelect.value = selectedStatsArtist;
      }}
    }}

    function renderArtistSongBreakdown(node) {{
      const container = document.getElementById('artist-inspector-songs');
      const songs = (node.songDetails || []).slice().sort((left, right) =>
        Number(right.playCount || 0) - Number(left.playCount || 0) || String(left.name || '').localeCompare(String(right.name || ''))
      );
      const maxPlays = Math.max(1, ...songs.map((song) => Number(song.playCount || 0)));
      const totalPlays = Math.max(1, Number(node.playCount || 0));
      container.innerHTML = songs.slice(0, 18).map((song) => {{
        const plays = Number(song.playCount || 0);
        const width = Math.max(2, 100 * plays / maxPlays);
        const share = 100 * plays / totalPlays;
        return `
          <div class=\"song-breakdown-row\" title=\"${{escapeHtml(song.name || '')}}\">
            <div class=\"song-breakdown-name\">${{escapeHtml(song.name || '')}}</div>
            <div class=\"song-breakdown-meta\">${{formatNumber(plays)}} · ${{share.toFixed(0)}}%</div>
            <div class=\"song-breakdown-track\"><div class=\"song-breakdown-fill\" style=\"width:${{width.toFixed(1)}}%\"></div></div>
          </div>
        `;
      }}).join('') || '<div class=\"muted\">No tagged songs for this artist.</div>';
    }}

    function renderSelectedArtistDetails() {{
      const node = findArtistNodeByName(selectedStatsArtist);
      renderArtistSongGravityPlot(node);
      renderArtistCatalogCoverage(node);
    }}

    function showArtistInspector(artistName) {{
      const summary = document.getElementById('artist-stats-selection');
      const metrics = document.getElementById('artist-inspector-metrics');
      const songs = document.getElementById('artist-inspector-songs');
      const node = findArtistNodeByName(artistName);
      if (!node) {{
        selectedStatsArtist = null;
        summary.textContent = 'Select an artist.';
        metrics.innerHTML = '';
        songs.innerHTML = '';
        renderSelectedArtistDetails();
        return;
      }}
      selectedStatsArtist = node.label;
      syncArtistInspectorSelect();
      const topSong = node.topSong
        ? `${{escapeHtml(node.topSong)}} · ${{formatNumber(node.topSongPlays)}} plays · ${{(Number(node.topSongShare || 0) * 100).toFixed(0)}}%`
        : 'No top song yet';
      summary.innerHTML = `
        <div class=\"artist-inspector-title\">
          <strong>${{escapeHtml(node.label)}}</strong>
          <span class=\"muted\">${{formatNumber(node.songCount)}} songs · ${{formatListenTime(node.listenTimeSeconds)}}</span>
        </div>
        <div class=\"muted\">Top song: ${{topSong}}</div>
      `;
      metrics.innerHTML = `
        <div class=\"metric\"><div class=\"label\">Plays</div><div class=\"value\">${{formatNumber(node.playCount)}}</div></div>
        <div class=\"metric\"><div class=\"label\">Recent 30d</div><div class=\"value\">${{formatNumber(node.recent30PlayCount)}}</div></div>
        <div class=\"metric\"><div class=\"label\">Active days</div><div class=\"value\">${{formatNumber(node.activeDayCount)}}</div></div>
        <div class=\"metric\"><div class=\"label\">Longest gap</div><div class=\"value\">${{node.longestGapDays === null || node.longestGapDays === undefined ? '-' : Number(node.longestGapDays).toFixed(1) + 'd'}}</div></div>
      `;
      renderArtistSongBreakdown(node);
      renderSelectedArtistDetails();
    }}

    function renderArtistStatsDashboard() {{
      renderArtistStatsRace();
      renderArtistTakeoverPlot();
      renderArtistOrbitPlot();
      renderSelectedArtistDetails();
    }}

    function wireStatsStageSelection(stage) {{
      if (!stage) {{
        return;
      }}
      stage.querySelectorAll('[data-artist]').forEach((element) => {{
        element.addEventListener('click', () => {{
          const artist = element.dataset.artist || '';
          showArtistInspector(artist);
        }});
      }});
    }}

    function renderArtistOrbitPlot() {{
      if (!artistOrbitStage) {{
        return;
      }}
      const timestamp = artistStatsTimestamp();
      document.getElementById('artist-orbit-caption').textContent =
        artistStatsTimeline.disabled ? 'No timestamped artist plays yet.' : `Through ${{formatDateTime(timestamp)}} · closer to center = stronger current momentum`;
      const ranked = [];
      for (const artist of artistTimelineRows) {{
        const scored = scoreArtist(artist, timestamp);
        if (scored.score > 0) {{
          ranked.push({{ ...artist, ...scored }});
        }}
      }}
      ranked.sort((left, right) => right.score - left.score || left.name.localeCompare(right.name));
      const selected = ranked.slice(0, 26);
      const maxScore = Math.max(1, ...selected.map((artist) => artist.score));
      artistOrbitStage.innerHTML = '<div class=\"moving-point\" style=\"left:50%; top:50%; background:#ffffff; border-color:#2f80ed; width:52px; height:52px;\">YOU</div>' + selected.map((artist, index) => {{
        const strength = Math.max(0, Math.min(1, artist.score / maxScore));
        const radius = 8 + (1 - strength) * 36 + (index % 4) * 4;
        const angle = (index * 137.508 + (Number(artistStatsTimeline.value || 0) * 0.08)) * Math.PI / 180;
        const left = 50 + Math.cos(angle) * radius;
        const top = 50 + Math.sin(angle) * radius * 0.72;
        const label = escapeHtml(artist.name);
        return `
          <button class=\"moving-point\" data-artist=\"${{label}}\" style=\"left:${{left.toFixed(2)}}%; top:${{top.toFixed(2)}}%; background:${{artistMapColor(artist.name)}};\" title=\"${{label}} · +${{formatScore(artist.score)}} momentum\">${{indexSafeLabel(artist.name)}}</button>
          <div class=\"moving-label\" style=\"left:${{left.toFixed(2)}}%; top:${{top.toFixed(2)}}%;\">${{label}}</div>
        `;
      }}).join('') || '<div class=\"muted\" style=\"padding:18px;\">No artist orbit at this point.</div>';
      wireStatsStageSelection(artistOrbitStage);
    }}

    function renderArtistTakeoverPlot() {{
      if (!artistTakeoverStage) {{
        return;
      }}
      const timestamp = artistStatsTimestamp();
      const ranked = [];
      for (const artist of artistTimelineRows) {{
        const scored = scoreArtist(artist, timestamp);
        if (scored.score > 0) {{
          ranked.push({{ ...artist, ...scored }});
        }}
      }}
      ranked.sort((left, right) => right.score - left.score || left.name.localeCompare(right.name));
      const selected = ranked.slice(0, 14);
      const total = selected.reduce((sum, artist) => sum + artist.score, 0) || 1;
      let offset = 0;
      artistTakeoverStage.innerHTML = selected.map((artist, index) => {{
        const share = artist.score / total;
        const width = Math.max(4, share * 100);
        const left = offset;
        offset += width;
        const label = escapeHtml(artist.name);
        return `
          <button class=\"stats-chip\" data-artist=\"${{label}}\" style=\"left:${{Math.min(94, left + width / 2).toFixed(2)}}%; top:${{(28 + (index % 7) * 44).toFixed(1)}}px; width:${{Math.max(150, width * 7).toFixed(1)}}px; border-color:${{artistMapColor(artist.name)}};\" title=\"${{label}} · ${{(share * 100).toFixed(0)}}% current share\">
            <strong>${{label}}</strong><span>${{(share * 100).toFixed(0)}}%</span>
          </button>
        `;
      }}).join('') || '<div class=\"muted\" style=\"padding:18px;\">No artist takeover at this point.</div>';
      wireStatsStageSelection(artistTakeoverStage);
    }}

    function renderArtistStatsRace() {{
      const container = document.getElementById('artist-stats-bars');
      if (!container) {{
        return;
      }}
      const timestamp = artistStatsTimestamp();
      document.getElementById('artist-stats-date').textContent = artistStatsTimeline.disabled
        ? 'No timestamped artist plays yet.'
        : `Through ${{formatDateTime(timestamp)}}`;
      const ranked = [];
      for (const artist of artistTimelineRows) {{
        const progress = artistProgressAt(artist, timestamp);
        const scored = scoreArtist(artist, timestamp);
        if (scored.score > 0) {{
          ranked.push({{ ...artist, ...progress, ...scored }});
        }}
      }}
      ranked.sort((left, right) => right.score - left.score || right.count - left.count || left.name.localeCompare(right.name));
      const selected = ranked.slice(0, 14);
      const maxScore = Math.max(1, ...selected.map((artist) => artist.score));
      container.innerHTML = selected.map((artist, index) => `
        <div class=\"artist-bar-row\" title=\"${{escapeHtml(artist.name)}}\">
          <div class=\"muted\">#${{index + 1}}</div>
          <div class=\"artist-bar-name\">${{escapeHtml(artist.name)}}</div>
          <div>+${{formatScore(artist.score)}}</div>
          <div class=\"artist-bar-track\"><div class=\"artist-bar-fill\" style=\"width:${{Math.max(2, 100 * artist.score / maxScore).toFixed(1)}}%; background:${{artistMapColor(artist.name)}};\"></div></div>
        </div>
      `).join('') || '<div class=\"muted\">No active artists at this point.</div>';
    }}

    function renderArtistPhaseMap() {{
      if (!artistPhaseStage) {{
        return;
      }}
      const timestamp = artistStatsTimestamp();
      const caption = document.getElementById('artist-phase-caption');
      if (caption) {{
        caption.textContent = artistStatsTimeline.disabled
          ? 'No timestamped artist plays yet.'
          : `Through ${{formatDateTime(timestamp)}} · dot = latest play position inside each artist arc`;
      }}
      const artists = artistTimelineRows
        .filter((artist) => (artist.timestamps || []).length)
        .sort((left, right) => right.historyCount - left.historyCount || left.name.localeCompare(right.name))
        .slice(0, 18);
      artistPhaseStage.style.minHeight = `${{Math.max(260, artists.length * 30 + 24)}}px`;
      artistPhaseStage.innerHTML = artists.map((artist) => {{
        const timestamps = artist.timestamps || [];
        const progress = artistProgressAt(artist, timestamp);
        const recent = artistRecentCount(artist, timestamp, 7);
        const left = Math.max(2, Math.min(98, progress.progress * 100));
        const color = artistMapColor(artist.name);
        return `
          <div class=\"phase-row\">
            <button class=\"phase-name\" data-artist=\"${{escapeHtml(artist.name)}}\" title=\"${{escapeHtml(artist.name)}}\">${{escapeHtml(artist.name)}}</button>
            <div class=\"phase-track\">
              <div class=\"phase-trail\" style=\"width:${{left.toFixed(1)}}%; background:${{color}};\"></div>
              <button class=\"phase-spark\" data-artist=\"${{escapeHtml(artist.name)}}\" style=\"left:${{left.toFixed(1)}}%; background:${{color}}; opacity:${{recent ? 1 : 0.38}};\" title=\"${{escapeHtml(artist.name)}} · ${{formatNumber(recent)}} plays in last 7 days\"></button>
            </div>
          </div>
        `;
      }}).join('') || '<div class=\"muted\" style=\"padding:18px;\">No artist phase data yet.</div>';
      wireStatsStageSelection(artistPhaseStage);
    }}

    function renderListeningWeatherPlot() {{
      if (!artistWeatherStage) {{
        return;
      }}
      const timestamp = artistStatsTimestamp();
      const caption = document.getElementById('artist-weather-caption');
      if (caption) {{
        caption.textContent = artistStatsTimeline.disabled
          ? 'No timestamped artist plays yet.'
          : `Through ${{formatDateTime(timestamp)}} · 14-day listening weather`;
      }}
      const clouds = artistTimelineRows.map((artist) => {{
        const recent14 = artistRecentCount(artist, timestamp, 14);
        const recent3 = artistRecentCount(artist, timestamp, 3);
        return {{ ...artist, recent14, recent3 }};
      }}).filter((artist) => artist.recent14 > 0)
        .sort((left, right) => right.recent14 - left.recent14 || left.name.localeCompare(right.name))
        .slice(0, 22);
      const maxRecent = Math.max(1, ...clouds.map((artist) => artist.recent14));
      artistWeatherStage.innerHTML = clouds.map((artist, index) => {{
        const drift = ((weekStartTimestamp(timestamp) / secondsPerDay) + index * 17) % 100;
        const left = 8 + ((drift * 0.84) % 84);
        const top = 16 + ((index * 29 + artist.recent3 * 11) % 72);
        const width = 62 + 82 * Math.sqrt(artist.recent14 / maxRecent);
        const label = escapeHtml(artist.name);
        return `
          <button class=\"weather-cloud\" data-artist=\"${{label}}\" style=\"left:${{left.toFixed(2)}}%; top:${{top.toFixed(2)}}%; width:${{width.toFixed(1)}}px; border-color:${{artistMapColor(artist.name)}};\" title=\"${{label}} · ${{formatNumber(artist.recent14)}} plays in last 14 days\">${{indexSafeLabel(artist.name)}}</button>
        `;
      }}).join('') || '<div class=\"muted\" style=\"padding:18px;\">No recent artist weather at this point.</div>';
      wireStatsStageSelection(artistWeatherStage);
    }}

    function renderArtistEraRiverPlot() {{
      if (!artistRiverStage) {{
        return;
      }}
      const weekly = getArtistWeeklyData();
      const currentWeek = weekStartTimestamp(artistStatsTimestamp());
      const rows = weekly.rows.map((artist) => ({{
        ...artist,
        currentCount: artist.weekCounts.get(currentWeek) || 0,
      }})).filter((artist) => artist.currentCount > 0)
        .sort((left, right) => right.currentCount - left.currentCount || left.name.localeCompare(right.name))
        .slice(0, 12);
      const total = rows.reduce((sum, artist) => sum + artist.currentCount, 0) || 1;
      artistRiverStage.innerHTML = rows.map((artist, index) => {{
        const share = artist.currentCount / total;
        const width = 10 + share * 78;
        const top = 24 + index * 26;
        const label = escapeHtml(artist.name);
        return `
          <button class=\"river-band\" data-artist=\"${{label}}\" style=\"top:${{top}}px; width:${{width.toFixed(1)}}%; background:${{artistMapColor(artist.name)}}; opacity:${{Math.max(0.34, share + 0.34).toFixed(2)}};\" title=\"${{label}} · ${{(share * 100).toFixed(0)}}% of week\">
            <span class=\"river-label\">${{label}} · ${{(share * 100).toFixed(0)}}%</span>
          </button>
        `;
      }}).join('') || '<div class=\"muted\" style=\"padding:18px;\">No artist river for this week.</div>';
      wireStatsStageSelection(artistRiverStage);
    }}

    function renderRediscoveryTimeline() {{
      if (!artistRediscoveryStage) {{
        return;
      }}
      const timestamp = artistStatsTimestamp();
      const discoveries = [];
      for (const artist of artistTimelineRows) {{
        const timestamps = (artist.timestamps || []).filter((playedAt) => playedAt <= timestamp);
        if (timestamps.length < 2) {{
          continue;
        }}
        const last = timestamps[timestamps.length - 1];
        if (timestamp - last > 14 * secondsPerDay) {{
          continue;
        }}
        let bestGap = 0;
        for (let index = 1; index < timestamps.length; index += 1) {{
          bestGap = Math.max(bestGap, timestamps[index] - timestamps[index - 1]);
        }}
        if (bestGap >= 14 * secondsPerDay) {{
          discoveries.push({{ ...artist, bestGapDays: bestGap / secondsPerDay, last }});
        }}
      }}
      discoveries.sort((left, right) => right.bestGapDays - left.bestGapDays || left.name.localeCompare(right.name));
      artistRediscoveryStage.innerHTML = discoveries.slice(0, 14).map((artist, index) => {{
        const left = 10 + 80 * Math.min(1, artist.bestGapDays / 180);
        const top = 28 + (index % 7) * 44;
        const label = escapeHtml(artist.name);
        return `
          <button class=\"event-burst\" data-artist=\"${{label}}\" style=\"left:${{left.toFixed(2)}}%; top:${{top}}px; border-color:${{artistMapColor(artist.name)}};\" title=\"${{label}} · returned after ${{artist.bestGapDays.toFixed(0)}} days\">
            <strong>${{label}}</strong><span>${{artist.bestGapDays.toFixed(0)}}d</span>
          </button>
        `;
      }}).join('') || '<div class=\"muted\" style=\"padding:18px;\">No big rediscoveries near this point.</div>';
      wireStatsStageSelection(artistRediscoveryStage);
    }}

    function renderObsessionDetector() {{
      if (!artistObsessionStage) {{
        return;
      }}
      const timestamp = artistStatsTimestamp();
      const rows = artistTimelineRows.map((artist) => {{
        const currentWeek = artistRecentCount(artist, timestamp, 7);
        const baseline = artistWeeklyBaseline(artist);
        const lift = baseline > 0 ? currentWeek / baseline : 0;
        return {{ ...artist, currentWeek, baseline, lift }};
      }}).filter((artist) => artist.currentWeek >= 2 && artist.lift >= 1.4)
        .sort((left, right) => right.lift - left.lift || right.currentWeek - left.currentWeek || left.name.localeCompare(right.name))
        .slice(0, 16);
      const maxLift = Math.max(1, ...rows.map((artist) => artist.lift));
      artistObsessionStage.innerHTML = rows.map((artist, index) => {{
        const left = 12 + 78 * Math.min(1, artist.lift / maxLift);
        const top = 24 + (index % 8) * 40;
        const label = escapeHtml(artist.name);
        return `
          <button class=\"event-burst\" data-artist=\"${{label}}\" style=\"left:${{left.toFixed(2)}}%; top:${{top}}px; border-color:${{artistMapColor(artist.name)}};\" title=\"${{label}} · ${{artist.lift.toFixed(1)}}x normal week\">
            <strong>${{label}}</strong><span>${{artist.lift.toFixed(1)}}x</span>
          </button>
        `;
      }}).join('') || '<div class=\"muted\" style=\"padding:18px;\">No unusual artist spikes at this point.</div>';
      wireStatsStageSelection(artistObsessionStage);
    }}

    function songDisplayName(name) {{
      return String(name || '').split('/').pop();
    }}

    function selectedArtistSongs(node) {{
      return (node?.songDetails || []).slice().sort((left, right) =>
        Number(right.playCount || 0) - Number(left.playCount || 0) || String(left.name || '').localeCompare(String(right.name || ''))
      );
    }}

    function renderArtistCatalogCoverage(node) {{
      if (!artistCoverageStage) {{
        return;
      }}
      const caption = document.getElementById('artist-coverage-caption');
      if (!node) {{
        caption.textContent = 'When each song was active across your history.';
        artistCoverageStage.innerHTML = '<div class=\"muted\" style=\"padding:18px;\">Select an artist to see song lifespans.</div>';
        return;
      }}
      const allSongs = selectedArtistSongs(node).map((song) => {{
        const timestamps = decodeArtistHistory(song.history).sort((left, right) => left - right);
        return {{
          ...song,
          timestamps,
          first: timestamps[0] || null,
          last: timestamps[timestamps.length - 1] || null,
        }};
      }});
      const timestampedSongs = allSongs.filter((song) => song.first !== null);
      const songs = allSongs.sort((left, right) => {{
        if (left.first !== null && right.first !== null) {{
          return left.first - right.first || Number(right.playCount || 0) - Number(left.playCount || 0) || String(left.name || '').localeCompare(String(right.name || ''));
        }}
        if (left.first !== null) {{
          return -1;
        }}
        if (right.first !== null) {{
          return 1;
        }}
        return Number(right.playCount || 0) - Number(left.playCount || 0) || String(left.name || '').localeCompare(String(right.name || ''));
      }});
      if (!songs.length) {{
        caption.textContent = 'No songs tagged for this artist.';
        artistCoverageStage.innerHTML = '<div class=\"muted\" style=\"padding:18px;\">No tagged songs.</div>';
        return;
      }}
      if (!timestampedSongs.length) {{
        const bucketCount = 12;
        caption.textContent = `${{formatNumber(songs.length)}} songs tagged · none have timestamp history, so only listen-count coverage is available`;
        artistCoverageStage.innerHTML = `
          <div class=\"coverage-map\">
            <div class=\"coverage-map-head\">Song</div>
            <div class=\"coverage-map-head\">Activity over time</div>
            <div class=\"coverage-map-head\">Plays</div>
            <div class=\"coverage-map-head\">Last</div>
            ${{songs.map((song) => {{
              const display = songDisplayName(song.name);
              const cells = Array.from({{ length: bucketCount }}, () =>
                `<div class=\"coverage-cell is-missing\" title=\"${{escapeHtml(display)}} · no timestamp history\"></div>`
              ).join('');
              return `
                <div class=\"coverage-name\" title=\"${{escapeHtml(display)}} · no timestamp history\">${{escapeHtml(display)}}</div>
                <div class=\"coverage-cells\" style=\"grid-template-columns:repeat(${{bucketCount}}, minmax(5px, 1fr));\">${{cells}}</div>
                <div>${{formatNumber(song.playCount)}}</div>
                <div class=\"muted\">No dates</div>
              `;
            }}).join('')}}
          </div>`;
        return;
      }}
      const first = Math.min(...timestampedSongs.map((song) => song.first));
      const last = Math.max(...timestampedSongs.map((song) => song.last));
      const spanSeconds = Math.max(secondsPerDay, last - first);
      const bucketCount = Math.max(8, Math.min(36, Math.ceil(spanSeconds / (7 * secondsPerDay))));
      const bucketSeconds = spanSeconds / bucketCount;
      const rows = songs.map((song) => {{
        const buckets = Array.from({{ length: bucketCount }}, () => 0);
        for (const timestamp of song.timestamps) {{
          const bucketIndex = Math.max(0, Math.min(bucketCount - 1, Math.floor((timestamp - first) / bucketSeconds)));
          buckets[bucketIndex] += 1;
        }}
        return {{ ...song, buckets, maxBucket: Math.max(1, ...buckets) }};
      }});
      const globalMaxBucket = Math.max(1, ...rows.flatMap((song) => song.buckets));
      const stickyCount = rows.filter((song) => song.last !== null && song.last >= last - 30 * secondsPerDay).length;
      const untimestampedCount = rows.length - timestampedSongs.length;
      caption.textContent = `${{formatNumber(rows.length)}} songs · ${{formatNumber(timestampedSongs.length)}} with timestamp history · ${{formatNumber(stickyCount)}} active in the last 30 days of this artist span${{untimestampedCount ? ` · ${{formatNumber(untimestampedCount)}} count-only` : ''}}`;
      artistCoverageStage.innerHTML = `
        <div class=\"coverage-map\">
          <div class=\"coverage-map-head\">Song</div>
          <div class=\"coverage-map-head\">Activity over time</div>
          <div class=\"coverage-map-head\">Plays</div>
          <div class=\"coverage-map-head\">Last</div>
          <div></div>
          <div class=\"coverage-axis\"><span>${{shortDateLabel(localDayKey(first))}}</span><span>${{shortDateLabel(localDayKey(first + spanSeconds / 2))}}</span><span>${{shortDateLabel(localDayKey(last))}}</span></div>
          <div></div>
          <div></div>
          ${{rows.map((song) => {{
        const display = songDisplayName(song.name);
        const cells = song.buckets.map((count, index) => {{
          const bucketStart = first + index * bucketSeconds;
          const bucketEnd = Math.min(last, bucketStart + bucketSeconds);
          if (song.first === null) {{
            return `<div class=\"coverage-cell is-missing\" title=\"${{escapeHtml(display)}} · ${{formatNumber(song.playCount)}} plays counted · no timestamp history\"></div>`;
          }}
          const strength = count ? Math.max(0.18, count / globalMaxBucket) : 0;
          const color = count ? artistMapColor(display) : '#e7eef4';
          return `<div class=\"coverage-cell${{count ? ' is-active' : ''}}\" style=\"background:${{color}}; opacity:${{count ? Math.min(1, 0.28 + strength * 0.72).toFixed(2) : '1'}};\" title=\"${{escapeHtml(display)}} · ${{formatNumber(count)}} plays · ${{shortDateLabel(localDayKey(bucketStart))}} to ${{shortDateLabel(localDayKey(bucketEnd))}}\"></div>`;
        }}).join('');
        return `
          <div class=\"coverage-name\" title=\"${{escapeHtml(display)}}${{song.first === null ? ' · no timestamp history' : ` · first ${{formatDateTime(song.first)}}`}}\">${{escapeHtml(display)}}</div>
          <div class=\"coverage-cells\" style=\"grid-template-columns:repeat(${{bucketCount}}, minmax(5px, 1fr));\">${{cells}}</div>
          <div>${{formatNumber(song.playCount)}}</div>
          <div class=\"muted\">${{song.last === null ? 'No dates' : shortDateLabel(localDayKey(song.last))}}</div>
        `;
      }}).join('')}}</div>`;
    }}

    function renderArtistSongGravityPlot(node) {{
      if (!artistSongGravityStage) {{
        return;
      }}
      const caption = document.getElementById('artist-song-gravity-caption');
      if (!node) {{
        caption.textContent = 'Select an artist; songs move closer when they are active.';
        artistSongGravityStage.innerHTML = '<div class=\"muted\" style=\"padding:18px;\">Select an artist in any artist chart.</div>';
        return;
      }}
      const timestamp = artistStatsTimestamp();
      caption.textContent = `Selected: ${{node.label}} · through ${{formatDateTime(timestamp)}}`;
      const songs = (node.songDetails || []).slice(0, 26).map((song) => {{
        const timestamps = decodeArtistHistory(song.history);
        const recent = countInWindow(timestamps, timestamp - 14 * secondsPerDay, timestamp);
        const total = Number(song.playCount || 0);
        const score = recent + Math.sqrt(total) * 0.18;
        return {{ ...song, timestamps, recent, score }};
      }}).filter((song) => Number(song.playCount || 0) > 0);
      const maxScore = Math.max(1, ...songs.map((song) => song.score));
      const maxPlays = Math.max(1, ...songs.map((song) => Number(song.playCount || 0)));
      artistSongGravityStage.innerHTML = `<button class=\"moving-point\" style=\"left:50%; top:50%; width:58px; height:58px; background:#ffffff; border-color:${{artistMapColor(node.label)}};\">${{indexSafeLabel(node.label)}}</button>` + songs.map((song, index) => {{
        const strength = Math.max(0, Math.min(1, song.score / maxScore));
        const radius = 10 + (1 - strength) * 38 + (index % 3) * 4;
        const angle = (index * 137.508 + Number(artistStatsTimeline.value || 0) * 0.11) * Math.PI / 180;
        const left = 50 + Math.cos(angle) * radius;
        const top = 50 + Math.sin(angle) * radius * 0.72;
        const size = 22 + 24 * Math.sqrt(Number(song.playCount || 0) / maxPlays);
        const display = songDisplayName(song.name);
        return `
          <button class=\"moving-point\" style=\"left:${{left.toFixed(2)}}%; top:${{top.toFixed(2)}}%; width:${{size.toFixed(1)}}px; height:${{size.toFixed(1)}}px; background:${{artistMapColor(display)}};\" title=\"${{escapeHtml(display)}} · ${{formatNumber(song.recent)}} recent plays · ${{formatNumber(song.playCount)}} total\">${{indexSafeLabel(display)}}</button>
          <div class=\"moving-label\" style=\"left:${{left.toFixed(2)}}%; top:${{top.toFixed(2)}}%;\">${{escapeHtml(display)}}</div>
        `;
      }}).join('');
    }}

    function stopArtistStatsPlayback() {{
      artistStatsIsPlaying = false;
      artistStatsPlay.textContent = '▶';
      artistStatsPlay.title = 'Play artist timeline';
      artistStatsLastFrame = null;
      if (artistStatsAnimationFrame !== null) {{
        cancelAnimationFrame(artistStatsAnimationFrame);
        artistStatsAnimationFrame = null;
      }}
    }}

    function stepArtistStatsPlayback(frameTimestamp) {{
      if (!artistStatsIsPlaying || artistStatsTimeline.disabled) {{
        return;
      }}
      if (artistStatsLastFrame === null) {{
        artistStatsLastFrame = frameTimestamp;
      }}
      const elapsedMs = Math.max(0, frameTimestamp - artistStatsLastFrame);
      artistStatsLastFrame = frameTimestamp;
      const current = Number(artistStatsTimeline.value) || 0;
      const sliderMax = Number(artistStatsTimeline.max) || 1000;
      if (current >= sliderMax) {{
        stopArtistStatsPlayback();
        return;
      }}
      const secondsPerTimelineDay = Math.max(0.1, Number(artistStatsSpeed.value) || 0.5);
      const timelineSpanSeconds = Math.max(1, artistLastTimestamp - artistFirstTimestamp);
      const advancedTimelineSeconds = elapsedMs * secondsPerDay / (secondsPerTimelineDay * 1000);
      const sliderDelta = sliderMax * advancedTimelineSeconds / timelineSpanSeconds;
      artistStatsTimeline.value = Math.min(sliderMax, current + sliderDelta);
      renderArtistStatsDashboard();
      artistStatsAnimationFrame = requestAnimationFrame(stepArtistStatsPlayback);
    }}

    function toggleArtistStatsPlayback() {{
      if (artistStatsIsPlaying) {{
        stopArtistStatsPlayback();
        return;
      }}
      if (artistStatsTimeline.disabled) {{
        return;
      }}
      const sliderMax = Number(artistStatsTimeline.max) || 1000;
      if (Number(artistStatsTimeline.value) >= sliderMax) {{
        artistStatsTimeline.value = 0;
      }}
      artistStatsIsPlaying = true;
      artistStatsLastFrame = null;
      artistStatsPlay.textContent = '❚❚';
      artistStatsPlay.title = 'Pause artist timeline';
      artistStatsAnimationFrame = requestAnimationFrame(stepArtistStatsPlayback);
    }}

    function showArtistNodeDetails(nodeId) {{
      if (!nodeId) {{
        document.getElementById('artist-selection-summary').textContent = 'Select an artist, song, or edge.';
        document.getElementById('artist-selection-content').innerHTML = '';
        return;
      }}
      const graph = getArtistGraphData();
      const node = (graph.nodes || []).find((item) => item.id === nodeId);
      if (!node) {{
        return;
      }}
      if (node.kind !== 'song') {{
        showArtistInspector(node.label);
      }}
      document.getElementById('artist-selection-summary').innerHTML = `<strong>${{escapeHtml(node.label)}}</strong>`;
      const songs = (node.songs || []).slice(0, 12).map((song) => `<div class=\"item\"><strong>${{escapeHtml(song)}}</strong></div>`).join('');
      const topSongLine = node.topSong
        ? `<div class=\"item\"><strong>Top song</strong><div class=\"muted\">${{escapeHtml(node.topSong)}} · ${{formatNumber(node.topSongPlays)}} plays · ${{(Number(node.topSongShare || 0) * 100).toFixed(0)}}% of artist plays</div></div>`
        : '';
      const timeWindowLine = node.firstListen || node.lastListen
        ? `<div class=\"item\"><strong>Listening window</strong><div class=\"muted\">${{formatDateTime(node.firstListen)}} → ${{formatDateTime(node.lastListen)}} · ${{formatNumber(node.activeDayCount)}} active days · ${{Number(node.activeSpanDays || 0).toFixed(1)}} day span</div></div>`
        : '';
      const peakLine = node.peakDay
        ? `<div class=\"item\"><strong>Peak day</strong><div class=\"muted\">${{escapeHtml(node.peakDay)}} · ${{formatNumber(node.peakDayCount)}} plays</div></div>`
        : '';
      const gapLine = node.longestGapDays !== null && node.longestGapDays !== undefined
        ? `<div class=\"item\"><strong>Gaps</strong><div class=\"muted\">avg ${{Number(node.averageGapDays || 0).toFixed(2)}} days · longest ${{Number(node.longestGapDays || 0).toFixed(2)}} days</div></div>`
        : '';
      document.getElementById('artist-selection-content').innerHTML = `
        <div class=\"metric-grid\">
          <div class=\"metric\"><div class=\"label\">Plays</div><div class=\"value\">${{formatNumber(node.playCount)}}</div></div>
          <div class=\"metric\"><div class=\"label\">Songs</div><div class=\"value\">${{formatNumber(node.songCount || 1)}}</div></div>
          <div class=\"metric\"><div class=\"label\">Listen time</div><div class=\"value\">${{formatListenTime(node.listenTimeSeconds)}}</div></div>
          <div class=\"metric\"><div class=\"label\">History</div><div class=\"value\">${{formatNumber(node.historyCount)}}</div></div>
          <div class=\"metric\"><div class=\"label\">Plays/song</div><div class=\"value\">${{Number(node.playsPerSong || 0).toFixed(1)}}</div></div>
          <div class=\"metric\"><div class=\"label\">Last 30d</div><div class=\"value\">${{formatNumber(node.recent30PlayCount)}}</div></div>
        </div>
        <div class=\"list\">${{topSongLine}}${{timeWindowLine}}${{peakLine}}${{gapLine}}</div>
        ${{songs ? `<h4>Tagged songs</h4><div class=\"list\">${{songs}}</div>` : ''}}
      `;
    }}

    function showArtistEdgeDetails(edgeId) {{
      const graph = getArtistGraphData();
      const edge = (graph.edges || []).find((item) => item.id === edgeId);
      if (!edge) {{
        return;
      }}
      document.getElementById('artist-selection-summary').innerHTML = `<strong>${{escapeHtml(edge.from)}} → ${{escapeHtml(edge.to)}}</strong>`;
      document.getElementById('artist-selection-content').innerHTML = `
        <div class=\"metric-grid\">
          <div class=\"metric\"><div class=\"label\">Count</div><div class=\"value\">${{formatNumber(edge.count)}}</div></div>
          <div class=\"metric\"><div class=\"label\">Probability</div><div class=\"value\">${{edge.probability ? (edge.probability * 100).toFixed(1) + '%' : '-'}}</div></div>
          <div class=\"metric\"><div class=\"label\">Avg gap</div><div class=\"value\">${{edge.averageGapSeconds ? (edge.averageGapSeconds / 60).toFixed(1) + 'm' : '-'}}</div></div>
        </div>
      `;
    }}

    function focusArtist() {{
      const query = document.getElementById('artist-search').value.trim().toLowerCase();
      if (!query || !artistNetwork) {{
        return;
      }}
      const graph = getArtistGraphData();
      const match = (graph.nodes || []).find((node) => String(node.label || '').toLowerCase().includes(query));
      if (!match) {{
        document.getElementById('artist-selection-summary').textContent = `No artist/song matched "${{query}}".`;
        document.getElementById('artist-selection-content').innerHTML = '';
        return;
      }}
      artistNetwork.selectNodes([match.id]);
      artistNetwork.focus(match.id, {{ scale: 1.1, animation: {{ duration: 500, easingFunction: 'easeInOutQuad' }} }});
      showArtistNodeDetails(match.id);
    }}

    function describePairingMode(pairingMode) {{
      if (pairingMode === 'consecutive') {{
        return 'immediate next song only';
      }}
      if (pairingMode === 'window') {{
        return 'all later songs inside the time window';
      }}
      return pairingMode;
    }}

    function getViewLabel(viewKey) {{
      const view = (liveConfig.views || []).find((item) => item.key === viewKey);
      return view ? view.label : viewKey;
    }}

    function graphDataUrl(viewKey) {{
      return `/graph-data.json?view=${{encodeURIComponent(viewKey)}}`;
    }}

    function setActiveGraphTab() {{
      document.querySelectorAll('#graph-tabs button').forEach((button) => {{
        button.classList.toggle('active', button.dataset.viewKey === activeViewKey);
      }});
    }}

    async function loadGraphView(viewKey) {{
      if (!liveConfig.enabled || viewKey === activeViewKey) {{
        return;
      }}
      setLiveStatus(`Live: loading ${{getViewLabel(viewKey)}}...`);
      try {{
        const response = await fetch(graphDataUrl(viewKey), {{ cache: 'no-store' }});
        if (!response.ok) {{
          throw new Error(`HTTP ${{response.status}}`);
        }}
        const nextPayload = await response.json();
        activeViewKey = nextPayload.view || viewKey;
        currentGraphHash = nextPayload.hash || null;
        graphHashesByView.set(activeViewKey, currentGraphHash);
        setActiveGraphTab();
        applyGraphData(nextPayload.graph, {{ clearSelection: true, rebuild: true }});
        applyTrendingData(nextPayload.trending);
        applyArtistData(nextPayload.artists);
        setLiveStatus(`Live: showing ${{getViewLabel(activeViewKey)}} after ${{graphData.summary.eventCount}} play events.`);
      }} catch (error) {{
        setLiveStatus(`Could not load ${{getViewLabel(viewKey)}}: ${{error.message}}`, true);
      }}
    }}

    function renderGraphTabs() {{
      const tabs = document.getElementById('graph-tabs');
      const views = liveConfig.views || [];
      if (!tabs || !views.length) {{
        return;
      }}
      tabs.innerHTML = views.map((view) => `
        <button type=\"button\" data-view-key=\"${{view.key}}\">${{view.label}}</button>
      `).join('');
      tabs.classList.add('visible');
      tabs.querySelectorAll('button').forEach((button) => {{
        button.addEventListener('click', () => loadGraphView(button.dataset.viewKey));
      }});
      setActiveGraphTab();
    }}

    function setAppMode(mode) {{
      const isTrending = mode === 'trending';
      const isArtist = mode === 'artist';
      const isStats = mode === 'stats';
      const isGraph = !isTrending && !isArtist && !isStats;
      document.body.classList.toggle('trending-mode', isTrending);
      document.body.classList.toggle('artist-mode', isArtist);
      document.body.classList.toggle('stats-mode', isStats);
      document.body.classList.toggle('graph-mode', isGraph);
      document.getElementById('graph-mode-button').classList.toggle('active', isGraph);
      document.getElementById('trending-mode-button').classList.toggle('active', isTrending);
      document.getElementById('artist-mode-button').classList.toggle('active', isArtist);
      document.getElementById('stats-mode-button').classList.toggle('active', isStats);
      if (isGraph && network) {{
        stopTrendingPlayback();
        stopArtistStatsPlayback();
        window.setTimeout(() => {{
          network.redraw();
        }}, 0);
      }}
      if (isTrending) {{
        stopArtistStatsPlayback();
        renderTrending();
      }}
      if (isArtist && artistNetwork) {{
        stopTrendingPlayback();
        stopArtistStatsPlayback();
        window.setTimeout(() => {{
          artistNetwork.redraw();
          artistNetwork.fit({{ animation: {{ duration: 250, easingFunction: 'easeInOutQuad' }} }});
        }}, 0);
      }}
      if (isStats) {{
        stopTrendingPlayback();
        renderArtistStatsDashboard();
      }}
    }}

    function rebuildLookups() {{
      nodeLookup = new Map(allNodes.map((node) => [node.id, node]));
      edgeLookup = new Map(allEdges.map((edge) => [edge.id, edge]));
    }}

    function updateSummary() {{
      const maxEdgeCount = Math.max(1, Number(graphData.summary.maxEdgeCount || 1));
      const countFilter = document.getElementById('count-filter');
      countFilter.max = String(maxEdgeCount);
      if (Number(countFilter.value) > maxEdgeCount) {{
        countFilter.value = String(maxEdgeCount);
        document.getElementById('count-filter-value').textContent = countFilter.value;
      }}

      document.getElementById('summary-song-count').textContent = graphData.summary.songCount;
      document.getElementById('summary-event-count').textContent = graphData.summary.eventCount;
      document.getElementById('summary-edge-count').textContent = graphData.summary.edgeCount;
      document.getElementById('summary-window').textContent = `${{graphData.summary.windowMinutes}}m`;
      document.getElementById('summary-folder').textContent = graphData.summary.folder;
      document.getElementById('summary-pairing').textContent =
        `Pairing: ${{graphData.summary.pairingMode}} (${{describePairingMode(graphData.summary.pairingMode)}})`;
      refreshSizingSummary();
    }}

    function setLiveStatus(message, isError = false) {{
      const liveStatus = document.getElementById('live-status');
      if (!liveStatus) {{
        return;
      }}
      if (!liveConfig.enabled) {{
        liveStatus.style.display = 'none';
        return;
      }}
      liveStatus.style.display = 'block';
      liveStatus.style.color = isError ? '#b42318' : '';
      liveStatus.textContent = message;
    }}

    function applyGraphData(nextGraphData, options = {{}}) {{
      graphData = nextGraphData;
      allNodes = graphData.nodes || [];
      allEdges = graphData.edges || [];
      rebuildLookups();

      if (options.clearSelection || (selectedNodeId && !nodeLookup.has(selectedNodeId))) {{
        selectedNodeId = null;
        if (network) {{
          network.unselectAll();
        }}
        document.getElementById('selection-summary').textContent = 'Select a node or edge to inspect it.';
        document.getElementById('selection-content').innerHTML = '';
      }}

      updateSummary();
      if (options.rebuild) {{
        createNetwork();
      }} else {{
        refreshGraph({{ relayout: Boolean(options.relayout) }});
      }}

      if (selectedNodeId) {{
        network.selectNodes([selectedNodeId]);
        showNodeDetails(selectedNodeId);
      }}
    }}

    function getFilteredEdges() {{
      const minimumCount = Number(document.getElementById('count-filter').value);
      const minimumProbability = Number(document.getElementById('probability-filter').value) / 100;
      const showLabels = document.getElementById('labels-toggle').checked;
      const neighborhoodOnly = document.getElementById('neighborhood-toggle').checked;

      let filteredEdges = allEdges.filter((edge) => edge.count >= minimumCount && edge.probability >= minimumProbability);

      if (neighborhoodOnly && selectedNodeId) {{
        filteredEdges = filteredEdges.filter((edge) => edge.from === selectedNodeId || edge.to === selectedNodeId);
      }}

      return filteredEdges.map((edge) => ({{
        ...edge,
        label: showLabels ? edge.label : '',
      }}));
    }}

    function reconcileDataSet(dataSet, nextItems) {{
      const nextIds = new Set(nextItems.map((item) => item.id));
      const idsToRemove = dataSet.getIds().filter((id) => !nextIds.has(id));
      if (idsToRemove.length) {{
        dataSet.remove(idsToRemove);
      }}
      if (nextItems.length) {{
        dataSet.update(nextItems);
      }}
    }}

    function rememberNodePositions(nodeIds = null) {{
      if (!network) {{
        return;
      }}
      const ids = nodeIds || nodeDataSet.getIds();
      if (!ids.length) {{
        return;
      }}
      const positions = network.getPositions(ids);
      ids.forEach((nodeId) => {{
        const position = positions[nodeId];
        if (position && Number.isFinite(position.x) && Number.isFinite(position.y)) {{
          nodePositions.set(nodeId, {{ x: position.x, y: position.y }});
        }}
      }});
    }}

    function updateNodeAppearance(nodeId) {{
      const node = nodeLookup.get(nodeId);
      if (!node || !nodeDataSet.get(nodeId)) {{
        return;
      }}
      const visibleNode = buildVisibleNode(node);
      nodeDataSet.update({{
        id: nodeId,
        color: visibleNode.color,
        size: visibleNode.size,
      }});
    }}

    function updateSelectedNodeStyles(previousNodeId, nextNodeId) {{
      new Set([previousNodeId, nextNodeId].filter(Boolean)).forEach(updateNodeAppearance);
    }}

    function settleGraphLayout() {{
      if (!network) {{
        return;
      }}
      network.setOptions({{
        physics: {{
          enabled: true,
          stabilization: {{ iterations: 120, updateInterval: 25 }},
          timestep: 0.3,
          minVelocity: 1.05,
          barnesHut: {{ gravitationalConstant: -2300, springLength: 135, springConstant: 0.011, damping: 0.78, avoidOverlap: 0.05 }}
        }}
      }});
      network.once('stabilized', () => {{
        rememberNodePositions();
      }});
      network.stabilize(120);
    }}

    function attachNetworkEvents() {{
      network.on('selectNode', (params) => {{
        const previousNodeId = selectedNodeId;
        selectedNodeId = params.nodes[0] || null;
        if (document.getElementById('neighborhood-toggle').checked) {{
          refreshGraph();
        }} else {{
          updateSelectedNodeStyles(previousNodeId, selectedNodeId);
        }}
        if (selectedNodeId) {{
          showNodeDetails(selectedNodeId);
        }}
      }});

      network.on('deselectNode', () => {{
        const previousNodeId = selectedNodeId;
        selectedNodeId = null;
        if (document.getElementById('neighborhood-toggle').checked) {{
          refreshGraph();
        }} else {{
          updateSelectedNodeStyles(previousNodeId, null);
        }}
        document.getElementById('selection-summary').textContent = 'Select a node or edge to inspect it.';
        document.getElementById('selection-content').innerHTML = '';
      }});

      network.on('selectEdge', (params) => {{
        const edgeId = params.edges[0];
        if (edgeId) {{
          showEdgeDetails(edgeId);
        }}
      }});

      network.on('dragStart', () => {{
        graphWasManuallyMoved = true;
        network.setOptions({{
          physics: {{
            enabled: true,
            timestep: 0.3,
            minVelocity: 0.8,
            barnesHut: {{ gravitationalConstant: -2300, springLength: 135, springConstant: 0.015, damping: 0.72, avoidOverlap: 0.05 }}
          }}
        }});
      }});

      network.on('dragEnd', (params) => {{
        rememberNodePositions(params.nodes);
        network.setOptions({{
          physics: {{
            enabled: true,
            timestep: 0.28,
            minVelocity: 1.25,
            barnesHut: {{ gravitationalConstant: -2300, springLength: 135, springConstant: 0.0075, damping: 0.84, avoidOverlap: 0.05 }}
          }}
        }});
      }});

      network.on('zoom', () => {{
        graphWasManuallyMoved = true;
      }});
    }}

    function createNetwork() {{
      const shouldKeepManualView = graphWasManuallyMoved || nodePositions.size > 0;
      if (network) {{
        rememberNodePositions();
        network.destroy();
      }}
      nodeDataSet = new vis.DataSet([]);
      edgeDataSet = new vis.DataSet([]);
      graphWasManuallyMoved = shouldKeepManualView;
      refreshGraph();
      network = new vis.Network(container, {{ nodes: nodeDataSet, edges: edgeDataSet }}, networkOptions);
      attachNetworkEvents();
      network.once('stabilized', () => {{
        rememberNodePositions();
        if (!graphWasManuallyMoved) {{
          network.fit({{ animation: {{ duration: 350, easingFunction: 'easeInOutQuad' }} }});
        }}
      }});
      if (networkStatus) {{
        networkStatus.classList.remove('visible');
      }}
    }}

    function refreshGraph(options = {{}}) {{
      rememberNodePositions();
      const filteredEdges = getFilteredEdges();
      const connectedNodeIds = new Set();
      filteredEdges.forEach((edge) => {{
        connectedNodeIds.add(edge.from);
        connectedNodeIds.add(edge.to);
      }});

      const neighborhoodOnly = document.getElementById('neighborhood-toggle').checked;
      const visibleNodes = allNodes.map((node) => ({{
        ...buildVisibleNode(node),
        hidden: neighborhoodOnly && selectedNodeId && !connectedNodeIds.has(node.id) && node.id !== selectedNodeId,
      }}));
      reconcileDataSet(nodeDataSet, visibleNodes);
      reconcileDataSet(edgeDataSet, filteredEdges);
      if (options.relayout) {{
        settleGraphLayout();
      }}
    }}

    function renderTopEdgeList(edges, heading) {{
      if (!edges.length) {{
        return `<div class=\"muted\">No ${{heading.toLowerCase()}} above the current filters.</div>`;
      }}
      const items = edges.slice(0, 10).map((edge) => `
        <div class=\"item\">
          <strong>${{edge.from}} → ${{edge.to}}</strong>
          <div class=\"muted\">Count: ${{edge.count}} · Probability: ${{formatPercent(edge.probability)}} · Avg gap: ${{formatMinutes(edge.averageGapSeconds)}}</div>
        </div>
      `).join('');
      return `<h4>${{heading}}</h4><div class=\"list\">${{items}}</div>`;
    }}

    function showNodeDetails(nodeId) {{
      const node = nodeLookup.get(nodeId);
      if (!node) {{
        return;
      }}
      const outgoingEdges = allEdges
        .filter((edge) => edge.from === nodeId)
        .sort((left, right) => right.count - left.count || left.to.localeCompare(right.to));
      const incomingEdges = allEdges
        .filter((edge) => edge.to === nodeId)
        .sort((left, right) => right.count - left.count || left.from.localeCompare(right.from));

      document.getElementById('selection-summary').innerHTML = `<strong>${{node.label}}</strong>`;
      document.getElementById('selection-content').innerHTML = `
        <div class=\"metric-grid\">
          <div class=\"metric\"><div class=\"label\">Play count</div><div class=\"value\">${{node.playCount}}</div></div>
          <div class=\"metric\"><div class=\"label\">Duration</div><div class=\"value\">${{formatDuration(node.durationSeconds)}}</div></div>
          <div class=\"metric\"><div class=\"label\">Listen time</div><div class=\"value\">${{formatListenTime(node.listenTimeSeconds)}}</div></div>
          <div class=\"metric\"><div class=\"label\">History entries</div><div class=\"value\">${{node.historyCount}}</div></div>
          <div class=\"metric\"><div class=\"label\">Observed outgoing</div><div class=\"value\">${{node.outgoingObservedCount}}</div></div>
          <div class=\"metric\"><div class=\"label\">Observed incoming</div><div class=\"value\">${{node.incomingObservedCount}}</div></div>
          <div class=\"metric\"><div class=\"label\">Observed neighbors</div><div class=\"value\">${{node.outgoingNeighborCount}}</div></div>
          <div class=\"metric\"><div class=\"label\">Fallback targets</div><div class=\"value\">${{node.missingNeighborCount}}</div></div>
        </div>
        ${{renderTopEdgeList(outgoingEdges, 'Top outgoing edges')}}
        ${{renderTopEdgeList(incomingEdges, 'Top incoming edges')}}
      `;
    }}

    function showEdgeDetails(edgeId) {{
      const edge = edgeLookup.get(edgeId);
      if (!edge) {{
        return;
      }}
      document.getElementById('selection-summary').innerHTML = `<strong>${{edge.from}} → ${{edge.to}}</strong>`;
      document.getElementById('selection-content').innerHTML = `
        <div class=\"metric-grid\">
          <div class=\"metric\"><div class=\"label\">Count</div><div class=\"value\">${{edge.count}}</div></div>
          <div class=\"metric\"><div class=\"label\">Probability</div><div class=\"value\">${{(edge.probability * 100).toFixed(1)}}%</div></div>
          <div class=\"metric\"><div class=\"label\">Average gap</div><div class=\"value\">${{(edge.averageGapSeconds / 60).toFixed(1)}}m</div></div>
        </div>
      `;
    }}

    function focusSong() {{
      const query = document.getElementById('song-search').value.trim().toLowerCase();
      if (!query) {{
        return;
      }}
      const match = allNodes.find((node) => node.label.toLowerCase().includes(query));
      if (!match) {{
        document.getElementById('selection-summary').textContent = `No song matched "${{query}}".`;
        document.getElementById('selection-content').innerHTML = '';
        return;
      }}
      const previousNodeId = selectedNodeId;
      selectedNodeId = match.id;
      if (document.getElementById('neighborhood-toggle').checked) {{
        refreshGraph();
      }} else {{
        updateSelectedNodeStyles(previousNodeId, selectedNodeId);
      }}
      network.selectNodes([match.id]);
      network.focus(match.id, {{ scale: 1.1, animation: {{ duration: 500, easingFunction: 'easeInOutQuad' }} }});
      showNodeDetails(match.id);
    }}

    document.getElementById('count-filter').addEventListener('input', (event) => {{
      document.getElementById('count-filter-value').textContent = event.target.value;
      refreshGraph();
    }});
    document.getElementById('probability-filter').addEventListener('input', (event) => {{
      document.getElementById('probability-filter-value').textContent = `${{Number(event.target.value).toFixed(1)}}%`;
      refreshGraph();
    }});
    document.getElementById('labels-toggle').addEventListener('change', refreshGraph);
    document.getElementById('neighborhood-toggle').addEventListener('change', refreshGraph);
    listenTimeSizeToggle.addEventListener('change', () => {{
      refreshSizingSummary();
      refreshGraph();
    }});
    document.getElementById('focus-button').addEventListener('click', focusSong);
    document.getElementById('song-search').addEventListener('keydown', (event) => {{
      if (event.key === 'Enter') {{
        focusSong();
      }}
    }});
    document.getElementById('graph-mode-button').addEventListener('click', () => setAppMode('graph'));
    document.getElementById('trending-mode-button').addEventListener('click', () => setAppMode('trending'));
    document.getElementById('artist-mode-button').addEventListener('click', () => setAppMode('artist'));
    document.getElementById('stats-mode-button').addEventListener('click', () => setAppMode('stats'));
    document.getElementById('artist-transition-button').addEventListener('click', () => {{
      artistMode = 'transitions';
      document.getElementById('artist-transition-button').classList.add('active');
      document.getElementById('artist-song-button').classList.remove('active');
      createArtistNetwork();
    }});
    document.getElementById('artist-song-button').addEventListener('click', () => {{
      artistMode = 'songs';
      document.getElementById('artist-song-button').classList.add('active');
      document.getElementById('artist-transition-button').classList.remove('active');
      createArtistNetwork();
    }});
    document.getElementById('artist-focus-button').addEventListener('click', focusArtist);
    document.getElementById('artist-search').addEventListener('keydown', (event) => {{
      if (event.key === 'Enter') {{
        focusArtist();
      }}
    }});
    artistTimeline.addEventListener('input', renderArtistTimeline);
    artistStatsTimeline.addEventListener('input', () => {{
      stopArtistStatsPlayback();
      renderArtistStatsDashboard();
    }});
    artistStatsPlay.addEventListener('click', toggleArtistStatsPlayback);
    artistStatsSpeed.addEventListener('change', () => {{
      if (artistStatsIsPlaying) {{
        artistStatsLastFrame = null;
      }}
    }});
    artistInspectorSelect.addEventListener('change', () => {{
      showArtistInspector(artistInspectorSelect.value);
    }});
    trendingTopLimit.addEventListener('input', renderTrending);
    trendingTimeline.addEventListener('input', () => {{
      stopTrendingPlayback();
      renderTrending();
    }});
    trendingPlayToggle.addEventListener('click', toggleTrendingPlayback);
    trendingPlaySpeed.addEventListener('change', () => {{
      if (trendingIsPlaying) {{
        trendingLastPlaybackFrame = null;
      }}
    }});

    async function pollGraphData() {{
      if (!liveConfig.enabled) {{
        return;
      }}
      try {{
        const response = await fetch(graphDataUrl(activeViewKey), {{ cache: 'no-store' }});
        if (!response.ok) {{
          throw new Error(`HTTP ${{response.status}}`);
        }}
        const nextPayload = await response.json();
        if (nextPayload.hash && nextPayload.hash !== currentGraphHash) {{
          currentGraphHash = nextPayload.hash;
          graphHashesByView.set(activeViewKey, currentGraphHash);
          applyGraphData(nextPayload.graph);
          applyTrendingData(nextPayload.trending);
          applyArtistData(nextPayload.artists);
          setLiveStatus(`Live: updated ${{getViewLabel(activeViewKey)}} after ${{graphData.summary.eventCount}} play events.`);
        }} else {{
          setLiveStatus(`Live: watching ${{getViewLabel(activeViewKey)}} every ${{liveConfig.refreshSeconds}}s.`);
        }}
      }} catch (error) {{
        setLiveStatus(`Live update failed: ${{error.message}}`, true);
      }}
    }}

    function startLiveUpdates() {{
      if (!liveConfig.enabled) {{
        return;
      }}
      const refreshMilliseconds = Math.max(250, Number(liveConfig.refreshSeconds || 2) * 1000);
      setLiveStatus(`Live: watching ${{getViewLabel(activeViewKey)}} every ${{liveConfig.refreshSeconds}}s.`);
      window.setInterval(pollGraphData, refreshMilliseconds);
    }}

    renderGraphTabs();
    updateSummary();
    applyTrendingData(trendingData);
    applyArtistData(artistData);
    createNetwork();
    startLiveUpdates();
  </script>
</body>
</html>
"""


def make_live_graph_handler(
    args: argparse.Namespace,
    repo_root: Path,
    folder: Path,
    use_all_folders: bool,
    fallback_probability: float,
):
    initial_view_key = initial_graph_view_key(folder, use_all_folders)
    view_keys = graph_view_keys()
    view_definitions = {
        key: graph_view_for_key(repo_root, key)
        for key in view_keys
    }
    view_source_paths = {
        key: source_paths_for_folder(
            repo_root,
            view_definition["folder"],
            view_definition["use_all_folders"],
        )
        for key, view_definition in view_definitions.items()
    }
    cached_responses = {
        key: {
            "signature": None,
            "combined_payload": None,
            "meta_path": None,
            "payload_hash": None,
        }
        for key in view_keys
    }

    def get_graph_response(view_key: str):
        if view_key not in view_keys:
            raise KeyError(view_key)

        view_definition = view_definitions[view_key]
        cached_response = cached_responses[view_key]
        source_paths = view_source_paths[view_key]
        current_signature = source_signature_for_paths(source_paths)
        if (
            cached_response["signature"] == current_signature
            and cached_response["combined_payload"] is not None
        ):
            return (
                cached_response["combined_payload"],
                cached_response["meta_path"],
                cached_response["payload_hash"],
            )

        combined_payload, meta_path = build_combined_payload_from_options(
            args=args,
            repo_root=repo_root,
            folder=view_definition["folder"],
            use_all_folders=view_definition["use_all_folders"],
            fallback_probability=fallback_probability,
            view_key=view_definition["key"],
            view_label=view_definition["label"],
        )
        payload_hash = hash_graph_payload(combined_payload)
        cached_response.update(
            {
                "signature": current_signature,
                "combined_payload": combined_payload,
                "meta_path": meta_path,
                "payload_hash": payload_hash,
            }
        )
        return combined_payload, meta_path, payload_hash

    class LiveGraphHandler(BaseHTTPRequestHandler):
        def send_bytes(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status: int, payload) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_bytes(status, body, "application/json; charset=utf-8")

        def do_GET(self) -> None:
            parsed_url = urlparse(self.path)
            route = parsed_url.path
            query = parse_qs(parsed_url.query)
            requested_view = query.get("view", [initial_view_key])[0]
            if route in ("/", "/index.html"):
                try:
                    combined_payload, _meta_path, payload_hash = get_graph_response(
                        requested_view
                    )
                    html = build_html(
                        combined_payload["graph"],
                        combined_payload["trending"],
                        combined_payload["artists"],
                        live_config={
                            "enabled": True,
                            "refreshSeconds": args.refresh_seconds,
                            "initialHash": payload_hash,
                            "initialViewKey": requested_view,
                            "views": graph_view_definitions(),
                        },
                    )
                    self.send_bytes(
                        200,
                        html.encode("utf-8"),
                        "text/html; charset=utf-8",
                    )
                except Exception as exc:
                    self.send_json(500, {"error": str(exc)})
                return

            if route == "/graph-data.json":
                try:
                    combined_payload, _meta_path, payload_hash = get_graph_response(
                        requested_view
                    )
                    self.send_json(
                        200,
                        {
                            "view": requested_view,
                            "hash": payload_hash,
                            "graph": combined_payload["graph"],
                            "trending": combined_payload["trending"],
                            "artists": combined_payload["artists"],
                        },
                    )
                except KeyError:
                    self.send_json(404, {"error": f"Unknown graph view: {requested_view}"})
                except Exception as exc:
                    self.send_json(500, {"error": str(exc)})
                return

            self.send_json(404, {"error": "Not found"})

        def log_message(self, _format, *args) -> None:
            return

    return LiveGraphHandler


def serve_live_graph(
    args: argparse.Namespace,
    repo_root: Path,
    folder: Path,
    use_all_folders: bool,
    fallback_probability: float,
) -> int:
    try:
        combined_payload, meta_path = build_combined_payload_from_options(
            args=args,
            repo_root=repo_root,
            folder=folder,
            use_all_folders=use_all_folders,
            fallback_probability=fallback_probability,
        )
    except (FileNotFoundError, ValueError) as exc:
        emit(f"Error: {exc}")
        return 1

    handler_class = make_live_graph_handler(
        args=args,
        repo_root=repo_root,
        folder=folder,
        use_all_folders=use_all_folders,
        fallback_probability=fallback_probability,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler_class)
    server.daemon_threads = True
    server_host, server_port = server.server_address[:2]
    browser_host = "127.0.0.1" if server_host in ("", "0.0.0.0") else server_host
    url = f"http://{browser_host}:{server_port}/"

    emit(f"Read metadata from '{meta_path}'.")
    emit(
        f"Serving live graph/trending dashboard at {url} with {combined_payload['graph']['summary']['songCount']} songs and {combined_payload['graph']['summary']['edgeCount']} observed edges."
    )
    emit(
        f"Pairing mode: {args.pairing}; window: {args.window_minutes} minutes; self loops: {'on' if args.include_self_loops else 'off'}."
    )
    emit(f"Live refresh: every {args.refresh_seconds} seconds. Press Ctrl+C to stop.")

    if args.no_open:
        emit("Skipped opening the browser (--no-open).")
    else:
        webbrowser.open(url)
        emit("Opened the live visualization in your default browser.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        emit("Stopped live graph server.")
    finally:
        server.server_close()

    return 0


def get_default_output_path(folder: Path) -> Path:
    folder_label = folder.name or folder.parent.name or "mp3"
    safe_folder_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in folder_label)
    return Path(DEFAULT_OUTPUT_TEMPLATE.format(folder=safe_folder_label))


def main() -> int:
    global console
    console = require_console()
    print_banner("[Autoplay Graph Viewer]", console)

    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    folder, use_all_folders = resolve_cli_folder(repo_root, args.folder)
    if not use_all_folders and not folder.is_dir():
        emit(f"Error: folder '{folder}' does not exist.")
        return 1

    if args.window_minutes <= 0:
        emit("Error: --window-minutes must be positive.")
        return 1

    if args.refresh_seconds <= 0:
        emit("Error: --refresh-seconds must be positive.")
        return 1

    fallback_probability = clamp_probability(args.fallback_probability)
    if fallback_probability != args.fallback_probability:
        emit(
            f"Clamped fallback probability from {args.fallback_probability} to {fallback_probability}."
        )

    if args.live:
        return serve_live_graph(
            args=args,
            repo_root=repo_root,
            folder=folder,
            use_all_folders=use_all_folders,
            fallback_probability=fallback_probability,
        )

    try:
        combined_payload, meta_path = build_combined_payload_from_options(
            args=args,
            repo_root=repo_root,
            folder=folder,
            use_all_folders=use_all_folders,
            fallback_probability=fallback_probability,
        )
    except (FileNotFoundError, ValueError) as exc:
        emit(f"Error: {exc}")
        return 1

    output_path = (
        Path(args.output).expanduser()
        if args.output
        else get_default_output_path(folder)
    )
    html = build_html(
        combined_payload["graph"],
        combined_payload["trending"],
        combined_payload["artists"],
    )
    output_path.write_text(html, encoding="utf-8")

    emit(f"Read metadata from '{meta_path}'.")
    emit(
        f"Wrote graph/trending dashboard to '{output_path}' with {combined_payload['graph']['summary']['songCount']} songs and {combined_payload['graph']['summary']['edgeCount']} observed edges."
    )
    emit(
        f"Pairing mode: {args.pairing}; window: {args.window_minutes} minutes; self loops: {'on' if args.include_self_loops else 'off'}."
    )
    if args.no_open:
        emit("Skipped opening the browser (--no-open).")
    else:
        webbrowser.open(output_path.resolve().as_uri())
        emit("Opened the visualization in your default browser.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
