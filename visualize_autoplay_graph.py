#!/usr/bin/env python3
"""Build an interactive autoplay-transition graph from an MP3 folder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
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
NODE_MIN_SIZE = 7
NODE_SIZE_RANGE = 63
NODE_SIZE_POWER = 1.25
VIS_NETWORK_CDN = "https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"
LISTEN_COUNTS_FILE = "listen_counts.json"
MID_LISTEN_COUNTS_FILE = "mid_listen_counts.json"
ALL_FOLDERS = (Path("static/mp3"), Path("static/mid-mp3s"))
GRAPH_VIEW_OPTIONS = (
    ("mp3", "MP3", Path("static/mp3"), False),
    ("mid-mp3s", "Mid MP3s", Path("static/mid-mp3s"), False),
    ("all", "All", Path("all"), True),
)

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
        default="mp3s",
        help=(
            "Folder containing MP3 files and .mp3meta.json (default: mp3s). "
            "Use 'all' to combine static/mp3 and static/mid-mp3s."
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

        return song_counts, history_mapping, durations, meta_path

    song_counts, counts_path = load_repo_listen_counts(folder, repo_root)
    history_mapping, timestamps_path = load_repo_history_mapping(
        folder, repo_root, song_counts
    )
    source_description = Path(
        f"{counts_path.name} + {timestamps_path.name}"
    )
    return song_counts, history_mapping, durations, source_description


def prefix_mapping_keys(mapping, prefix: str):
    return {f"{prefix}/{key}": value for key, value in mapping.items()}


def load_all_folder_data(repo_root: Path):
    combined_counts = {}
    combined_history = {}
    combined_durations = {}
    sources = []

    for relative_folder in ALL_FOLDERS:
        folder = repo_root / relative_folder
        if not folder.is_dir():
            raise FileNotFoundError(f"folder '{folder}' does not exist")

        song_counts, counts_path = load_repo_listen_counts(folder, repo_root)
        history_mapping, timestamps_path = load_repo_history_mapping(
            folder, repo_root, song_counts
        )
        prefix = folder.name
        combined_counts.update(prefix_mapping_keys(song_counts, prefix))
        combined_history.update(prefix_mapping_keys(history_mapping, prefix))
        combined_durations.update(prefix_mapping_keys(load_folder_durations(folder), prefix))
        sources.extend([counts_path.name, timestamps_path.name])

    return combined_counts, combined_history, combined_durations, Path(" + ".join(sources))


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
        song_counts, history_mapping, durations, meta_path = load_all_folder_data(repo_root)
    else:
        song_counts, history_mapping, durations, meta_path = load_folder_data(folder)

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


def graph_view_definitions():
    return [
        {
            "key": key,
            "label": label,
        }
        for key, label, _path, _use_all_folders in GRAPH_VIEW_OPTIONS
    ]


def graph_view_keys():
    return {key for key, _label, _path, _use_all_folders in GRAPH_VIEW_OPTIONS}


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
                    repo_root / listen_counts_filename_for_folder(source_folder),
                    repo_root / listen_timestamps_filename_for_folder(source_folder),
                ]
            )
        return source_paths

    meta_path = folder / META_FILENAME
    if meta_path.is_file():
        return [meta_path]

    return [
        repo_root / listen_counts_filename_for_folder(folder),
        repo_root / listen_timestamps_filename_for_folder(folder),
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


def build_html(graph_payload, live_config=None):
    graph_json = json.dumps(graph_payload, ensure_ascii=False)
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
      .sidebar {{ border-left: none; border-top: 1px solid var(--border); }}
    }}
  </style>
</head>
<body>
  <div class=\"toolbar\">
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
  <script>
    let graphData = {graph_json};
    const liveConfig = {live_config_json};
    let activeViewKey = liveConfig.initialViewKey || graphData.summary.viewKey || 'all';
    const graphHashesByView = new Map([[activeViewKey, liveConfig.initialHash || null]]);
    let currentGraphHash = graphHashesByView.get(activeViewKey) || null;
    let allNodes = graphData.nodes;
    let allEdges = graphData.edges;
    let nodeLookup = new Map(allNodes.map((node) => [node.id, node]));
    let edgeLookup = new Map(allEdges.map((edge) => [edge.id, edge]));
    let selectedNodeId = null;

    const container = document.getElementById('network');
    const networkStatus = document.getElementById('network-status');
    const listenTimeSizeToggle = document.getElementById('listen-time-size-toggle');

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
          : getNodeColor(node)
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
        stabilization: {{ iterations: 250 }},
        barnesHut: {{ gravitationalConstant: -3200, springLength: 135, springConstant: 0.03, damping: 0.18 }}
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
        network.unselectAll();
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

    function settleGraphLayout() {{
      if (!network) {{
        return;
      }}
      network.setOptions({{
        physics: {{
          enabled: true,
          stabilization: {{ iterations: 180 }},
          barnesHut: {{ gravitationalConstant: -3200, springLength: 135, springConstant: 0.03, damping: 0.18 }}
        }}
      }});
      network.once('stabilized', () => {{
        network.setOptions({{ physics: false }});
        network.fit({{ animation: {{ duration: 350, easingFunction: 'easeInOutQuad' }} }});
      }});
      network.stabilize(180);
    }}

    function attachNetworkEvents() {{
      network.on('selectNode', (params) => {{
        selectedNodeId = params.nodes[0] || null;
        refreshGraph();
        if (selectedNodeId) {{
          showNodeDetails(selectedNodeId);
        }}
      }});

      network.on('deselectNode', () => {{
        selectedNodeId = null;
        refreshGraph();
        document.getElementById('selection-summary').textContent = 'Select a node or edge to inspect it.';
        document.getElementById('selection-content').innerHTML = '';
      }});

      network.on('selectEdge', (params) => {{
        const edgeId = params.edges[0];
        if (edgeId) {{
          showEdgeDetails(edgeId);
        }}
      }});
    }}

    function createNetwork() {{
      if (network) {{
        network.destroy();
      }}
      nodeDataSet = new vis.DataSet([]);
      edgeDataSet = new vis.DataSet([]);
      refreshGraph();
      network = new vis.Network(container, {{ nodes: nodeDataSet, edges: edgeDataSet }}, networkOptions);
      attachNetworkEvents();
      network.once('stabilized', () => {{
        network.setOptions({{ physics: false }});
        network.fit({{ animation: {{ duration: 350, easingFunction: 'easeInOutQuad' }} }});
      }});
      if (networkStatus) {{
        networkStatus.classList.remove('visible');
      }}
    }}

    function refreshGraph(options = {{}}) {{
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
      selectedNodeId = match.id;
      refreshGraph();
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
            "graph_payload": None,
            "meta_path": None,
            "graph_hash": None,
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
            and cached_response["graph_payload"] is not None
        ):
            return (
                cached_response["graph_payload"],
                cached_response["meta_path"],
                cached_response["graph_hash"],
            )

        graph_payload, meta_path = build_graph_payload_from_options(
            args=args,
            repo_root=repo_root,
            folder=view_definition["folder"],
            use_all_folders=view_definition["use_all_folders"],
            fallback_probability=fallback_probability,
            view_key=view_definition["key"],
            view_label=view_definition["label"],
        )
        graph_hash = hash_graph_payload(graph_payload)
        cached_response.update(
            {
                "signature": current_signature,
                "graph_payload": graph_payload,
                "meta_path": meta_path,
                "graph_hash": graph_hash,
            }
        )
        return graph_payload, meta_path, graph_hash

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
                    graph_payload, _meta_path, graph_hash = get_graph_response(
                        requested_view
                    )
                    html = build_html(
                        graph_payload,
                        live_config={
                            "enabled": True,
                            "refreshSeconds": args.refresh_seconds,
                            "initialHash": graph_hash,
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
                    graph_payload, _meta_path, graph_hash = get_graph_response(
                        requested_view
                    )
                    self.send_json(
                        200,
                        {
                            "view": requested_view,
                            "hash": graph_hash,
                            "graph": graph_payload,
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
        graph_payload, meta_path = build_graph_payload_from_options(
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
        f"Serving live graph at {url} with {graph_payload['summary']['songCount']} songs and {graph_payload['summary']['edgeCount']} observed edges."
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
    folder_label = folder.name or folder.parent.name or "mp3s"
    safe_folder_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in folder_label)
    return Path(DEFAULT_OUTPUT_TEMPLATE.format(folder=safe_folder_label))


def main() -> int:
    global console
    console = require_console()
    print_banner("[Autoplay Graph Viewer]", console)

    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    use_all_folders = args.folder == "all"
    folder = Path("all") if use_all_folders else Path(args.folder).expanduser().resolve()
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
        graph_payload, meta_path = build_graph_payload_from_options(
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
    html = build_html(graph_payload)
    output_path.write_text(html, encoding="utf-8")

    emit(f"Read metadata from '{meta_path}'.")
    emit(
        f"Wrote graph to '{output_path}' with {graph_payload['summary']['songCount']} songs and {graph_payload['summary']['edgeCount']} observed edges."
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
