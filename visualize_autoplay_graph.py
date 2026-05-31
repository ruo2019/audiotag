#!/usr/bin/env python3
"""Build an interactive autoplay-transition graph from an MP3 folder."""

from __future__ import annotations

import argparse
import json
import math
import sys
import webbrowser
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

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
VIS_NETWORK_CDN = "https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"
LISTEN_COUNTS_FILE = "listen_counts.json"
MID_LISTEN_COUNTS_FILE = "mid_listen_counts.json"
ALL_FOLDERS = (Path("static/mp3"), Path("static/mid-mp3s"))

console = None


def emit(message: str) -> None:
    print_message(message, console)


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
        help="Maximum time gap for linking plays (default: 10)",
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
            "Defaults to 0 so observed transitions sum to 100%."
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

        return song_counts, history_mapping, meta_path

    song_counts, counts_path = load_repo_listen_counts(folder, repo_root)
    history_mapping, timestamps_path = load_repo_history_mapping(
        folder, repo_root, song_counts
    )
    source_description = Path(
        f"{counts_path.name} + {timestamps_path.name}"
    )
    return song_counts, history_mapping, source_description


def prefix_mapping_keys(mapping, prefix: str):
    return {f"{prefix}/{key}": value for key, value in mapping.items()}


def load_all_folder_data(repo_root: Path):
    combined_counts = {}
    combined_history = {}
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
        sources.extend([counts_path.name, timestamps_path.name])

    return combined_counts, combined_history, Path(" + ".join(sources))


def build_global_events(history_mapping):
    events = []
    for song_name, timestamps in history_mapping.items():
        for timestamp in timestamps:
            if isinstance(timestamp, (int, float)):
                events.append((int(timestamp), song_name))
    events.sort(key=lambda item: (item[0], item[1]))
    return events


def build_transition_counts(events, window_seconds: int, pairing_mode: str, include_self_loops: bool):
    edge_counts = Counter()
    edge_gaps = defaultdict(list)

    if pairing_mode == "consecutive":
        for current_event, next_event in zip(events, events[1:]):
            current_timestamp, current_song = current_event
            next_timestamp, next_song = next_event
            gap_seconds = next_timestamp - current_timestamp
            if gap_seconds < 0 or gap_seconds > window_seconds:
                continue
            if not include_self_loops and current_song == next_song:
                continue
            edge_counts[(current_song, next_song)] += 1
            edge_gaps[(current_song, next_song)].append(gap_seconds)
        return edge_counts, edge_gaps

    for start_index, (current_timestamp, current_song) in enumerate(events):
        next_index = start_index + 1
        while next_index < len(events):
            next_timestamp, next_song = events[next_index]
            gap_seconds = next_timestamp - current_timestamp
            if gap_seconds > window_seconds:
                break
            if gap_seconds >= 0 and (include_self_loops or current_song != next_song):
                edge_counts[(current_song, next_song)] += 1
                edge_gaps[(current_song, next_song)].append(gap_seconds)
            next_index += 1

    return edge_counts, edge_gaps


def clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_graph_payload(
    folder: Path,
    song_counts,
    history_mapping,
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

        node_size = 18 + 28 * math.sqrt(play_count / max_play_count) if max_play_count > 0 else 18
        nodes.append(
            {
                "id": song_name,
                "label": display_label,
                "sourceGroup": source_group,
                "value": play_count,
                "size": round(node_size, 2),
                "playCount": play_count,
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


def build_html(graph_payload):
    graph_json = json.dumps(graph_payload, ensure_ascii=False)
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
  </div>
  <div class=\"layout\">
    <div class=\"network-panel\">
      <div id=\"network\"></div>
      <div id=\"network-status\" class=\"network-status\"></div>
    </div>
    <aside class=\"sidebar\">
      <section class=\"card\">
        <h2>Autoplay Graph</h2>
        <div class=\"muted\">Built from global play history in <code>{graph_payload['summary']['folder']}</code>.</div>
        <div class=\"metric-grid\" style=\"margin-top: 12px;\">
          <div class=\"metric\"><div class=\"label\">Songs</div><div class=\"value\" id=\"summary-song-count\"></div></div>
          <div class=\"metric\"><div class=\"label\">Play Events</div><div class=\"value\" id=\"summary-event-count\"></div></div>
          <div class=\"metric\"><div class=\"label\">Observed Edges</div><div class=\"value\" id=\"summary-edge-count\"></div></div>
          <div class=\"metric\"><div class=\"label\">Window</div><div class=\"value\" id=\"summary-window\"></div></div>
        </div>
        <div style=\"margin-top: 12px;\">
          <span class=\"pill\" id=\"summary-pairing\"></span>
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
    const graphData = {graph_json};
    const allNodes = graphData.nodes;
    const allEdges = graphData.edges;
    const nodeLookup = new Map(allNodes.map((node) => [node.id, node]));
    const edgeLookup = new Map(allEdges.map((edge) => [edge.id, edge]));
    let selectedNodeId = null;

    const container = document.getElementById('network');
    const networkStatus = document.getElementById('network-status');

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

    const nodeDataSet = new vis.DataSet(allNodes.map((node) => ({{
      ...node,
      font: {{ color: '#142130', size: 18, face: 'Inter' }},
      color: getNodeColor(node)
    }})));
    const edgeDataSet = new vis.DataSet([]);

    const network = new vis.Network(container, {{ nodes: nodeDataSet, edges: edgeDataSet }}, {{
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
    }});
    if (networkStatus) {{
      networkStatus.classList.remove('visible');
    }}

    function formatPercent(probability) {{
      return `${{(probability * 100).toFixed(2)}}%`;
    }}

    function formatMinutes(seconds) {{
      return `${{(seconds / 60).toFixed(2)}} min`;
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

    function refreshGraph() {{
      const filteredEdges = getFilteredEdges();
      const connectedNodeIds = new Set();
      filteredEdges.forEach((edge) => {{
        connectedNodeIds.add(edge.from);
        connectedNodeIds.add(edge.to);
      }});

      const neighborhoodOnly = document.getElementById('neighborhood-toggle').checked;
      nodeDataSet.clear();
      nodeDataSet.add(allNodes
        .filter((node) => !neighborhoodOnly || !selectedNodeId || connectedNodeIds.has(node.id) || node.id === selectedNodeId)
        .map((node) => ({{
          ...node,
          hidden: neighborhoodOnly && selectedNodeId && !connectedNodeIds.has(node.id) && node.id !== selectedNodeId,
          font: {{ color: '#142130', size: 18, face: 'Inter' }},
          color: node.id === selectedNodeId
            ? {{
                background: '#ff8a65',
                border: '#eb5757',
                highlight: {{ background: '#ff8a65', border: '#eb5757' }}
              }}
            : getNodeColor(node)
        }})));
      edgeDataSet.clear();
      edgeDataSet.add(filteredEdges);
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
    document.getElementById('focus-button').addEventListener('click', focusSong);
    document.getElementById('song-search').addEventListener('keydown', (event) => {{
      if (event.key === 'Enter') {{
        focusSong();
      }}
    }});

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

    document.getElementById('summary-song-count').textContent = graphData.summary.songCount;
    document.getElementById('summary-event-count').textContent = graphData.summary.eventCount;
    document.getElementById('summary-edge-count').textContent = graphData.summary.edgeCount;
    document.getElementById('summary-window').textContent = `${{graphData.summary.windowMinutes}}m`;
    document.getElementById('summary-pairing').textContent =
      `Pairing: ${{graphData.summary.pairingMode}} (${{describePairingMode(graphData.summary.pairingMode)}})`;
    refreshGraph();
    network.fit({{ animation: {{ duration: 600, easingFunction: 'easeInOutQuad' }} }});
  </script>
</body>
</html>
"""


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

    fallback_probability = clamp_probability(args.fallback_probability)
    if fallback_probability != args.fallback_probability:
        emit(
            f"Clamped fallback probability from {args.fallback_probability} to {fallback_probability}."
        )

    try:
        if use_all_folders:
            song_counts, history_mapping, meta_path = load_all_folder_data(repo_root)
        else:
            song_counts, history_mapping, meta_path = load_folder_data(folder)
    except (FileNotFoundError, ValueError) as exc:
        emit(f"Error: {exc}")
        return 1

    events = build_global_events(history_mapping)
    window_seconds = int(round(args.window_minutes * 60))
    edge_counts, edge_gaps = build_transition_counts(
        events=events,
        window_seconds=window_seconds,
        pairing_mode=args.pairing,
        include_self_loops=args.include_self_loops,
    )

    graph_payload = build_graph_payload(
        folder=folder,
        song_counts=song_counts,
        history_mapping=history_mapping,
        events=events,
        edge_counts=edge_counts,
        edge_gaps=edge_gaps,
        window_minutes=args.window_minutes,
        pairing_mode=args.pairing,
        fallback_probability=fallback_probability,
        generated_from=str(meta_path),
    )

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
