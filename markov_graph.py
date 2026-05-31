from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
from mutagen.mp3 import MP3

MARKOV_SESSION_CUTOFF_SECONDS = 10 * 60
DEFAULT_MP3_FOLDER = Path("static/mp3")
LISTEN_DB_FILE = "listen_counts.json"
MID_LISTEN_DB_FILE = "mid_listen_counts.json"


def listen_timestamps_filename_for_folder(mp3_folder: Path) -> str:
    folder_name = mp3_folder.name or "default"
    return f"listen_timestamps_{folder_name}.json"


def listen_db_path(filename: str) -> Path:
    return Path(__file__).resolve().parent / filename


def listen_counts_filename_for_folder(mp3_folder: Path) -> str:
    if mp3_folder.name == "mid-mp3s":
        return MID_LISTEN_DB_FILE
    return LISTEN_DB_FILE


def safe_read_json(path: Path, default):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


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


def load_listen_counts(mp3_dir: Path, db_filename: str) -> Dict[str, int]:
    raw = safe_read_json(listen_db_path(db_filename), {})
    counts: Dict[str, int] = raw if isinstance(raw, dict) else {}
    try:
        for path in mp3_dir.iterdir():
            if path.is_file() and path.suffix.lower() == ".mp3":
                counts.setdefault(path.stem, 0)
    except Exception:
        pass
    out: Dict[str, int] = {}
    for key, value in counts.items():
        try:
            out[str(key)] = int(value)
        except Exception:
            out[str(key)] = 0
    return out


def get_audio_duration_seconds(mp3_path: Path) -> float:
    try:
        return float(MP3(str(mp3_path)).info.length or 0.0)
    except Exception:
        return 0.0


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


def scale_value(value: float, min_value: float, max_value: float, low: float, high: float) -> float:
    if max_value <= min_value:
        return (low + high) / 2.0
    ratio = (value - min_value) / (max_value - min_value)
    return low + ratio * (high - low)


def resolve_node_overlaps(
    nodes: List[dict],
    width: int,
    height: int,
    min_x: float = 0.0,
    min_y: float = 0.0,
    max_x: float | None = None,
    max_y: float | None = None,
) -> None:
    if len(nodes) < 2:
        return

    padding = 40.0
    bound_max_x = float(width) if max_x is None else float(max_x)
    bound_max_y = float(height) if max_y is None else float(max_y)
    for _ in range(160):
        moved = False
        for i in range(len(nodes)):
            a = nodes[i]
            ax = float(a["x"])
            ay = float(a["y"])
            ar = float(a["radius"])
            for j in range(i + 1, len(nodes)):
                b = nodes[j]
                bx = float(b["x"])
                by = float(b["y"])
                br = float(b["radius"])
                dx = bx - ax
                dy = by - ay
                dist = math.hypot(dx, dy)
                min_dist = ar + br + 14.0
                if dist >= min_dist:
                    continue

                if dist < 1e-6:
                    angle = (i * 0.73) + (j * 1.17)
                    dx = math.cos(angle)
                    dy = math.sin(angle)
                    dist = 1.0

                overlap = min_dist - dist
                ux = dx / dist
                uy = dy / dist
                shift = overlap * 0.52
                a["x"] = float(a["x"]) - ux * shift
                a["y"] = float(a["y"]) - uy * shift
                b["x"] = float(b["x"]) + ux * shift
                b["y"] = float(b["y"]) + uy * shift
                moved = True

        for node in nodes:
            r = float(node["radius"])
            node["x"] = min(
                max(min_x + padding + r, float(node["x"])),
                bound_max_x - padding - r,
            )
            node["y"] = min(
                max(min_y + padding + r, float(node["y"])),
                bound_max_y - padding - r,
            )

        if not moved:
            break


def force_layout(
    nodes: List[dict],
    edges: List[dict],
    width: int,
    height: int,
    min_x: float = 0.0,
    min_y: float = 0.0,
) -> None:
    if not nodes:
        return

    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node["id"])
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        if source == target:
            continue
        strength = max(0.05, float(edge["probability"]))
        if graph.has_edge(source, target):
            graph[source][target]["weight"] += strength
        else:
            graph.add_edge(source, target, weight=strength)

    node_count = max(1, len(nodes))
    avg_radius = sum(float(node.get("radius", 0.0)) for node in nodes) / float(node_count)
    area_scale = math.sqrt(float(width * height))
    k = max(0.48, (avg_radius * 10.5) / max(1.0, area_scale / math.sqrt(node_count)))
    positions = nx.spring_layout(
        graph,
        weight="weight",
        seed=7,
        k=k,
        iterations=700,
        scale=1.0,
    )

    padding = 220.0
    usable_w = max(200.0, float(width) - 2.0 * padding)
    usable_h = max(200.0, float(height) - 2.0 * padding)
    for node in nodes:
        x_norm, y_norm = positions.get(node["id"], (0.0, 0.0))
        node["x"] = min_x + padding + ((float(x_norm) + 1.0) / 2.0) * usable_w
        node["y"] = min_y + padding + ((float(y_norm) + 1.0) / 2.0) * usable_h

    resolve_node_overlaps(
        nodes,
        int(min_x + width),
        int(min_y + height),
        min_x=min_x,
        min_y=min_y,
        max_x=min_x + width,
        max_y=min_y + height,
    )


def grid_layout(
    nodes: List[dict],
    min_x: float,
    min_y: float,
    width: float,
    height: float,
) -> None:
    if not nodes:
        return

    cols = max(1, math.ceil(math.sqrt(len(nodes))))
    rows = max(1, math.ceil(len(nodes) / cols))
    cell_w = max(90.0, width / cols)
    cell_h = max(70.0, height / rows)
    for index, node in enumerate(nodes):
        row = index // cols
        col = index % cols
        node["x"] = min_x + (col + 0.5) * cell_w
        node["y"] = min_y + (row + 0.5) * cell_h

    resolve_node_overlaps(
        nodes,
        int(min_x + width),
        int(min_y + height),
        min_x=min_x,
        min_y=min_y,
        max_x=min_x + width,
        max_y=min_y + height,
    )


def build_graph_data(
    mp3_folder: Path,
    session_cutoff_seconds: float,
    max_edges_per_node: int,
    min_probability: float,
) -> dict:
    timestamp_file = listen_timestamps_filename_for_folder(mp3_folder)
    listen_counts_file = listen_counts_filename_for_folder(mp3_folder)
    history = load_listen_timestamps(mp3_folder, timestamp_file)
    listen_counts = load_listen_counts(mp3_folder, listen_counts_file)
    durations = build_duration_by_track_name(mp3_folder)
    transitions, global_counts = build_markov_transition_counts(
        mp3_folder,
        history,
        durations,
        session_cutoff_seconds=session_cutoff_seconds,
    )

    nodes: List[dict] = []
    for stem, count in sorted(
        listen_counts.items(), key=lambda item: (-item[1], item[0].lower())
    ):
        track_name = f"{stem}.mp3"
        nodes.append(
            {
                "id": track_name,
                "label": stem,
                "full_name": track_name,
                "count": int(count),
            }
        )

    width = 4200
    height = 3000
    count_values = [node["count"] for node in nodes] or [1]
    min_count = min(count_values)
    max_count = max(count_values)
    for node in nodes:
        node["radius"] = scale_value(
            math.log1p(float(node["count"])),
            math.log1p(float(min_count)),
            math.log1p(float(max_count)),
            4.5,
            13.0,
        )

    node_lookup = {node["id"]: node for node in nodes}
    edges: List[dict] = []
    for source_name, outgoing in transitions.items():
        total = sum(int(v) for v in outgoing.values())
        if total <= 0:
            continue
        sorted_outgoing = sorted(
            outgoing.items(), key=lambda item: (-item[1], item[0].lower())
        )[: max(1, max_edges_per_node)]
        for target_name, count in sorted_outgoing:
            probability = float(count) / float(total)
            if probability < min_probability:
                continue
            if source_name not in node_lookup or target_name not in node_lookup:
                continue
            edges.append(
                {
                    "source": source_name,
                    "target": target_name,
                    "count": int(count),
                    "probability": probability,
                }
            )

    probability_values = [edge["probability"] for edge in edges] or [1.0]
    min_probability_seen = min(probability_values)
    max_probability_seen = max(probability_values)
    for edge in edges:
        edge["stroke_width"] = scale_value(
            edge["probability"],
            min_probability_seen,
            max_probability_seen,
            0.9,
            4.8,
        )
        edge["opacity"] = scale_value(
            edge["probability"],
            min_probability_seen,
            max_probability_seen,
            0.14,
            0.72,
        )

    connected_ids = {edge["source"] for edge in edges} | {edge["target"] for edge in edges}
    connected_nodes = [node for node in nodes if node["id"] in connected_ids]
    isolated_nodes = [node for node in nodes if node["id"] not in connected_ids]

    if connected_nodes:
        force_layout(
            connected_nodes,
            edges,
            int(width * 0.70),
            height,
            min_x=80.0,
            min_y=0.0,
        )
    if isolated_nodes:
        grid_layout(
            isolated_nodes,
            min_x=width * 0.75,
            min_y=140.0,
            width=width * 0.22,
            height=height - 280.0,
        )
    if connected_nodes and isolated_nodes:
        resolve_node_overlaps(nodes, width, height)

    return {
        "folder": str(mp3_folder),
        "timestamp_file": timestamp_file,
        "listen_counts_file": listen_counts_file,
        "session_cutoff_seconds": session_cutoff_seconds,
        "nodes": nodes,
        "edges": edges,
        "width": width,
        "height": height,
    }


def render_html(graph: dict) -> str:
    payload = json.dumps(graph, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Markov Graph</title>
  <style>
    :root {{
      --bg: #dce8f2;
      --ink: #13202b;
      --muted: #587184;
      --edge: #f26b5e;
      --node: #2d6ea3;
      --node-active: #20a4b8;
      --panel: rgba(245, 250, 255, 0.82);
      --grid: rgba(19, 32, 43, 0.05);
      --canvas: rgba(249, 252, 255, 0.74);
      --panel-border: rgba(19, 32, 43, 0.1);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", Helvetica, Arial, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(32, 164, 184, 0.16), transparent 28%),
        radial-gradient(circle at bottom right, rgba(242, 107, 94, 0.12), transparent 24%),
        linear-gradient(135deg, #eef6fb 0%, #d8e7f3 52%, #e7edf7 100%);
    }}
    .shell {{
      display: grid;
      grid-template-columns: 320px 1fr;
      min-height: 100vh;
    }}
    .sidebar {{
      padding: 28px 24px;
      border-right: 1px solid var(--panel-border);
      background: var(--panel);
      backdrop-filter: blur(14px);
      box-shadow: inset -1px 0 0 rgba(255, 255, 255, 0.45);
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 30px;
      line-height: 1;
      letter-spacing: 0.01em;
      font-weight: 750;
    }}
    .sub {{
      color: var(--muted);
      margin-bottom: 18px;
      font-size: 14px;
      line-height: 1.5;
    }}
    .meta {{
      display: grid;
      gap: 10px;
      margin-bottom: 22px;
      font-size: 14px;
    }}
    .meta strong {{
      display: block;
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 4px;
    }}
    .legend {{
      display: grid;
      gap: 12px;
      margin-top: 18px;
      font-size: 14px;
    }}
    .chip {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid rgba(19, 32, 43, 0.1);
      background: rgba(255, 255, 255, 0.62);
      margin-right: 8px;
      margin-bottom: 8px;
    }}
    .hint {{
      margin-top: 18px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    .canvas-wrap {{
      position: relative;
      overflow: auto;
      padding: 24px;
      background:
        linear-gradient(var(--grid) 1px, transparent 1px),
        linear-gradient(90deg, var(--grid) 1px, transparent 1px);
      background-size: 32px 32px;
    }}
    svg {{
      width: 100%;
      min-width: 1200px;
      height: auto;
      border-radius: 18px;
      background: var(--canvas);
      box-shadow: 0 22px 70px rgba(24, 42, 59, 0.1);
    }}
    .edge {{
      stroke: var(--edge);
      fill: none;
      transition: opacity 140ms ease, stroke 140ms ease;
    }}
    .edge.faded {{
      opacity: 0.06 !important;
    }}
    .node-circle {{
      fill: var(--node);
      stroke: rgba(255,255,255,0.92);
      stroke-width: 1.6;
      cursor: pointer;
      transition: fill 140ms ease, transform 140ms ease, opacity 140ms ease;
    }}
    .node-circle.active {{
      fill: var(--node-active);
    }}
    .node-circle.faded {{
      opacity: 0.16;
    }}
    .label {{
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.01em;
      text-anchor: middle;
      pointer-events: none;
      fill: var(--ink);
      opacity: 0.82;
    }}
    .label.faded {{
      opacity: 0.15;
    }}
    .tooltip {{
      position: fixed;
      right: 22px;
      bottom: 22px;
      width: min(360px, calc(100vw - 44px));
      padding: 16px 18px;
      border-radius: 16px;
      border: 1px solid rgba(19, 32, 43, 0.1);
      background: rgba(246, 251, 255, 0.93);
      box-shadow: 0 16px 48px rgba(24, 42, 59, 0.12);
      backdrop-filter: blur(14px);
    }}
    .tooltip h2 {{
      margin: 0 0 10px;
      font-size: 20px;
      line-height: 1.1;
    }}
    .tooltip .small {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 10px;
    }}
    .tooltip ul {{
      margin: 0;
      padding-left: 18px;
      font-size: 14px;
      line-height: 1.45;
    }}
  </style>
</head>
<body>
  <div class="shell">
    <aside class="sidebar">
      <h1>Markov Map</h1>
      <div class="sub">Nodes are tracks. Bigger nodes mean more full listens. Edges show next-track probabilities inside the 10-minute session window.</div>
      <div class="meta">
        <div><strong>Folder</strong>{graph["folder"]}</div>
        <div><strong>Timestamp File</strong>{graph["timestamp_file"]}</div>
        <div><strong>Listen Counts File</strong>{graph["listen_counts_file"]}</div>
        <div><strong>Nodes</strong>{len(graph["nodes"])}</div>
        <div><strong>Edges</strong>{len(graph["edges"])}</div>
        <div><strong>Session Cutoff</strong>{int(graph["session_cutoff_seconds"] // 60)} minutes</div>
      </div>
      <div class="legend">
        <div><span class="chip">Large node = listened more</span></div>
        <div><span class="chip">Thick edge = stronger transition</span></div>
        <div><span class="chip">Click a node = isolate its neighborhood</span></div>
      </div>
      <div class="hint">If the graph gets too dense later, regenerate it with a higher `--min-prob` or lower `--max-edges-per-node` value.</div>
    </aside>
    <main class="canvas-wrap">
      <svg id="graph" viewBox="0 0 {graph["width"]} {graph["height"]}" preserveAspectRatio="xMidYMid meet">
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#c65d3b"></path>
          </marker>
        </defs>
      </svg>
    </main>
  </div>
  <section class="tooltip" id="tooltip">
    <h2>Nothing selected</h2>
    <div class="small">Click any node to inspect its outgoing probabilities.</div>
  </section>
  <script>
    const graph = {payload};
    const svg = document.getElementById("graph");
    const tooltip = document.getElementById("tooltip");

    const nodeMap = new Map(graph.nodes.map(node => [node.id, node]));
    const outgoing = new Map();
    const incoming = new Map();
    for (const edge of graph.edges) {{
      if (!outgoing.has(edge.source)) outgoing.set(edge.source, []);
      if (!incoming.has(edge.target)) incoming.set(edge.target, []);
      outgoing.get(edge.source).push(edge);
      incoming.get(edge.target).push(edge);
    }}
    for (const [key, value] of outgoing.entries()) {{
      value.sort((a, b) => b.probability - a.probability || a.target.localeCompare(b.target));
    }}

    function el(name, attrs = {{}}, parent = null) {{
      const node = document.createElementNS("http://www.w3.org/2000/svg", name);
      for (const [key, value] of Object.entries(attrs)) {{
        node.setAttribute(key, String(value));
      }}
      if (parent) parent.appendChild(node);
      return node;
    }}

    const edgeGroup = el("g", {{}}, svg);
    const nodeGroup = el("g", {{}}, svg);

    const edgeEls = [];
    for (const edge of graph.edges) {{
      const source = nodeMap.get(edge.source);
      const target = nodeMap.get(edge.target);
      if (!source || !target) continue;

      const dx = target.x - source.x;
      const dy = target.y - source.y;
      const distance = Math.max(1, Math.hypot(dx, dy));
      const ux = dx / distance;
      const uy = dy / distance;
      const startX = source.x + ux * source.radius * 0.9;
      const startY = source.y + uy * source.radius * 0.9;
      const endX = target.x - ux * target.radius * 0.95;
      const endY = target.y - uy * target.radius * 0.95;
      const midX = (startX + endX) / 2 + (-uy * Math.min(40, distance * 0.08));
      const midY = (startY + endY) / 2 + (ux * Math.min(40, distance * 0.08));
      const path = `M ${{startX}} ${{startY}} Q ${{midX}} ${{midY}} ${{endX}} ${{endY}}`;
      const edgeEl = el("path", {{
        d: path,
        class: "edge",
        "stroke-width": edge.stroke_width,
        "stroke-opacity": edge.opacity,
        "marker-end": "url(#arrow)"
      }}, edgeGroup);
      edgeEl.dataset.source = edge.source;
      edgeEl.dataset.target = edge.target;
      edgeEl.dataset.probability = edge.probability.toFixed(4);
      edgeEl.dataset.count = String(edge.count);
      edgeEls.push(edgeEl);
    }}

    const labelEls = [];
    const nodeCircleEls = [];
    for (const node of graph.nodes) {{
      const g = el("g", {{}}, nodeGroup);
      const circle = el("circle", {{
        cx: node.x,
        cy: node.y,
        r: node.radius,
        class: "node-circle"
      }}, g);
      const label = el("text", {{
        x: node.x,
        y: node.y + node.radius + 16,
        class: "label"
      }}, g);
      label.textContent = node.label;
      circle.dataset.id = node.id;
      label.dataset.id = node.id;
      nodeCircleEls.push(circle);
      labelEls.push(label);
      circle.addEventListener("click", () => selectNode(node.id));
    }}

    function resetStyles() {{
      for (const edgeEl of edgeEls) edgeEl.classList.remove("faded");
      for (const circle of nodeCircleEls) {{
        circle.classList.remove("active");
        circle.classList.remove("faded");
      }}
      for (const label of labelEls) label.classList.remove("faded");
    }}

    function selectNode(nodeId) {{
      const node = nodeMap.get(nodeId);
      if (!node) return;
      const related = new Set([nodeId]);
      for (const edge of outgoing.get(nodeId) || []) related.add(edge.target);
      for (const edge of incoming.get(nodeId) || []) related.add(edge.source);

      resetStyles();
      for (const edgeEl of edgeEls) {{
        const keep = edgeEl.dataset.source === nodeId || edgeEl.dataset.target === nodeId;
        if (!keep) edgeEl.classList.add("faded");
      }}
      for (const circle of nodeCircleEls) {{
        const keep = related.has(circle.dataset.id);
        if (!keep) circle.classList.add("faded");
        if (circle.dataset.id === nodeId) circle.classList.add("active");
      }}
      for (const label of labelEls) {{
        if (!related.has(label.dataset.id)) label.classList.add("faded");
      }}

      const nextRows = (outgoing.get(nodeId) || []).slice(0, 8).map(edge => {{
        const target = nodeMap.get(edge.target);
        const label = target ? target.label : edge.target;
        return `<li><strong>${{label}}</strong> - ${{(edge.probability * 100).toFixed(1)}}% (${{edge.count}} transition${{edge.count === 1 ? "" : "s"}})</li>`;
      }});
      const incomingCount = (incoming.get(nodeId) || []).length;
      tooltip.innerHTML = `
        <h2>${{node.label}}</h2>
        <div class="small">${{node.count}} full listens recorded • ${{incomingCount}} incoming edge${{incomingCount === 1 ? "" : "s"}}</div>
        ${{nextRows.length ? `<ul>${{nextRows.join("")}}</ul>` : "<div class='small'>No outgoing transitions recorded yet.</div>"}}
      `;
    }}

    svg.addEventListener("click", (event) => {{
      if (event.target === svg) {{
        resetStyles();
        tooltip.innerHTML = `
          <h2>Nothing selected</h2>
          <div class="small">Click any node to inspect its outgoing probabilities.</div>
        `;
      }}
    }});
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an HTML graph of Markov transitions between MP3 tracks."
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=str(DEFAULT_MP3_FOLDER),
        help="MP3 folder path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HTML path. Defaults to markov_graph_<folder>.html.",
    )
    parser.add_argument(
        "--cutoff-minutes",
        type=float,
        default=10.0,
        help="Session cutoff in minutes for linking transitions.",
    )
    parser.add_argument(
        "--max-edges-per-node",
        type=int,
        default=6,
        help="Maximum outgoing edges to render per node.",
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.08,
        help="Minimum transition probability to render, from 0.0 to 1.0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mp3_folder = Path(args.folder)
    output = (
        Path(args.output)
        if args.output
        else Path(f"markov_graph_{mp3_folder.name or 'default'}.html")
    )

    graph = build_graph_data(
        mp3_folder=mp3_folder,
        session_cutoff_seconds=max(1.0, float(args.cutoff_minutes) * 60.0),
        max_edges_per_node=max(1, int(args.max_edges_per_node)),
        min_probability=max(0.0, min(1.0, float(args.min_prob))),
    )
    html = render_html(graph)
    output.write_text(html, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
