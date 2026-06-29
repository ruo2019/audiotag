#!/usr/bin/env python3
"""Build an animated HTML viewer for Trending rankings over time."""

from __future__ import annotations

import argparse
import json
import sys
import time
import webbrowser
from pathlib import Path

from cli_helpers import print_banner, print_message, require_console

try:
    from player.constants import (
        ANALYSIS_CACHE_FILENAME,
        CONFIG_KEY,
        META_FILENAME,
        PLAY_HISTORY_CONFIG_KEY,
        TRENDING_BASELINE_DURATION_SECONDS,
        TRENDING_HALF_LIFE_SECONDS,
    )
except ModuleNotFoundError:  # pragma: no cover - support running from scripts dir
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from player.constants import (
        ANALYSIS_CACHE_FILENAME,
        CONFIG_KEY,
        META_FILENAME,
        PLAY_HISTORY_CONFIG_KEY,
        TRENDING_BASELINE_DURATION_SECONDS,
        TRENDING_HALF_LIFE_SECONDS,
    )

DEFAULT_OUTPUT_TEMPLATE = "trending_rankings_{folder}.html"
DEFAULT_TOP_LIMIT = 20

console = None


def emit(message: str) -> None:
    print_message(message, console)


def format_duration_label(seconds: int) -> str:
    seconds = max(1, int(seconds))
    if seconds % 86400 == 0 and seconds >= 172800:
        return f"{seconds // 86400}d"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an animated HTML timeline of the player's Trending "
            "exponential-decay rankings from .mp3meta.json play history."
        )
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default="mp3s",
        help="Folder containing MP3 files and .mp3meta.json (default: mp3s)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP_LIMIT,
        help=f"Initial number of ranked songs to show (default: {DEFAULT_TOP_LIMIT})",
    )
    parser.add_argument(
        "--output",
        help=(
            "Output HTML path (default: trending_rankings_<folder>.html in "
            "the repo root)"
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


def encode_timestamps(timestamps):
    encoded = []
    previous = None
    for timestamp in sorted(timestamps):
        timestamp = int(timestamp)
        if previous is None:
            encoded.append(timestamp)
        else:
            encoded.append(timestamp - previous)
        previous = timestamp
    return encoded


def load_duration_mapping(folder: Path):
    analysis_path = folder / ANALYSIS_CACHE_FILENAME
    if not analysis_path.is_file():
        return {}, analysis_path

    try:
        data = json.loads(analysis_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, analysis_path

    duration_mapping = {}
    if isinstance(data, dict):
        for song_name, payload in data.items():
            if str(song_name).startswith("_") or not isinstance(payload, dict):
                continue
            duration = payload.get("duration")
            if isinstance(duration, (int, float)) and duration > 0:
                duration_mapping[str(song_name)] = float(duration)
    return duration_mapping, analysis_path


def trending_weight(duration_seconds):
    if not isinstance(duration_seconds, (int, float)) or duration_seconds <= 0:
        return 1.0
    baseline = max(1.0, float(TRENDING_BASELINE_DURATION_SECONDS))
    return max(0.0, float(duration_seconds) / baseline)


def load_folder_data(folder: Path):
    meta_path = folder / META_FILENAME
    if not meta_path.is_file():
        raise FileNotFoundError(f"Could not find metadata file: {meta_path}")

    data = json.loads(meta_path.read_text(encoding="utf-8"))
    song_counts = {
        name: count
        for name, count in data.items()
        if not str(name).startswith("__") and isinstance(count, int)
    }

    config = data.get(CONFIG_KEY, {})
    raw_history_mapping = config.get(PLAY_HISTORY_CONFIG_KEY, {})
    history_mapping = {
        song_name: sorted(decode_play_history(raw_history_mapping.get(song_name, [])))
        for song_name in song_counts
    }

    return song_counts, history_mapping, meta_path


def build_trending_payload(folder: Path, song_counts, history_mapping, duration_mapping):
    all_timestamps = []
    songs = []
    history_event_count = 0
    songs_with_history = 0
    mismatched_history_count = 0
    adjusted_history_count = 0
    historyless_play_count = 0

    for song_name in sorted(song_counts, key=str.casefold):
        play_count = int(song_counts[song_name])
        timestamps = [
            int(timestamp)
            for timestamp in history_mapping.get(song_name, [])
            if isinstance(timestamp, (int, float))
        ]
        timestamps.sort()
        if timestamps:
            songs_with_history += 1
        if len(timestamps) != play_count:
            mismatched_history_count += 1
            if timestamps:
                adjusted_history_count += 1
            elif play_count > 0:
                historyless_play_count += 1

        history_event_count += len(timestamps)
        all_timestamps.extend(timestamps)
        duration_seconds = duration_mapping.get(song_name)
        songs.append(
            {
                "name": song_name,
                "playCount": play_count,
                "historyCount": len(timestamps),
                "history": encode_timestamps(timestamps),
                "durationSeconds": round(float(duration_seconds), 3) if duration_seconds else None,
                "trendingWeight": round(trending_weight(duration_seconds), 6),
            }
        )

    timeline = sorted(set(all_timestamps))
    payload = {
        "summary": {
            "folder": str(folder),
            "songCount": len(songs),
            "songsWithHistory": songs_with_history,
            "metadataPlayCount": sum(int(count) for count in song_counts.values()),
            "historyEventCount": history_event_count,
            "timelinePointCount": len(timeline),
            "firstTimestamp": timeline[0] if timeline else None,
            "lastTimestamp": timeline[-1] if timeline else None,
            "generatedTimestamp": int(time.time()),
            "trendingHalfLifeSeconds": TRENDING_HALF_LIFE_SECONDS,
            "trendingBaselineDurationSeconds": TRENDING_BASELINE_DURATION_SECONDS,
            "mismatchedHistoryCount": mismatched_history_count,
            "adjustedHistoryCount": adjusted_history_count,
            "historylessPlayCount": historyless_play_count,
        },
        "timeline": encode_timestamps(timeline),
        "songs": songs,
    }
    return payload


def build_html(payload, initial_top_limit: int):
    payload_json = json.dumps(payload, ensure_ascii=False)
    initial_top_limit = max(1, int(initial_top_limit))
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trending Rankings</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0f172a;
      --panel: #111827;
      --panel-2: #1f2937;
      --panel-3: #0b1220;
      --border: #334155;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent-2: #a78bfa;
      --accent-3: #f472b6;
      --success: #34d399;
      --warning: #f59e0b;
      --track: rgba(148, 163, 184, 0.14);
      --row-height: 56px;
      --row-gap: 7px;
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      min-height: 100%;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }
    body {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      padding-bottom: 132px;
    }
    button, input, select {
      font: inherit;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: end;
      padding: 14px 16px;
      border-bottom: 1px solid var(--border);
      background: rgba(15, 23, 42, 0.92);
      position: sticky;
      top: 0;
      z-index: 20;
      backdrop-filter: blur(10px);
    }
    .toolbar label, .timeline-label {
      display: flex;
      flex-direction: column;
      gap: 6px;
      min-width: 150px;
      font-size: 13px;
      color: var(--muted);
    }
    .toolbar input, .toolbar select, .timeline-label select {
      color: var(--text);
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 10px;
    }
    .mode-tabs {
      display: flex;
      gap: 8px;
      align-items: center;
      padding: 4px;
      border: 1px solid var(--border);
      border-radius: 999px;
      background: rgba(15, 23, 42, 0.58);
    }
    .mode-tab {
      border: 0;
      border-radius: 999px;
      padding: 8px 13px;
      color: var(--muted);
      background: transparent;
      cursor: pointer;
      font-weight: 750;
    }
    .mode-tab:hover {
      color: var(--text);
      background: rgba(148, 163, 184, 0.12);
    }
    .mode-tab.is-active {
      color: #e0f2fe;
      background: rgba(56, 189, 248, 0.18);
      box-shadow: inset 0 0 0 1px rgba(56, 189, 248, 0.38);
    }
    .page {
      display: grid;
      gap: 16px;
      padding: 16px;
    }
    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 16px;
      align-items: stretch;
    }
    .card {
      background: rgba(17, 24, 39, 0.92);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 20px 45px rgba(15, 23, 42, 0.22);
    }
    .card h1, .card h2, .card h3, .card p {
      margin-top: 0;
    }
    .muted { color: var(--muted); }
    .code-pill {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      color: #bae6fd;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }
    .metric {
      background: rgba(15, 23, 42, 0.82);
      border: 1px solid rgba(51, 65, 85, 0.8);
      border-radius: 12px;
      padding: 12px;
      min-width: 0;
    }
    .metric .label {
      color: var(--muted);
      font-size: 12px;
    }
    .metric .value {
      margin-top: 4px;
      font-size: 22px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }
    .section-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
      margin-bottom: 12px;
    }
    .section-head h2 {
      margin-bottom: 6px;
    }
    .ranking-shell h2 {
      margin: 0 0 6px;
    }
    .ranking-shell {
      min-height: 440px;
      padding: 8px 12px 0 14px;
      position: relative;
      z-index: 0;
    }
    .chart-stage {
      position: relative;
      isolation: isolate;
      z-index: 0;
      min-height: var(--row-height);
      transition: height 420ms cubic-bezier(0.2, 0.8, 0.2, 1);
    }
    .rank-row {
      --song-color: #38bdf8;
      --song-color-soft: rgba(56, 189, 248, 0.16);
      position: absolute;
      left: 0;
      right: 0;
      height: var(--row-height);
      display: grid;
      grid-template-columns: 52px minmax(0, 1fr) 112px;
      gap: 10px;
      align-items: center;
      padding: 7px 4px 7px 8px;
      border-radius: 0;
      background: transparent;
      opacity: 1;
      transform: translateY(0);
      transition:
        transform 520ms cubic-bezier(0.2, 0.8, 0.2, 1),
        opacity 260ms ease,
        background 260ms ease;
      will-change: transform;
    }
    .rank-row.is-leader {
      background: linear-gradient(90deg, var(--song-color-soft), rgba(15, 23, 42, 0));
    }
    .rank-row:hover {
      background: linear-gradient(90deg, var(--song-color-soft), rgba(15, 23, 42, 0.28));
    }
    .rank-row.is-exiting {
      opacity: 0;
    }
    .rank {
      display: grid;
      place-items: center;
      width: 42px;
      height: 32px;
      border-radius: 999px;
      background: var(--song-color-soft);
      color: var(--song-color);
      font-weight: 800;
      font-size: 14px;
      border: 1px solid color-mix(in srgb, var(--song-color) 52%, transparent);
      font-variant-numeric: tabular-nums;
    }
    .song-block {
      display: grid;
      gap: 6px;
      min-width: 0;
      padding-left: 10px;
      border-left: 3px solid var(--song-color);
    }
    .song-line {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      min-width: 0;
    }
    .song-title {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 14px;
      font-weight: 720;
    }
    .song-subtitle {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .track {
      width: min(100%, 860px);
      height: 11px;
      overflow: hidden;
      border-radius: 999px;
      background: var(--track);
      border: 1px solid rgba(51, 65, 85, 0.65);
    }
    .bar-fill {
      width: 0%;
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(
        90deg,
        var(--song-color),
        color-mix(in srgb, var(--song-color) 58%, white)
      );
      transition: width 520ms cubic-bezier(0.2, 0.8, 0.2, 1);
      box-shadow: 0 0 18px var(--song-color-soft);
    }
    .score {
      text-align: right;
      font-variant-numeric: tabular-nums;
    }
    .score strong {
      display: block;
      font-size: 15px;
    }
    .score span {
      display: block;
      margin-top: 3px;
      color: var(--muted);
      font-size: 12px;
    }
    .notes {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .note {
      border-radius: 12px;
      border: 1px solid rgba(51, 65, 85, 0.75);
      background: rgba(15, 23, 42, 0.62);
      padding: 12px;
    }
    .note strong {
      display: block;
      margin-bottom: 4px;
    }
    .empty-state {
      padding: 18px;
      border-radius: 12px;
      border: 1px dashed rgba(51, 65, 85, 0.9);
      color: var(--muted);
      text-align: center;
      background: rgba(15, 23, 42, 0.45);
    }
    .timeline-dock {
      position: fixed;
      left: 0;
      right: 0;
      bottom: 0;
      z-index: 3000;
      display: grid;
      gap: 10px;
      padding: 14px 16px 16px;
      border-top: 1px solid var(--border);
      background: rgba(15, 23, 42, 0.94);
      backdrop-filter: blur(12px);
    }
    .timeline-top {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr) auto;
      gap: 12px;
      align-items: center;
    }
    .timeline-meta {
      display: grid;
      gap: 2px;
      min-width: 0;
    }
    .timeline-date {
      font-weight: 750;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .timeline-count {
      color: var(--muted);
      font-size: 12px;
    }
    .play-button {
      width: 42px;
      height: 42px;
      border: 1px solid rgba(56, 189, 248, 0.65);
      border-radius: 999px;
      color: #e0f2fe;
      background: rgba(56, 189, 248, 0.14);
      cursor: pointer;
      font-weight: 800;
    }
    .play-button:hover {
      background: rgba(56, 189, 248, 0.22);
    }
    input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }
    @media (max-width: 1000px) {
      .hero {
        grid-template-columns: 1fr;
      }
      .metric-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      .rank-row {
        grid-template-columns: 46px minmax(0, 1fr);
      }
      .score {
        grid-column: 2;
        text-align: left;
        display: flex;
        gap: 8px;
        align-items: baseline;
      }
      .song-subtitle {
        white-space: normal;
      }
    }
    @media (max-width: 640px) {
      body {
        padding-bottom: 156px;
      }
      .toolbar label {
        flex: 1 1 140px;
      }
      .metric-grid {
        grid-template-columns: 1fr;
      }
      .section-head, .timeline-top {
        grid-template-columns: 1fr;
      }
      .timeline-top {
        display: grid;
      }
      .timeline-label {
        min-width: 0;
      }
    }
  </style>
</head>
<body>
  <div class="toolbar">
    <div class="mode-tabs" aria-label="Ranking mode">
      <button class="mode-tab is-active" type="button" data-mode="trending">Trending</button>
      <button class="mode-tab" type="button" data-mode="listenTime">Listen Time</button>
    </div>
    <label>
      Top songs
      <input id="top-limit" type="number" min="1" max="100" step="1" value="__INITIAL_TOP_LIMIT__">
    </label>
    <label>
      Rank spacing
      <select id="rank-density">
        <option value="comfortable">Comfortable</option>
        <option value="compact">Compact</option>
      </select>
    </label>
  </div>
  <main class="page">
    <section class="hero">
      <section class="card">
        <h1 id="page-title">Trending Rankings</h1>
        <p class="muted" id="page-subtitle">Duration-weighted exponential-decay Trending score over time for <span class="code-pill">__FOLDER__</span>, using a __TRENDING_HALF_LIFE__ half-life.</p>
        <div class="metric-grid">
          <div class="metric"><div class="label">Songs</div><div class="value" id="summary-song-count"></div></div>
          <div class="metric"><div class="label">History Events</div><div class="value" id="summary-history-count"></div></div>
          <div class="metric"><div class="label">Timeline Points</div><div class="value" id="summary-timeline-count"></div></div>
          <div class="metric"><div class="label">Range</div><div class="value" id="summary-range"></div></div>
        </div>
      </section>
      <aside class="card">
        <h3>Current Slice</h3>
        <div class="notes">
          <div class="note">
            <strong id="slice-leader"></strong>
            <div class="muted" id="slice-leader-subtitle"></div>
          </div>
          <div class="note">
            <strong id="slice-active"></strong>
            <div class="muted" id="slice-active-subtitle"></div>
          </div>
          <div class="note">
            <strong id="slice-note"></strong>
            <div class="muted" id="slice-note-subtitle"></div>
          </div>
        </div>
      </aside>
    </section>
    <section class="ranking-shell">
      <div class="section-head">
        <div>
          <h2 id="chart-title">Trending Bar Race</h2>
          <div class="muted" id="chart-caption"></div>
        </div>
        <div class="muted" id="rank-caption"></div>
      </div>
      <div id="empty-state" class="empty-state" hidden>No play history found.</div>
      <div id="chart-stage" class="chart-stage"></div>
    </section>
  </main>
  <section class="timeline-dock">
    <div class="timeline-top">
      <button id="play-toggle" class="play-button" type="button" title="Play timeline">▶</button>
      <div class="timeline-meta">
        <div class="timeline-date" id="timeline-date"></div>
        <div class="timeline-count" id="timeline-count"></div>
      </div>
      <label class="timeline-label">
        Playback speed
        <select id="play-speed">
          <option value="10">Slow · 10s/day</option>
          <option value="5" selected>Medium · 5s/day</option>
          <option value="3">Fast · 3s/day</option>
        </select>
      </label>
    </div>
    <input id="timeline-slider" type="range" min="0" max="0" step="any" value="0">
  </section>
  <script>
    const rawData = __DATA_JSON__;
    const summary = rawData.summary;
    const trendingHalfLifeSeconds = Math.max(1, Number(summary.trendingHalfLifeSeconds || 172800));
    const trendingHalfLifeLabel = '__TRENDING_HALF_LIFE__';
    const minVisibleScore = 1e-9;
    const secondsPerDay = 86400;
    const maxBarWidthPercent = 100;
    const minBarWidthPercent = 0.75;
    const rankingModes = {
      trending: {
        title: 'Trending Rankings',
        chartTitle: 'Trending Bar Race',
        subtitle: `Duration-weighted exponential-decay Trending score over time for <span class="code-pill">__FOLDER__</span>, using a ${trendingHalfLifeLabel} half-life.`,
      },
      listenTime: {
        title: 'Listen Time Rankings',
        chartTitle: 'Listen Time Bar Race',
        subtitle: 'Cumulative global listen-time rankings over time for <span class="code-pill">__FOLDER__</span>, matching the player’s metadata listens × track duration ranking.',
      },
    };
    const rowHeightByDensity = {
      comfortable: { height: 56, gap: 7 },
      compact: { height: 46, gap: 6 },
    };
    const state = {
      mode: 'trending',
      rows: new Map(),
      latestSnapshot: null,
      rafId: null,
      isPlaying: false,
      playAnimationFrame: null,
      lastPlaybackFrame: null,
    };

    function decodeTimestamps(encoded) {
      if (!Array.isArray(encoded)) {
        return [];
      }
      const decoded = [];
      let current = null;
      for (let index = 0; index < encoded.length; index += 1) {
        const value = Number(encoded[index]);
        if (!Number.isFinite(value)) {
          continue;
        }
        if (index === 0 || current === null) {
          current = value;
        } else {
          current += value;
        }
        decoded.push(current);
      }
      return decoded;
    }

    const timeline = decodeTimestamps(rawData.timeline);
    const songs = rawData.songs.map((song) => ({
      ...song,
      timestamps: decodeTimestamps(song.history),
    }));
    const timelineStart = Number(summary.firstTimestamp || 0);
    const timelineEnd = Math.max(
      Number(summary.lastTimestamp || 0),
      Number(summary.generatedTimestamp || 0),
    );
    const sliderMax = timeline.length
      ? Math.max(1, Math.min(12000, Math.max(1000, timeline.length * 4)))
      : 0;

    const elements = {
      topLimit: document.getElementById('top-limit'),
      rankDensity: document.getElementById('rank-density'),
      chartStage: document.getElementById('chart-stage'),
      emptyState: document.getElementById('empty-state'),
      chartCaption: document.getElementById('chart-caption'),
      chartTitle: document.getElementById('chart-title'),
      rankCaption: document.getElementById('rank-caption'),
      pageTitle: document.getElementById('page-title'),
      pageSubtitle: document.getElementById('page-subtitle'),
      slider: document.getElementById('timeline-slider'),
      playToggle: document.getElementById('play-toggle'),
      playSpeed: document.getElementById('play-speed'),
      timelineDate: document.getElementById('timeline-date'),
      timelineCount: document.getElementById('timeline-count'),
      modeTabs: Array.from(document.querySelectorAll('.mode-tab')),
    };

    function formatNumber(value) {
      return new Intl.NumberFormat().format(value);
    }

    function formatRate(value) {
      const rate = Number(value || 0);
      if (rate >= 100) {
        return rate.toFixed(0);
      }
      if (rate >= 10) {
        return rate.toFixed(1);
      }
      return rate.toFixed(2);
    }

    function formatHours(value) {
      const hours = Number(value || 0);
      if (hours >= 1000) {
        return new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 }).format(hours);
      }
      if (hours >= 100) {
        return hours.toFixed(1);
      }
      if (hours >= 10) {
        return hours.toFixed(2);
      }
      return hours.toFixed(3);
    }

    function formatListenDuration(seconds) {
      const totalSeconds = Math.max(0, Number(seconds || 0));
      const hours = totalSeconds / 3600;
      if (hours >= 1) {
        return `${formatHours(hours)}h`;
      }
      const minutes = totalSeconds / 60;
      if (minutes >= 1) {
        return `${minutes.toFixed(minutes >= 10 ? 1 : 2)}m`;
      }
      return `${Math.round(totalSeconds)}s`;
    }

    function formatDate(timestamp) {
      if (!Number.isFinite(timestamp)) {
        return '-';
      }
      return new Intl.DateTimeFormat(undefined, {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
      }).format(new Date(timestamp * 1000));
    }

    function formatDateTime(timestamp) {
      if (!Number.isFinite(timestamp)) {
        return '-';
      }
      return new Intl.DateTimeFormat(undefined, {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
      }).format(new Date(timestamp * 1000));
    }

    function hashString(value) {
      let hash = 2166136261;
      for (let index = 0; index < value.length; index += 1) {
        hash ^= value.charCodeAt(index);
        hash = Math.imul(hash, 16777619);
      }
      return hash >>> 0;
    }

    function colorForSong(songName) {
      const hue = hashString(songName) % 360;
      return {
        color: `hsl(${hue} 84% 64%)`,
        soft: `hsl(${hue} 84% 64% / 0.16)`,
      };
    }

    function countBefore(timestamps, timestamp) {
      let low = 0;
      let high = timestamps.length;
      while (low < high) {
        const middle = Math.floor((low + high) / 2);
        if (timestamps[middle] < timestamp) {
          low = middle + 1;
        } else {
          high = middle;
        }
      }
      return low;
    }

    function countThrough(timestamps, timestamp) {
      let low = 0;
      let high = timestamps.length;
      while (low < high) {
        const middle = Math.floor((low + high) / 2);
        if (timestamps[middle] <= timestamp) {
          low = middle + 1;
        } else {
          high = middle;
        }
      }
      return low;
    }

    function getTopLimit() {
      const requestedLimit = Math.max(1, Math.min(100, Number(elements.topLimit.value) || 1));
      elements.topLimit.value = requestedLimit;
      return requestedLimit;
    }

    function timestampForSlider(sliderValue) {
      if (!timeline.length) {
        return null;
      }
      if (sliderMax <= 0 || timelineEnd <= timelineStart) {
        return timelineStart;
      }
      const ratio = Math.max(0, Math.min(1, Number(sliderValue || 0) / sliderMax));
      return timelineStart + ((timelineEnd - timelineStart) * ratio);
    }

    function computeSnapshot(sliderValue) {
      const timestamp = timestampForSlider(sliderValue);
      const ranked = [];
      let totalScore = 0;
      let totalListenSeconds = 0;
      let prefixHistoryEvents = 0;
      let activeSongCount = 0;
      const mode = rankingModes[state.mode] ? state.mode : 'trending';

      for (const song of songs) {
        const prefixCount = countThrough(song.timestamps, timestamp);
        prefixHistoryEvents += prefixCount;
        const legacyCount = Math.max(0, Number(song.playCount || 0) - Number(song.historyCount || 0));
        const hasRetainedHistory = song.timestamps.length > 0;
        const legacyBaselineActive = legacyCount > 0 && (
          hasRetainedHistory ? timestamp >= song.timestamps[0] : timestamp >= timelineStart
        );
        const activeLegacyCount = legacyBaselineActive ? legacyCount : 0;

        if (mode !== 'listenTime' && prefixCount <= 0) {
          continue;
        }

        let score = 0;
        let listenSeconds = 0;
        let rankedListenCount = prefixCount;
        if (mode === 'listenTime') {
          rankedListenCount = prefixCount + activeLegacyCount;
          listenSeconds = rankedListenCount * Math.max(0, Number(song.durationSeconds || 0));
          score = listenSeconds / 3600;
        } else {
          const trendingWeight = Number(song.trendingWeight || 1);
          for (let index = 0; index < prefixCount; index += 1) {
            const age = timestamp - song.timestamps[index];
            if (age >= 0) {
              score += Math.pow(2, -age / trendingHalfLifeSeconds) * trendingWeight;
            }
          }
        }
        if (score <= minVisibleScore) {
          continue;
        }

        totalScore += score;
        totalListenSeconds += listenSeconds;
        activeSongCount += 1;
        ranked.push({
          name: song.name,
          score,
          count: rankedListenCount,
          timestampedCount: prefixCount,
          legacyCount: activeLegacyCount,
          listenSeconds,
          durationSeconds: Number(song.durationSeconds || 0),
          historyCount: prefixCount,
          firstListen: song.timestamps[0],
          lastListen: song.timestamps[prefixCount - 1],
          totalPlayCount: song.playCount,
          totalHistoryCount: song.historyCount,
        });
      }

      ranked.sort((left, right) => {
        const delta = right.score - left.score;
        if (Math.abs(delta) > 1e-12) {
          return delta;
        }
        return left.name.toLocaleLowerCase().localeCompare(right.name.toLocaleLowerCase());
      });

      const limit = getTopLimit();
      const selected = ranked.slice(0, limit);
      const maxScore = Math.max(...selected.map((song) => song.score), 1);
      return {
        sliderValue,
        timestamp,
        ranked,
        selected,
        maxScore,
        totalScore,
        totalListenSeconds,
        prefixHistoryEvents,
        activeSongCount,
        mode,
      };
    }

    function createRow(songName) {
      const row = document.createElement('article');
      row.className = 'rank-row';
      row.dataset.songName = songName;
      const color = colorForSong(songName);
      row.style.setProperty('--song-color', color.color);
      row.style.setProperty('--song-color-soft', color.soft);
      row.innerHTML = `
        <div class="rank"></div>
        <div class="song-block">
          <div class="song-line">
            <div class="song-title"></div>
            <div class="song-subtitle"></div>
          </div>
          <div class="track"><div class="bar-fill"></div></div>
        </div>
        <div class="score"><strong></strong><span></span></div>
      `;
      row.style.opacity = '0';
      elements.chartStage.appendChild(row);
      return row;
    }

    function updateRow(row, song, rank, maxScore, yPosition) {
      const ratio = maxScore > 0 ? song.score / maxScore : 0;
      const isListenTime = state.mode === 'listenTime';
      row.classList.toggle('is-leader', rank === 1);
      row.classList.remove('is-exiting');
      row.style.transform = `translateY(${yPosition}px)`;
      row.style.zIndex = String(1000 - rank);
      row.style.opacity = '1';
      row.querySelector('.rank').textContent = `#${rank}`;
      row.querySelector('.song-title').textContent = song.name;
      row.querySelector('.song-subtitle').textContent = isListenTime
        ? `${formatNumber(song.count)} listens (${formatNumber(song.timestampedCount)} timestamped + ${formatNumber(song.legacyCount)} legacy) × ${formatListenDuration(song.durationSeconds)} track`
        : `${formatNumber(song.count)} listens so far - last ${formatDate(song.lastListen)}`;
      row.querySelector('.bar-fill').style.width = `${Math.max(ratio * maxBarWidthPercent, minBarWidthPercent).toFixed(2)}%`;
      row.querySelector('.score strong').textContent = isListenTime
        ? formatListenDuration(song.listenSeconds)
        : `+${formatRate(song.score)}`;
      row.querySelector('.score span').textContent = isListenTime
        ? 'cumulative listen time'
        : `${trendingHalfLifeLabel} weighted decay`;
    }

    function setDensity() {
      const density = rowHeightByDensity[elements.rankDensity.value] || rowHeightByDensity.comfortable;
      document.documentElement.style.setProperty('--row-height', `${density.height}px`);
      document.documentElement.style.setProperty('--row-gap', `${density.gap}px`);
      return density;
    }

    function renderSnapshot(snapshot) {
      state.latestSnapshot = snapshot;
      renderModeHeader();
      const density = setDensity();
      const selectedNames = new Set(snapshot.selected.map((song) => song.name));
      const totalHeight = snapshot.selected.length
        ? snapshot.selected.length * density.height + Math.max(0, snapshot.selected.length - 1) * density.gap
        : density.height;
      elements.chartStage.style.height = `${totalHeight}px`;

      snapshot.selected.forEach((song, index) => {
        const yPosition = index * (density.height + density.gap);
        let row = state.rows.get(song.name);
        if (!row) {
          row = createRow(song.name);
          state.rows.set(song.name, row);
          row.style.transform = `translateY(${yPosition + 10}px)`;
          requestAnimationFrame(() => updateRow(row, song, index + 1, snapshot.maxScore, yPosition));
        } else {
          updateRow(row, song, index + 1, snapshot.maxScore, yPosition);
        }
      });

      for (const [songName, row] of state.rows.entries()) {
        if (selectedNames.has(songName)) {
          continue;
        }
        row.classList.add('is-exiting');
        row.style.opacity = '0';
        window.setTimeout(() => {
          if (!state.rows.has(songName)) {
            return;
          }
          const currentRow = state.rows.get(songName);
          if (currentRow === row && row.classList.contains('is-exiting')) {
            row.remove();
            state.rows.delete(songName);
          }
        }, 280);
      }

      renderSliceText(snapshot);
    }

    function renderModeHeader() {
      const mode = rankingModes[state.mode] || rankingModes.trending;
      elements.pageTitle.textContent = mode.title;
      elements.pageSubtitle.innerHTML = mode.subtitle;
      elements.chartTitle.textContent = mode.chartTitle;
      elements.modeTabs.forEach((tab) => {
        tab.classList.toggle('is-active', tab.dataset.mode === state.mode);
      });
    }

    function renderSliceText(snapshot) {
      const leader = snapshot.selected[0];
      const isListenTime = snapshot.mode === 'listenTime';
      elements.chartCaption.textContent = `Through ${formatDateTime(snapshot.timestamp)}`;
      elements.rankCaption.textContent = `${snapshot.selected.length} of ${formatNumber(snapshot.activeSongCount)} active songs`;
      elements.timelineDate.textContent = `Through ${formatDateTime(snapshot.timestamp)}`;
      elements.timelineCount.textContent = isListenTime
        ? `${formatListenDuration(snapshot.totalListenSeconds)} cumulative listen time - ${formatNumber(snapshot.prefixHistoryEvents)} prefix events`
        : `+${formatRate(snapshot.totalScore)} weighted score - ${formatNumber(snapshot.prefixHistoryEvents)} prefix events`;

      document.getElementById('slice-leader').textContent = leader
        ? `Leader: ${leader.name}`
        : 'Leader: -';
      if (isListenTime) {
        document.getElementById('slice-leader-subtitle').textContent = leader
          ? `${formatListenDuration(leader.listenSeconds)} from ${formatNumber(leader.count)} listens so far`
          : 'No songs with metadata listens and known durations yet.';
        document.getElementById('slice-active').textContent = `Ranked songs: ${formatNumber(snapshot.activeSongCount)}`;
        document.getElementById('slice-active-subtitle').textContent = `${formatListenDuration(snapshot.totalListenSeconds)} cumulative listen time across ranked songs`;
      } else {
        document.getElementById('slice-leader-subtitle').textContent = leader
          ? `+${formatRate(leader.score)} from ${formatNumber(leader.count)} listens so far`
          : 'No songs active yet.';
        document.getElementById('slice-active').textContent = `Active songs: ${formatNumber(snapshot.activeSongCount)}`;
        document.getElementById('slice-active-subtitle').textContent = `+${formatRate(snapshot.totalScore)} duration-weighted score with ${trendingHalfLifeLabel} half-life`;
      }

      if (summary.adjustedHistoryCount > 0 || summary.historylessPlayCount > 0) {
        document.getElementById('slice-note').textContent = 'History coverage';
        const parts = [];
        if (summary.adjustedHistoryCount > 0) {
          parts.push(`${formatNumber(summary.adjustedHistoryCount)} songs have legacy untimestamped counts`);
        }
        if (summary.historylessPlayCount > 0) {
          parts.push(`${formatNumber(summary.historylessPlayCount)} songs are count-only`);
        }
        if (isListenTime) {
          parts.push('listen-time tab includes legacy counts as a baseline');
        }
        document.getElementById('slice-note-subtitle').textContent = parts.join(' - ');
      } else if (isListenTime) {
        document.getElementById('slice-note').textContent = 'Formula';
        document.getElementById('slice-note-subtitle').textContent = 'Σ metadata listens so far × track duration';
      } else {
        document.getElementById('slice-note').textContent = 'Formula';
        document.getElementById('slice-note-subtitle').textContent = `Σ duration/210s × 2^(-(now - listen) / H), H = ${trendingHalfLifeLabel}`;
      }
    }

    function renderAtSlider() {
      if (!timeline.length) {
        return;
      }
      const sliderValue = Math.max(0, Math.min(sliderMax, Number(elements.slider.value) || 0));
      renderSnapshot(computeSnapshot(sliderValue));
    }

    function scheduleRender() {
      if (state.rafId !== null) {
        cancelAnimationFrame(state.rafId);
      }
      state.rafId = requestAnimationFrame(() => {
        state.rafId = null;
        renderAtSlider();
      });
    }

    function setMode(mode) {
      if (!rankingModes[mode] || state.mode === mode) {
        return;
      }
      state.mode = mode;
      scheduleRender();
    }

    function stopPlayback() {
      state.isPlaying = false;
      elements.playToggle.textContent = '▶';
      elements.playToggle.title = 'Play timeline';
      state.lastPlaybackFrame = null;
      if (state.playAnimationFrame !== null) {
        cancelAnimationFrame(state.playAnimationFrame);
        state.playAnimationFrame = null;
      }
    }

    function stepPlayback(frameTimestamp) {
      if (!state.isPlaying || !timeline.length) {
        return;
      }

      if (state.lastPlaybackFrame === null) {
        state.lastPlaybackFrame = frameTimestamp;
      }

      const elapsedMs = Math.max(0, frameTimestamp - state.lastPlaybackFrame);
      state.lastPlaybackFrame = frameTimestamp;
      const current = Number(elements.slider.value) || 0;
      if (current >= sliderMax) {
        stopPlayback();
        return;
      }

      const secondsPerTimelineDay = Math.max(0.1, Number(elements.playSpeed.value) || 5);
      const timelineSpanSeconds = Math.max(1, timelineEnd - timelineStart);
      const advancedTimelineSeconds = elapsedMs * secondsPerDay / (secondsPerTimelineDay * 1000);
      const sliderDelta = sliderMax * advancedTimelineSeconds / timelineSpanSeconds;
      const nextValue = Math.min(sliderMax, current + sliderDelta);
      elements.slider.value = nextValue;
      renderAtSlider();
      state.playAnimationFrame = requestAnimationFrame(stepPlayback);
    }

    function togglePlayback() {
      if (state.isPlaying) {
        stopPlayback();
        return;
      }
      if (!timeline.length) {
        return;
      }
      if (Number(elements.slider.value) >= sliderMax) {
        elements.slider.value = 0;
      }
      state.isPlaying = true;
      state.lastPlaybackFrame = null;
      elements.playToggle.textContent = '❚❚';
      elements.playToggle.title = 'Pause timeline';
      state.playAnimationFrame = requestAnimationFrame(stepPlayback);
    }

    function renderSummary() {
      document.getElementById('summary-song-count').textContent = formatNumber(summary.songCount);
      document.getElementById('summary-history-count').textContent = formatNumber(summary.historyEventCount);
      document.getElementById('summary-timeline-count').textContent = formatNumber(summary.timelinePointCount);
      document.getElementById('summary-range').textContent = summary.firstTimestamp
        ? `${formatDate(timelineStart)} to ${formatDate(timelineEnd)}`
        : '-';
    }

    function initialize() {
      renderSummary();
      renderModeHeader();
      elements.slider.max = sliderMax;
      elements.slider.value = sliderMax;

      if (!timeline.length) {
        elements.emptyState.hidden = false;
        elements.chartCaption.textContent = 'No play history available.';
        elements.rankCaption.textContent = '';
        elements.timelineDate.textContent = 'No timeline points';
        elements.timelineCount.textContent = '+0.00 weighted score';
        elements.playToggle.disabled = true;
        return;
      }

      elements.slider.addEventListener('input', () => {
        stopPlayback();
        scheduleRender();
      });
      elements.topLimit.addEventListener('input', scheduleRender);
      elements.rankDensity.addEventListener('change', scheduleRender);
      elements.modeTabs.forEach((tab) => {
        tab.addEventListener('click', () => setMode(tab.dataset.mode));
      });
      elements.playToggle.addEventListener('click', togglePlayback);
      elements.playSpeed.addEventListener('change', () => {
        if (state.isPlaying) {
          state.lastPlaybackFrame = null;
        }
      });

      renderAtSlider();
    }

    initialize();
  </script>
</body>
</html>
"""
    return (
        html.replace("__DATA_JSON__", payload_json)
        .replace("__FOLDER__", str(payload["summary"]["folder"]))
        .replace("__TRENDING_HALF_LIFE__", format_duration_label(TRENDING_HALF_LIFE_SECONDS))
        .replace("__INITIAL_TOP_LIMIT__", str(initial_top_limit))
    )


def main() -> int:
    global console
    console = require_console()
    args = parse_args()
    folder = Path(args.folder).expanduser().resolve()

    if not folder.is_dir():
        emit(f"Folder not found: {folder}")
        return 1

    try:
        song_counts, history_mapping, meta_path = load_folder_data(folder)
    except FileNotFoundError as exc:
        emit(str(exc))
        return 1

    duration_mapping, analysis_path = load_duration_mapping(folder)
    payload = build_trending_payload(
        folder=folder,
        song_counts=song_counts,
        history_mapping=history_mapping,
        duration_mapping=duration_mapping,
    )
    html = build_html(payload, initial_top_limit=max(1, args.top))
    output_path = (
        Path(args.output).expanduser()
        if args.output
        else Path.cwd() / DEFAULT_OUTPUT_TEMPLATE.format(folder=folder.name)
    )
    output_path.write_text(html, encoding="utf-8")

    print_banner("Trending Rankings", console)
    emit(f"Loaded metadata from {meta_path}")
    if duration_mapping:
        emit(f"Loaded durations from {analysis_path}")
    emit(f"Packed {payload['summary']['historyEventCount']} history events")
    emit(f"Wrote dashboard to {output_path.resolve()}")

    if payload["summary"]["mismatchedHistoryCount"] > 0:
        emit(
            "Note: "
            f"{payload['summary']['mismatchedHistoryCount']} songs have metadata "
            "counts that differ from timestamp history; Trending uses timestamped "
            "listens, while Listen Time includes legacy metadata counts as a baseline"
        )

    if not args.no_open:
        webbrowser.open(output_path.resolve().as_uri())
        emit("Opened dashboard in your browser")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
