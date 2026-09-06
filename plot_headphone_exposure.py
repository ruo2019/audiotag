#!/usr/bin/env python3
"""Build an offline, interactive dashboard from headphone_exposure JSONL logs.

Uses only the Python standard library. Reads logs without importing or starting
the audio recorder. All charts and data are embedded in the generated HTML.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import webbrowser
from collections import defaultdict
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "headphone_exposure"
DEFAULT_OUTPUT = ROOT / "headphone_exposure_dashboard.html"
TEMPLATE = ROOT / "templates" / "headphone_exposure_dashboard.html"


def number(value: object) -> float | None:
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        value = float(value)
        if math.isfinite(value):
            return value
    return None


def energy_average(values: list[float]) -> float | None:
    """Equal-weight logarithmic average, evaluated without exponent overflow."""
    if not values:
        return None
    highest = max(values)
    return highest + 10 * math.log10(
        sum(10 ** ((value - highest) / 10) for value in values) / len(values)
    )


def average(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def read_records(directory: Path) -> tuple[list[dict], dict]:
    paths = sorted({*directory.glob("????-??.jsonl"), *directory.glob("????-??.jsonl.gz")})
    records = []
    seen = set()
    quality = {"files": len(paths), "invalid_rows": 0, "duplicate_rows": 0, "dose_resets": 0}
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("expected an object")
                    timestamp = datetime.fromisoformat(str(row.get("timestamp", "")))
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.astimezone()
                    level = number(row.get("average_db_a"))
                    if level is None or not -200 <= level <= 200:
                        raise ValueError("invalid level")
                except (ValueError, TypeError, OverflowError):
                    quality["invalid_rows"] += 1
                    continue
                volume = number(row.get("mac_volume_percent"))
                dose = number(row.get("weekly_dose_percent"))
                source_level = number(row.get("source_average_dbfs_a"))
                record = {
                    "time": timestamp,
                    "level": level,
                    "volume": volume if volume is not None and 0 <= volume <= 100 else None,
                    "dose": dose if dose is not None and dose >= 0 else None,
                    "source": source_level if source_level is not None and -200 <= source_level <= 200 else None,
                }
                identity = tuple(record.values())
                if identity in seen:
                    quality["duplicate_rows"] += 1
                    continue
                seen.add(identity)
                records.append(record)
    records.sort(key=lambda row: row["time"].timestamp())
    return records, quality


def build_report(directory: Path) -> dict:
    records, quality = read_records(directory)
    buckets: dict[int, list[dict]] = defaultdict(list)
    daily_dose: dict[str, float] = defaultdict(float)
    weekly_dose = {}
    previous_dose = {}
    for row in records:
        timestamp = row["time"]
        day = timestamp.date()
        week = (day - timedelta(days=day.weekday())).isoformat()
        buckets[int(timestamp.timestamp() // 60)].append(row)
        dose = row["dose"]
        if dose is not None:
            previous = previous_dose.get(week)
            # The first reading is a baseline: it may include earlier history
            # absent from these files. Never assign that history to its date.
            if previous is not None:
                if dose >= previous:
                    daily_dose[day.isoformat()] += dose - previous
                else:
                    quality["dose_resets"] += 1
            previous_dose[week] = dose
            weekly_dose[week] = dose

    days: dict[str, list[dict]] = defaultdict(list)
    for epoch_minute, rows in sorted(buckets.items()):
        timestamp = rows[0]["time"].replace(second=0, microsecond=0)
        days[timestamp.date().isoformat()].append({
            "minute": timestamp.hour * 60 + timestamp.minute,
            "epoch": epoch_minute,
            "time": timestamp.isoformat(timespec="minutes"),
            "level": energy_average([row["level"] for row in rows]),
            "volume": average([row["volume"] for row in rows]),
            "source": energy_average([row["source"] for row in rows if row["source"] is not None]),
        })

    output_days = []
    for day, minutes in sorted(days.items()):
        date = datetime.fromisoformat(day).date()
        levels = [minute["level"] for minute in minutes]
        hours = [0] * 24
        for minute in minutes:
            hours[minute["minute"] // 60] += 1
        output_days.append({
            "date": day,
            "week": (date - timedelta(days=date.weekday())).isoformat(),
            "count": len(minutes),
            "level": energy_average(levels),
            "maximum": max(levels),
            "volume": average([minute["volume"] for minute in minutes]),
            "dose": daily_dose.get(day),
            "hours": hours,
            "minutes": minutes,
        })

    return {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "days": output_days,
        "weekly_dose": weekly_dose,
        "quality": quality,
        "records": len(records),
    }


def render_html(report: dict, *, live: bool = False) -> str:
    # JSON is inert script data, but an HTML parser still recognizes </script>.
    payload = json.dumps(report, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return TEMPLATE.read_text(encoding="utf-8").replace("__REPORT_JSON__", payload).replace(
        "__LIVE_MODE__", "true" if live else "false"
    )


def serve(directory: Path, port: int, open_browser: bool) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            route = urlsplit(self.path).path
            if route not in ("/", "/data"):
                self.send_error(404)
                return
            try:
                report = build_report(directory)
                if route == "/data":
                    body = json.dumps(report, allow_nan=False).encode("utf-8")
                    content_type = "application/json; charset=utf-8"
                else:
                    body = render_html(report, live=True).encode("utf-8")
                    content_type = "text/html; charset=utf-8"
            except (OSError, ValueError) as error:
                print(f"Could not refresh logs: {error}", file=sys.stderr)
                self.send_error(500, "Could not read exposure logs")
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *args: object) -> None:
            pass

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        url = f"http://127.0.0.1:{server.server_port}"
        print(f"Live dashboard: {url} (refreshes every 15 seconds; Ctrl-C to stop)", flush=True)
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="directory of monthly .jsonl / .jsonl.gz logs")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="standalone HTML output")
    parser.add_argument("--no-open", action="store_true", help="do not open the browser")
    parser.add_argument("--live", action="store_true", help="serve a dashboard that refreshes every 15 seconds")
    parser.add_argument("--port", type=int, default=0, help="live server port (default: choose a free port)")
    args = parser.parse_args()
    if not args.input.is_dir():
        parser.error(f"log directory does not exist: {args.input}")
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    try:
        report = build_report(args.input)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render_html(report), encoding="utf-8")
        count = sum(day["count"] for day in report["days"])
        print(f"Dashboard: {args.output.resolve()}\n{count:,} recorded minutes across {len(report['days'])} days.")
        if report["quality"]["invalid_rows"]:
            print(f"Skipped {report['quality']['invalid_rows']} invalid/incomplete rows.", file=sys.stderr)
        if args.live:
            serve(args.input, args.port, not args.no_open)
        elif not args.no_open:
            webbrowser.open(args.output.resolve().as_uri())
    except (OSError, ValueError) as error:
        print(f"Could not build dashboard: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
