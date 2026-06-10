from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, url_for
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SOURCE_DIRS = (STATIC_DIR / "mp3", STATIC_DIR / "mid-mp3s")
UPLOAD_DIR = STATIC_DIR / "trim-uploads"
OUTPUT_DIR = STATIC_DIR / "trimmed"
ALLOWED_DIRS = SOURCE_DIRS + (UPLOAD_DIR, OUTPUT_DIR)

app = Flask(__name__)


def ensure_dirs() -> None:
    for folder in ALLOWED_DIRS:
        folder.mkdir(parents=True, exist_ok=True)


def require_tools() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(f"Missing required tool(s): {names}. Install FFmpeg first.")


def is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def resolve_audio_path(rel_path: str) -> Path:
    if not rel_path:
        raise ValueError("Missing MP3 path.")

    path = (BASE_DIR / rel_path).resolve()
    if path.suffix.lower() != ".mp3":
        raise ValueError("Only MP3 files are supported.")
    if not path.is_file():
        raise ValueError("MP3 file was not found.")
    if not any(is_under(path, root) for root in ALLOWED_DIRS):
        raise ValueError("MP3 path is outside the allowed folders.")
    return path


def rel_for(path: Path) -> str:
    return path.resolve().relative_to(BASE_DIR).as_posix()


def static_url(path: Path) -> str:
    return url_for("static", filename=path.resolve().relative_to(STATIC_DIR).as_posix())


def bytes_label(size: int) -> str:
    units = ("B", "KB", "MB", "GB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def ffprobe(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=True)
    data = json.loads(proc.stdout or "{}")
    fmt = data.get("format") or {}
    streams = data.get("streams") or []
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})

    duration = audio_stream.get("duration") or fmt.get("duration") or 0
    bit_rate = audio_stream.get("bit_rate") or fmt.get("bit_rate") or 0
    sample_rate = audio_stream.get("sample_rate") or 0

    try:
        duration_seconds = max(0.0, float(duration))
    except (TypeError, ValueError):
        duration_seconds = 0.0
    try:
        bit_rate_int = int(float(bit_rate))
    except (TypeError, ValueError):
        bit_rate_int = 0
    try:
        sample_rate_int = int(float(sample_rate))
    except (TypeError, ValueError):
        sample_rate_int = 0

    return {
        "duration": duration_seconds,
        "duration_ms": int(round(duration_seconds * 1000)),
        "bit_rate": bit_rate_int,
        "bit_rate_label": f"{round(bit_rate_int / 1000)} kbps" if bit_rate_int else "unknown",
        "sample_rate": sample_rate_int,
        "codec": audio_stream.get("codec_name") or "mp3",
        "size": path.stat().st_size,
        "size_label": bytes_label(path.stat().st_size),
    }


def clean_output_name(name: str, input_path: Path, start_ms: int, end_ms: int) -> str:
    fallback = f"{input_path.stem}_trim_{start_ms}-{end_ms}.mp3"
    cleaned = secure_filename(name or fallback)
    if not cleaned:
        cleaned = secure_filename(fallback)
    if not cleaned.lower().endswith(".mp3"):
        cleaned += ".mp3"
    return cleaned


def next_available_output(filename: str) -> Path:
    path = OUTPUT_DIR / filename
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    for index in range(2, 1000):
        candidate = OUTPUT_DIR / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise ValueError("Could not find an available output filename.")


def parse_ms(value: Any, label: str) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number of milliseconds.") from None
    if parsed < 0:
        raise ValueError(f"{label} cannot be negative.")
    return parsed


def trim_mp3_copy(input_path: Path, output_path: Path, start_ms: int, end_ms: int) -> None:
    duration_seconds = (end_ms - start_ms) / 1000
    start_seconds = start_ms / 1000

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-ss",
        f"{start_seconds:.6f}",
        "-i",
        str(input_path),
        "-t",
        f"{duration_seconds:.6f}",
        "-map",
        "0",
        "-c",
        "copy",
        "-map_metadata",
        "0",
        "-id3v2_version",
        "3",
        "-avoid_negative_ts",
        "make_zero",
        str(output_path),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        message = proc.stderr.strip() or "FFmpeg failed while trimming the MP3."
        raise RuntimeError(message[-1200:])


def list_mp3_files() -> list[dict[str, Any]]:
    ensure_dirs()
    files: list[dict[str, Any]] = []
    labels = {
        SOURCE_DIRS[0]: "static/mp3",
        SOURCE_DIRS[1]: "static/mid-mp3s",
        UPLOAD_DIR: "uploads",
        OUTPUT_DIR: "trimmed",
    }
    for folder in ALLOWED_DIRS:
        for path in sorted(folder.glob("*.mp3"), key=lambda p: p.name.lower()):
            files.append(
                {
                    "name": path.name,
                    "folder": labels.get(folder, folder.name),
                    "path": rel_for(path),
                    "size": path.stat().st_size,
                    "size_label": bytes_label(path.stat().st_size),
                }
            )
    return files


@app.route("/")
def index():
    return render_template("trimmer.html")


@app.route("/api/files")
def api_files():
    return jsonify({"files": list_mp3_files()})


@app.route("/api/probe")
def api_probe():
    try:
        path = resolve_audio_path(request.args.get("path", ""))
        meta = ffprobe(path)
        return jsonify(
            {
                "file": {
                    "name": path.name,
                    "path": rel_for(path),
                    "url": static_url(path),
                    **meta,
                }
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/upload", methods=["POST"])
def api_upload():
    try:
        ensure_dirs()
        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename:
            raise ValueError("Choose an MP3 file to upload.")
        if not uploaded.filename.lower().endswith(".mp3"):
            raise ValueError("Only MP3 uploads are supported.")

        filename = secure_filename(uploaded.filename)
        if not filename:
            filename = "upload.mp3"
        if not filename.lower().endswith(".mp3"):
            filename += ".mp3"

        target = UPLOAD_DIR / filename
        if target.exists():
            stem = target.stem
            suffix = target.suffix
            for index in range(2, 1000):
                candidate = UPLOAD_DIR / f"{stem}-{index}{suffix}"
                if not candidate.exists():
                    target = candidate
                    break
        uploaded.save(target)
        return jsonify({"file": {"name": target.name, "path": rel_for(target), "url": static_url(target)}})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/trim", methods=["POST"])
def api_trim():
    try:
        require_tools()
        ensure_dirs()
        payload = request.get_json(silent=True) or {}
        input_path = resolve_audio_path(str(payload.get("path") or ""))
        start_ms = parse_ms(payload.get("start_ms"), "Start")
        end_ms = parse_ms(payload.get("end_ms"), "End")
        if end_ms <= start_ms:
            raise ValueError("End must be after start.")

        source_meta = ffprobe(input_path)
        source_duration_ms = int(source_meta.get("duration_ms") or 0)
        if source_duration_ms and end_ms > source_duration_ms + 100:
            raise ValueError("End is past the MP3 duration.")

        filename = clean_output_name(str(payload.get("output_name") or ""), input_path, start_ms, end_ms)
        filename = re.sub(r"[/\\\\]+", "_", filename)
        output_path = next_available_output(filename)

        trim_mp3_copy(input_path, output_path, start_ms, end_ms)
        output_meta = ffprobe(output_path)

        return jsonify(
            {
                "output": {
                    "name": output_path.name,
                    "path": rel_for(output_path),
                    "url": static_url(output_path),
                    **output_meta,
                },
                "source": source_meta,
                "requested": {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "duration_ms": end_ms - start_ms,
                },
                "mode": "copy",
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    ensure_dirs()
    require_tools()
    app.run(debug=True, port=5111)
