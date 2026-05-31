#!/usr/bin/env python3
"""
Interactive YouTube search + audio player from the terminal (macOS-friendly).

Features:
- Prints top N YouTube search results (sorted by relevance).
- Enter a 1-indexed number to play **audio only** for that result.
- While audio is playing, press **Ctrl+G** to stop and return to the selector.
- Enter another number to play a different result, or 'q' to quit.

Install (macOS):
    pip install yt-dlp
    brew install ffmpeg       # for ffplay
    # or
    brew install mpv

Usage:
    python3 link_player.py "SEARCH WORDS" -n 15 --titles
"""

import argparse
import os
import select
import shutil
import subprocess
import sys

# POSIX-only (macOS/Linux).
import termios
import threading
import time
import tty
import re
from typing import Dict, List, Tuple, Optional

from yt_dlp import YoutubeDL

Result = Tuple[str, str, str]  # (title, url, uploader)


def search_youtube(query: str, limit: int) -> List[Result]:
    """Return [(title, url, uploader), ...] for the top-N results sorted by relevance."""
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)

    if not info:
        return []
    entries = info["entries"] if "entries" in info else [info]

    results: List[Result] = []
    for e in entries:
        url = e.get("webpage_url") or ""
        uploader = (
            e.get("uploader")
            or e.get("channel")
            or e.get("uploader_id")
            or e.get("channel_id")
            or "Unknown uploader"
        )
        if "/watch?v=" in url:
            results.append((e.get("title", ""), url, uploader))
            continue
        if (
            e.get("ie_key") == "Youtube" or e.get("extractor_key") == "Youtube"
        ) and e.get("id"):
            results.append(
                (
                    e.get("title", ""),
                    f"https://www.youtube.com/watch?v={e['id']}",
                    uploader,
                )
            )
    return results


def resolve_bestaudio_url(video_url: str) -> Tuple[str, str, Dict[str, str]]:
    """
    Use yt-dlp to resolve a *direct* bestaudio stream URL for the given YouTube link.
    Returns (direct_audio_url, title, http_headers).
    """
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "format": "bestaudio/best",  # prefer audio-only formats when available
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        direct_url = info.get("url")
        title = info.get("title", video_url)
        headers = info.get("http_headers") or {}
        if not direct_url:
            raise RuntimeError(
                "Could not resolve a direct audio stream URL for this video."
            )
        return direct_url, title, headers


def pick_player() -> Tuple[str, list]:
    """
    Return (name, base_command_list) for an available player.

    We REQUIRE mpv because we need --audio-device to force "External Headphones".
    """
    mpv = shutil.which("mpv")
    if not mpv:
        raise RuntimeError(
            "This script is configured to pin audio to 'External Headphones', "
            "which requires mpv.\n\nInstall mpv with:\n  brew install mpv"
        )
    return "mpv", [
        mpv,
        "--no-video",
        "--quiet",
        "--force-window=no",
        "--no-input-terminal",
        "--volume=75",
    ]



def _headers_to_ffplay(headers: Dict[str, str]) -> List[str]:
    """
    Convert headers dict to ffplay -headers argument list.
    ffplay expects a single string with CRLF line endings.
    """
    if not headers:
        return []
    header_lines = "".join(f"{k}: {v}\r\n" for k, v in headers.items())
    return ["-headers", header_lines]


def _headers_to_mpv(headers: Dict[str, str]) -> List[str]:
    """
    Convert headers dict to mpv --http-header-fields arguments.
    One --http-header-fields=Key: Value per header.
    """
    args = []
    for k, v in (headers or {}).items():
        args.append(f"--http-header-fields={k}: {v}")
    return args



FORCED_DEVICE_HUMAN_NAME = "External Headphones"

def mpv_audio_device_help(mpv_path: str) -> str:
    """Return mpv's device list output."""
    p = subprocess.run(
        [mpv_path, "--audio-device=help"],
        capture_output=True,
        text=True,
    )
    return (p.stdout or "") + (p.stderr or "")


def find_mpv_audio_device_token(mpv_path: str, wanted: str) -> str:
    """
    Find the actual mpv --audio-device token that corresponds to a human name,
    by scanning `mpv --audio-device=help`.

    Example return value (macOS): "coreaudio/External Headphones"
    """
    help_text = mpv_audio_device_help(mpv_path)
    wanted_lc = wanted.lower()

    candidates: List[str] = []

    for line in help_text.splitlines():
        if wanted_lc not in line.lower():
            continue

        # mpv help lines are usually like:
        #   'coreaudio/External Headphones' (External Headphones)
        # Extract the first token on the line, handling optional quotes.
        token = None
        m = re.search(r"^\s*['\"]([^'\"]+)['\"]", line)
        if m:
            token = m.group(1)
        else:
            m = re.search(r"^\s*(\S+)", line)
            if m:
                token = m.group(1)

        if token:
            candidates.append(token)

    # Prefer "coreaudio/..." on macOS if present
    for tok in candidates:
        if tok.lower().startswith("coreaudio/"):
            return tok

    if candidates:
        return candidates[0]

    raise RuntimeError(
        f"Could not find an mpv audio device matching '{wanted}'.\n\n"
        f"mpv --audio-device=help output:\n{help_text}"
    )


def spawn_player(
    direct_audio_url: str,
    headers: Dict[str, str],
    forced_audio_device: Optional[str],
) -> subprocess.Popen:
    """Start mpv pinned to the forced audio device (macOS)."""
    name, base = pick_player()
    assert name == "mpv"

    cmd = base
    if forced_audio_device:
        cmd = cmd + ["--audio-device="+forced_audio_device]

    cmd = cmd + _headers_to_mpv(headers) + [direct_audio_url]

    # Detach the player's stdin so our script can read Ctrl+G uninterrupted.
    return subprocess.Popen(cmd, stdin=subprocess.DEVNULL)


def _ctrl_g_watcher(stop_event: threading.Event, on_ctrl_g) -> None:
    """
    Watch stdin for Ctrl+G (BEL, ASCII 7) while playback is running.
    Puts the tty into cbreak mode so reads don't require Enter.
    """
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not stop_event.is_set():
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not rlist:
                continue
            ch = os.read(fd, 1)  # raw byte
            if not ch:
                continue
            if ch == b"\x07":  # Ctrl+G (BEL)
                on_ctrl_g()
                return
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def play_with_ctrl_g(
    direct_audio_url: str,
    title: str,
    headers: Dict[str, str],
    forced_audio_device: Optional[str],
) -> None:
    """
    Play the given direct audio URL. While playing, Ctrl+G stops playback.
    This returns only after playback stops/finishes.
    """
    print(f"\n Playing: {title}\n    (Press Ctrl+G to stop)\n")
    proc = spawn_player(direct_audio_url, headers, forced_audio_device)
    stop_event = threading.Event()

    def _stop_proc():
        if proc.poll() is None:
            try:
                proc.terminate()
                for _ in range(15):  # ~1.5s
                    if proc.poll() is not None:
                        break
                    time.sleep(0.1)
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass
        stop_event.set()

    watcher = threading.Thread(
        target=_ctrl_g_watcher, args=(stop_event, _stop_proc), daemon=True
    )
    watcher.start()

    try:
        proc.wait()
    finally:
        stop_event.set()
        watcher.join(timeout=0.5)
    print(" Stopped.\n")


def print_results(results: List[Result], show_titles: bool) -> None:
    for idx, (title, url, uploader) in enumerate(results, start=1):
        if show_titles:
            print(f"{idx:>2}. {url}\t# {title} | by {uploader}")
        else:
            print(f"{idx:>2}. {url}\t# by {uploader}")


def main():
    parser = argparse.ArgumentParser(
        description="YouTube search + interactive audio player."
    )
    parser.add_argument("query", nargs="*", help="search text")
    parser.add_argument("-n", "--num", type=int, default=10, help="number of results")
    parser.add_argument(
        "--titles", action="store_true", help="print titles next to URLs"
    )
    args = parser.parse_args()

    forced_audio_device: Optional[str] = None
    if sys.platform == "darwin":
        mpv_path = shutil.which("mpv")
        if not mpv_path:
            print(
                "mpv is required to pin audio to 'External Headphones'. "
                "Install with: brew install mpv",
                file=sys.stderr,
            )
            sys.exit(2)

        forced_audio_device = find_mpv_audio_device_token(
            mpv_path, FORCED_DEVICE_HUMAN_NAME
        )

    query = " ".join(args.query).strip() if args.query else input("Search: ").strip()
    if not query:
        print("No search text provided.", file=sys.stderr)
        sys.exit(1)

    try:
        results = search_youtube(query, args.num)
    except Exception as e:
        print(f"Error searching YouTube: {e}", file=sys.stderr)
        sys.exit(2)

    if not results:
        print("No results.")
        sys.exit(3)

    print_results(results, args.titles)

    while True:
        try:
            sel = input("\nEnter a 1-indexed number to play (or 'q' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if sel.lower() in {"q", "quit", "exit"}:
            break
        if not sel.isdigit():
            print("Please enter a number or 'q' to quit.")
            continue

        idx = int(sel)
        if idx < 1 or idx > len(results):
            print(f"Out of range (1..{len(results)}).")
            continue

        title, watch_url, _uploader = results[idx - 1]
        try:
            direct_url, real_title, headers = resolve_bestaudio_url(watch_url)
        except Exception as e:
            print(f"Failed to resolve audio stream: {e}")
            continue

        try:
            play_with_ctrl_g(direct_url, real_title or title, headers, forced_audio_device)

        except RuntimeError as e:
            print(str(e))
        except FileNotFoundError:
            print(
                "Could not start a media player. Install FFmpeg (ffplay) with:\n"
                "  brew install ffmpeg\n"
                "or install mpv with:\n"
                "  brew install mpv"
            )
        except Exception as e:
            print(f"Playback failed: {e}")

    print("Bye!")


if __name__ == "__main__":
    main()
