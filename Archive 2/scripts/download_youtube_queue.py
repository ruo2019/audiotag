#!/usr/bin/env python3
"""Download YouTube URLs from a queue file and convert them to MP3."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from cli_helpers import print_banner, print_message, require_console

console = None


def ensure_cli_tools_available(console=None) -> None:
    """Validate that yt-dlp and ffmpeg are installed."""
    missing = [tool for tool in ("yt-dlp", "ffmpeg") if shutil.which(tool) is None]
    if missing:
        message = (
            f"Missing required tool(s): {', '.join(missing)}. Install them and try again."
        )
        print_message(f"[red]{message}[/red]" if console else message, console)
        sys.exit(1)


def parse_queue_file(queue_path: Path):
    """Parse lines in the queue file into (url, output_path) tuples."""
    entries = []
    with queue_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(
                    f"Line {line_number} is invalid. Each line must include a URL and an output filename."
                )

            url, filename = parts
            entries.append((url, Path(filename)))
    return entries


def run_command(command):
    """Run a shell command and capture output."""
    return subprocess.run(command, capture_output=True, text=True)


def download_and_convert(url: str, output_path: Path, temp_path: Path, console=None):
    """Download audio via yt-dlp then convert to MP3 with ffmpeg."""
    try:
        if temp_path.exists():
            temp_path.unlink()
    except OSError as exc:
        return False, f"Could not remove existing temp file '{temp_path}': {exc}"

    try:
        yt_command = [
            "yt-dlp",
            "--cookies-from-browser",
            "chrome",
            "-x",
            url,
            "--audio-format",
            "wav",
            "-o",
            str(temp_path),
        ]
        if console:
            console.print(f"[dim]$ {' '.join(yt_command)}[/dim]")
        else:
            print(" ".join(yt_command))
        yt_result = run_command(yt_command)
        if yt_result.returncode != 0:
            error_text = yt_result.stderr.strip() or yt_result.stdout.strip() or "yt-dlp failed."
            return False, f"yt-dlp error: {error_text}"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        ffmpeg_command = [
            "ffmpeg",
            "-y",
            "-i",
            str(temp_path),
            "-c:a",
            "libmp3lame",
            "-q:a",
            "0",
            str(output_path),
        ]
        ff_result = run_command(ffmpeg_command)
        if ff_result.returncode != 0:
            error_text = ff_result.stderr.strip() or ff_result.stdout.strip() or "ffmpeg failed."
            return False, f"ffmpeg error: {error_text}"

        return True, "Downloaded and converted"
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass


def main():
    global console
    console = require_console()
    from rich.progress import (
        Progress,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeRemainingColumn,
    )
    from rich.panel import Panel
    from rich.align import Align
    from rich.live import Live

    print_banner("[Queue Downloader]", console)
    parser = argparse.ArgumentParser(
        description="Download a queue of YouTube URLs to MP3 files."
    )
    parser.add_argument(
        "queue_file",
        nargs="?",
        default="download_queue.txt",
        help="Path to the download queue file (default: download_queue.txt)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files even if the target MP3 already exists.",
    )
    parser.add_argument(
        "--temp-file",
        default="temp_output.wav",
        help="Temporary filename used between yt-dlp and ffmpeg (default: temp_output.wav)",
    )

    args = parser.parse_args()

    queue_path = Path(args.queue_file)
    temp_path = Path(args.temp_file)

    if not queue_path.exists():
        console.print(f"[red]Error: Queue file '{queue_path}' does not exist.[/red]")
        sys.exit(1)

    ensure_cli_tools_available(console)

    try:
        queue_entries = parse_queue_file(queue_path)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    if not queue_entries:
        console.print(f"[yellow]No downloads found in '{queue_path}'.[/yellow]")
        sys.exit(0)

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    task = progress.add_task(
        f"Downloading ({len(queue_entries)} item(s)) ", total=len(queue_entries)
    )
    panel = Panel(Align.center(progress), expand=False)

    results = []
    with Live(panel, console=console, refresh_per_second=10) as live:
        for url, output_path in queue_entries:
            if output_path.exists() and not args.overwrite:
                results.append(
                    {
                        "name": str(output_path),
                        "status": "skipped",
                        "message": "Already exists, skipping.",
                    }
                )
            else:
                success, message = download_and_convert(
                    url,
                    output_path,
                    temp_path,
                    console,
                )
                results.append(
                    {
                        "name": str(output_path),
                        "status": "success" if success else "error",
                        "message": message,
                    }
                )

            progress.update(task, advance=1)
            live.update(Panel(Align.center(progress), expand=False))

    success_count = len([r for r in results if r["status"] == "success"])
    skip_count = len([r for r in results if r["status"] == "skipped"])

    print_message(
        f"[bold green]Finished processing queue.[/bold green] "
        f"{success_count}/{len(queue_entries)} downloaded, {skip_count} skipped.",
        console,
    )

    for result in results:
        if result["status"] == "success":
            console.print(f"[green]✓[/green] {result['name']}: {result['message']}")
        elif result["status"] == "skipped":
            console.print(f"[yellow]-[/yellow] {result['name']}: {result['message']}")
        else:
            console.print(f"[red]✗[/red] {result['name']}: {result['message']}")


if __name__ == "__main__":
    main()
