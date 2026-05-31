#!/usr/bin/env python3
"""Search YouTube, select a video, and download it directly as MP3."""

import argparse
import json
import re
import subprocess
from pathlib import Path

from download_youtube_queue import download_and_convert, ensure_cli_tools_available
from cli_helpers import print_banner, print_message, require_console


def run_search(query: str, count: int):
    command = ["yt-dlp", "--dump-json", f"ytsearch{count}:{query}"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        error_text = result.stderr.strip() or result.stdout.strip() or "yt-dlp failed."
        raise RuntimeError(error_text)

    entries = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        title = data.get("title") or "Untitled"
        url = data.get("webpage_url") or ""
        if not url:
            video_id = data.get("id") or data.get("url")
            if video_id:
                url = f"https://www.youtube.com/watch?v={video_id}"
        duration = data.get("duration")
        uploader = data.get("uploader") or data.get("channel") or ""

        if url:
            entries.append(
                {
                    "title": title,
                    "url": url,
                    "duration": duration,
                    "uploader": uploader,
                }
            )

    return entries


def format_duration(seconds):
    if seconds is None:
        return ""
    try:
        total = int(seconds)
    except (TypeError, ValueError):
        return ""
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|]", "-", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.strip(".")
    return cleaned or "output"


def prompt_query(console, initial_query: str) -> str:
    from rich.prompt import Prompt

    query = initial_query.strip() if initial_query else ""
    while not query:
        query = Prompt.ask("Search YouTube for").strip()
        if not query:
            print_message("[red]Search query cannot be empty.[/red]", console)
    return query


def show_results(entries, console) -> None:
    from rich.table import Table

    table = Table(title="Search Results", header_style="bold cyan")
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Title", style="white")
    table.add_column("Duration", style="magenta", no_wrap=True)
    table.add_column("Uploader", style="green")

    for idx, entry in enumerate(entries, start=1):
        table.add_row(
            str(idx),
            entry["title"],
            format_duration(entry.get("duration")),
            entry.get("uploader") or "",
        )

    console.print(table)


def select_entry(entries, console):
    from rich.prompt import Prompt

    if not entries:
        return None

    show_results(entries, console)
    while True:
        choice = Prompt.ask(
            "Select a video by number (or 'q' to cancel)",
            default="1",
        ).strip()
        if choice.lower() in {"q", "quit", "exit"}:
            return None
        try:
            index = int(choice)
        except ValueError:
            print_message("[red]Please enter a number.[/red]", console)
            continue

        if 1 <= index <= len(entries):
            return entries[index - 1]

        print_message("[red]Selection out of range.[/red]", console)


def prompt_output_name(console, title: str) -> str:
    from rich.prompt import Prompt

    default_name = sanitize_filename(title)
    if not default_name.lower().endswith(".mp3"):
        default_name = f"{default_name}.mp3"

    output_name = Prompt.ask("Output MP3 name", default=default_name).strip()
    if not output_name:
        output_name = default_name
    if not output_name.lower().endswith(".mp3"):
        output_name = f"{output_name}.mp3"
    return output_name


def main():
    console = require_console()
    print_banner("[YT Search]", console)
    parser = argparse.ArgumentParser(
        description="Search YouTube, choose a result, and download it as MP3"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of search results to show (default: 5)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output MP3 if it already exists.",
    )
    parser.add_argument(
        "--temp-file",
        default="temp_output.wav",
        help="Temporary filename used between yt-dlp and ffmpeg (default: temp_output.wav)",
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Search terms (if omitted, you will be prompted)",
    )

    args = parser.parse_args()

    ensure_cli_tools_available(console)

    query = prompt_query(console, " ".join(args.query))

    try:
        entries = run_search(query, args.count)
    except RuntimeError as exc:
        print_message(f"[red]Search failed: {exc}[/red]", console)
        return

    if not entries:
        print_message("[yellow]No results found.[/yellow]", console)
        return

    selected = select_entry(entries, console)
    if not selected:
        print_message("[yellow]No selection made. Exiting.[/yellow]", console)
        return

    output_name = prompt_output_name(console, selected["title"])
    output_path = Path(output_name)
    if output_path.exists() and not args.overwrite:
        print_message(
            f"[red]Output file '{output_path}' already exists. Use --overwrite to replace it.[/red]",
            console,
        )
        return

    success, message = download_and_convert(
        selected["url"],
        output_path,
        Path(args.temp_file),
        console,
    )
    if success:
        print_message(f"[green]Downloaded:[/green] {selected['url']} -> {output_path}", console)
    else:
        print_message(f"[red]Download failed:[/red] {message}", console)


if __name__ == "__main__":
    main()
