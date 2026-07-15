import argparse
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import pygame
except ImportError:
    pygame = None


DEFAULT_MP3_FOLDER = Path("static") / "mp3"
DEFAULT_DELAY_BETWEEN_PLAYS = 10
DEFAULT_DELAY_BETWEEN_TRACKS = 1
DEFAULT_OUTPUT_DEVICE = "External Headphones"
DEFAULT_VOLUME = 0.1
DEFAULT_CHOICE_COUNT = 20
BRACKET_ARTIST_RE = re.compile(r"^\[([^\]]+)\]\s*(.+)$")


def list_pygame_output_devices() -> List[str]:
    if pygame is None:
        return []
    try:
        import pygame._sdl2.audio as sdl2_audio  # type: ignore

        if not pygame.get_init():
            pygame.init()
        return [str(d) for d in sdl2_audio.get_audio_device_names(False)]
    except Exception:
        return []


def init_audio(output_device: Optional[str], strict: bool, volume: float) -> None:
    if pygame is None:
        raise RuntimeError("pygame is required for playback. Install with: pip install -U pygame")

    if output_device:
        devices = list_pygame_output_devices()
        if strict and devices and output_device not in devices:
            raise RuntimeError(
                f"Output device '{output_device}' not found.\n"
                f"Available SDL devices: {devices}\n"
                "Use --list-output-devices and copy an exact name."
            )

    try:
        pygame.mixer.pre_init(44100, -16, 2, 2048, devicename=output_device)
        pygame.init()
        pygame.mixer.init(44100, -16, 2, 2048, devicename=output_device)
    except TypeError:
        if output_device and strict:
            raise RuntimeError(
                "Your pygame build does not support devicename=. "
                "Upgrade pygame or pass --no-strict-output-device."
            )
        pygame.mixer.pre_init(44100, -16, 2, 2048)
        pygame.init()
        pygame.mixer.init()
    pygame.mixer.music.set_volume(max(0.0, min(1.0, volume)))


def shutdown_audio() -> None:
    if pygame is None:
        return
    try:
        pygame.mixer.quit()
    except Exception:
        pass
    try:
        pygame.quit()
    except Exception:
        pass


def load_json(path: Path) -> object:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{path} is not valid JSON, so I will not overwrite it. "
            f"Fix the file or restore {path.name}.bak if it exists. ({exc})"
        ) from exc


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    backup = path.with_suffix(path.suffix + ".bak")
    if path.exists():
        backup.write_bytes(path.read_bytes())
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    tmp.replace(path)


def atomic_write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def default_listen_counts_file(mp3_folder: Path) -> Path:
    if mp3_folder.name == "mid-mp3s":
        return Path("mid_listen_counts.json")
    return Path("listen_counts.json")


def load_listen_counts(path: Path, mp3_folder: Path) -> Dict[str, int]:
    raw = load_json(path)
    counts = raw if isinstance(raw, dict) else {}
    out: Dict[str, int] = {}
    for key, value in counts.items():
        try:
            out[str(key)] = int(value)
        except Exception:
            out[str(key)] = 0
    for track in get_mp3_files(mp3_folder):
        out.setdefault(track.stem, 0)
    return out


def save_listen_counts(path: Path, counts: Dict[str, int]) -> None:
    atomic_write_json(path, counts)


def increment_listen_count(path: Path, mp3_folder: Path, track: Path) -> int:
    counts = load_listen_counts(path, mp3_folder)
    counts[track.stem] = counts.get(track.stem, 0) + 1
    save_listen_counts(path, counts)
    return counts[track.stem]


def normalize_artist_names(values: object) -> List[str]:
    if isinstance(values, str):
        raw_items: Iterable[object] = [values]
    elif isinstance(values, list):
        raw_items = values
    elif isinstance(values, Iterable):
        raw_items = values
    else:
        raw_items = []

    seen = set()
    names: List[str] = []
    for item in raw_items:
        name = " ".join(str(item).strip().split())
        key = name.casefold()
        if name and key not in seen:
            seen.add(key)
            names.append(name)
    return names


def load_artists(path: Path) -> Dict[str, List[str]]:
    raw = load_json(path)
    if not isinstance(raw, dict):
        return {}
    artists: Dict[str, List[str]] = {}
    for key, value in raw.items():
        names = normalize_artist_names(value)
        if names:
            artists[str(key)] = names
    return artists


def default_artists_file(mp3_folder: Path) -> Path:
    if mp3_folder.name == "mid-mp3s":
        return Path("mid_artists.json")
    return Path("artists.json")


def default_reviewed_file(mp3_folder: Path) -> Path:
    if mp3_folder.name == "mid-mp3s":
        return Path("mid_artists_reviewed.json")
    return Path("artists_reviewed.json")


def get_mp3_files(mp3_folder: Path) -> List[Path]:
    if not mp3_folder.is_dir():
        return []
    return sorted(
        p
        for p in mp3_folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".mp3" and not p.name.startswith(".")
    )


def load_reviewed(path: Path) -> set[str]:
    raw = load_json(path)
    if isinstance(raw, list):
        return {str(item) for item in raw if str(item).strip()}
    if isinstance(raw, dict):
        return {str(key) for key, value in raw.items() if value}
    return set()


def save_reviewed(path: Path, reviewed: set[str]) -> None:
    save_json(path, sorted(reviewed))


def choose_tracks(
    mp3_folder: Path,
    artists: Dict[str, List[str]],
    include_existing: bool,
    review_existing: bool,
    reviewed: set[str],
) -> List[Path]:
    tracks = get_mp3_files(mp3_folder)
    if review_existing:
        return [
            track
            for track in tracks
            if artists.get(track.stem) and track.stem not in reviewed
        ]
    if include_existing:
        return tracks
    return [track for track in tracks if not artists.get(track.stem)]


def inferred_artist_from_filename(track: Path) -> List[str]:
    match = BRACKET_ARTIST_RE.match(track.stem)
    if not match:
        return []
    return split_artist_input(match.group(1).strip())


def iter_artist_names(artists: Dict[str, List[str]]) -> Iterable[str]:
    for names in artists.values():
        for name in names:
            if name:
                yield name


def format_artist_list(names: List[str]) -> str:
    return ", ".join(names)


def split_artist_input(value: str) -> List[str]:
    normalized = re.sub(r"\s+(?:and|&|\+)\s+", ",", value.strip(), flags=re.IGNORECASE)
    parts = re.split(r"\s*[,;/]\s*", normalized)
    return normalize_artist_names(parts)


def artist_choices(
    artists: Dict[str, List[str]],
    track: Path,
    current_artists: List[str],
    recent_artists: List[str],
    max_choices: int,
) -> List[str]:
    seen = set()
    choices: List[str] = []

    def add(name: Optional[str]) -> None:
        if not name:
            return
        normalized = " ".join(name.strip().split())
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            choices.append(normalized)

    for name in current_artists:
        add(name)
    for name in inferred_artist_from_filename(track):
        add(name)
    for name in recent_artists:
        add(name)

    counts = Counter(iter_artist_names(artists))
    for name, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0].casefold())):
        add(name)
        if len(choices) >= max_choices:
            break

    return choices[:max_choices]


def all_known_artist_names(artists: Dict[str, List[str]], recent_artists: List[str]) -> List[str]:
    seen = set()
    names: List[str] = []

    def add(name: str) -> None:
        normalized = " ".join(str(name).strip().split())
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            names.append(normalized)

    for name in recent_artists:
        add(name)
    for name, _count in sorted(
        Counter(iter_artist_names(artists)).items(),
        key=lambda item: (-item[1], item[0].casefold()),
    ):
        add(name)
    return names


def resolve_artist_text(value: str, known_names: List[str]) -> str:
    exact = [name for name in known_names if name.casefold() == value.casefold()]
    if exact:
        return exact[0]

    prefix = [name for name in known_names if name.casefold().startswith(value.casefold())]
    if len(prefix) == 1:
        return prefix[0]

    contains = [name for name in known_names if value.casefold() in name.casefold()]
    if len(contains) == 1:
        return contains[0]

    return value


def resolve_artist_names(values: List[str], known_names: List[str]) -> List[str]:
    return normalize_artist_names(resolve_artist_text(value, known_names) for value in values)


def parse_artist_tokens(value: str, choices: List[str], known_names: List[str]) -> Optional[List[str]]:
    picked: List[str] = []
    for token in split_artist_input(value):
        if token.isdigit() and choices:
            idx = int(token)
            if not (1 <= idx <= len(choices)):
                print(f"No choice {idx}; skipped.")
                return None
            picked.append(choices[idx - 1])
        else:
            picked.append(resolve_artist_text(token, known_names))
    return normalize_artist_names(picked)


def play_mp3_loop(
    mp3_path: Path,
    stop_event: threading.Event,
    delay_between_plays: int,
    listen_counts_file: Optional[Path],
    mp3_folder: Path,
) -> None:
    if pygame is None:
        print("pygame not installed; cannot play.")
        return

    while not stop_event.is_set():
        try:
            pygame.mixer.music.load(str(mp3_path))
            pygame.mixer.music.play()
        except Exception as exc:
            print(f"Error: could not play '{mp3_path}': {exc}")
            return

        while pygame.mixer.music.get_busy() and not stop_event.is_set():
            time.sleep(0.2)

        if stop_event.is_set():
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
            break

        if listen_counts_file is not None:
            try:
                increment_listen_count(listen_counts_file, mp3_folder, mp3_path)
            except Exception as exc:
                print(f"Warning: could not update listen count for '{mp3_path.stem}': {exc}")

        for _ in range(max(0, delay_between_plays) * 10):
            if stop_event.is_set():
                break
            time.sleep(0.1)


def prompt_artist(
    track: Path,
    current_artists: List[str],
    choices: List[str],
    known_names: List[str],
) -> Optional[Tuple[str, List[str]]]:
    print(f"\n[{track.stem}]")
    if current_artists:
        print(f"Current artist: {format_artist_list(current_artists)}")
    print("Playing...")
    if choices:
        print("Pick an artist:")
        for idx, artist in enumerate(choices, start=1):
            print(f"  {idx}. {artist}")
        print("Enter = skip, numbers/names = replace, +numbers/+names = add, - = clear.")
    else:
        print("Type artist name(s), then Enter. Press Enter to skip. Type '-' to clear.")
    print("Multiple artists: 1,3 or Artist A, Artist B")

    try:
        value = input("artist: ").strip()
    except EOFError:
        return None
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
        raise

    if not value or value.casefold() == "s":
        return None
    if value == "-":
        return ("clear", [])

    action = "set"
    if value.startswith("+"):
        action = "add"
        value = value[1:].strip()
        if not value:
            return None

    artist_names = parse_artist_tokens(value, choices, known_names)
    if artist_names is None:
        return None
    return (action, artist_names)


def tag_artists(
    mp3_folder: Path,
    artists_file: Path,
    reviewed_file: Path,
    listen_counts_file: Path,
    include_existing: bool,
    review_existing: bool,
    random_order: bool,
    no_play: bool,
    count_listens: bool,
    limit: Optional[int],
    choice_count: int,
    delay_between_plays: int,
    delay_between_tracks: int,
    output_device: Optional[str],
    strict_output_device: bool,
    volume: float,
) -> None:
    artists = load_artists(artists_file)
    reviewed = load_reviewed(reviewed_file)
    recent_artists: List[str] = []

    if not mp3_folder.is_dir():
        print(f"MP3 folder not found: {mp3_folder}")
        return

    tracks = choose_tracks(mp3_folder, artists, include_existing, review_existing, reviewed)
    if random_order:
        random.shuffle(tracks)
    if limit is not None:
        tracks = tracks[: max(0, limit)]

    total_tracks = len(get_mp3_files(mp3_folder))
    filled = sum(1 for track in get_mp3_files(mp3_folder) if artists.get(track.stem))
    reviewed_count = sum(
        1
        for track in get_mp3_files(mp3_folder)
        if artists.get(track.stem) and track.stem in reviewed
    )
    print(f"Artist file: {artists_file}")
    print(f"Folder: {mp3_folder}")
    print(f"Already filled: {filled}/{total_tracks}")
    if review_existing:
        print(f"Already reviewed: {reviewed_count}/{filled}")

    if not tracks:
        print("No tracks to process.")
        return

    if not no_play:
        init_audio(output_device=output_device, strict=strict_output_device, volume=volume)

    try:
        for track in tracks:
            stop_event = threading.Event()
            playback_thread: Optional[threading.Thread] = None
            if not no_play:
                playback_thread = threading.Thread(
                    target=play_mp3_loop,
                    args=(track, stop_event, delay_between_plays),
                    kwargs={
                        "listen_counts_file": listen_counts_file if count_listens else None,
                        "mp3_folder": mp3_folder,
                    },
                    daemon=True,
                )
                playback_thread.start()

            try:
                current_artists = artists.get(track.stem, [])
                choices = artist_choices(
                    artists,
                    track,
                    current_artists,
                    recent_artists,
                    choice_count,
                )
                known_names = all_known_artist_names(artists, recent_artists)
                artist_result = prompt_artist(track, current_artists, choices, known_names)
            except KeyboardInterrupt:
                stop_event.set()
                if pygame is not None:
                    try:
                        pygame.mixer.music.stop()
                    except Exception:
                        pass
                if playback_thread is not None:
                    playback_thread.join()
                return

            stop_event.set()
            if pygame is not None:
                try:
                    pygame.mixer.music.stop()
                except Exception:
                    pass
            if playback_thread is not None:
                playback_thread.join()

            if artist_result is None:
                print("Skipped.")
            else:
                artist_action, artist_names = artist_result
                if artist_action == "add":
                    artist_names = normalize_artist_names([*current_artists, *artist_names])

            if artist_result is None:
                if review_existing:
                    reviewed.add(track.stem)
                    save_reviewed(reviewed_file, reviewed)
                    print(f"Marked reviewed: '{track.stem}'.")
            elif artist_action == "clear" or not artist_names:
                artists.pop(track.stem, None)
                reviewed.discard(track.stem)
                save_json(artists_file, artists)
                if review_existing:
                    save_reviewed(reviewed_file, reviewed)
                print(f"Cleared artist for '{track.stem}'.")
            else:
                artists[track.stem] = artist_names
                if review_existing:
                    reviewed.add(track.stem)
                for artist in reversed(artist_names):
                    recent_artists = [
                        artist,
                        *[name for name in recent_artists if name.casefold() != artist.casefold()],
                    ]
                recent_artists = recent_artists[: max(1, choice_count)]
                save_json(artists_file, artists)
                if review_existing:
                    save_reviewed(reviewed_file, reviewed)
                print(f"Saved artist for '{track.stem}': {format_artist_list(artist_names)}")

            if delay_between_tracks > 0:
                time.sleep(delay_between_tracks)
    finally:
        if not no_play:
            shutdown_audio()


def migrate_artists_file(artists_file: Path) -> None:
    artists = load_artists(artists_file)
    save_json(artists_file, artists)
    song_count = len(artists)
    artist_count = len(set(iter_artist_names(artists)))
    print(f"Migrated {artists_file}: {song_count} songs, {artist_count} artists.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Terminal artist tagger. Plays tracks and stores artist names separately from tags.json."
    )
    parser.add_argument(
        "--folder",
        default=str(DEFAULT_MP3_FOLDER),
        help="Folder containing MP3 files (default: static/mp3).",
    )
    parser.add_argument(
        "--artists",
        default=None,
        help="Artist JSON path (default: artists.json, or mid_artists.json for static/mid-mp3s).",
    )
    parser.add_argument(
        "--mid",
        action="store_true",
        help="Shortcut for --folder static/mid-mp3s --artists mid_artists.json.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Review all tracks, including ones that already have an artist.",
    )
    parser.add_argument(
        "--review-existing",
        action="store_true",
        help="Review already-assigned artist entries once, marking each skipped/saved song done.",
    )
    parser.add_argument(
        "--reviewed",
        default=None,
        help="Reviewed-song JSON path (default: artists_reviewed.json, or mid_artists_reviewed.json).",
    )
    parser.add_argument(
        "--listen-counts",
        default=None,
        help="Listen-count JSON path (default: listen_counts.json, or mid_listen_counts.json).",
    )
    parser.add_argument(
        "--ordered",
        action="store_true",
        help="Process tracks alphabetically instead of shuffled.",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Prompt without playing audio.",
    )
    parser.add_argument(
        "--count-listens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Increment listen counts when a prompted MP3 reaches the end. Does not write timestamps.",
    )
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Convert existing artist JSON values to list format and exit.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum tracks to process this run.",
    )
    parser.add_argument(
        "--choices",
        type=int,
        default=DEFAULT_CHOICE_COUNT,
        help="Number of existing artist choices to show for each track.",
    )
    parser.add_argument(
        "--delay",
        dest="delay_between_plays",
        type=int,
        default=DEFAULT_DELAY_BETWEEN_PLAYS,
        help="Seconds between repeat plays of the same track.",
    )
    parser.add_argument(
        "--next-delay",
        dest="delay_between_tracks",
        type=int,
        default=DEFAULT_DELAY_BETWEEN_TRACKS,
        help="Seconds to wait after each answer before starting the next track.",
    )
    parser.add_argument(
        "--output-device",
        default=DEFAULT_OUTPUT_DEVICE,
        help="Pin playback to this output device.",
    )
    parser.add_argument(
        "--list-output-devices",
        action="store_true",
        help="List available output devices and exit.",
    )
    parser.add_argument(
        "--strict-output-device",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refuse to play if the chosen output device is not available.",
    )
    parser.add_argument(
        "--volume",
        type=float,
        default=DEFAULT_VOLUME,
        help="Playback volume from 0.0 to 1.0.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.list_output_devices:
        devices = list_pygame_output_devices()
        if not devices:
            print("No devices found, or pygame is not installed.")
            return 1
        print("SDL/pygame playback devices:")
        for device in devices:
            print(f" - {device}")
        return 0

    mp3_folder = Path("static") / "mid-mp3s" if args.mid else Path(args.folder)
    artists_file = Path(args.artists) if args.artists else default_artists_file(mp3_folder)
    reviewed_file = Path(args.reviewed) if args.reviewed else default_reviewed_file(mp3_folder)
    listen_counts_file = (
        Path(args.listen_counts)
        if args.listen_counts
        else default_listen_counts_file(mp3_folder)
    )
    if args.mid and args.artists is None:
        artists_file = Path("mid_artists.json")
    if args.mid and args.reviewed is None:
        reviewed_file = Path("mid_artists_reviewed.json")
    if args.mid and args.listen_counts is None:
        listen_counts_file = Path("mid_listen_counts.json")

    try:
        if args.migrate:
            migrate_artists_file(artists_file)
            return 0
        tag_artists(
            mp3_folder=mp3_folder,
            artists_file=artists_file,
            reviewed_file=reviewed_file,
            listen_counts_file=listen_counts_file,
            include_existing=args.all,
            review_existing=args.review_existing,
            random_order=not args.ordered,
            no_play=args.no_play,
            count_listens=args.count_listens,
            limit=args.limit,
            choice_count=max(0, args.choices),
            delay_between_plays=args.delay_between_plays,
            delay_between_tracks=args.delay_between_tracks,
            output_device=args.output_device,
            strict_output_device=args.strict_output_device,
            volume=args.volume,
        )
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
