import argparse
import json
import os
import random
import sys
import threading
import time
from typing import List, Optional

try:
    import pygame
except ImportError:
    pygame = None


# Default configuration
DEFAULT_MP3_FOLDER = os.path.join("static", "mp3")
DEFAULT_TAGS_FILE = "tags.json"
DEFAULT_DELAY_BETWEEN_PLAYS = 10  # seconds between plays of the same track
DEFAULT_DELAY_BETWEEN_TRACKS = 10  # seconds after submitting tags before next track

# On macOS, "afplay" is a built-in command-line audio player.
# If you want to use something else, edit this.
# PLAYER_COMMAND = ["afplay", "-v", "0.5"]


DEFAULT_VOLUME = 0.1

def list_pygame_output_devices() -> List[str]:
    """List SDL/pygame playback device names (these are what devicename= expects)."""
    if pygame is None:
        return []
    try:
        import pygame._sdl2.audio as sdl2_audio  # type: ignore
        if not pygame.get_init():
            pygame.init()
        return [str(d) for d in sdl2_audio.get_audio_device_names(False)]  # False = playback
    except Exception:
        return []


def init_audio(output_device: Optional[str], strict: bool = False) -> None:
    """
    Initialize pygame mixer. If output_device is set, pin playback to that device.
    If strict=True, refuse to start if we can't confirm/select that device.
    """
    if pygame is None:
        raise RuntimeError("pygame is required for headphone-only output. Install with: pip install -U pygame")

    # If user asked for a specific device, optionally validate it before init.
    if output_device:
        devs = list_pygame_output_devices()
        if strict and devs and output_device not in devs:
            raise RuntimeError(
                f"Output device '{output_device}' not found.\n"
                f"Available SDL devices: {devs}\n"
                "Use --list-output-devices and copy an exact name."
            )

    # devicename= pins the stream to that device (pygame 2.0+). :contentReference[oaicite:2]{index=2}
    try:
        pygame.mixer.pre_init(44100, -16, 2, 2048, devicename=output_device)
        pygame.init()
        pygame.mixer.init(44100, -16, 2, 2048, devicename=output_device)
    except TypeError:
        # No devicename= support => cannot guarantee headphone-only.
        if output_device and strict:
            raise RuntimeError(
                "Your pygame build doesn't support devicename=. "
                "Upgrade pygame (pip install -U pygame) or disable --strict-output-device."
            )
        pygame.mixer.pre_init(44100, -16, 2, 2048)
        pygame.init()
        pygame.mixer.init()
    pygame.mixer.music.set_volume(DEFAULT_VOLUME)


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



def get_mp3_files(mp3_folder: str):
    """Return a sorted list of MP3 filenames (no paths) in the given folder."""
    if not os.path.isdir(mp3_folder):
        return []
    return sorted(
        f
        for f in os.listdir(mp3_folder)
        if f.lower().endswith(".mp3") and not f.startswith(".")
    )


def load_tags(tags_file: str):
    """Load tags from the JSON file. Returns empty dict if file doesn't exist."""
    if not os.path.exists(tags_file):
        return {}

    try:
        with open(tags_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: {tags_file} is not valid JSON. Starting with empty tags.")
        return {}


def save_tags(tags_file: str, tags_data):
    """Save tags dict to the JSON file."""
    directory = os.path.dirname(os.path.abspath(tags_file))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    with open(tags_file, "w", encoding="utf-8") as f:
        json.dump(tags_data, f, indent=4, ensure_ascii=False)


def get_untagged_mp3s(mp3_folder: str, tags_data):
    """Return list of MP3 filenames that don't have tags yet."""
    all_mp3s = get_mp3_files(mp3_folder)
    if not all_mp3s:
        return []

    untagged = []
    for mp3 in all_mp3s:
        key = os.path.splitext(mp3)[0]  # remove extension
        if key not in tags_data or not tags_data[key]:
            untagged.append(mp3)
    return untagged


def play_mp3_loop(mp3_path: str, stop_event: threading.Event, delay_between_plays: int):
    """
    Play the MP3 on repeat until stop_event is set.
    There will be delay_between_plays seconds between each play.
    """
    if pygame is None:
        print("pygame not installed; cannot play.")
        return

    while not stop_event.is_set():
        try:
            pygame.mixer.music.load(mp3_path)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Error: could not play '{mp3_path}': {e}")
            return

        # Wait for playback to finish or until stop_event is set
        while pygame.mixer.music.get_busy() and not stop_event.is_set():
            time.sleep(0.2)

        if stop_event.is_set():
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
            break

        # Finished one playthrough; wait before starting again
        for _ in range(int(delay_between_plays * 10)):  # check stop flag every 0.1s
            if stop_event.is_set():
                break
            time.sleep(0.1)



def collect_tags_for_track(track_name: str):
    """
    Collect tags from the user via stdin until Ctrl+G is pressed.
    Each non-empty line is stored as a tag.

    Note: In most terminals you'll need to press Ctrl+G and then Enter
    so that the line is sent to the program.
    """
    tags = []

    # These two lines match the format you described.
    print(f"\n[{track_name}]")
    print("Playing...")
    print(
        "Type tags, one per line. Press Ctrl+G then Enter when you're done with this track.\n"
    )

    while True:
        try:
            line = input()
        except EOFError:
            # End of input (e.g., user closed stdin)
            break
        except KeyboardInterrupt:
            print("\nInterrupted. Exiting.")
            raise

        # Detect Ctrl+G (BEL, ASCII code 7) anywhere in the line
        if "\x07" in line:
            # If there is other text on the line besides Ctrl+G, keep it as a tag.
            cleaned = line.replace("\x07", "").strip()
            if cleaned:
                tags.append(cleaned)
                print(cleaned)
            break

        cleaned = line.strip()
        if not cleaned:
            # ignore empty lines
            continue

        tags.append(cleaned)
        # Echo the tag so you see each one on a new line
        print(cleaned)

    return tags


def tag_mp3s(
    mp3_folder: str,
    tags_file: str,
    delay_between_plays: int,
    delay_between_tracks: int,
    output_device: Optional[str],
    volume: float,
    strict_output_device: bool,
):
    tags_data = load_tags(tags_file)

    if not os.path.isdir(mp3_folder):
        print(f"MP3 folder not found: {mp3_folder}")
        return

    all_mp3s = get_mp3_files(mp3_folder)
    if not all_mp3s:
        print(f"No MP3 files found in {mp3_folder}")
        return

    untagged_mp3s = get_untagged_mp3s(mp3_folder, tags_data)
    if not untagged_mp3s:
        print("All MP3 files already have tags.")
        return

    # Initialize audio once for the whole run
    init_audio(output_device=output_device, strict=strict_output_device)
    pygame.mixer.music.set_volume(volume)

    try:
        # Shuffle so the order is random but each untagged track is visited once.
        random.shuffle(untagged_mp3s)

        print(f"Found {len(untagged_mp3s)} untagged MP3 file(s).")

        for mp3 in untagged_mp3s:
            mp3_path = os.path.join(mp3_folder, mp3)
            track_name = os.path.splitext(mp3)[0]

            stop_event = threading.Event()
            playback_thread = threading.Thread(
                target=play_mp3_loop,
                args=(mp3_path, stop_event, delay_between_plays),
                daemon=True,
            )

            playback_thread.start()

            try:
                tags_for_track = collect_tags_for_track(track_name)
            except KeyboardInterrupt:
                # Stop playback thread and exit
                stop_event.set()
                try:
                    pygame.mixer.music.stop()
                except Exception:
                    pass
                playback_thread.join()
                print("\nExiting without tagging the remaining files.")
                return

            # Stop the playback and wait for thread to end
            stop_event.set()
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
            playback_thread.join()

            # Save tags for this track
            tags_data[track_name] = tags_for_track
            save_tags(tags_file, tags_data)
            print(f"Saved {len(tags_for_track)} tag(s) for '{track_name}'.")

            # Wait before starting the next track
            time.sleep(delay_between_tracks)
        pass
    finally:
        shutdown_audio()

    print("Done. All MP3 files have been processed.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Terminal-based MP3 tagger. Plays each untagged MP3 on repeat "
            "and lets you type tags in the terminal."
        )
    )
    parser.add_argument(
        "--tag",
        dest="tags_file",
        default=DEFAULT_TAGS_FILE,
        help="Path to the JSON tags file (default: tags.json).",
    )
    parser.add_argument(
        "--folder",
        dest="mp3_folder",
        default=DEFAULT_MP3_FOLDER,
        help="Folder containing MP3 files (default: static/mp3).",
    )
    parser.add_argument(
        "--delay",
        dest="delay_between_plays",
        type=int,
        default=DEFAULT_DELAY_BETWEEN_PLAYS,
        help="Seconds between each repeat play of the same track (default: 10).",
    )
    parser.add_argument(
        "--next-delay",
        dest="delay_between_tracks",
        type=int,
        default=DEFAULT_DELAY_BETWEEN_TRACKS,
        help=(
            "Seconds to wait after submitting tags before starting the next "
            "track (default: 10)."
        ),
    )
    parser.add_argument(
        "--output-device",
        dest="output_device",
        default="External Headphones",
        help="Pin playback to this output device (SDL/pygame device name).",
    )
    parser.add_argument(
        "--list-output-devices",
        action="store_true",
        help="List available output devices (SDL/pygame names) and exit.",
    )
    parser.add_argument(
        "--strict-output-device",
        action="store_true",
        default=True,
        help="Refuse to play if the chosen --output-device isn't available.",
    )
    parser.add_argument(
        "--volume",
        type=float,
        default=DEFAULT_VOLUME,
        help="Playback volume (0.0 to 1.0).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()

    if args.list_output_devices:
        devs = list_pygame_output_devices()
        if not devs:
            print("No devices found (or pygame not installed). Try: pip install -U pygame")
            sys.exit(1)
        print("SDL/pygame playback devices:")
        for d in devs:
            print(f" - {d}")
        sys.exit(0)

    try:
        tag_mp3s(
            mp3_folder=args.mp3_folder,
            tags_file=args.tags_file,
            delay_between_plays=args.delay_between_plays,
            delay_between_tracks=args.delay_between_tracks,
            output_device=args.output_device,
            volume=args.volume,
            strict_output_device=args.strict_output_device,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
