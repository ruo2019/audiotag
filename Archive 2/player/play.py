import argparse
import curses
import locale
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame
from pydub import AudioSegment

from player.audio_analysis import analyze_audio_folder
from player.constants import (
    ANALYSIS_CACHE_FILENAME,
    DEFAULT_MP3_FOLDER,
    DEFAULT_SAMPLE_FILENAME,
    META_FILENAME,
)
from player.meta_manager import MetaManager
from player.tui import MusicPlayerTUI

try:
    locale.setlocale(locale.LC_ALL, "")
except locale.Error:
    pass

# --- Main Application Logic ---

def main(stdscr, args):
    """Initializes components and runs the TUI."""
    if not os.path.isdir(args.folder):
        print(f"Error: MP3 folder not found at '{args.folder}'", file=sys.stderr)
        sys.exit(1)

    meta_manager = MetaManager(args.folder)

    try:
        pygame.mixer.pre_init(44100, -16, 2, 2048)
        pygame.init()
        pygame.mixer.init()
    except pygame.error as e:
        print(f"Error initializing Pygame: {e}", file=sys.stderr)
        print("Please ensure you have a working audio output device.", file=sys.stderr)
        sys.exit(1)

    player = MusicPlayerTUI(stdscr, args, meta_manager)
    player.run()

    pygame.mixer.quit()
    pygame.quit()

def run_analysis_only(args):
    """Analyze volumes and update caches without launching the TUI."""
    if not os.path.isdir(args.folder):
        print(f"Error: MP3 folder not found at '{args.folder}'", file=sys.stderr)
        sys.exit(1)

    MetaManager(args.folder) # Sync and persist meta file

    def status_print(message):
        print(message)

    audio_analysis, target_loudness = analyze_audio_folder(
        args.folder,
        args.sample,
        target_loudness_override=args.vol,
        status_callback=status_print,
    )

    print(f"Analysis complete. {len(audio_analysis)} file(s) processed.")
    print(f"Target loudness: {target_loudness:.2f}")
    print(f"Updated cache at {os.path.join(args.folder, ANALYSIS_CACHE_FILENAME)}")
    print(f"Updated metadata at {os.path.join(args.folder, META_FILENAME)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='A terminal-based music player with volume normalization and play tracking.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--folder', type=str, default=DEFAULT_MP3_FOLDER,
                        help=f'Path to the MP3 folder (default: {DEFAULT_MP3_FOLDER}).')
    parser.add_argument('--sample', type=str, default=DEFAULT_SAMPLE_FILENAME,
                        help=f'Reference MP3 for auto volume scaling (default: {DEFAULT_SAMPLE_FILENAME}).')
    parser.add_argument('--vol', type=float, default=None,
                        help='Target loudness in LUFS (e.g., -20.0). Overrides --sample.')
    parser.add_argument('--log', action='store_true',
                        help='Enable detailed logging (currently not used by TUI).')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Analyze volumes, update cache/meta files, and exit without launching the TUI.')

    args = parser.parse_args()

    try:
        if not AudioSegment.converter:
            print("Warning: FFmpeg or libav (pydub dependency) not found.", file=sys.stderr)
            print("Audio analysis and playback will likely fail.", file=sys.stderr)
            print("Please install FFmpeg and ensure it's in your system's PATH.", file=sys.stderr)
            time.sleep(3)
    except Exception:
        pass

    if args.analyze_only:
        run_analysis_only(args)
        sys.exit(0)

    try:
        curses.wrapper(main, args)
    except curses.error as e:
        print(f"A Curses error occurred: {e}", file=sys.stderr)
        print("Your terminal might be too small to run this application.", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nPlayer stopped by user.")
    finally:
        print("Exiting.")
