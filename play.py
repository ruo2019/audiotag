import argparse
import json
import random
import os
import time
import math
import sys
import curses
from curses import textpad
import textwrap

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame
import jellyfish
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

# --- Configuration ---
META_FILENAME = ".mp3meta.json"
ANALYSIS_CACHE_FILENAME = ".mp3analysis.json"
ANALYSIS_CACHE_VERSION = 2
DEFAULT_MP3_FOLDER = "mp3s"
DEFAULT_SAMPLE_FILENAME = "[D2] Deep Stone Crypt Theme.mp3"
PBAR_TIP = "─"

# --- Audio Helper Functions (Volume Normalization) ---

def get_audio_loudness(full_filepath):
    """Calculates the loudness in dBFS using the top 25% loudest seconds."""
    try:
        audio = AudioSegment.from_mp3(full_filepath)
        if audio.duration_seconds <= 0:
            return -100.0

        # Analyze 1-second windows; use the mean of the loudest 25% seconds.
        second_loudness = []
        for start_ms in range(0, len(audio), 1000):
            segment = audio[start_ms:start_ms + 1000]
            segment_dbfs = segment.dBFS
            if segment_dbfs == -math.inf:
                segment_dbfs = -100.0
            second_loudness.append(segment_dbfs)

        if not second_loudness:
            return -100.0

        second_loudness.sort(reverse=True)
        top_count = max(1, math.ceil(0.25 * len(second_loudness)))
        top_slice = second_loudness[:top_count]
        return sum(top_slice) / len(top_slice)
    except (CouldntDecodeError, FileNotFoundError, Exception):
        return None

def calculate_volume_scale(target_dbfs, current_dbfs):
    """Calculates the pygame volume scale factor (0.0 to 1.0)."""
    if target_dbfs is None or current_dbfs is None:
        return 0.5
    if target_dbfs <= -100.0 or current_dbfs <= -100.0:
        return 0.5

    db_difference = target_dbfs - current_dbfs
    scale_factor = 10 ** (db_difference / 20)
    return max(0.0, min(1.0, 0.5 * scale_factor))

def analyze_audio_folder(folder_path, sample_filename, target_loudness_override=None, status_callback=None, default_target=-20.0):
    """Analyze MP3 loudness, update cache, and return analysis + target loudness."""
    def notify(message):
        if status_callback:
            status_callback(message)

    cache_path = os.path.join(folder_path, ANALYSIS_CACHE_FILENAME)
    notify("Analyzing audio (using cache)...")

    needs_saving = False

    try:
        with open(cache_path, 'r') as f:
            analysis_cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        analysis_cache = {}
    if not isinstance(analysis_cache, dict):
        analysis_cache = {}

    if analysis_cache.get("_version") != ANALYSIS_CACHE_VERSION:
        analysis_cache = {"_version": ANALYSIS_CACHE_VERSION}
        needs_saving = True

    audio_analysis = {}
    all_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.mp3')]
    disk_basenames = {os.path.splitext(f)[0] for f in all_files}

    for i, filename in enumerate(all_files):
        basename = os.path.splitext(filename)[0]
        full_path = os.path.join(folder_path, filename)

        try:
            mtime = os.path.getmtime(full_path)
            cached_data = analysis_cache.get(basename)

            if cached_data and cached_data.get('mtime') == mtime:
                loudness = cached_data.get('loudness')
            else:
                notify(f"Analyzing [{i + 1}/{len(all_files)}]: {basename[:50]}")
                loudness = get_audio_loudness(full_path)
                analysis_cache[basename] = {'loudness': loudness, 'mtime': mtime}
                needs_saving = True

            audio_analysis[basename] = {'loudness': loudness}
        except FileNotFoundError:
            continue

    for basename in set(analysis_cache.keys()) - disk_basenames:
        if basename.startswith("_"):
            continue
        del analysis_cache[basename]
        needs_saving = True

    if needs_saving:
        try:
            with open(cache_path, 'w') as f:
                json.dump(analysis_cache, f, indent=2)
        except IOError:
            pass # Fail silently

    target_loudness = default_target
    if target_loudness_override is not None:
        target_loudness = target_loudness_override
    else:
        sample_path = os.path.join(folder_path, sample_filename)
        if os.path.exists(sample_path):
            sample_loudness = get_audio_loudness(sample_path)
            if sample_loudness:
                target_loudness = sample_loudness

    for basename, data in audio_analysis.items():
        data['scale'] = calculate_volume_scale(target_loudness, data['loudness'])

    return audio_analysis, target_loudness

# --- Metadata Management ---

class MetaManager:
    """Handles loading, saving, and updating the .mp3meta.json file."""
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.meta_path = os.path.join(self.folder_path, META_FILENAME)
        self.data = self._load()
        self.sync_with_disk()

    def _load(self):
        try:
            if os.path.exists(self.meta_path):
                with open(self.meta_path, 'r') as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
        return {}

    def save(self):
        try:
            with open(self.meta_path, 'w') as f:
                json.dump(self.data, f, indent=2, sort_keys=True)
        except IOError:
            pass

    def sync_with_disk(self):
        """Ensures metadata is synced with the MP3s on disk."""
        try:
            disk_files = {os.path.splitext(f)[0] for f in os.listdir(self.folder_path) if f.lower().endswith('.mp3')}
            meta_files = set(self.data.keys())

            for basename in disk_files - meta_files:
                self.data[basename] = 0

            for basename in meta_files - disk_files:
                del self.data[basename]

            self.save()
        except OSError:
            pass

    def increment_count(self, basename):
        if basename in self.data:
            self.data[basename] += 1
            self.save()

    def get_all_sorted_data(self):
        return sorted(self.data.items(), key=lambda item: item[0])

# --- Curses TUI ---

class MusicPlayerTUI:
    """Manages the Curses-based Terminal User Interface."""

    def __init__(self, stdscr, args, meta_manager):
        self.stdscr = stdscr
        self.args = args
        self.meta_manager = meta_manager

        self.audio_analysis = {}
        self.target_loudness = -20.0
        self.current_pool_basenames = []
        self.playback_queue = []

        self.all_songs_data = [] # Full list for scrolling
        self.scroll_pos = 0     # Current scroll position for the playlist view
        self.playback_stopped = True # Flag to track if user manually stopped audio

        self.mode = 'typing' # 'typing' or 'scroll'

        self._reset_current_song_info()

    def _reset_current_song_info(self):
        self.current_song_info = {
            "basename": "None",
            "loudness": "N/A",
            "scale": "N/A"
        }
        self.current_song_duration = 0
        self.current_song_start_time = 0

    def _find_closest_match(self, input_name, known_basenames, max_distance):
        """Find the closest match using Levenshtein distance."""
        input_name_lower = input_name.lower()
        closest_match = None
        min_distance = float('inf')

        for basename in known_basenames:
            distance = jellyfish.levenshtein_distance(input_name_lower, basename.lower())
            if distance <= max_distance and distance < min_distance:
                min_distance = distance
                closest_match = basename

        return closest_match, min_distance

    def run(self):
        """Main entry point for the TUI application."""
        self._setup_curses()
        self._analyze_all_audio_with_cache()
        self.all_songs_data = self.meta_manager.get_all_sorted_data()

        self._draw_static_layout()

        h, w = self.stdscr.getmaxyx()
        input_box = curses.newwin(1, w - 4, h - 2, 2)
        # Enable insert mode so typing at a cursor position shifts text instead of overwriting
        textbox = textpad.Textbox(input_box, insert_mode=True)

        def draw_input_mode():
            """Clears and redraws the input window based on the current mode."""
            self.win_input.clear()
            self.win_input.box()
            if self.mode == 'typing':
                curses.curs_set(1)
            else: # scroll
                self.win_input.addstr(0, 2, " Scroll Mode ")
                curses.curs_set(0)
            self.win_input.refresh() # Use refresh on this small window for immediate effect
            input_box.touchwin()
            input_box.refresh()

        self.mode = 'typing'
        draw_input_mode()

        # Main application loop
        while True:
            # Playback Logic (runs if no new key press)
            if not pygame.mixer.get_busy() and not self.playback_stopped:
                self._play_next_song()

            # Drawing
            self._update_all_windows()
            if self.mode == 'typing':
                input_box.noutrefresh()
            curses.doupdate()

            # Input Handling
            key = self.stdscr.getch()

            if key == -1: # Timeout, no key pressed
                continue

            if key == ord('\t') or key == 9:
                self.mode = 'scroll' if self.mode == 'typing' else 'typing'
                draw_input_mode()
                continue

            if self.mode == 'typing':
                if key == curses.KEY_ENTER or key == 10 or key == 13:
                    content = textbox.gather().strip()
                    if self._process_input(content):
                        # Clear textbox content by sending backspace
                        for _ in range(len(content) + 1):
                            textbox.do_command(curses.KEY_BACKSPACE)
                        input_box.refresh()

                    draw_input_mode()
                elif key == curses.KEY_DC:
                    # Delete key: remove character under cursor if present
                    try:
                        textbox.win.delch()
                    except curses.error:
                        pass
                    input_box.refresh()
                elif key == curses.KEY_HOME:
                    textbox.win.move(0, 0)
                    input_box.refresh()
                elif key == curses.KEY_END:
                    # Move to end of current line without exceeding window width
                    line = textbox.gather().split('\n', 1)[0]
                    max_x = textbox.win.getmaxyx()[1] - 1
                    textbox.win.move(0, min(len(line), max_x))
                    input_box.refresh()
                else:
                    textbox.do_command(key)
                    input_box.refresh()

            elif self.mode == 'scroll':
                if key in (ord('q'), ord('Q')):
                    break # Quit application

                elif key == curses.KEY_DOWN:
                    max_h, _ = self.win_playlist.getmaxyx()
                    max_items = max_h - 2
                    if self.scroll_pos < len(self.all_songs_data) - max_items:
                        self.scroll_pos += 1

                elif key == curses.KEY_UP:
                    if self.scroll_pos > 0:
                        self.scroll_pos -= 1

                elif key == ord('s'): # Stop and prepare for new input
                    pygame.mixer.stop()
                    self._reset_current_song_info()
                    self.current_pool_basenames = []
                    self.playback_queue = []
                    self.playback_stopped = True
                    self.mode = 'typing'
                    # Clear textbox content
                    content = textbox.gather()
                    for _ in range(len(content) + 1):
                        textbox.do_command(curses.KEY_BACKSPACE)
                    draw_input_mode()

                elif key == ord('n'): # Next
                    self.playback_stopped = False
                    pygame.mixer.stop()

    def _process_input(self, content):
        """Handles user input from the textbox to define or modify a song pool."""
        is_playing = pygame.mixer.get_busy() or self.current_song_start_time > 0

        # Case 1: Add to an existing playlist
        if is_playing and content.startswith('||'):
            potential_songs = [s.strip() for s in content.split('||') if s.strip()]
            valid_new_songs = []
            auto_corrected_songs = []
            not_found = []
            all_known_basenames = [basename for basename, _ in self.all_songs_data]

            for song_name in potential_songs:
                # Check exact case-insensitive match first
                for known_basename in all_known_basenames:
                    if song_name.lower() == known_basename.lower():
                        valid_new_songs.append(known_basename)
                        break
                else:
                    # Try Levenshtein distance auto-correction
                    closest_match, distance = self._find_closest_match(song_name, all_known_basenames, max_distance=5)
                    if closest_match:
                        auto_corrected_songs.append(f"'{song_name}' -> '{closest_match}'")
                        valid_new_songs.append(closest_match)
                    else:
                        not_found.append(song_name)

            if not_found:
                err_msg = f"Error: Song(s) not found: {', '.join(not_found)}"
                self._display_message(err_msg, 4, 3)
                return False # Failure

            if valid_new_songs:
                if auto_corrected_songs:
                    correction_msg = f"Auto-corrected: {', '.join(auto_corrected_songs)}"
                    self._display_message(correction_msg, 2, 2)
                self.current_pool_basenames.extend(valid_new_songs)
                self.playback_queue.extend(valid_new_songs)
                random.shuffle(self.playback_queue)
                self._display_message(f"Added {len(valid_new_songs)} songs to the pool.", 2, 1.5)
                return True # Success
            return False

        # Case 2: Create a new playlist
        else:
            pool_to_set = []
            if not content:
                self._display_message("Pool is empty. Playing all songs randomly.", 2, 1.5)
                pool_to_set = [basename for basename, count in self.all_songs_data]
            else:
                potential_songs = [s.strip() for s in content.split('||')]
                valid_pool = []
                auto_corrected_songs = []
                not_found = []
                all_known_basenames = [basename for basename, _ in self.all_songs_data]

                for song_name in potential_songs:
                    # Check exact case-insensitive match first
                    for known_basename in all_known_basenames:
                        if song_name.lower() == known_basename.lower():
                            valid_pool.append(known_basename)
                            break
                    else:
                        # Try Levenshtein distance auto-correction
                        closest_match, distance = self._find_closest_match(song_name, all_known_basenames, max_distance=5)
                        if closest_match:
                            auto_corrected_songs.append(f"'{song_name}' -> '{closest_match}'")
                            valid_pool.append(closest_match)
                        else:
                            not_found.append(song_name)

                if not_found:
                    err_msg = f"Error: Song(s) not found: {', '.join(not_found)}"
                    self._display_message(err_msg, 4, 3)
                    return False

                if valid_pool:
                    if auto_corrected_songs:
                        correction_msg = f"Auto-corrected: {', '.join(auto_corrected_songs)}"
                        self._display_message(correction_msg, 2, 2)
                    self._display_message("Pool accepted. Starting playback.", 2, 1.5)
                    pool_to_set = valid_pool
                else:
                    return False # No valid songs found

            self.current_pool_basenames = pool_to_set
            self.playback_queue = []
            self.playback_stopped = False
            pygame.mixer.stop() # Stop current to play from new pool
            self.mode = 'scroll' # Switch to scroll mode automatically
            return True

    def _setup_curses(self):
        curses.curs_set(0)
        self.stdscr.nodelay(1)
        self.stdscr.timeout(100)
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_RED, -1)
        self._create_windows()

    def _create_windows(self):
        h, w = self.stdscr.getmaxyx()
        info_h, playlist_h = 5, h - 9
        input_h = 3

        self.win_header = curses.newwin(1, w, 0, 0)
        self.win_info = curses.newwin(info_h, w, 1, 0)
        self.win_playlist = curses.newwin(playlist_h, w, info_h + 1, 0)
        self.win_input = curses.newwin(input_h, w, h - input_h, 0)

    def _add_win_info_separator(self):
        self.win_info.addstr(0, 16, "┬")
        self.win_info.addstr(4, 16, "┴")

    def _draw_static_layout(self):
        h, w = self.stdscr.getmaxyx()

        self.win_info.box()

        self._add_win_info_separator()

        self.win_playlist.box()
        self.win_input.box()

        self.win_playlist.addstr(0, 4, " #  Name ")

        self.stdscr.noutrefresh()
        self.win_info.noutrefresh()
        self.win_playlist.noutrefresh()
        self.win_input.noutrefresh()
        curses.doupdate()

    def _analyze_all_audio_with_cache(self):
        """Analyzes all MP3s, using a cache to speed up the process."""
        self.audio_analysis, self.target_loudness = analyze_audio_folder(
            self.args.folder,
            self.args.sample,
            target_loudness_override=self.args.vol,
            status_callback=self._update_info_win_message,
            default_target=self.target_loudness,
        )

    def _play_next_song(self):
        if not self.current_pool_basenames:
            self.playback_stopped = True # Stop trying to play if pool is empty
            return

        if not self.playback_queue:
            self.playback_queue = self.current_pool_basenames[:]
            random.shuffle(self.playback_queue)

        if not self.playback_queue:
            self.playback_stopped = True
            return

        basename = self.playback_queue.pop()
        self.meta_manager.increment_count(basename)

        self.all_songs_data = self.meta_manager.get_all_sorted_data()

        full_path = os.path.join(self.args.folder, basename + ".mp3")
        if not os.path.exists(full_path):
            self.current_song_info['basename'] = f"'{basename}' NOT FOUND!"
            return

        analysis = self.audio_analysis.get(basename, {'loudness': 'N/A', 'scale': 0.5})
        volume_scale = analysis['scale']

        self.current_song_info = {
            "basename": basename,
            "loudness": f"{analysis['loudness']:.2f}" if isinstance(analysis['loudness'], float) else 'N/A',
            "scale": f"{volume_scale:.3f}"
        }

        try:
            sound = pygame.mixer.Sound(full_path)
            self.current_song_duration = sound.get_length()
            sound.set_volume(volume_scale)
            sound.play()
            self.current_song_start_time = time.time()
            self.playback_stopped = False
        except pygame.error as e:
            self.current_song_duration = 0
            self.current_song_start_time = 0
            self.current_song_info['basename'] = f"ERROR PLAYING '{basename}'"

    def _update_all_windows(self):
        self._update_header_win()
        self._update_info_win()
        self._update_playlist_win()

        self.stdscr.noutrefresh()
        self.win_header.noutrefresh()
        self.win_info.noutrefresh()
        self.win_playlist.noutrefresh()

    def _format_time(self, seconds):
        if seconds is None or seconds < 0:
            seconds = 0
        minutes, sec = divmod(int(seconds), 60)
        return f"{minutes:02d}:{sec:02d}"

    def _update_header_win(self):
        self.win_header.clear()
        self.win_header.bkgd(' ', curses.color_pair(1))
        _, w = self.win_header.getmaxyx()

        is_playing = self.current_song_start_time > 0 and self.current_song_duration > 0 and pygame.mixer.get_busy()

        if is_playing:
            elapsed_time = time.time() - self.current_song_start_time
            elapsed_time = max(0, min(elapsed_time, self.current_song_duration))

            time_str = f"{self._format_time(elapsed_time)} / {self._format_time(self.current_song_duration)}"
            bar_max_width = w - len(time_str) - 6

            if bar_max_width > 0:
                progress_percent = elapsed_time / self.current_song_duration if self.current_song_duration > 0 else 0
                filled_len = int(bar_max_width * progress_percent)
                bar = ('─' * filled_len) + (' ' * (bar_max_width - filled_len))
                progress_bar_str = f" [{bar}] {time_str} "
            else:
                progress_bar_str = f" {time_str} ".center(w)

            self.win_header.addstr(0, 0, progress_bar_str)
        else:
            title = " MPlayer3 "
            self.win_header.addstr(0, (w - len(title)) // 2, title, curses.A_BOLD)

    def _update_info_win(self):
        self.win_info.clear()
        self.win_info.box()
        self._add_win_info_separator()
        h, w = self.win_info.getmaxyx()

        song_name = textwrap.shorten(self.current_song_info['basename'], width=w - 20)

        self.win_info.addstr(1, 3, "Now Playing  │ ")
        self.win_info.addstr(song_name, curses.color_pair(2) | curses.A_BOLD)

        # --- Volume Info ---
        self.win_info.addstr(2, 3, f"Song LUFS    │ {self.current_song_info['loudness']} (Target: {self.target_loudness:.2f})")
        self.win_info.addstr(3, 3, f"Volume Scale │ {self.current_song_info['scale']}")

    def _update_info_win_message(self, message):
         self.win_info.clear()
         self.win_info.box()
         self.win_info.addstr(0, 2, " Status ")
         h, w = self.win_info.getmaxyx()
         self.win_info.addstr(h // 2, (w - len(message)) // 2, message, curses.color_pair(3))
         self.win_info.refresh()

    def _display_message(self, message, color_pair_id, duration_sec):
        self.win_input.clear()
        self.win_input.box()
        self.win_input.addstr(0, 2, " Message ")
        h, w = self.win_input.getmaxyx()
        self.win_input.addstr(h // 2, (w - len(message)) // 2, message, curses.color_pair(color_pair_id))
        self.win_input.refresh()
        time.sleep(duration_sec)
        self.win_input.clear()
        self.win_input.box()
        self.win_input.refresh()

    def _update_playlist_win(self):
        self.win_playlist.clear()
        self.win_playlist.box()
        self.win_playlist.addstr(0, 4, " #  Name ")

        h, w = self.win_playlist.getmaxyx()
        max_items = h - 2
        currently_playing_basename = self.current_song_info['basename']

        controls = " (q:Quit s:Stop&New n:Next Up/Down:Scroll Tab:Switch) "
        self.win_playlist.addstr(0, w - len(controls) - 2, controls)

        display_data = self.all_songs_data[self.scroll_pos:self.scroll_pos + max_items]

        for i, (basename, count) in enumerate(display_data):
            line = f"[{str(int(count)).rjust(3, ' ')}] {basename}"

            attr = curses.A_NORMAL
            if basename == currently_playing_basename:
                attr = curses.color_pair(2) | curses.A_BOLD

            try:
                self.win_playlist.addstr(i + 1, 2, line, attr)
            except curses.error:
                pass # Avoid crashing if line is too long

        if self.scroll_pos > 0:
            self.win_playlist.addstr(1, w-3, "▲", curses.A_BOLD)
        if self.scroll_pos + max_items < len(self.all_songs_data):
            self.win_playlist.addstr(h-2, w-3, "▼", curses.A_BOLD)


# --- Main Application Logic ---

def main(stdscr, args):
    """Initializes components and runs the TUI."""
    if not os.path.isdir(args.folder):
        print(f"Error: MP3 folder not found at '{args.folder}'", file=sys.stderr)
        sys.exit(1)

    try:
        pygame.mixer.pre_init(44100, -16, 2, 2048)
        pygame.init()
        pygame.mixer.init()
    except pygame.error as e:
        print(f"Error initializing Pygame: {e}", file=sys.stderr)
        print("Please ensure you have a working audio output device.", file=sys.stderr)
        sys.exit(1)

    meta_manager = MetaManager(args.folder)
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
