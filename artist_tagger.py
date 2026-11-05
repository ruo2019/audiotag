import argparse
import curses
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
        (
            p
            for p in mp3_folder.iterdir()
            if p.is_file() and p.suffix.lower() == ".mp3" and not p.name.startswith(".")
        ),
        key=lambda p: p.name.casefold(),
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


def search_known_artist_names(
    query: str,
    known_names: List[str],
    max_results: int,
) -> Tuple[List[str], int]:
    """Return saved artist names matching a case-insensitive local query."""
    normalized_query = " ".join(query.casefold().split())
    if not normalized_query:
        return ([], 0)

    query_words = normalized_query.split()
    ranked: List[Tuple[int, str, str]] = []
    for name in known_names:
        normalized_name = " ".join(name.casefold().split())
        name_words = normalized_name.split()
        if normalized_name == normalized_query:
            rank = 0
        elif normalized_name.startswith(normalized_query):
            rank = 1
        elif any(word.startswith(normalized_query) for word in name_words):
            rank = 2
        elif normalized_query in normalized_name:
            rank = 3
        elif all(
            any(query_word in name_word for name_word in name_words)
            for query_word in query_words
        ):
            rank = 4
        else:
            continue
        ranked.append((rank, normalized_name, name))

    ranked.sort(key=lambda item: (item[0], item[1]))
    matches = [name for _rank, _normalized, name in ranked]
    return (matches[: max(0, max_results)], len(matches))


_PICKER_PENDING = object()


class ArtistPicker:
    def __init__(
        self,
        track: Path,
        current_artists: List[str],
        choices: List[str],
        known_names: List[str],
        initial_status: str = "",
    ) -> None:
        self.track = track
        self.current_artists = list(current_artists)
        self.choices = list(choices)
        self.known_names = list(known_names)
        self.query = ""
        self.selected: List[str] = []
        self.add_mode = False
        self.list_focused = True
        self.focus = 0
        self.scroll = 0
        self.status = initial_status or "Type to search; Down Arrow focuses results."
        self.title_attr = curses.A_BOLD | curses.A_REVERSE
        self.heading_attr = curses.A_BOLD
        self.focus_attr = curses.A_REVERSE
        self.footer_attr = curses.A_DIM

    def configure_theme(self, screen: "curses.window") -> None:
        try:
            if not curses.has_colors():
                return
            curses.start_color()
            if curses.COLORS >= 256:
                background = 231  # xterm-256 pure white: #ffffff
                highlight = 159  # pale blue
                heading = 25  # readable blue on white
            elif curses.COLORS >= 16:
                background = 15  # bright white
                highlight = 14  # bright cyan
                heading = curses.COLOR_BLUE
            else:
                background = curses.COLOR_WHITE
                highlight = curses.COLOR_CYAN
                heading = curses.COLOR_BLUE
            curses.init_pair(1, curses.COLOR_BLACK, background)
            curses.init_pair(2, curses.COLOR_BLACK, highlight)
            curses.init_pair(3, heading, background)
            base_attr = curses.color_pair(1)
            self.title_attr = curses.color_pair(2) | curses.A_BOLD
            self.heading_attr = curses.color_pair(3) | curses.A_BOLD
            self.focus_attr = curses.color_pair(2)
            self.footer_attr = base_attr | curses.A_DIM
            screen.bkgd(" ", base_attr)
            screen.attrset(base_attr)
        except curses.error:
            pass

    def options(self) -> List[str]:
        if not self.query.strip():
            return list(self.choices)
        matches, _total = search_known_artist_names(
            self.query,
            self.known_names,
            len(self.known_names),
        )
        return matches

    def focus_rows(self) -> List[Tuple[str, str]]:
        pinned = [("pinned", name) for name in self.selected]
        results = [("result", name) for name in self.options()]
        return [*pinned, *results]

    def _focus_search_results(self) -> None:
        options = self.options()
        if options:
            self.focus = len(self.selected)
        elif self.selected:
            self.focus = 0
        else:
            self.focus = 0
        self.scroll = 0

    def _focused_row(self) -> Optional[Tuple[str, str]]:
        rows = self.focus_rows()
        if not rows:
            return None
        self.focus = max(0, min(self.focus, len(rows) - 1))
        return rows[self.focus]

    def _toggle_focused(self) -> None:
        row = self._focused_row()
        if row is None:
            self.status = "There is no artist row to select."
            return
        _section, name = row
        self._toggle_names([name])

    def _toggle_names(self, names: List[str]) -> None:
        added: List[str] = []
        removed: List[str] = []
        for name in names:
            existing_index = next(
                (
                    index
                    for index, selected_name in enumerate(self.selected)
                    if selected_name.casefold() == name.casefold()
                ),
                None,
            )
            if existing_index is None:
                self.selected.append(name)
                added.append(name)
            else:
                removed.append(self.selected.pop(existing_index))
        if added and removed:
            self.status = (
                f"Selected {format_artist_list(added)}; "
                f"removed {format_artist_list(removed)}."
            )
        elif added:
            self.status = (
                f"Selected: {format_artist_list(added)}. "
                "Search for another, or press Right Arrow to save."
            )
        elif removed:
            self.status = f"Removed from selection: {format_artist_list(removed)}."
        else:
            self.status = "No selection changed."
        self.query = ""
        self.list_focused = True
        self.focus = 0
        self.scroll = 0

    def _add_new_artist(self) -> None:
        name = " ".join(self.query.split())
        if not name:
            self.status = "Type a new artist name first."
            return
        if self.options():
            self.status = "This search has matches. Use Down Arrow, then Space to select one."
            return
        if any(selected.casefold() == name.casefold() for selected in self.selected):
            self.status = f"{name} is already selected."
        else:
            self.selected.append(name)
            self.status = (
                f"Added new artist to the pinned selection: {name}. "
                "Press Right Arrow to save."
            )
        self.query = ""
        self.list_focused = True
        self.focus = 0
        self.scroll = 0

    def _selection_result(self, names: List[str]) -> Tuple[str, List[str]]:
        combined = normalize_artist_names([*self.selected, *names])
        return ("add" if self.add_mode else "set", combined)

    def handle_key(self, key: object) -> object:
        rows = self.focus_rows()
        if key == curses.KEY_UP:
            if not rows:
                return _PICKER_PENDING
            if not self.list_focused:
                self.list_focused = True
                self.focus = len(self.selected) - 1 if self.selected else len(rows) - 1
            else:
                self.focus = (self.focus - 1) % len(rows)
            return _PICKER_PENDING
        if key == curses.KEY_DOWN:
            if not rows:
                return _PICKER_PENDING
            if not self.list_focused:
                self.list_focused = True
                self.focus = len(self.selected) if self.options() else 0
            else:
                self.focus = (self.focus + 1) % len(rows)
            return _PICKER_PENDING
        if key == curses.KEY_PPAGE:
            if rows:
                self.list_focused = True
                self.focus = max(0, self.focus - 10)
            return _PICKER_PENDING
        if key == curses.KEY_NPAGE:
            if rows:
                self.list_focused = True
                self.focus = min(len(rows) - 1, self.focus + 10)
            return _PICKER_PENDING
        if key == curses.KEY_HOME:
            if rows:
                self.list_focused = True
                self.focus = 0
            return _PICKER_PENDING
        if key == curses.KEY_END:
            if rows:
                self.list_focused = True
                self.focus = len(rows) - 1
            return _PICKER_PENDING
        if key == curses.KEY_RIGHT:
            if not self.selected:
                return None
            return self._selection_result([])
        if key in ("\n", "\r", curses.KEY_ENTER):
            if self.query.strip() and not self.options():
                self._add_new_artist()
            elif self.query.strip():
                self.status = "Matches found. Use Down Arrow, then Space to select."
            else:
                self.status = "Type a missing artist name before pressing Enter."
            return _PICKER_PENDING
        if key == " ":
            if self.list_focused:
                self._toggle_focused()
            else:
                self.query += " "
                self._focus_search_results()
            return _PICKER_PENDING
        if key == "\x1b":
            return None
        if key == "-" and not self.query:
            return ("clear", [])
        if key == "+" and not self.query:
            self.add_mode = True
            self.status = "Add mode: the selected artist will be added to the current tag."
            return _PICKER_PENDING
        if key in (curses.KEY_BACKSPACE, curses.KEY_DC, "\b", "\x7f"):
            if self.query:
                self.query = self.query[:-1]
            elif self.add_mode:
                self.add_mode = False
                self.status = "Replace mode."
            self.list_focused = False
            self._focus_search_results()
            return _PICKER_PENDING
        if isinstance(key, str) and key.isprintable():
            self.query += key
            self.list_focused = False
            self._focus_search_results()
            return _PICKER_PENDING
        return _PICKER_PENDING

    @staticmethod
    def _addstr(
        screen: "curses.window",
        row: int,
        col: int,
        value: str,
        attr: int = 0,
    ) -> None:
        height, width = screen.getmaxyx()
        if not (0 <= row < height and 0 <= col < width - 1):
            return
        try:
            screen.addnstr(row, col, value, max(0, width - col - 1), attr)
        except curses.error:
            pass

    def draw(self, screen: "curses.window") -> None:
        screen.erase()
        height, width = screen.getmaxyx()
        if height < 14 or width < 42:
            self._addstr(screen, 0, 0, "Terminal too small; resize to at least 42x14.")
            screen.refresh()
            return

        self._addstr(screen, 0, 0, " Artist Tagger ", self.title_attr)
        self._addstr(screen, 2, 0, f"Track: {self.track.stem}", self.heading_attr)
        current = format_artist_list(self.current_artists) or "None"
        self._addstr(screen, 3, 0, f"Current: {current}")
        self._addstr(screen, 4, 0, f"Selected: {len(self.selected)} artist(s)")
        mode = "Add to current" if self.add_mode else "Replace"
        self._addstr(screen, 5, 0, f"Mode: {mode}")

        search_prefix = "Input: "
        available = max(1, width - len(search_prefix) - 1)
        visible_query = self.query[-available:]
        self._addstr(screen, 6, 0, search_prefix, self.heading_attr)
        self._addstr(screen, 6, len(search_prefix), visible_query)

        options = self.options()
        selected_keys = {name.casefold() for name in self.selected}
        row = 8
        list_bottom = height - 3

        if self.selected and row < list_bottom:
            self._addstr(
                screen,
                row,
                0,
                "Selected artists (pinned)",
                self.heading_attr | curses.A_UNDERLINE,
            )
            row += 1
            for index, name in enumerate(self.selected):
                if row >= list_bottom:
                    break
                attr = (
                    self.focus_attr
                    if self.list_focused and self.focus == index
                    else 0
                )
                self._addstr(screen, row, 0, f"[x] {name}", attr)
                row += 1
            if row < list_bottom:
                self._addstr(screen, row, 0, "-" * min(36, width - 1), self.footer_attr)
                row += 1

        if row < list_bottom:
            result_label = (
                "Search results"
                if self.query.strip()
                else "Recent and common artists"
            )
            self._addstr(
                screen,
                row,
                0,
                result_label,
                self.heading_attr | curses.A_UNDERLINE,
            )
            row += 1

        result_height = max(0, list_bottom - row)
        if options and result_height:
            if self.list_focused and self.focus >= len(self.selected):
                result_focus = self.focus - len(self.selected)
                if result_focus < self.scroll:
                    self.scroll = result_focus
                elif result_focus >= self.scroll + result_height:
                    self.scroll = result_focus - result_height + 1
            self.scroll = max(
                0,
                min(self.scroll, max(0, len(options) - result_height)),
            )
            for row_offset, name in enumerate(
                options[self.scroll : self.scroll + result_height]
            ):
                option_index = self.scroll + row_offset
                focus_index = len(self.selected) + option_index
                marker = "[x]" if name.casefold() in selected_keys else "[ ]"
                attr = (
                    self.focus_attr
                    if self.list_focused and self.focus == focus_index
                    else 0
                )
                self._addstr(screen, row + row_offset, 0, f"{marker} {name}", attr)
        elif result_height:
            if self.query.strip():
                message = f'No saved artist matches "{" ".join(self.query.split())}".'
                self._addstr(screen, row, 0, message)
                if row + 1 < list_bottom:
                    self._addstr(screen, row + 1, 0, "Press Enter to add it as new.")
            else:
                self._addstr(screen, row, 0, "No saved artists yet. Type a name to begin.")

        self._addstr(screen, height - 3, 0, self.status)
        self._addstr(
            screen,
            height - 2,
            0,
            "Type search  Down focuses results  Space toggles focused artist",
            self.footer_attr,
        )
        self._addstr(
            screen,
            height - 1,
            0,
            "Enter adds unmatched name  Right Arrow saves/skips  - clears tag",
            self.footer_attr,
        )
        cursor_col = min(width - 2, len(search_prefix) + len(visible_query))
        try:
            curses.curs_set(0 if self.list_focused else 1)
            if not self.list_focused:
                screen.move(6, cursor_col)
        except curses.error:
            pass
        screen.refresh()

    def run(self, screen: "curses.window") -> Optional[Tuple[str, List[str]]]:
        screen.keypad(True)
        self.configure_theme(screen)
        try:
            curses.curs_set(1)
        except curses.error:
            pass
        while True:
            self.draw(screen)
            try:
                key = screen.get_wch()
            except curses.error:
                continue
            result = self.handle_key(key)
            if result is not _PICKER_PENDING:
                return result  # type: ignore[return-value]


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
    screen: Optional["curses.window"] = None,
    initial_status: str = "",
) -> Optional[Tuple[str, List[str]]]:
    picker = ArtistPicker(
        track,
        current_artists,
        choices,
        known_names,
        initial_status=initial_status,
    )
    if screen is not None:
        return picker.run(screen)
    try:
        return curses.wrapper(picker.run)
    except curses.error as exc:
        raise RuntimeError(
            "Could not start the artist picker. "
            "Run this script in an interactive terminal."
        ) from exc


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

    session_messages: List[str] = []
    notice = ""

    def run_session(screen: "curses.window") -> None:
        nonlocal notice, recent_artists
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
                artist_result = prompt_artist(
                    track,
                    current_artists,
                    choices,
                    known_names,
                    screen=screen,
                    initial_status=notice,
                )
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

            if artist_result is not None:
                artist_action, artist_names = artist_result
                if artist_action == "add":
                    artist_names = normalize_artist_names([*current_artists, *artist_names])

            if artist_result is None:
                if review_existing:
                    reviewed.add(track.stem)
                    save_reviewed(reviewed_file, reviewed)
                    notice = f"Skipped and marked reviewed: {track.stem}"
                else:
                    notice = f"Skipped: {track.stem}"
            elif artist_action == "clear" or not artist_names:
                artists.pop(track.stem, None)
                reviewed.discard(track.stem)
                save_json(artists_file, artists)
                if review_existing:
                    save_reviewed(reviewed_file, reviewed)
                notice = f"Cleared artist: {track.stem}"
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
                notice = (
                    f"Saved {format_artist_list(artist_names)} for {track.stem}"
                )

            session_messages.append(notice)

            if delay_between_tracks > 0:
                time.sleep(delay_between_tracks)

    try:
        try:
            curses.wrapper(run_session)
        except curses.error as exc:
            raise RuntimeError(
                "Could not start the artist picker. "
                "Run this script in an interactive terminal."
            ) from exc
    finally:
        if not no_play:
            shutdown_audio()

    for message in session_messages:
        print(message)


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
        help="Process tracks alphabetically. This is now the default.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Shuffle tracks before processing.",
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
            random_order=args.random,
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
