#!/usr/bin/env python3
from __future__ import annotations

import argparse
import curses
import locale
import math
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from external_listen_tracker import run_tracker
from external_listens import (
    EXTERNAL_MAPPINGS_FILE,
    EXTERNAL_PLAYS_FILE,
    LocalTrackRef,
    delete_external_id,
    distinct_tracks,
    load_mappings,
    load_plays,
    restore_last_deleted,
    save_mapping,
    store_revision,
)


ROOT = Path(__file__).resolve().parent
LIBRARIES = (
    ("mp3", "m", ROOT / "static/mp3"),
    ("mid-mp3s", "mm", ROOT / "static/mid-mp3s"),
)
DOUBLE_CLICK_SECONDS = 0.38
DATA_REFRESH_SECONDS = 0.35
UI_POLL_HZ = 20
UI_INTERACTIVE_HZ = 60
UI_INTERACTION_SECONDS = 0.5
INPUT_DRAIN_LIMIT = 64
MAX_WHEEL_STEPS_PER_RENDER = 1


@dataclass(frozen=True)
class LocalTrack:
    library: str
    tag: str
    filename: str

    @property
    def stem(self) -> str:
        return Path(self.filename).stem

    @property
    def label(self) -> str:
        return f"[{self.tag}] {self.stem}"


def folded(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def load_local_tracks() -> List[LocalTrack]:
    tracks: List[LocalTrack] = []
    for library, tag, folder in LIBRARIES:
        try:
            files = sorted(
                (
                    path
                    for path in folder.iterdir()
                    if path.is_file() and path.suffix.casefold() == ".mp3"
                ),
                key=lambda path: path.stem.casefold(),
            )
        except OSError:
            files = []
        tracks.extend(LocalTrack(library, tag, path.name) for path in files)
    return tracks


def display_width(value: str) -> int:
    width = 0
    for char in value:
        if unicodedata.combining(char) or unicodedata.category(char) == "Cf":
            continue
        width += 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
    return width


def truncate_display(value: str, width: int) -> str:
    width = max(0, width)
    result: List[str] = []
    used = 0
    for char in value:
        char_width = display_width(char)
        if used + char_width > width:
            break
        result.append(char)
        used += char_width
    return "".join(result)


def fit_display(value: str, width: int) -> str:
    value = truncate_display(value, width)
    return value + " " * max(0, width - display_width(value))


def _fit_left_and_right(left: str, right: str, width: int) -> str:
    """Keep the right-hand value visible while shortening the name as needed."""
    width = max(0, width)
    right_width = display_width(right)
    if width < right_width + 2:
        return right[-width:].rjust(width) if width else ""
    left_width = width - right_width - 2
    if display_width(left) > left_width:
        ellipsis = "…" if left_width else ""
        left = truncate_display(left, max(0, left_width - display_width(ellipsis)))
        left += ellipsis
    return f"{fit_display(left, left_width)}  {right}"


def external_listen_summary(item: dict) -> str:
    try:
        plays = max(0, int(item.get("qualified_plays") or 0))
    except (TypeError, ValueError):
        plays = 0
    try:
        duration = max(0.0, float(item.get("duration") or 0.0))
    except (TypeError, ValueError):
        duration = 0.0

    if plays and not duration:
        hours = "?h"
    else:
        total_hours = plays * duration / 3600.0
        hours = "<0.1h" if 0.0 < total_hours < 0.05 else f"{total_hours:.1f}h"
    return f"({plays:,} · {hours})"


def external_label(item: dict, width: int) -> str:
    title = str(item.get("title") or "Untitled")
    artist = str(item.get("artist") or "")
    suffix = f" · {artist}" if artist else ""
    return _fit_left_and_right(
        f"{title}{suffix}", external_listen_summary(item), width
    )


class MatcherUI:
    PANELS = ("external", "local", "history")

    def __init__(self, screen: "curses._CursesWindow") -> None:
        self.screen = screen
        self.focus = "external"
        self.search = ""
        self.status = (
            "Select an unmatched song, then double-click its local MP3 to match."
        )
        self.local_tracks = load_local_tracks()
        self.external_by_key: Dict[str, dict] = {}
        self.mappings: Dict[str, dict] = {}
        self.unmatched: List[dict] = []
        self.filtered_local: List[LocalTrack] = list(self.local_tracks)
        self.history: List[Tuple[str, dict]] = []
        self.selected = {panel: 0 for panel in self.PANELS}
        self.scroll = {panel: 0 for panel in self.PANELS}
        self.active_external_key: Optional[str] = None
        self.rematching = False
        self.bounds: Dict[str, Tuple[int, int, int, int]] = {}
        self.last_click_panel: Optional[str] = None
        self.last_click_identity: Optional[str] = None
        self.last_click_at = 0.0
        self.pending_delete_key: Optional[str] = None
        self.last_revision: object = None
        self.last_refresh_at = -math.inf
        self.last_input_at = -math.inf
        self.wheel_steps_this_render = 0
        self.running = True
        self._configure_curses()
        self.refresh_data(force=True)

    def _configure_curses(self) -> None:
        locale.setlocale(locale.LC_ALL, "")
        curses.noecho()
        curses.cbreak()
        self.screen.keypad(True)
        self.screen.nodelay(True)
        try:
            curses.curs_set(0)
            curses.set_escdelay(25)
        except Exception:
            pass
        try:
            curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
            curses.mouseinterval(0)
        except Exception:
            pass
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_CYAN, -1)
            curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_CYAN)
            curses.init_pair(3, curses.COLOR_YELLOW, -1)
            curses.init_pair(4, curses.COLOR_GREEN, -1)

    def refresh_data(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_refresh_at < DATA_REFRESH_SECONDS:
            return
        self.last_refresh_at = now
        revision = store_revision((EXTERNAL_PLAYS_FILE, EXTERNAL_MAPPINGS_FILE))
        if not force and revision == self.last_revision:
            return
        self.last_revision = revision

        old_keys = {
            "external": self._selected_external_key(),
            "local": self._selected_local_identity(),
            "history": self._selected_history_key(),
        }
        events = load_plays()
        self.external_by_key = distinct_tracks(events)
        self.mappings = load_mappings()
        self.unmatched = sorted(
            (
                item
                for key, item in self.external_by_key.items()
                if key not in self.mappings
            ),
            key=lambda item: str(item.get("last_played_at") or ""),
            reverse=True,
        )
        self.history = sorted(
            self.mappings.items(),
            key=lambda item: int(item[1].get("order") or 0),
            reverse=True,
        )
        self._filter_local_tracks()
        self._restore_selection("external", old_keys["external"])
        self._restore_selection("local", old_keys["local"])
        self._restore_selection("history", old_keys["history"])

    def _filter_local_tracks(self) -> None:
        query = folded(" ".join(self.search.split()))
        if not query:
            self.filtered_local = list(self.local_tracks)
        else:
            self.filtered_local = [
                track for track in self.local_tracks if query in folded(track.label)
            ]
        self.selected["local"] = min(
            self.selected["local"], max(0, len(self.filtered_local) - 1)
        )

    def _items(self, panel: str) -> list:
        if panel == "external":
            return self.unmatched
        if panel == "local":
            return self.filtered_local
        return self.history

    def _identity(self, panel: str, item: object) -> str:
        if panel == "external" and isinstance(item, dict):
            return str(item.get("external_key") or "")
        if panel == "local" and isinstance(item, LocalTrack):
            return f"{item.library}/{item.filename}"
        if panel == "history" and isinstance(item, tuple):
            return str(item[0])
        return ""

    def _selected_identity(self, panel: str) -> Optional[str]:
        items = self._items(panel)
        if not items:
            return None
        index = min(self.selected[panel], len(items) - 1)
        return self._identity(panel, items[index])

    def _selected_external_key(self) -> Optional[str]:
        return self._selected_identity("external")

    def _selected_local_identity(self) -> Optional[str]:
        return self._selected_identity("local")

    def _selected_history_key(self) -> Optional[str]:
        return self._selected_identity("history")

    def _restore_selection(self, panel: str, identity: Optional[str]) -> None:
        items = self._items(panel)
        if identity:
            for index, item in enumerate(items):
                if self._identity(panel, item) == identity:
                    self.selected[panel] = index
                    return
        self.selected[panel] = min(self.selected[panel], max(0, len(items) - 1))

    def _external_for_key(self, key: str) -> dict:
        item = self.external_by_key.get(key)
        if item:
            return item
        return {"external_key": key, "title": key}

    def select_external(self) -> None:
        key = self._selected_external_key()
        if not key:
            self.status = "There are no unmatched external songs to select."
            return
        self.active_external_key = key
        self.rematching = False
        item = self._external_for_key(key)
        self.status = f"Selected: {item.get('title') or 'Untitled'}"

    def delete_selected(self) -> None:
        if self.focus != "external":
            self.status = "Select an unmatched external song first."
            return
        key = self._selected_external_key()
        if not key:
            self.status = "There are no unmatched external songs to delete."
            return
        item = self._external_for_key(key)
        title = str(item.get("title") or key)
        self.pending_delete_key = key
        self.status = f'Move "{title}" to Trash? Press y to confirm.'

    def confirm_delete(self) -> None:
        key = self.pending_delete_key
        self.pending_delete_key = None
        if not key:
            return
        item = self._external_for_key(key)
        title = str(item.get("title") or key)
        deleted = delete_external_id(key)
        if deleted is None:
            self.status = f'Could not find "{title}" to move to Trash.'
            return
        if self.active_external_key == key:
            self.active_external_key = None
        self.refresh_data(force=True)
        self.status = f'Moved "{title}" to Trash. Press u to undo.'

    def undo_delete(self) -> None:
        self.pending_delete_key = None
        record = restore_last_deleted()
        if record is None:
            self.status = "Trash is empty; there is nothing to undo."
            return
        self.refresh_data(force=True)
        self._restore_selection("external", str(record.get("id") or ""))
        self.status = f'Restored "{record.get("title") or "Untitled"} from Trash.'

    def begin_rematch(self) -> None:
        key = self._selected_history_key()
        if not key:
            self.status = "There are no recorded matches to change."
            return
        self.active_external_key = key
        self.rematching = True
        self.focus = "local"
        item = self._external_for_key(key)
        self.status = (
            f"Changing match for {item.get('title') or 'Untitled'}; "
            "double-click the replacement MP3."
        )

    def commit_match(self) -> None:
        if not self.active_external_key:
            self.status = "Select an external song on the left first."
            return
        if not self.filtered_local:
            self.status = "No local MP3 is selected."
            return
        track = self.filtered_local[
            min(self.selected["local"], len(self.filtered_local) - 1)
        ]
        external = self._external_for_key(self.active_external_key)
        save_mapping(
            self.active_external_key,
            LocalTrackRef(track.library, track.filename),
        )
        title = str(external.get("title") or "Untitled")
        verb = "Changed" if self.rematching else "Matched"
        self.status = f"{verb} {title} → {track.label}"
        mapped_key = self.active_external_key
        self.active_external_key = None
        self.rematching = False
        self.refresh_data(force=True)
        self.focus = "history"
        self._restore_selection("history", mapped_key)
        self.scroll["history"] = 0

    def move_selection(self, amount: int) -> None:
        items = self._items(self.focus)
        if not items:
            return
        self.selected[self.focus] = min(
            max(0, self.selected[self.focus] + amount), len(items) - 1
        )

    def page_selection(self, amount: int) -> None:
        bounds = self.bounds.get(self.focus, (0, 0, 0, 10))
        self.move_selection(amount * max(1, bounds[3] - 3))

    def activate_focused(self) -> None:
        if self.focus == "external":
            self.select_external()
        elif self.focus == "local":
            self.commit_match()
        else:
            self.begin_rematch()

    def handle_click(self, mouse_x: int, mouse_y: int) -> None:
        for panel in self.PANELS:
            x, y, width, height = self.bounds.get(panel, (0, 0, 0, 0))
            if not (x <= mouse_x < x + width and y <= mouse_y < y + height):
                continue
            self.focus = panel
            row = mouse_y - (y + 1)
            index = self.scroll[panel] + row
            items = self._items(panel)
            if row < 0 or index < 0 or index >= len(items):
                return
            self.selected[panel] = index
            identity = self._identity(panel, items[index])
            now = time.monotonic()
            double = (
                self.last_click_panel == panel
                and self.last_click_identity == identity
                and now - self.last_click_at <= DOUBLE_CLICK_SECONDS
            )
            self.last_click_panel = panel
            self.last_click_identity = identity
            self.last_click_at = now
            if panel == "external":
                self.select_external()
            elif double and panel == "local":
                self.commit_match()
            elif double and panel == "history":
                self.begin_rematch()
            return

    def handle_wheel(
        self, mouse_x: int, mouse_y: int, state: int
    ) -> bool:
        wheel_up = 0
        for name in ("BUTTON4_PRESSED", "BUTTON4_CLICKED", "BUTTON4_RELEASED"):
            wheel_up |= int(getattr(curses, name, 0))
        if wheel_up == 0:
            wheel_up = 0x00080000

        wheel_down = 0
        for name in ("BUTTON5_PRESSED", "BUTTON5_CLICKED", "BUTTON5_RELEASED"):
            wheel_down |= int(getattr(curses, name, 0))
        wheel_down |= 0x00200000 | 0x08000000

        if not (state & wheel_up or state & wheel_down):
            return False
        if self.wheel_steps_this_render >= MAX_WHEEL_STEPS_PER_RENDER:
            return True
        self.wheel_steps_this_render += 1
        direction = -1 if state & wheel_up else 1
        for panel in self.PANELS:
            x, y, width, height = self.bounds.get(panel, (0, 0, 0, 0))
            if x <= mouse_x < x + width and y <= mouse_y < y + height:
                self.focus = panel
                self.move_selection(direction)
                break
        return True

    def handle_key(self, key: int) -> None:
        if key == -1:
            return
        if key in (3, curses.KEY_F10):
            self.running = False
            return
        if key == curses.KEY_RESIZE:
            return
        if self.pending_delete_key:
            if key in (ord("y"), ord("Y")):
                self.confirm_delete()
                return
            self.pending_delete_key = None
            if key in (ord("n"), ord("N"), 27):
                self.status = "Delete cancelled."
                return
        if key == curses.KEY_DC or (
            key in (127, curses.KEY_BACKSPACE)
            and self.focus == "external"
        ):
            self.delete_selected()
            return
        if key == 26 or (
            key in (ord("u"), ord("U"))
            and self.focus == "external"
            and not self.search
        ):
            self.undo_delete()
            return
        if key == curses.KEY_MOUSE:
            try:
                _, mouse_x, mouse_y, _, state = curses.getmouse()
            except curses.error:
                return
            if self.handle_wheel(mouse_x, mouse_y, state):
                return
            click_mask = 0
            for name in ("BUTTON1_PRESSED", "BUTTON1_CLICKED"):
                click_mask |= int(getattr(curses, name, 0))
            if state & click_mask:
                self.handle_click(mouse_x, mouse_y)
            return
        if key == 9:
            index = (self.PANELS.index(self.focus) + 1) % len(self.PANELS)
            self.focus = self.PANELS[index]
            return
        if key in (curses.KEY_UP, 16):
            self.move_selection(-1)
            return
        if key in (curses.KEY_DOWN, 14):
            self.move_selection(1)
            return
        if key == curses.KEY_PPAGE:
            self.page_selection(-1)
            return
        if key == curses.KEY_NPAGE:
            self.page_selection(1)
            return
        if key in (10, 13, curses.KEY_ENTER):
            self.activate_focused()
            return
        if key == 27:
            if self.search:
                self.search = ""
                self._filter_local_tracks()
            elif self.active_external_key:
                self.active_external_key = None
                self.rematching = False
                self.status = "Match selection cancelled."
            return
        if key in (8, 127, curses.KEY_BACKSPACE):
            if self.search:
                self.search = self.search[:-1]
                self._filter_local_tracks()
                self.selected["local"] = 0
                self.scroll["local"] = 0
            return
        if 32 <= key <= 0x10FFFF:
            try:
                char = chr(key)
            except ValueError:
                return
            if char.isprintable():
                self.search += char
                self._filter_local_tracks()
                self.selected["local"] = 0
                self.scroll["local"] = 0
                self.focus = "local"

    def _add(
        self, y: int, x: int, text: str, attr: int = 0, width: Optional[int] = None
    ) -> None:
        height, screen_width = self.screen.getmaxyx()
        if y < 0 or y >= height or x < 0 or x >= screen_width:
            return
        maximum = max(0, screen_width - x - 1)
        if width is not None:
            maximum = min(maximum, max(0, width))
        if maximum <= 0:
            return
        try:
            self.screen.addnstr(y, x, text, maximum, attr)
        except curses.error:
            pass

    def _box(
        self, x: int, y: int, width: int, height: int, title: str, focused: bool
    ) -> None:
        if width < 4 or height < 3:
            return
        attr = curses.color_pair(1) | curses.A_BOLD if focused else curses.A_DIM
        self._add(y, x, "+" + "-" * (width - 2) + "+", attr, width)
        for row in range(y + 1, y + height - 1):
            self._add(row, x, "|", attr)
            self._add(row, x + width - 1, "|", attr)
        self._add(y + height - 1, x, "+" + "-" * (width - 2) + "+", attr, width)
        self._add(y, x + 2, f" {title} ", attr, max(0, width - 4))

    def _row_text(self, panel: str, item: object, width: int) -> str:
        if panel == "external" and isinstance(item, dict):
            return external_label(item, width)
        if panel == "local" and isinstance(item, LocalTrack):
            return item.label
        if panel == "history" and isinstance(item, tuple):
            key, mapping = item
            external = self._external_for_key(key)
            target = mapping.get("target", {})
            library = str(target.get("library") or "")
            tag = "mm" if library == "mid-mp3s" else "m"
            local_name = Path(str(target.get("track") or "")).stem
            return f"{external.get('title') or 'Untitled'} → [{tag}] {local_name}"
        return ""

    def _render_panel(self, panel: str, title: str) -> None:
        x, y, width, height = self.bounds[panel]
        self._box(x, y, width, height, title, self.focus == panel)
        items = self._items(panel)
        visible_rows = max(0, height - 2)
        if items:
            selected = min(self.selected[panel], len(items) - 1)
            self.selected[panel] = selected
            if selected < self.scroll[panel]:
                self.scroll[panel] = selected
            elif selected >= self.scroll[panel] + visible_rows:
                self.scroll[panel] = selected - visible_rows + 1
            self.scroll[panel] = min(
                max(0, self.scroll[panel]), max(0, len(items) - visible_rows)
            )
        else:
            self.scroll[panel] = 0
        for row in range(visible_rows):
            index = self.scroll[panel] + row
            if index >= len(items):
                break
            item = items[index]
            selected = index == self.selected[panel]
            identity = self._identity(panel, item)
            active = (
                panel in {"external", "history"}
                and identity == self.active_external_key
            )
            attr = 0
            if selected:
                attr = curses.color_pair(2) | curses.A_BOLD
            elif active:
                attr = curses.color_pair(3) | curses.A_BOLD
            prefix = "> " if active else "  "
            text = fit_display(
                prefix + self._row_text(panel, item, width - 4),
                width - 2,
            )
            self._add(
                y + 1 + row, x + 1, text, attr, width - 2
            )
        if not items and visible_rows:
            empty = {
                "external": "No unmatched external songs",
                "local": "No MP3s match the search",
                "history": "No matches recorded yet",
            }[panel]
            self._add(y + 1, x + 2, empty, curses.A_DIM, width - 4)

    def render(self) -> None:
        self.screen.erase()
        height, width = self.screen.getmaxyx()
        if height < 10 or width < 72:
            self._add(0, 0, "Enlarge the terminal to at least 72x10.", curses.A_BOLD)
            self.screen.refresh()
            return
        content_top = 2
        content_height = height - 6
        left_width = width // 3
        middle_width = width // 3
        right_width = width - left_width - middle_width
        self.bounds = {
            "external": (0, content_top, left_width, content_height),
            "local": (left_width, content_top, middle_width, content_height),
            "history": (
                left_width + middle_width,
                content_top,
                right_width,
                content_height,
            ),
        }
        self._add(0, 0, "External Listen Matcher", curses.color_pair(1) | curses.A_BOLD)
        self._add(
            1,
            0,
            "Click an unmatched song to select · Delete then y moves it to Trash · "
            "u restores · double-click a local MP3 to match",
            curses.A_DIM,
        )
        self._render_panel("external", f"Unmatched ({len(self.unmatched)})")
        self._render_panel("local", f"Local MP3s ({len(self.filtered_local)})")
        self._render_panel("history", f"Match history ({len(self.history)})")

        search_y = height - 3
        status_y = height - 2
        help_y = height - 1
        self._add(search_y, 0, "Search: ", curses.color_pair(1) | curses.A_BOLD)
        search_text = self.search or "type to filter both local folders"
        search_attr = 0 if self.search else curses.A_DIM
        self._add(search_y, 8, search_text, search_attr)
        self._add(status_y, 0, self.status, curses.color_pair(4))
        self._add(
            help_y,
            0,
            "Delete then y trash  u undo  Tab columns  ↑↓ select  "
            "mouse wheel scroll  F10/Ctrl-C quit",
            curses.A_DIM,
        )
        try:
            self.screen.refresh()
        except curses.error:
            pass

    def run(self) -> None:
        while self.running:
            self.refresh_data()
            self.wheel_steps_this_render = 0
            saw_input = False
            for _ in range(INPUT_DRAIN_LIMIT):
                try:
                    key = self.screen.get_wch()
                except curses.error:
                    break
                saw_input = True
                if isinstance(key, str):
                    self.handle_key(ord(key))
                else:
                    self.handle_key(key)
            if saw_input:
                self.last_input_at = time.monotonic()
            self.render()
            interactive = (
                time.monotonic() - self.last_input_at < UI_INTERACTION_SECONDS
            )
            poll_hz = UI_INTERACTIVE_HZ if interactive else UI_POLL_HZ
            time.sleep(1.0 / poll_hz)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Three-column TUI for explicitly matching external plays to local MP3s."
        )
    )
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Do not monitor macOS Now Playing while this interface is open.",
    )
    return parser.parse_args()


def curses_main(screen: "curses._CursesWindow") -> None:
    MatcherUI(screen).run()


def main() -> int:
    args = parse_args()
    stop_event = threading.Event()
    tracker_thread: Optional[threading.Thread] = None
    if not args.no_monitor:
        tracker_thread = threading.Thread(
            target=run_tracker,
            args=(1.0, stop_event, True),
            daemon=True,
        )
        tracker_thread.start()
    try:
        curses.wrapper(curses_main)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        if tracker_thread is not None:
            tracker_thread.join(timeout=3.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
