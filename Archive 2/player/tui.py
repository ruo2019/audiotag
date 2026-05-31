import locale
import os
import queue
import random
import subprocess
import sys
import textwrap
import time
from collections import deque

import curses
import jellyfish
import pygame

from .audio_analysis import analyze_audio_folder
from .autoplay import build_autoplay_snapshot, choose_autoplay_song
from .constants import (
    ACTIVE_PLAYLIST_TAB_CONFIG_KEY,
    ALL_SONGS_TAB_ID,
    AUTOPLAY_FALLBACK_PROBABILITY,
    AUTOPLAY_WINDOW_SECONDS,
    AUTO_CORRECT_DISTANCE,
    CMD_ADD_PREFIX,
    CMD_AUTOPLAY_PREFIX,
    CMD_DEL_PREFIX,
    CMD_DROP_PREFIX,
    CMD_NEW_PREFIX,
    CMD_SORT_PREFIX,
    MAX_LOG_LINES,
    PLAYLISTS_CONFIG_KEY,
    PLAYLIST_ORDER_CONFIG_KEY,
    SCROLLBAR_CHAR,
    TRENDING_WINDOW_DAYS,
    VOLUME_STEPS,
)


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
        self.autoplay_session = None

        self.all_songs_data = [] # Full list for scrolling
        self.scroll_pos = 0     # Current scroll position for the playlist view
        self.playback_stopped = True # Flag to track if user manually stopped audio

        self.mode = 'typing' # 'typing' or 'scroll'
        self.sort_mode = "name"
        self.search_query = ""
        self.display_data = []
        self.logs = deque(maxlen=MAX_LOG_LINES)
        self.search_select_index = 0
        self.search_highlight_attr = curses.A_REVERSE
        self.add_search_highlight_attr = curses.A_REVERSE
        self.del_search_highlight_attr = curses.A_REVERSE
        self.playlist_name_attr = curses.A_REVERSE
        self.playlist_name_active_attr = curses.A_REVERSE | curses.A_BOLD
        self.input_header_search_attr = curses.A_BOLD
        self.input_header_add_attr = curses.A_BOLD
        self.input_header_del_attr = curses.A_BOLD
        self.input_header_new_attr = curses.A_BOLD
        self.input_header_drop_attr = curses.A_BOLD
        self.input_header_sort_attr = curses.A_BOLD
        self.input_header_inactive_attr = curses.A_DIM
        self.sort_accent_attr = curses.A_BOLD
        self.sort_accent_pair_id = 1
        self.volume_level = VOLUME_STEPS
        self.volume_multiplier = 1.0
        self.current_channel = None
        self.global_actions = queue.Queue()
        self._media_listener = None
        self.playlists = {}
        self.playlist_order = []
        self.active_tab_id = ALL_SONGS_TAB_ID
        self.tab_context = {}
        self.command_context = {
            "mode": None,
            "query": "",
            "scroll_pos": 0,
            "select_index": 0,
        }
        self._active_view_state = None
        self._active_view_has_query = False
        self._active_command_mode = None
        self.add_sidebar_rows = []
        self.add_sidebar_query_hits = 0
        self.input_text = ""
        self.input_cursor = 0
        self._ui_get_input_line = None
        self._ui_clear_input_line = None
        self._ui_set_input_text = None
        self._ui_draw_input_mode = None
        self.input_box = None
        self._last_screen_size = None
        self._tab_click_regions = []
        self._last_progress_second = None
        self._getch_delay = 100
        self.help_mode = False
        self._last_screen_state_check = 0.0
        self._screen_locked_or_saver = False
        self._screen_lock_poll_s = 2.0
        self._load_playlists_from_config()
        self._reset_current_song_info()

    def _reset_current_song_info(self):
        self.current_song_info = {
            "basename": "None",
            "loudness": "N/A",
            "base_scale": None,
            "scale": "N/A",
            "rank_info": "N/A",
            "analysis": "N/A"
        }
        self.current_song_duration = 0
        self.current_song_start_time = 0
        self.current_song_basename = None
        self.current_song_listen_recorded = False
        self.current_channel = None
        self._last_progress_second = None

    def _new_view_context(self):
        return {
            "query": "",
            "scroll_pos": 0,
            "select_index": 0,
        }

    def _sanitize_playlist_name(self, name):
        if not isinstance(name, str):
            return ""
        cleaned = " ".join(name.strip().split())
        return cleaned[:80]

    def _ensure_tab_context(self, tab_id):
        if tab_id not in self.tab_context:
            self.tab_context[tab_id] = self._new_view_context()

    def _load_playlists_from_config(self):
        raw_playlists = self.meta_manager.get_config_value(PLAYLISTS_CONFIG_KEY, {})
        raw_order = self.meta_manager.get_config_value(PLAYLIST_ORDER_CONFIG_KEY, [])
        raw_active = self.meta_manager.get_config_value(ACTIVE_PLAYLIST_TAB_CONFIG_KEY, ALL_SONGS_TAB_ID)

        parsed = {}
        if isinstance(raw_playlists, dict):
            for key, songs in raw_playlists.items():
                name = self._sanitize_playlist_name(key)
                if not name:
                    continue
                cleaned_songs = []
                seen = set()
                if isinstance(songs, list):
                    for song in songs:
                        if not isinstance(song, str):
                            continue
                        basename = song.strip()
                        if not basename or basename in seen:
                            continue
                        seen.add(basename)
                        cleaned_songs.append(basename)
                parsed[name] = cleaned_songs

        ordered_names = []
        seen_names = set()
        if isinstance(raw_order, list):
            for name in raw_order:
                clean_name = self._sanitize_playlist_name(name)
                if clean_name and clean_name in parsed and clean_name not in seen_names:
                    ordered_names.append(clean_name)
                    seen_names.add(clean_name)
        for name in sorted(parsed.keys(), key=lambda item: item.lower()):
            if name not in seen_names:
                ordered_names.append(name)
                seen_names.add(name)

        self.playlist_order = ordered_names
        self.playlists = {name: list(parsed.get(name, [])) for name in self.playlist_order}

        self.tab_context = {ALL_SONGS_TAB_ID: self._new_view_context()}
        for name in self.playlist_order:
            self._ensure_tab_context(name)

        if isinstance(raw_active, str) and (raw_active == ALL_SONGS_TAB_ID or raw_active in self.playlists):
            self.active_tab_id = raw_active
        else:
            self.active_tab_id = ALL_SONGS_TAB_ID

    def _persist_playlists_to_config(self):
        payload = {name: list(self.playlists.get(name, [])) for name in self.playlist_order if name in self.playlists}
        self.meta_manager.set_config_value(PLAYLISTS_CONFIG_KEY, payload)
        self.meta_manager.set_config_value(PLAYLIST_ORDER_CONFIG_KEY, list(self.playlist_order))
        self.meta_manager.set_config_value(ACTIVE_PLAYLIST_TAB_CONFIG_KEY, self.active_tab_id)

    def _sync_playlists_with_library(self):
        known_basenames = {basename for basename, _ in self.all_songs_data}
        changed = False

        for name in self.playlist_order:
            songs = self.playlists.get(name, [])
            filtered = []
            seen = set()
            for basename in songs:
                if basename in known_basenames and basename not in seen:
                    filtered.append(basename)
                    seen.add(basename)
            if filtered != songs:
                self.playlists[name] = filtered
                changed = True
            self._ensure_tab_context(name)

        self._ensure_tab_context(ALL_SONGS_TAB_ID)
        for tab_id in list(self.tab_context.keys()):
            if tab_id != ALL_SONGS_TAB_ID and tab_id not in self.playlists:
                del self.tab_context[tab_id]

        if self.active_tab_id != ALL_SONGS_TAB_ID and self.active_tab_id not in self.playlists:
            self.active_tab_id = ALL_SONGS_TAB_ID
            changed = True

        if changed:
            self._persist_playlists_to_config()

    def _get_all_song_basenames_sorted(self):
        return [basename for basename, _ in self._get_sorted_all_songs()]

    def _get_tab_song_basenames(self, tab_id=None):
        target_tab = self.active_tab_id if tab_id is None else tab_id
        if target_tab == ALL_SONGS_TAB_ID:
            return self._get_all_song_basenames_sorted()
        playlist_songs = set(self.playlists.get(target_tab, []))
        return [basename for basename, _ in self._get_sorted_all_songs() if basename in playlist_songs]

    def _get_tab_ids(self):
        return [ALL_SONGS_TAB_ID] + list(self.playlist_order)

    def _get_tab_label(self, tab_id):
        if tab_id == ALL_SONGS_TAB_ID:
            return "All Songs"
        return tab_id

    def _set_active_tab(self, tab_id):
        if tab_id != ALL_SONGS_TAB_ID and tab_id not in self.playlists:
            return
        if self.active_tab_id == tab_id:
            return
        self.active_tab_id = tab_id
        self.meta_manager.set_config_value(ACTIVE_PLAYLIST_TAB_CONFIG_KEY, self.active_tab_id)
        self._refresh_display_data(reset_scroll=False)

    def _switch_tab(self, delta):
        tab_ids = self._get_tab_ids()
        if not tab_ids:
            return
        try:
            idx = tab_ids.index(self.active_tab_id)
        except ValueError:
            idx = 0
        self._set_active_tab(tab_ids[(idx + delta) % len(tab_ids)])

    def _create_playlist(self, raw_name):
        clean_name = self._sanitize_playlist_name(raw_name)
        if not clean_name:
            return False, "Playlist name cannot be empty."

        for existing in self.playlist_order:
            if existing.lower() == clean_name.lower():
                self._set_active_tab(existing)
                return False, f"Switched to existing playlist '{existing}'."

        self.playlists[clean_name] = []
        self.playlist_order.append(clean_name)
        self._ensure_tab_context(clean_name)
        self.active_tab_id = clean_name
        self._persist_playlists_to_config()
        return True, f"Created playlist '{clean_name}'."

    def _delete_playlist(self, raw_name):
        clean_name = self._sanitize_playlist_name(raw_name)
        if not clean_name:
            return False, "Playlist name cannot be empty."
        if clean_name == ALL_SONGS_TAB_ID:
            return False, "Cannot delete the All Songs tab."

        target_name = None
        for existing in self.playlist_order:
            if existing.lower() == clean_name.lower():
                target_name = existing
                break
        if target_name is None:
            return False, f"Playlist '{clean_name}' was not found."

        self.playlist_order = [name for name in self.playlist_order if name != target_name]
        self.playlists.pop(target_name, None)
        self.tab_context.pop(target_name, None)
        if self.active_tab_id == target_name:
            self.active_tab_id = ALL_SONGS_TAB_ID
        self._persist_playlists_to_config()
        return True, f"Deleted playlist '{target_name}'."

    def _add_song_to_active_playlist(self, basename):
        if self.active_tab_id == ALL_SONGS_TAB_ID:
            return False, "Switch to a playlist tab before using /add."
        songs = self.playlists.get(self.active_tab_id, [])
        if basename in songs:
            return False, f"'{basename}' is already in '{self.active_tab_id}'."
        songs.append(basename)
        self.playlists[self.active_tab_id] = songs
        self._persist_playlists_to_config()
        return True, f"Added '{basename}' to '{self.active_tab_id}'."

    def _delete_song_from_active_playlist(self, basename):
        if self.active_tab_id == ALL_SONGS_TAB_ID:
            return False, "Switch to a playlist tab before using /del."
        songs = self.playlists.get(self.active_tab_id, [])
        if basename not in songs:
            return False, f"'{basename}' is not in '{self.active_tab_id}'."
        self.playlists[self.active_tab_id] = [song for song in songs if song != basename]
        self._persist_playlists_to_config()
        return True, f"Removed '{basename}' from '{self.active_tab_id}'."

    def _match_command_prefix(self, lowered, prefix):
        if not lowered.startswith(prefix):
            return None
        if len(lowered) == len(prefix):
            return len(prefix)
        if lowered[len(prefix)].isspace():
            return len(prefix) + 1
        return None

    def _parse_input_command(self, line):
        lowered = line.lower()
        for mode, prefix in (
            ("add", CMD_ADD_PREFIX),
            ("del", CMD_DEL_PREFIX),
            ("new", CMD_NEW_PREFIX),
            ("drop", CMD_DROP_PREFIX),
            ("sort", CMD_SORT_PREFIX),
            ("autoplay", CMD_AUTOPLAY_PREFIX),
        ):
            prefix_len = self._match_command_prefix(lowered, prefix)
            if prefix_len is not None:
                return mode, prefix_len
        return None, 0

    def _get_pending_input_command(self, line=None):
        content = self.input_text if line is None else line
        mode, _ = self._parse_input_command(content)
        if mode:
            return mode
        token = content.strip().lower()
        if token in (CMD_ADD_PREFIX, CMD_DEL_PREFIX, CMD_NEW_PREFIX, CMD_DROP_PREFIX, CMD_SORT_PREFIX, CMD_AUTOPLAY_PREFIX):
            return token.lstrip("/")
        return None

    def _get_input_header(self, line=None):
        mode = self._get_pending_input_command(line)
        if mode == "add":
            return "Add to Playlist", self.input_header_add_attr
        if mode == "del":
            return "Delete from Playlist", self.input_header_del_attr
        if mode == "new":
            return "New Playlist", self.input_header_new_attr
        if mode == "drop":
            return "Drop Playlist", self.input_header_drop_attr
        if mode == "sort":
            return "Sort Songs", self.input_header_sort_attr
        if mode == "autoplay":
            return "Autoplay", self.input_header_sort_attr
        return None

    def _get_input_anchor(self, line=None):
        content = self.input_text if line is None else line
        mode, prefix_len = self._parse_input_command(content)
        if mode in ("add", "del", "new", "drop", "sort", "autoplay"):
            return prefix_len
        return 0

    def _normalize_sort_token(self, text):
        return "".join(ch for ch in text.lower() if ch.isalnum())

    def _get_sort_option_specs(self):
        return [
            {"mode": "name", "label": "Alphabetical (A-Z)", "tokens": ("name", "alpha", "alphabetical", "az")},
            {"mode": "listens", "label": "Most listens", "tokens": ("listens", "listen", "plays", "count")},
            {"mode": "trending", "label": "Most trending", "tokens": ("trending", "trend", "delta", "7day", "7days", "5day", "5days")},
            {"mode": "time", "label": "Most listen time", "tokens": ("time", "hours", "duration", "listentime", "listenhours")},
            {"mode": "recent", "label": "Most recent listen", "tokens": ("recent", "last", "lastplayed", "lastlisten", "date")},
        ]

    def _resolve_sort_mode(self, text):
        normalized = self._normalize_sort_token(text)
        if not normalized:
            return None
        for spec in self._get_sort_option_specs():
            label_norm = self._normalize_sort_token(spec["label"])
            if normalized == label_norm or normalized.startswith(label_norm) or label_norm.startswith(normalized):
                return spec["mode"]
            for token in spec["tokens"]:
                if normalized == token or normalized.startswith(token):
                    return spec["mode"]
        return None

    def _build_sort_display_data(self, query):
        options = self._get_sort_option_specs()
        if query:
            query_lower = query.lower()
            query_norm = self._normalize_sort_token(query)
            filtered = []
            for spec in options:
                label_lower = spec["label"].lower()
                tokens = spec["tokens"]
                token_match = False
                if query_norm:
                    for token in tokens:
                        if query_norm == token or token.startswith(query_norm) or query_norm.startswith(token):
                            token_match = True
                            break
                if query_lower in label_lower or token_match:
                    filtered.append(spec)
            options = filtered
        return [(spec["label"], 0, spec["mode"]) for spec in options]

    def _describe_sort_mode(self, mode):
        return {
            "name": "alphabetical",
            "listens": "most listens",
            "trending": f"most trending ({TRENDING_WINDOW_DAYS} days)",
            "time": "most listen time",
            "recent": "most recent listen",
        }.get(mode, "alphabetical")

    def _get_trending_listen_count(self, basename, now=None):
        return self.meta_manager.get_recent_play_count(
            basename,
            TRENDING_WINDOW_DAYS,
            now=now,
        )

    def _playlist_max_items(self, command_mode=None, query=None):
        max_h, _ = self.win_playlist.getmaxyx()
        reserved_rows = 2
        return max(1, max_h - reserved_rows)

    def _is_autoplay_active(self):
        return isinstance(self.autoplay_session, dict) and bool(self.autoplay_session)

    def _get_scroll_box_highlight_attr(self):
        if self._is_autoplay_active():
            return self.sort_accent_attr
        return curses.color_pair(2)

    def _build_autoplay_session(self, pool_basenames, seed_basename=None):
        library_basenames = [basename for basename, _ in self.all_songs_data]
        history_mapping = {
            basename: self.meta_manager.get_play_history(basename)
            for basename in library_basenames
        }
        snapshot = build_autoplay_snapshot(
            library_basenames,
            history_mapping,
            window_seconds=AUTOPLAY_WINDOW_SECONDS,
            fallback_probability=AUTOPLAY_FALLBACK_PROBABILITY,
        )
        return {
            "snapshot": snapshot,
            "pool": list(pool_basenames),
            "pending_seed": seed_basename,
            "tab_id": self.active_tab_id,
        }

    def _resolve_autoplay_seed(self, seed_query, pool_basenames):
        query = seed_query.strip()
        if not query:
            return None, None

        if self.display_data:
            selected_index = max(0, min(self.search_select_index, len(self.display_data) - 1))
            selected = self.display_data[selected_index][0]
            if selected in pool_basenames:
                return selected, None

        for known_basename in pool_basenames:
            if query.lower() == known_basename.lower():
                return known_basename, None

        closest_match, _ = self._find_closest_match(
            query,
            pool_basenames,
            max_distance=AUTO_CORRECT_DISTANCE,
        )
        if closest_match:
            return closest_match, f"Auto-corrected autoplay seed to '{closest_match}'."
        return None, f"Error: Song not found: {query}"

    def _start_autoplay(self, seed_basename=None):
        pool_to_set = self._get_tab_song_basenames(self.active_tab_id)
        if not pool_to_set:
            self._display_message("This tab has no songs to autoplay.", 4, 1.5)
            return False

        self.autoplay_session = self._build_autoplay_session(pool_to_set, seed_basename=seed_basename)
        self.current_pool_basenames = list(pool_to_set)
        self.playback_queue = []
        self.mode = 'scroll'
        self.playback_stopped = False
        pygame.mixer.stop()
        self._reset_current_song_info()
        self._set_search_query("")

        tab_name = self._get_tab_label(self.active_tab_id)
        if seed_basename:
            self._log(f"Autoplay snapshot locked for '{tab_name}' with seed '{seed_basename}'.")
            message = f"Autoplay from '{seed_basename}'."
        else:
            self._log(f"Autoplay snapshot locked for '{tab_name}' ({len(pool_to_set)} songs).")
            message = f"Autoplaying '{tab_name}'."
        if self.win_input:
            _, input_w = self.win_input.getmaxyx()
            message = self._truncate_line(message, max(1, input_w - 4))
        self._display_message(message, 2, 1.2)
        return True

    def _choose_autoplay_song(self):
        if not self._is_autoplay_active():
            return None
        pending_seed = self.autoplay_session.get("pending_seed")
        if pending_seed:
            self.autoplay_session["pending_seed"] = None
            return pending_seed
        return choose_autoplay_song(
            self.autoplay_session.get("snapshot"),
            self.autoplay_session.get("pool") or self.current_pool_basenames,
            current_song=self.current_song_basename,
            rng=random,
        )

    def _quit_autoplay_mode(self):
        if not self._is_autoplay_active():
            return False
        pygame.mixer.stop()
        self._reset_current_song_info()
        self.current_pool_basenames = []
        self.playback_queue = []
        self.playback_stopped = True
        self.autoplay_session = None
        self.mode = 'scroll'
        self._log("Exited autoplay.")
        if self._ui_draw_input_mode:
            self._ui_draw_input_mode()
        return True

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

    def _log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)

    def _set_volume_level(self, level, log_change=True):
        clamped = max(0, min(VOLUME_STEPS, int(level)))
        if clamped != self.volume_level:
            self.volume_level = clamped
            self.volume_multiplier = self.volume_level / VOLUME_STEPS
            if log_change:
                self._log(f"Volume scale set to {self.volume_level}/{VOLUME_STEPS}")
        self._apply_current_volume()

    def _apply_current_volume(self):
        if not self.current_channel:
            return
        self.current_channel.set_volume(max(0.0, min(1.0, self.volume_multiplier)))
        base_scale = self.current_song_info.get("base_scale")
        if isinstance(base_scale, float):
            self.current_song_info["scale"] = f"{base_scale * self.volume_multiplier:.3f}"

    def _get_sorted_all_songs(self):
        data = list(self.all_songs_data)
        now = time.time()

        def recent_timestamp(basename):
            history = self.meta_manager.get_play_history(basename)
            return history[-1] if history else 0

        if self.sort_mode == "listens":
            data.sort(key=lambda item: (-item[1], item[0].lower()))
        elif self.sort_mode == "trending":
            data.sort(key=lambda item: (-self._get_trending_listen_count(item[0], now=now), item[0].lower()))
        elif self.sort_mode == "time":
            data.sort(key=lambda item: (-self._get_listen_time_seconds(item[0], item[1]), item[0].lower()))
        elif self.sort_mode == "recent":
            data.sort(key=lambda item: (-recent_timestamp(item[0]), item[0].lower()))
        else:
            data.sort(key=lambda item: item[0].lower())
        return data

    def _refresh_display_data(self, reset_scroll=False):
        self._ensure_tab_context(self.active_tab_id)
        command_mode = None
        query = ""
        source_basenames = self._get_tab_song_basenames(self.active_tab_id)
        use_command_context = False

        if self.mode == 'typing':
            line = self.input_text
            command_mode, prefix_len = self._parse_input_command(line)
            if command_mode == "add":
                query = line[prefix_len:].strip()
                source_basenames = self._get_all_song_basenames_sorted()
                use_command_context = True
            elif command_mode == "del":
                query = line[prefix_len:].strip()
                if self.active_tab_id == ALL_SONGS_TAB_ID:
                    source_basenames = []
                else:
                    source_basenames = self._get_tab_song_basenames(self.active_tab_id)
                use_command_context = True
            elif command_mode == "new":
                query = ""
                source_basenames = self._get_tab_song_basenames(self.active_tab_id)
                use_command_context = True
            elif command_mode == "drop":
                query = line[prefix_len:].strip()
                source_basenames = list(self.playlist_order)
                use_command_context = True
            elif command_mode == "sort":
                query = line[prefix_len:].strip()
                use_command_context = True
            elif command_mode == "autoplay":
                query = line[prefix_len:].strip()
                use_command_context = True
            else:
                query = line.strip()
                # Legacy multi-song flow for pool building:
                # treat the text after the last "||" as the active search token
                # while preserving the full input for Enter submission.
                if "||" in line:
                    query = line.split("||")[-1].strip()

        if use_command_context:
            tab_state = self.tab_context[self.active_tab_id]
            prefix_word = line[:prefix_len].strip().lower()
            tab_query = tab_state.get("query", "").strip().lower()
            if prefix_word and tab_query and prefix_word.startswith(tab_query):
                tab_state["query"] = ""
                tab_state["scroll_pos"] = 0
                tab_state["select_index"] = 0
            state = self.command_context
            query_changed = state.get("query", "") != query
            mode_changed = state.get("mode") != command_mode
            if reset_scroll or query_changed or mode_changed:
                state["scroll_pos"] = 0
                state["select_index"] = 0
            state["mode"] = command_mode
            state["query"] = query
        else:
            state = self.tab_context[self.active_tab_id]
            if self.mode == 'typing':
                query_changed = state.get("query", "") != query
                state["query"] = query
                if reset_scroll or query_changed:
                    state["scroll_pos"] = 0
                    state["select_index"] = 0
            else:
                query = state.get("query", "")
                if reset_scroll:
                    state["scroll_pos"] = 0
                    state["select_index"] = 0

        query_lower = query.lower()
        counts = {}
        if command_mode == "sort" and use_command_context:
            self.display_data = self._build_sort_display_data(query)
        else:
            if command_mode == "drop" and use_command_context:
                counts = {name: len(self.playlists.get(name, [])) for name in source_basenames}
            else:
                if self.sort_mode == "trending":
                    now = time.time()
                    counts = {
                        basename: self._get_trending_listen_count(basename, now=now)
                        for basename, _ in self.all_songs_data
                    }
                else:
                    counts = {basename: count for basename, count in self.all_songs_data}
            if query:
                self.display_data = [
                    (basename, counts.get(basename, 0), None)
                    for basename in source_basenames
                    if query_lower in basename.lower()
                ]
            else:
                self.display_data = [(basename, counts.get(basename, 0), None) for basename in source_basenames]

        self.add_sidebar_rows = []
        self.add_sidebar_query_hits = 0
        if command_mode == "add" and use_command_context and self.active_tab_id != ALL_SONGS_TAB_ID:
            active_playlist_basenames = self._get_tab_song_basenames(self.active_tab_id)
            active_playlist_set = set(active_playlist_basenames)
            self.display_data = [row for row in self.display_data if row[0] not in active_playlist_set]
            self.add_sidebar_rows = [
                (basename, counts.get(basename, 0), None)
                for basename in active_playlist_basenames
            ]
            if query_lower:
                self.add_sidebar_query_hits = sum(
                    1 for basename in active_playlist_basenames if query_lower in basename.lower()
                )

        max_items = self._playlist_max_items(
            command_mode=command_mode if use_command_context else None,
            query=query,
        )
        max_scroll = max(0, len(self.display_data) - max_items)
        state["scroll_pos"] = max(0, min(state.get("scroll_pos", 0), max_scroll))
        if self.display_data:
            state["select_index"] = max(0, min(state.get("select_index", 0), len(self.display_data) - 1))
        else:
            state["select_index"] = 0

        self.search_query = query
        self.scroll_pos = state["scroll_pos"]
        self.search_select_index = state["select_index"]
        self._active_view_state = state
        self._active_view_has_query = bool(query)
        self._active_command_mode = command_mode if use_command_context else None

    def _set_search_query(self, query, preferred_basename=None):
        self._ensure_tab_context(self.active_tab_id)
        state = self.tab_context[self.active_tab_id]
        state["query"] = query
        state["scroll_pos"] = 0
        state["select_index"] = 0
        if preferred_basename:
            source = self._get_tab_song_basenames(self.active_tab_id)
            try:
                state["select_index"] = source.index(preferred_basename)
            except ValueError:
                pass
        self._refresh_display_data(reset_scroll=False)

    def _truncate_line(self, text, width):
        if width <= 0:
            return ""
        if len(text) <= width:
            return text
        if width <= 3:
            return text[:width]
        return text[:width - 3] + "..."

    def _draw_scrollbar(self, win, top_index, total_items, view_height, start_row=1, x=None):
        if total_items <= view_height or view_height <= 0:
            return
        h, w = win.getmaxyx()
        if w < 3 or h < 3:
            return
        ch_on = SCROLLBAR_CHAR
        if isinstance(ch_on, str):
            try:
                enc = locale.getpreferredencoding(False) or ""
            except Exception:
                enc = ""
            if "UTF-8" not in enc.upper():
                ch_on = "#"
        bar_height = max(1, int(view_height * view_height / total_items))
        max_top = total_items - view_height
        if max_top <= 0:
            bar_top = 0
        else:
            bar_top = int((top_index / max_top) * (view_height - bar_height))
        if x is None:
            x = w - 2
        x = max(1, min(x, w - 2))
        for i in range(view_height):
            ch = ch_on if bar_top <= i < bar_top + bar_height else " "
            try:
                if isinstance(ch, int):
                    win.addch(start_row + i, x, ch)
                else:
                    win.addstr(start_row + i, x, ch)
            except curses.error:
                pass

    def _scroll_playlist(self, delta):
        max_items = self._playlist_max_items()
        max_scroll = max(0, len(self.display_data) - max_items)
        new_scroll = max(0, min(self.scroll_pos + delta, max_scroll))
        self.scroll_pos = new_scroll
        if self._active_view_state is not None:
            self._active_view_state["scroll_pos"] = new_scroll
        self._ensure_tab_context(self.active_tab_id)
        if self._active_command_mode in ("add", "del", "new", "drop", "sort", "autoplay"):
            self.command_context["scroll_pos"] = new_scroll
        else:
            self.tab_context[self.active_tab_id]["scroll_pos"] = new_scroll

    def _move_search_selection(self, delta):
        use_selection = self._active_view_has_query or self._active_command_mode in ("add", "del", "drop", "sort", "autoplay")
        if not use_selection:
            self._scroll_playlist(delta)
            return
        if not self.display_data:
            return
        self.search_select_index = max(0, min(self.search_select_index + delta, len(self.display_data) - 1))
        max_items = self._playlist_max_items()
        if self.search_select_index < self.scroll_pos:
            self.scroll_pos = self.search_select_index
        elif self.search_select_index >= self.scroll_pos + max_items:
            self.scroll_pos = self.search_select_index - max_items + 1
        if self._active_view_state is not None:
            self._active_view_state["scroll_pos"] = self.scroll_pos
            self._active_view_state["select_index"] = self.search_select_index
        self._ensure_tab_context(self.active_tab_id)
        if self._active_command_mode in ("add", "del", "new", "drop", "sort", "autoplay"):
            self.command_context["scroll_pos"] = self.scroll_pos
            self.command_context["select_index"] = self.search_select_index
        else:
            self.tab_context[self.active_tab_id]["scroll_pos"] = self.scroll_pos
            self.tab_context[self.active_tab_id]["select_index"] = self.search_select_index

    def _start_media_key_listener(self):
        try:
            from pynput import keyboard as pynput_keyboard
        except Exception:
            self._log("Global media keys disabled (install pynput to enable).")
            return

        def on_press(key):
            if key == pynput_keyboard.Key.media_play_pause:
                self.global_actions.put("play_pause")

        try:
            listener = pynput_keyboard.Listener(on_press=on_press)
            listener.daemon = True
            listener.start()
            self._media_listener = listener
            self._log("Global media keys enabled (Play/Pause).")
        except Exception as exc:
            self._log(f"Global media keys disabled: {exc}")

    def _stop_media_key_listener(self):
        if not self._media_listener:
            return
        try:
            self._media_listener.stop()
        except Exception:
            pass
        self._media_listener = None

    def _process_global_actions(self):
        did_action = False
        while True:
            try:
                action = self.global_actions.get_nowait()
            except queue.Empty:
                break
            if action == "play_pause":
                self._handle_global_play_pause()
                did_action = True
        return did_action

    def _handle_global_play_pause(self):
        if not self.playback_stopped:
            self._stop_playback()
        else:
            self._restart_from_active_tab()

    def _is_screen_locked_or_saver(self):
        if sys.platform != "darwin":
            return False

        locked = False
        try:
            if os.path.exists("/usr/sbin/ioreg"):
                output = subprocess.check_output(
                    ["/usr/sbin/ioreg", "-n", "Root", "-d1"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                for line in output.splitlines():
                    if "CGSSessionScreenIsLocked" in line:
                        locked = "Yes" in line or "1" in line
                        break
        except Exception:
            locked = False

        if locked:
            return True

        try:
            if os.path.exists("/usr/bin/pgrep"):
                subprocess.check_output(
                    ["/usr/bin/pgrep", "-x", "ScreenSaverEngine"],
                    stderr=subprocess.DEVNULL,
                )
                return True
        except subprocess.CalledProcessError:
            return False
        except Exception:
            return False

        return False

    def _maybe_stop_on_screen_lock(self):
        now = time.monotonic()
        if now - self._last_screen_state_check < self._screen_lock_poll_s:
            return False

        self._last_screen_state_check = now
        locked = self._is_screen_locked_or_saver()
        if locked and not self._screen_locked_or_saver:
            self._screen_locked_or_saver = True
            if not self.playback_stopped:
                self._stop_playback()
                return True
        else:
            self._screen_locked_or_saver = locked

        return False

    def _stop_playback(self):
        pygame.mixer.stop()
        self._reset_current_song_info()
        self.current_pool_basenames = []
        self.playback_queue = []
        self.playback_stopped = True
        self.autoplay_session = None
        self.mode = 'typing'
        self._log("Stopped playback. Ready for new input.")
        if self._ui_clear_input_line:
            self._ui_clear_input_line()
        if self._ui_draw_input_mode:
            self._ui_draw_input_mode()

    def _restart_from_active_tab(self):
        started = self._process_input("")
        if started and self._ui_clear_input_line:
            self._ui_clear_input_line()
        if self._ui_draw_input_mode:
            self._ui_draw_input_mode()

    def _handle_command_enter(self):
        line = self.input_text
        command_mode, prefix_len = self._parse_input_command(line)

        if command_mode == "add":
            if not self.display_data:
                return True
            selected = self.display_data[self.search_select_index][0]
            success, _ = self._add_song_to_active_playlist(selected)
            self._refresh_display_data(reset_scroll=False)
            return True

        if command_mode == "del":
            if not self.display_data:
                return True
            selected = self.display_data[self.search_select_index][0]
            success, _ = self._delete_song_from_active_playlist(selected)
            if success and self._ui_clear_input_line:
                self._ui_clear_input_line()
            self._refresh_display_data(reset_scroll=False)
            return True

        if command_mode == "new":
            playlist_name = line[prefix_len:].strip()
            if not playlist_name:
                return True
            success, _ = self._create_playlist(playlist_name)
            if self._ui_clear_input_line:
                self._ui_clear_input_line()
            self._refresh_display_data(reset_scroll=False)
            return True

        if command_mode == "drop":
            if not self.display_data:
                return True
            selected = self.display_data[self.search_select_index][0]
            success, _ = self._delete_playlist(selected)
            if self._ui_clear_input_line:
                self._ui_clear_input_line()
            self._refresh_display_data(reset_scroll=False)
            return True

        if command_mode == "sort":
            sort_arg = line[prefix_len:].strip()
            chosen_mode = None
            if sort_arg:
                chosen_mode = self._resolve_sort_mode(sort_arg)
            if not chosen_mode and self.display_data:
                selected = self.display_data[self.search_select_index]
                if len(selected) > 2:
                    chosen_mode = selected[2]
            if not chosen_mode:
                return True
            self.sort_mode = chosen_mode
            self._refresh_display_data(reset_scroll=True)
            if self._ui_clear_input_line:
                self._ui_clear_input_line()
            return True

        if command_mode == "autoplay":
            pool_basenames = self._get_tab_song_basenames(self.active_tab_id)
            if not pool_basenames:
                self._display_message("This tab has no songs to autoplay.", 4, 1.5)
                return True
            seed_query = line[prefix_len:].strip()
            seed_basename, seed_message = self._resolve_autoplay_seed(seed_query, pool_basenames)
            if seed_query and seed_basename is None:
                self._display_message(seed_message or "No matching songs.", 4, 2)
                return True
            if seed_message:
                self._log(seed_message)
            if self._ui_clear_input_line:
                self._ui_clear_input_line()
            self._start_autoplay(seed_basename=seed_basename)
            return True

        return False

    def _try_fill_search_selection(self):
        line = self.input_text
        command_mode, _ = self._parse_input_command(line)
        if command_mode:
            return False
        if not self.display_data:
            return False
        if not (self._active_view_has_query or "||" in line):
            return False

        selected_index = max(0, min(self.search_select_index, len(self.display_data) - 1))
        selected = self.display_data[selected_index][0]
        parts = [part.strip() for part in line.split("||")]
        if not parts:
            parts = [""]
        active_token = parts[-1].strip()

        # If the current token already matches the selected song, Enter should submit.
        if active_token and active_token.lower() == selected.lower():
            return False

        parts[-1] = selected
        rebuilt = " || ".join(parts).strip()
        if self._ui_set_input_text:
            self._ui_set_input_text(rebuilt)
        else:
            self.input_text = rebuilt
            self.input_cursor = len(self.input_text)
        return True

    def _handle_typing_enter(self):
        if self._handle_command_enter():
            return
        if self._try_fill_search_selection():
            return

        content = self.input_text.strip()
        if self._process_input(content) and self._ui_clear_input_line:
            self._ui_clear_input_line()

    def _move_input_cursor_left(self):
        anchor = self._get_input_anchor()
        if self.input_cursor > anchor:
            self.input_cursor -= 1

    def _move_input_cursor_right(self):
        if self.input_cursor < len(self.input_text):
            self.input_cursor += 1

    def _move_input_cursor_home(self):
        self.input_cursor = self._get_input_anchor()

    def _move_input_cursor_end(self):
        self.input_cursor = len(self.input_text)

    def _backspace_input(self):
        anchor = self._get_input_anchor()
        if self.input_cursor > anchor:
            self.input_text = self.input_text[:self.input_cursor - 1] + self.input_text[self.input_cursor:]
            self.input_cursor -= 1
        elif anchor and self.input_cursor == anchor:
            self.input_text = ""
            self.input_cursor = 0

        new_anchor = self._get_input_anchor()
        if self.input_cursor < new_anchor:
            self.input_cursor = new_anchor

    def _delete_input_char(self):
        anchor = self._get_input_anchor()
        if self.input_cursor < anchor:
            self.input_cursor = anchor
            return
        if self.input_cursor < len(self.input_text):
            self.input_text = self.input_text[:self.input_cursor] + self.input_text[self.input_cursor + 1:]

    def _insert_input_char(self, key):
        if key < 32 or key == 127:
            return
        try:
            ch = chr(key)
        except ValueError:
            return
        if not ch.isprintable():
            return

        max_x = self.input_box.getmaxyx()[1] - 1 if self.input_box else len(self.input_text) + 1
        if max_x >= 0 and len(self.input_text) >= max_x:
            return
        self.input_text = self.input_text[:self.input_cursor] + ch + self.input_text[self.input_cursor:]
        self.input_cursor += 1

    def _handle_typing_key(self, key):
        if key == curses.KEY_ENTER or key == 10 or key == 13:
            self._handle_typing_enter()
        elif key in (curses.KEY_LEFT,):
            self._move_input_cursor_left()
        elif key in (curses.KEY_RIGHT,):
            self._move_input_cursor_right()
        elif key == curses.KEY_HOME:
            self._move_input_cursor_home()
        elif key == curses.KEY_END:
            self._move_input_cursor_end()
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            self._backspace_input()
        elif key == curses.KEY_DC:
            self._delete_input_char()
        elif key == curses.KEY_UP:
            self._move_search_selection(-1)
        elif key == curses.KEY_DOWN:
            self._move_search_selection(1)
        else:
            self._insert_input_char(key)

        if self._ui_draw_input_mode:
            self._ui_draw_input_mode()

    def run(self):
        """Main entry point for the TUI application."""
        self._setup_curses()
        self._analyze_all_audio_with_cache()
        self.all_songs_data = self.meta_manager.get_all_sorted_data()
        self._sync_playlists_with_library()
        self._refresh_display_data(reset_scroll=True)

        self._draw_static_layout()
        self._last_screen_size = self.stdscr.getmaxyx()
        self._setup_input_box()

        self.mode = 'typing'
        if self._ui_draw_input_mode:
            self._ui_draw_input_mode()

        self._start_media_key_listener()

        # Main application loop
        try:
            self._redraw()
            while True:
                if self._handle_resize_if_needed():
                    self._last_progress_second = self._current_progress_second()
                    self._redraw()
                    continue

                needs_redraw = False

                if self._process_global_actions():
                    needs_redraw = True

                if self._maybe_stop_on_screen_lock():
                    needs_redraw = True

                if self._maybe_record_current_listen():
                    needs_redraw = True

                # Playback Logic (runs if no new key press)
                if not pygame.mixer.get_busy() and not self.playback_stopped:
                    if self._play_next_song():
                        needs_redraw = True

                progress_second = self._current_progress_second()
                if progress_second != self._last_progress_second:
                    self._last_progress_second = progress_second
                    needs_redraw = True

                # Input Handling
                key = self._read_key()

                if key == -1: # Timeout, no key pressed
                    if needs_redraw:
                        self._redraw()
                    continue
                if key == curses.KEY_RESIZE:
                    self._handle_resize_if_needed()
                    self._last_progress_second = self._current_progress_second()
                    self._redraw()
                    continue

                if key == curses.KEY_MOUSE:
                    if self.help_mode:
                        continue
                    if self._handle_mouse_event():
                        needs_redraw = True
                    if needs_redraw:
                        self._redraw()
                    continue

                if key == ord('\t') or key == 9:
                    if self.help_mode:
                        continue
                    command_mode, _ = self._parse_input_command(self.input_text)
                    if command_mode in ("add", "del", "new", "drop", "sort", "autoplay"):
                        self._flash_input_mode_highlight()
                        continue
                    if self.mode == 'typing':
                        self.mode = 'scroll'
                    else:
                        self.mode = 'typing'
                    if self._ui_draw_input_mode:
                        self._ui_draw_input_mode()
                    needs_redraw = True
                    if needs_redraw:
                        self._redraw()
                    continue

                if self.mode == 'typing':
                    self._handle_typing_key(key)

                elif self.mode == 'scroll':
                    if self.help_mode:
                        if key in (ord('h'), ord('H'), 27, ord('q'), ord('Q')):
                            self.help_mode = False
                            if self._ui_draw_input_mode:
                                self._ui_draw_input_mode()
                            needs_redraw = True
                        if needs_redraw:
                            self._redraw()
                        continue

                    if key == ord('/'):
                        self.mode = 'typing'
                        if self._ui_draw_input_mode:
                            self._ui_draw_input_mode()
                        self._handle_typing_key(key)
                        continue

                    if key in (ord('h'), ord('H')):
                        self.help_mode = True
                        try:
                            curses.curs_set(0)
                        except curses.error:
                            pass
                        self._redraw()
                        continue

                    if key in (ord('q'), ord('Q')):
                        if self._is_autoplay_active():
                            if self._quit_autoplay_mode():
                                needs_redraw = True
                            if needs_redraw:
                                self._redraw()
                            continue
                        break # Quit application

                    elif key == curses.KEY_DOWN:
                        self._scroll_playlist(1)

                    elif key == curses.KEY_UP:
                        self._scroll_playlist(-1)

                    elif key == curses.KEY_LEFT:
                        self._switch_tab(-1)

                    elif key == curses.KEY_RIGHT:
                        self._switch_tab(1)

                    elif key == ord('s'): # Stop and prepare for new input
                        self._stop_playback()

                    elif key == ord('n'): # Next
                        self.playback_stopped = False
                        pygame.mixer.stop()
                        self._log("Skipping to next track.")
                self._redraw()
        finally:
            self._stop_media_key_listener()

    def _process_input(self, content):
        """Handles user input from the textbox to define or modify a song pool."""
        self.autoplay_session = None
        is_playing = pygame.mixer.get_busy() or self.current_song_start_time > 0
        known_basenames = self._get_tab_song_basenames(self.active_tab_id)

        # Case 1: Add to an existing playlist
        if is_playing and content.startswith('||'):
            potential_songs = [s.strip() for s in content.split('||') if s.strip()]
            valid_new_songs = []
            auto_corrected_songs = []
            not_found = []

            for song_name in potential_songs:
                # Check exact case-insensitive match first
                for known_basename in known_basenames:
                    if song_name.lower() == known_basename.lower():
                        valid_new_songs.append(known_basename)
                        break
                else:
                    # Try Levenshtein distance auto-correction
                    closest_match, distance = self._find_closest_match(song_name, known_basenames, max_distance=AUTO_CORRECT_DISTANCE)
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
                pool_to_set = self._get_tab_song_basenames(self.active_tab_id)
                if not pool_to_set:
                    self._display_message("This tab has no songs to play.", 4, 1.5)
                    return False
                tab_name = self._get_tab_label(self.active_tab_id)
                self._display_message(f"Starting '{tab_name}' ({len(pool_to_set)} songs).", 2, 1.5)
            else:
                potential_songs = [s.strip() for s in content.split('||')]
                valid_pool = []
                auto_corrected_songs = []
                not_found = []

                for song_name in potential_songs:
                    # Check exact case-insensitive match first
                    matched = False
                    for known_basename in known_basenames:
                        if song_name.lower() == known_basename.lower():
                            valid_pool.append(known_basename)
                            matched = True
                            break
                    if not matched:
                        # Try Levenshtein distance auto-correction
                        closest_match, distance = self._find_closest_match(
                            song_name,
                            known_basenames,
                            max_distance=AUTO_CORRECT_DISTANCE,
                        )
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
            self._set_search_query("")
            return True

    def _setup_curses(self):
        curses.curs_set(0)
        self.stdscr.keypad(True)
        self.stdscr.nodelay(1)
        self._getch_delay = 100
        self.stdscr.timeout(self._getch_delay)
        try:
            curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
            curses.mouseinterval(0)
        except curses.error:
            pass
        curses.start_color()
        curses.use_default_colors()
        # Prefer a brighter sky blue when the terminal supports custom colors.
        sky_blue_id = None
        if curses.COLORS >= 256:
            sky_blue_id = 117
        elif curses.COLORS >= 16:
            sky_blue_id = 12
        if sky_blue_id is not None and curses.can_change_color():
            try:
                curses.init_color(sky_blue_id, 350, 700, 1000)
            except curses.error:
                sky_blue_id = None
        if sky_blue_id is None:
            sky_blue_id = curses.COLOR_CYAN
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_RED, -1)
        curses.init_pair(5, curses.COLOR_BLUE, -1)
        sort_pair_id = 16 if curses.COLOR_PAIRS > 16 else 1
        try:
            curses.init_pair(sort_pair_id, sky_blue_id, -1)
        except curses.error:
            sort_pair_id = 1
            try:
                curses.init_pair(sort_pair_id, sky_blue_id, -1)
            except curses.error:
                sort_pair_id = 1
        self.sort_accent_pair_id = sort_pair_id
        self.sort_accent_attr = curses.color_pair(sort_pair_id)
        # Highlight: prefer dark gray background when available, otherwise fall back.
        highlight_attr = None
        if curses.COLORS >= 16:
            gray_id = 8
            try:
                if curses.can_change_color() and gray_id < curses.COLORS:
                    curses.init_color(gray_id, 300, 300, 300)
                curses.init_pair(6, curses.COLOR_WHITE, gray_id)
                highlight_attr = curses.color_pair(6)
            except curses.error:
                highlight_attr = None
        if highlight_attr is None:
            curses.init_pair(6, curses.COLOR_BLACK, curses.COLOR_WHITE)
            highlight_attr = curses.color_pair(6) | curses.A_DIM
        self.search_highlight_attr = highlight_attr
        add_highlight = None
        del_highlight = None
        try:
            curses.init_pair(7, curses.COLOR_BLACK, curses.COLOR_GREEN)
            curses.init_pair(8, curses.COLOR_BLACK, curses.COLOR_RED)
            add_highlight = curses.color_pair(7) | curses.A_BOLD
            del_highlight = curses.color_pair(8) | curses.A_BOLD
        except curses.error:
            add_highlight = None
            del_highlight = None
        if add_highlight is None:
            add_highlight = (curses.color_pair(2) | curses.A_REVERSE | curses.A_BOLD)
        if del_highlight is None:
            del_highlight = (curses.color_pair(4) | curses.A_REVERSE | curses.A_BOLD)
        self.add_search_highlight_attr = add_highlight
        self.del_search_highlight_attr = del_highlight
        playlist_attr = None
        playlist_active_attr = None
        try:
            if curses.COLORS >= 16:
                gray_id = 8
                if curses.can_change_color() and gray_id < curses.COLORS:
                    curses.init_color(gray_id, 300, 300, 300)
                curses.init_pair(9, gray_id, -1)
                curses.init_pair(10, curses.COLOR_GREEN, gray_id)
                playlist_attr = curses.color_pair(9) | curses.A_DIM
                playlist_active_attr = curses.color_pair(10) | curses.A_BOLD
        except curses.error:
            playlist_attr = None
            playlist_active_attr = None
        if playlist_attr is None:
            playlist_attr = curses.A_DIM
        if playlist_active_attr is None:
            playlist_active_attr = curses.color_pair(2) | curses.A_REVERSE | curses.A_BOLD
        self.playlist_name_attr = playlist_attr
        self.playlist_name_active_attr = playlist_active_attr
        header_search_attr = None
        header_add_attr = None
        header_del_attr = None
        header_new_attr = None
        header_drop_attr = None
        header_sort_attr = None
        header_inactive_attr = None
        try:
            if curses.COLORS >= 16:
                gray_id = 8
                if curses.can_change_color() and gray_id < curses.COLORS:
                    curses.init_color(gray_id, 300, 300, 300)
                curses.init_pair(11, curses.COLOR_WHITE, gray_id)
                curses.init_pair(12, curses.COLOR_GREEN, gray_id)
                curses.init_pair(13, curses.COLOR_RED, gray_id)
                curses.init_pair(14, curses.COLOR_YELLOW, gray_id)
                curses.init_pair(15, sky_blue_id, gray_id)
                header_search_attr = curses.color_pair(11) | curses.A_BOLD
                header_add_attr = curses.color_pair(12) | curses.A_BOLD
                header_del_attr = curses.color_pair(13) | curses.A_BOLD
                header_new_attr = curses.color_pair(14) | curses.A_BOLD
                header_drop_attr = header_del_attr
                header_sort_attr = curses.color_pair(15) | curses.A_BOLD
                header_inactive_attr = curses.color_pair(11) | curses.A_DIM
        except curses.error:
            header_search_attr = None
            header_add_attr = None
            header_del_attr = None
            header_new_attr = None
            header_drop_attr = None
            header_sort_attr = None
            header_inactive_attr = None
        if header_search_attr is None:
            header_search_attr = curses.A_DIM | curses.A_BOLD
        if header_add_attr is None:
            header_add_attr = curses.color_pair(2) | curses.A_BOLD
        if header_del_attr is None:
            header_del_attr = curses.color_pair(4) | curses.A_BOLD
        if header_new_attr is None:
            header_new_attr = curses.color_pair(3) | curses.A_BOLD
        if header_drop_attr is None:
            header_drop_attr = curses.color_pair(4) | curses.A_BOLD
        if header_sort_attr is None:
            header_sort_attr = self.sort_accent_attr | curses.A_BOLD
        if header_inactive_attr is None:
            header_inactive_attr = curses.A_DIM
        self.input_header_search_attr = header_search_attr
        self.input_header_add_attr = header_add_attr
        self.input_header_del_attr = header_del_attr
        self.input_header_new_attr = header_new_attr
        self.input_header_drop_attr = header_drop_attr
        self.input_header_sort_attr = header_sort_attr
        self.input_header_inactive_attr = header_inactive_attr
        self._create_windows()

    def _read_key(self):
        key = self.stdscr.getch()
        if key != 27:
            return key
        if self.help_mode:
            return 27

        # Normalize arrow key escape sequences that sometimes bypass keypad mode.
        self.stdscr.nodelay(False)
        self.stdscr.timeout(10)
        try:
            seq = []
            for _ in range(6):
                ch = self.stdscr.getch()
                if ch == -1:
                    break
                seq.append(ch)

            if not seq:
                return 27

            if seq[0] in (ord('['), ord('O')):
                last = seq[-1]
                mapping = {
                    ord('A'): curses.KEY_UP,
                    ord('B'): curses.KEY_DOWN,
                    ord('C'): curses.KEY_RIGHT,
                    ord('D'): curses.KEY_LEFT,
                }
                if last in mapping:
                    return mapping[last]

            for ch in reversed(seq):
                self.stdscr.ungetch(ch)
            return 27
        finally:
            self.stdscr.nodelay(True)
            self.stdscr.timeout(self._getch_delay)

    def _draw_box(self, win, active=False, active_attr=None):
        if active:
            attr = active_attr if active_attr is not None else curses.color_pair(2)
            win.attron(attr)
        win.box()
        if active:
            win.attroff(attr)

    def _get_input_box_highlight_attr(self):
        mode = self._get_pending_input_command()
        if mode == "del" or mode == "drop":
            return curses.color_pair(4)
        if mode == "new":
            return curses.color_pair(3)
        if mode == "sort" or mode == "autoplay":
            return self.sort_accent_attr
        if self._is_autoplay_active():
            return self.sort_accent_attr
        return curses.color_pair(2)

    def _create_windows(self):
        h, w = self.stdscr.getmaxyx()
        input_h = 3
        content_h = max(3, h - input_h - 1)
        left_w = w

        info_h = 5 if content_h >= 7 else max(3, content_h // 2)
        playlist_h = max(3, content_h - info_h)

        self.left_w = left_w

        self.win_header = curses.newwin(1, w, 0, 0)
        self.win_info = curses.newwin(info_h, left_w, 1, 0)
        self.win_playlist = curses.newwin(playlist_h, left_w, 1 + info_h, 0)
        self.win_input = curses.newwin(input_h, w, h - input_h, 0)

    def _render_input_box(self):
        if not self.input_box:
            return
        self.input_box.erase()
        max_x = self.input_box.getmaxyx()[1] - 1
        if max_x < 0:
            return

        clipped = self.input_text[:max_x] if max_x >= 0 else ""
        if clipped != self.input_text:
            self.input_text = clipped

        command_mode, prefix_len = self._parse_input_command(clipped)
        prefix_attr = curses.A_NORMAL
        if command_mode == "add":
            prefix_attr = curses.color_pair(2) | curses.A_BOLD
        elif command_mode == "del":
            prefix_attr = curses.color_pair(4) | curses.A_BOLD
        elif command_mode == "new":
            prefix_attr = curses.color_pair(3) | curses.A_BOLD
        elif command_mode == "drop":
            prefix_attr = curses.color_pair(4) | curses.A_BOLD
        elif command_mode == "sort":
            prefix_attr = self.sort_accent_attr | curses.A_BOLD
        elif command_mode == "autoplay":
            prefix_attr = self.sort_accent_attr | curses.A_BOLD

        try:
            if command_mode in ("add", "del", "new", "drop", "sort", "autoplay") and prefix_len <= len(clipped):
                self.input_box.addstr(0, 0, clipped[:prefix_len], prefix_attr)
                if len(clipped) > prefix_len:
                    self.input_box.addstr(0, prefix_len, clipped[prefix_len:])
            elif clipped:
                self.input_box.addstr(0, 0, clipped)
        except curses.error:
            pass

        anchor = self._get_input_anchor(clipped)
        self.input_cursor = max(anchor, min(self.input_cursor, len(clipped)))
        try:
            self.input_box.move(0, min(self.input_cursor, max_x))
        except curses.error:
            pass

    def _setup_input_box(self, preserve_text=None, preserve_cursor=None):
        h, w = self.stdscr.getmaxyx()
        box_width = max(1, w - 4)
        box_x = max(0, min(2, w - box_width))
        input_box = curses.newwin(1, box_width, h - 2, box_x)
        input_box.keypad(True)

        def get_input_line():
            return self.input_text

        def clear_input_line():
            self.input_text = ""
            self.input_cursor = 0
            self._render_input_box()
            input_box.refresh()

        def set_input_text(text):
            max_x = input_box.getmaxyx()[1] - 1
            clipped = text[:max_x] if max_x >= 0 else ""
            self.input_text = clipped
            self.input_cursor = len(clipped)
            anchor = self._get_input_anchor()
            if self.input_cursor < anchor:
                self.input_cursor = anchor
            self._render_input_box()
            input_box.refresh()

        def draw_input_mode():
            """Clears and redraws the input window based on the current mode."""
            self.win_input.clear()
            self._draw_box(
                self.win_input,
                active=(self.mode == 'typing'),
                active_attr=self._get_input_box_highlight_attr(),
            )
            self._draw_input_header()
            if self.mode == 'typing':
                curses.curs_set(1)
            else: # scroll
                curses.curs_set(0)
            self.win_input.refresh() # Use refresh on this small window for immediate effect
            self._render_input_box()
            input_box.refresh()

        self.input_box = input_box
        self._ui_get_input_line = get_input_line
        self._ui_clear_input_line = clear_input_line
        self._ui_set_input_text = set_input_text
        self._ui_draw_input_mode = draw_input_mode

        if preserve_text is not None:
            set_input_text(preserve_text)
            if preserve_cursor is not None:
                max_x = input_box.getmaxyx()[1] - 1
                self.input_cursor = max(0, min(preserve_cursor, max_x, len(self.input_text)))
                anchor = self._get_input_anchor()
                if self.input_cursor < anchor:
                    self.input_cursor = anchor
                self._render_input_box()
                input_box.refresh()

    def _handle_resize_if_needed(self):
        new_size = self.stdscr.getmaxyx()
        if self._last_screen_size == new_size:
            return False
        self._last_screen_size = new_size
        try:
            curses.resizeterm(new_size[0], new_size[1])
        except curses.error:
            pass

        preserve_text = ""
        preserve_cursor = 0
        if self._ui_get_input_line and self.input_box:
            preserve_text = self._ui_get_input_line().split('\n', 1)[0]
            try:
                _, preserve_cursor = self.input_box.getyx()
            except curses.error:
                preserve_cursor = 0

        self.stdscr.erase()
        self._create_windows()
        self._draw_static_layout()
        self._setup_input_box(preserve_text=preserve_text, preserve_cursor=preserve_cursor)
        self._refresh_display_data(reset_scroll=False)
        if self._ui_draw_input_mode:
            self._ui_draw_input_mode()
        return True

    def _add_win_info_separator(self):
        h, w = self.win_info.getmaxyx()
        if w > 17 and h > 4:
            self.win_info.addch(0, 16, curses.ACS_TTEE)
            self.win_info.addch(4, 16, curses.ACS_BTEE)

    def _draw_static_layout(self):
        h, w = self.stdscr.getmaxyx()

        self.win_info.box()

        self._add_win_info_separator()

        self.win_playlist.box()
        self.win_input.box()

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
            if not self.playback_stopped:
                self.playback_stopped = True # Stop trying to play if pool is empty
                return True
            return False

        if self._is_autoplay_active():
            basename = self._choose_autoplay_song()
            if not basename:
                if not self.playback_stopped:
                    self.playback_stopped = True
                    return True
                return False
        else:
            if not self.playback_queue:
                self.playback_queue = self.current_pool_basenames[:]
                random.shuffle(self.playback_queue)

            if not self.playback_queue:
                if not self.playback_stopped:
                    self.playback_stopped = True
                    return True
                return False

            basename = self.playback_queue.pop()

        full_path = os.path.join(self.args.folder, basename + ".mp3")
        if not os.path.exists(full_path):
            self.current_song_info['basename'] = f"'{basename}' NOT FOUND!"
            return True

        analysis = self.audio_analysis.get(basename, {'loudness': 'N/A', 'scale': 0.5})
        base_scale = analysis['scale']
        base_scale = max(0.0, min(1.0, base_scale)) if isinstance(base_scale, float) else base_scale

        listen_rank_data = self._get_listen_rank(basename)
        time_rank_data = self._get_listen_time_rank(basename)
        listen_count = listen_rank_data[2] if listen_rank_data else 0
        rank_info = self._format_rank_info(listen_rank_data, time_rank_data)

        try:
            sound = pygame.mixer.Sound(full_path)
            self.current_song_duration = sound.get_length()
            analysis_fact = self._build_analysis_fact(
                basename,
                listen_count,
                song_duration_sec=self.current_song_duration,
            )

            self.current_song_info = {
                "basename": basename,
                "loudness": f"{analysis['loudness']:.2f}" if isinstance(analysis['loudness'], float) else 'N/A',
                "base_scale": base_scale if isinstance(base_scale, float) else None,
                "scale": f"{(base_scale * self.volume_multiplier):.3f}" if isinstance(base_scale, float) else "N/A",
                "rank_info": rank_info,
                "analysis": analysis_fact
            }

            if isinstance(base_scale, float):
                sound.set_volume(base_scale)
            self.current_channel = sound.play()
            self._apply_current_volume()
            self.current_song_start_time = time.time()
            self.current_song_basename = basename
            self.current_song_listen_recorded = False
            self.playback_stopped = False
        except pygame.error as e:
            self.current_song_duration = 0
            self.current_song_start_time = 0
            self.current_song_basename = None
            self.current_song_listen_recorded = False
            self.current_channel = None
            analysis_fact = self._build_analysis_fact(basename, listen_count)
            self.current_song_info = {
                "basename": basename,
                "loudness": f"{analysis['loudness']:.2f}" if isinstance(analysis['loudness'], float) else 'N/A',
                "base_scale": base_scale if isinstance(base_scale, float) else None,
                "scale": f"{(base_scale * self.volume_multiplier):.3f}" if isinstance(base_scale, float) else "N/A",
                "rank_info": rank_info,
                "analysis": analysis_fact
            }
            self.current_song_info['basename'] = f"ERROR PLAYING '{basename}'"
        return True

    def _maybe_record_current_listen(self):
        if (
            self.current_song_listen_recorded
            or not self.current_song_basename
            or self.current_song_start_time <= 0
            or self.current_song_duration <= 0
        ):
            return False

        elapsed_time = time.time() - self.current_song_start_time
        if elapsed_time < (self.current_song_duration / 2):
            return False

        self.meta_manager.increment_count(self.current_song_basename)
        self.current_song_listen_recorded = True
        self.all_songs_data = self.meta_manager.get_all_sorted_data()
        self._refresh_display_data(reset_scroll=False)

        listen_rank_data = self._get_listen_rank(self.current_song_basename)
        time_rank_data = self._get_listen_time_rank(self.current_song_basename)
        self.current_song_info["rank_info"] = self._format_rank_info(listen_rank_data, time_rank_data)
        return True

    def _update_all_windows(self):
        self._update_header_win()
        self._update_info_win()
        self._update_playlist_win()

        self.stdscr.noutrefresh()
        self.win_header.noutrefresh()
        self.win_info.noutrefresh()
        self.win_playlist.noutrefresh()
        self.win_input.noutrefresh()

    def _dim_window(self, win):
        if not win:
            return
        h, w = win.getmaxyx()
        if h <= 0 or w <= 0:
            return
        for y in range(h):
            try:
                win.chgat(y, 0, w, curses.A_DIM)
            except curses.error:
                pass
        win.noutrefresh()

    def _dim_active_windows(self):
        self._dim_window(self.win_header)
        self._dim_window(self.win_info)
        self._dim_window(self.win_playlist)
        self._dim_window(self.win_input)
        if self.input_box:
            try:
                _, w = self.input_box.getmaxyx()
                if w > 0:
                    self.input_box.chgat(0, 0, w, curses.A_DIM)
            except curses.error:
                pass
            self.input_box.noutrefresh()

    def _redraw(self):
        if self.help_mode:
            self._refresh_display_data(reset_scroll=False)
            self._update_all_windows()
            if self.input_box:
                self.input_box.noutrefresh()
            self._dim_active_windows()
            self._update_help_win()
            curses.doupdate()
            return
        self._refresh_display_data(reset_scroll=False)
        self._update_all_windows()
        if self.input_box:
            self.input_box.noutrefresh()
        curses.doupdate()

    def _format_time(self, seconds):
        if seconds is None or seconds < 0:
            seconds = 0
        minutes, sec = divmod(int(seconds), 60)
        return f"{minutes:02d}:{sec:02d}"

    def _get_song_duration_sec(self, basename):
        analysis = self.audio_analysis.get(basename)
        if not isinstance(analysis, dict):
            return None
        duration = analysis.get("duration")
        if isinstance(duration, (int, float)) and duration > 0:
            return duration
        return None

    def _get_listen_time_seconds(self, basename, count):
        if not isinstance(count, (int, float)) or count <= 0:
            return 0.0
        duration = self._get_song_duration_sec(basename)
        if duration is None:
            return 0.0
        return max(0.0, float(count) * float(duration))

    def _get_longest_continuous_listen_seconds(self, basename, song_duration_sec):
        if not isinstance(song_duration_sec, (int, float)) or song_duration_sec <= 0:
            return 0.0

        history = self.meta_manager.get_play_history(basename)
        if not history:
            return 0.0
        history = sorted(ts for ts in history if isinstance(ts, (int, float)))
        if not history:
            return 0.0

        half = float(song_duration_sec) / 2.0
        tolerance = min(120.0, max(15.0, float(song_duration_sec) * 0.15))

        block_start = history[0] - half
        block_end = history[0] + half
        max_seconds = float(song_duration_sec)

        for ts in history[1:]:
            start = ts - half
            end = ts + half
            if start <= block_end + tolerance:
                if end > block_end:
                    block_end = end
            else:
                max_seconds = max(max_seconds, block_end - block_start)
                block_start = start
                block_end = end

        max_seconds = max(max_seconds, block_end - block_start)
        return max(0.0, max_seconds)

    def _format_listen_time_hours(self, seconds):
        hours = max(0.0, float(seconds)) / 3600.0
        return f"{hours:.1f}h"

    def _get_listen_rank(self, basename):
        if not self.all_songs_data:
            return None
        sorted_data = sorted(self.all_songs_data, key=lambda item: (-item[1], item[0].lower()))
        for index, (name, count) in enumerate(sorted_data, start=1):
            if name == basename:
                return index, len(sorted_data), count
        return None

    def _get_listen_time_rank(self, basename):
        if not self.all_songs_data:
            return None
        entries = []
        for name, count in self.all_songs_data:
            listen_time = self._get_listen_time_seconds(name, count)
            entries.append((name, listen_time))
        if not entries:
            return None
        entries.sort(key=lambda item: (-item[1], item[0].lower()))
        for index, (name, listen_time) in enumerate(entries, start=1):
            if name == basename:
                return index, len(entries), listen_time
        return None

    def _build_playlist_stats_text(self):
        counts = {basename: count for basename, count in self.all_songs_data}
        if self.active_tab_id == ALL_SONGS_TAB_ID:
            basenames = list(counts.keys())
        else:
            basenames = self.playlists.get(self.active_tab_id, [])
            if not isinstance(basenames, list):
                basenames = []

        total_listens = sum(counts.get(name, 0) for name in basenames)
        total_seconds = 0.0
        for name in basenames:
            count = counts.get(name, 0)
            if count:
                total_seconds += self._get_listen_time_seconds(name, count)

        listen_label = "listen" if total_listens == 1 else "listens"
        time_label = self._format_listen_time_hours(total_seconds)
        return f"[ Playlist Stats: {total_listens:,} {listen_label} • {time_label} ]"

    def _get_loudness_rank(self, basename):
        entries = []
        for name, data in self.audio_analysis.items():
            if not isinstance(data, dict):
                continue
            loudness = data.get("loudness")
            if isinstance(loudness, (int, float)):
                entries.append((name, loudness))
        if not entries:
            return None
        entries.sort(key=lambda item: item[1], reverse=True)
        for index, (name, loudness) in enumerate(entries, start=1):
            if name == basename:
                return index, len(entries), loudness
        return None

    def _format_rank_info(self, listen_rank_data, time_rank_data):
        parts = []
        if listen_rank_data:
            rank, total, count = listen_rank_data
            if total > 0:
                parts.append(f"Listens #{rank}/{total} ({count})")
            else:
                parts.append(f"Listens #{rank} ({count})")
        if time_rank_data:
            rank, total, seconds = time_rank_data
            listen_time = self._format_listen_time_hours(seconds)
            if total > 0:
                parts.append(f"Time #{rank}/{total} ({listen_time})")
            else:
                parts.append(f"Time #{rank} ({listen_time})")
        if not parts:
            return "N/A"
        return " | ".join(parts)

    def _format_time_ago(self, seconds):
        if seconds < 60:
            return "just now"
        minutes = int(seconds // 60)
        if minutes < 60:
            unit = "minute" if minutes == 1 else "minutes"
            return f"{minutes} {unit} ago"
        hours = int(seconds // 3600)
        if hours < 48:
            unit = "hour" if hours == 1 else "hours"
            return f"{hours} {unit} ago"
        days = int(seconds // 86400)
        if days < 60:
            unit = "day" if days == 1 else "days"
            return f"{days} {unit} ago"
        months = int(round(days / 30))
        unit = "month" if months == 1 else "months"
        return f"about {months} {unit} ago"

    def _build_analysis_fact(self, basename, count, song_duration_sec=None):
        now = time.time()
        facts = []
        solve_time_sec = 3.06

        history = self.meta_manager.get_play_history(basename)
        if history:
            history = sorted(history)

        if len(history) >= 2:
            prev_ts = history[-2]
            gap = now - prev_ts
            if gap >= 3600:
                facts.append(f"Last played {self._format_time_ago(gap)}")
            else:
                facts.append("Played earlier today")
        elif count <= 1:
            facts.append("First time playing this")

        if history:
            for window in (7, 30, 90):
                window_sec = window * 86400
                recent = sum(1 for ts in history if ts >= now - window_sec)
                if recent > 0:
                    facts.append(f"Listens in the last {window} days: {recent}")

        if self.all_songs_data:
            total_listens = sum(item[1] for item in self.all_songs_data)
            avg_listens = total_listens / len(self.all_songs_data) if self.all_songs_data else 0
            if avg_listens > 0:
                ratio = count / avg_listens
                if ratio >= 1.5:
                    facts.append(f"{ratio:.1f}x the library average listens")
                elif 0 < ratio <= 0.6:
                    facts.append(f"Below average listens ({ratio:.1f}x)")

        if (
            song_duration_sec is not None
            and isinstance(song_duration_sec, (int, float))
            and song_duration_sec > 0
        ):
            longest_continuous = self._get_longest_continuous_listen_seconds(
                basename,
                song_duration_sec,
            )
            longest_hours = longest_continuous / 3600.0
            if longest_hours > 1.0:
                facts.append(f"You once listened to this for {longest_hours:.1f} hours continuously")
            if solve_time_sec > 0:
                solves = int(song_duration_sec // solve_time_sec)
                if solves > 0:
                    solve_label = "time" if solves == 1 else "times"
                    facts.append(
                        f"Yiheng Wang can solve a 3x3 {solves} {solve_label} before this MP3 ends"
                    )

        history = self.meta_manager.get_play_history(basename)
        first_played = history[0] if history else None
        if isinstance(first_played, (int, float)) and count > 1:
            age_days = int((now - first_played) // 86400)
            if age_days >= 1:
                facts.append(f"First played {age_days} days ago")

        loud_rank = self._get_loudness_rank(basename)
        if loud_rank:
            position, total, _ = loud_rank
            if position == 1:
                facts.append("Loudest track in the library")
            elif position == total:
                facts.append("Quietest track in the library")
            else:
                facts.append(f"Loudness rank #{position} of {total}")

        if not facts:
            listen_label = "listen" if count == 1 else "listens"
            return f"{count} total {listen_label}"
        return random.choice(facts)

    def _current_progress_second(self):
        is_playing = (
            self.current_song_start_time > 0
            and self.current_song_duration > 0
            and pygame.mixer.get_busy()
        )
        if not is_playing:
            return None
        elapsed_time = time.time() - self.current_song_start_time
        elapsed_time = max(0, min(elapsed_time, self.current_song_duration))
        return int(elapsed_time)

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
            title = "❮❮ MPlayer3 ❯❯"
            self.win_header.addstr(0, (w - len(title)) // 2, title, curses.A_BOLD)

    def _update_info_win(self):
        self.win_info.clear()
        self.win_info.box()
        self._add_win_info_separator()
        h, w = self.win_info.getmaxyx()

        song_width = max(1, w - 20)
        song_name = textwrap.shorten(self.current_song_info['basename'], width=song_width, placeholder="...")

        try:
            self.win_info.addstr(1, 3, "Now Playing  │ ")
            self.win_info.addstr(song_name, curses.color_pair(2) | curses.A_BOLD)
        except curses.error:
            pass

        # --- Insight Info ---
        rank_info = self.current_song_info.get("rank_info", "N/A")
        analysis_info = self.current_song_info.get("analysis", "N/A")
        rank_line = f"{'Rank Info':<13}│ {rank_info}"
        analysis_line = f"{'Analysis':<13}│ {analysis_info}"
        try:
            self.win_info.addstr(2, 3, self._truncate_line(rank_line, w - 6))
        except curses.error:
            pass
        try:
            self.win_info.addstr(3, 3, self._truncate_line(analysis_line, w - 6))
        except curses.error:
            pass

    def _update_help_win(self):
        h, w = self.stdscr.getmaxyx()
        if h < 6 or w < 20:
            return

        try:
            title_attr = self.sort_accent_attr | curses.A_BOLD
        except curses.error:
            title_attr = curses.A_BOLD

        sections = [
            ("Modes", [
                "Tab toggles between Typing mode (input + search) and Playlist mode (scroll).",
                "Typing mode shows the input box. Playlist mode lets you navigate tabs and lists.",
            ]),
            ("Playlist Mode (scroll)", [
                "↑/↓: scroll songs   ←/→: switch tabs   Tab: switch mode",
                "h: help   q: quit or exit autoplay   s: stop playback   n: next track",
                "Mouse wheel also scrolls.",
            ]),
            ("Typing Mode (input/search)", [
                "Type to filter the current tab. ↑/↓ selects results. Enter plays the selected pool.",
                "Use commands with a leading / to manage playlists or sorting.",
            ]),
            ("Commands", [
                "/add   Add selected song to the current playlist tab.",
                "/del   Remove selected song from the current playlist tab.",
                "/new   Create a new playlist (type the name after /new).",
                "/drop  Delete a playlist (select a playlist name).",
                "/sort  Sort songs globally: name (A-Z), listens, trending, time, or recent.",
                "/autoplay  Start Markov autoplay for the current tab, with an optional seed song.",
            ]),
            ("Pool Building", [
                "Enter starts playback from the current tab (or the filtered search list).",
                "Use ' || ' between names to start a custom pool.",
                "While playing, typing '|| name1 || name2' adds to the active pool.",
            ]),
            ("Close Help", [
                "Press h or Esc to return to Playlist mode.",
                "In autoplay mode, q exits autoplay; pressing q again quits the app.",
            ]),
        ]

        popup_w = min(86, w - 4)
        popup_w = max(20, min(popup_w, w - 2))
        content_w = max(1, popup_w - 4)

        content_lines = []
        for heading, lines in sections:
            content_lines.append((heading, title_attr))
            for line in lines:
                wrapped = textwrap.wrap(
                    line,
                    width=content_w,
                    break_long_words=False,
                    replace_whitespace=False,
                )
                if not wrapped:
                    wrapped = [""]
                for segment in wrapped:
                    content_lines.append((segment, curses.A_NORMAL))
            content_lines.append(("", curses.A_NORMAL))
        if content_lines:
            content_lines.pop()

        popup_h = min(h - 4, len(content_lines) + 2)
        popup_h = max(6, min(popup_h, h - 2))

        start_y = max(0, (h - popup_h) // 2)
        start_x = max(0, (w - popup_w) // 2)
        help_win = curses.newwin(popup_h, popup_w, start_y, start_x)
        help_win.bkgd(' ', curses.A_NORMAL)
        help_win.box()

        title = " Help "
        try:
            help_win.addstr(0, max(2, (popup_w - len(title)) // 2), title, title_attr)
        except curses.error:
            pass

        max_lines = max(0, popup_h - 2)
        y = 1
        for line, attr in content_lines[:max_lines]:
            try:
                help_win.addstr(y, 2, self._truncate_line(line, content_w), attr)
            except curses.error:
                pass
            y += 1

        help_win.noutrefresh()

    def _update_info_win_message(self, message):
         self.win_info.clear()
         self.win_info.box()
         self.win_info.addstr(0, 2, " Status ")
         h, w = self.win_info.getmaxyx()
         self.win_info.addstr(h // 2, (w - len(message)) // 2, message, curses.color_pair(3))
         self.win_info.refresh()

    def _display_message(self, message, color_pair_id, duration_sec):
        self._log(message)
        self.win_input.clear()
        self._draw_box(
            self.win_input,
            active=(self.mode == 'typing'),
            active_attr=self._get_input_box_highlight_attr(),
        )
        h, w = self.win_input.getmaxyx()
        self.win_input.addstr(h // 2, (w - len(message)) // 2, message, curses.color_pair(color_pair_id))
        self.win_input.refresh()
        time.sleep(duration_sec)
        if self._ui_draw_input_mode:
            self._ui_draw_input_mode()
        else:
            self.win_input.clear()
            self._draw_box(
                self.win_input,
                active=(self.mode == 'typing'),
                active_attr=self._get_input_box_highlight_attr(),
            )
            self.win_input.refresh()

    def _flash_input_mode_highlight(self, duration_sec=0.10):
        if not self._ui_draw_input_mode:
            return
        original_mode = self.mode
        if original_mode == 'typing':
            self.mode = 'scroll'
            self._ui_draw_input_mode()
            time.sleep(max(0.03, duration_sec / 2))
            self.mode = 'typing'
            self._ui_draw_input_mode()
            return
        self.mode = 'typing'
        self._ui_draw_input_mode()
        time.sleep(max(0.03, duration_sec))
        self.mode = original_mode
        self._ui_draw_input_mode()

    def _draw_input_header(self):
        if not self.win_input:
            return
        _, w = self.win_input.getmaxyx()
        if w <= 4:
            return
        header = self._get_input_header()
        chip = ""
        if header:
            label, attr = header
            if self.mode != 'typing':
                attr = self.input_header_inactive_attr
            chip = self._truncate_line(f" {label} ", w - 4)
            try:
                self.win_input.addstr(0, 2, chip, attr)
            except curses.error:
                pass

    def _handle_mouse_event(self):
        try:
            _, mouse_x, mouse_y, _, bstate = curses.getmouse()
        except curses.error:
            return False
        wheel_up = 0
        for name in ("BUTTON4_PRESSED", "BUTTON4_CLICKED", "BUTTON4_RELEASED"):
            wheel_up |= getattr(curses, name, 0)
        if wheel_up == 0:
            wheel_up = 0x00080000

        wheel_down = 0
        for name in ("BUTTON5_PRESSED", "BUTTON5_CLICKED", "BUTTON5_RELEASED"):
            wheel_down |= getattr(curses, name, 0)
        wheel_down_mask = wheel_down | 0x00200000 | 0x08000000

        if bstate & wheel_up:
            self._scroll_playlist(-1)
            return True
        if bstate & wheel_down_mask:
            self._scroll_playlist(1)
            return True

        click_mask = 0
        for name in ("BUTTON1_CLICKED", "BUTTON1_PRESSED", "BUTTON1_RELEASED"):
            click_mask |= getattr(curses, name, 0)
        if click_mask:
            if not (bstate & click_mask):
                return False

        target_tab = None
        for region in self._tab_click_regions:
            if mouse_y == region["y"] and region["x1"] <= mouse_x <= region["x2"]:
                target_tab = region["tab_id"]
                break
        if target_tab is None:
            return False

        command_mode, _ = self._parse_input_command(self.input_text)
        if command_mode in ("add", "del", "new", "drop", "sort", "autoplay"):
            self._flash_input_mode_highlight()
            return True
        self._set_active_tab(target_tab)
        return True

    def _update_playlist_win(self):
        self.win_playlist.clear()
        self._draw_box(
            self.win_playlist,
            active=(self.mode == 'scroll'),
            active_attr=self._get_scroll_box_highlight_attr(),
        )
        h, w = self.win_playlist.getmaxyx()
        currently_playing_basename = self.current_song_info['basename']
        self._tab_click_regions = []
        try:
            begin_y, begin_x = self.win_playlist.getbegyx()
        except curses.error:
            begin_y, begin_x = 0, 0

        stats_text = None
        stats_x = None
        if self.mode == 'typing':
            stats_text = self._build_playlist_stats_text()
            if stats_text:
                stats_x = max(2, w - len(stats_text) - 2)

        if self.mode == 'typing':
            controls = None
        else:
            quit_label = "q:Exit Autoplay" if self._is_autoplay_active() else "q:Quit"
            controls = f" (h:Help {quit_label} s:Stop n:Next ←/→:Tabs ↑/↓:Scroll Tab:Mode) "
        if controls:
            controls_x = max(2, w - len(controls) - 2)
            try:
                self.win_playlist.addstr(0, controls_x, controls)
            except curses.error:
                pass
        else:
            controls_x = w - 2
            if stats_x is not None:
                controls_x = max(2, stats_x - 1)

        tabs = self._get_tab_ids()
        tab_x = 2
        max_tabs_x = max(2, controls_x - 1)
        drop_mode = self._active_command_mode == "drop"
        for tab_id in tabs:
            label = f" {self._get_tab_label(tab_id)} "
            if drop_mode:
                attr = self.playlist_name_attr
            else:
                attr = self.playlist_name_active_attr if tab_id == self.active_tab_id else self.playlist_name_attr
            if tab_x + len(label) >= max_tabs_x:
                if tab_x < max_tabs_x - 3:
                    try:
                        self.win_playlist.addstr(0, tab_x, "...", curses.A_DIM)
                    except curses.error:
                        pass
                break
            try:
                self.win_playlist.addstr(0, tab_x, label, attr)
            except curses.error:
                pass
            self._tab_click_regions.append({
                "tab_id": tab_id,
                "x1": begin_x + tab_x,
                "x2": begin_x + tab_x + len(label) - 1,
                "y": begin_y,
            })
            tab_x += len(label) + 1

        if stats_text and stats_x is not None and stats_x > tab_x:
            try:
                self.win_playlist.addstr(0, stats_x, stats_text, curses.A_DIM)
            except curses.error:
                pass

        row_start = 1

        max_items = self._playlist_max_items()
        display_data = self.display_data[self.scroll_pos:self.scroll_pos + max_items]
        current_pool = set(self.current_pool_basenames)
        active_playlist_set = set()
        sidebar_rows = []
        sidebar_enabled = False
        sidebar_header = None
        if self._active_command_mode == "add" and self.active_tab_id != ALL_SONGS_TAB_ID:
            active_playlist_set = set(self.playlists.get(self.active_tab_id, []))
            sidebar_rows = list(self.add_sidebar_rows)
            if sidebar_rows:
                sidebar_enabled = True
                total_added = len(sidebar_rows)
                if self.search_query:
                    sidebar_header = f"In Playlist ({total_added})"
                else:
                    sidebar_header = f"In Playlist ({total_added})"
        show_scrollbar = len(self.display_data) > max_items
        inner_w = max(1, w - 2)
        left_w = inner_w
        sidebar_w = 0
        sep_x = None
        if sidebar_enabled:
            min_sidebar_w = 22
            min_left_w = 28
            if inner_w >= min_left_w + min_sidebar_w + 1:
                sidebar_w = min(max(min_sidebar_w, inner_w // 3), inner_w - min_left_w - 1)
                left_w = inner_w - sidebar_w - 1
                sep_x = 1 + left_w
            else:
                sidebar_enabled = False
        text_width = max(1, left_w - 2 - (1 if show_scrollbar else 0))

        if not display_data:
            msg = None
            if self._active_command_mode == "del" and self.active_tab_id == ALL_SONGS_TAB_ID:
                msg = "Switch to a playlist tab to use '/del'."
            elif self._active_command_mode == "sort":
                msg = "No matching sort options."
            elif self._active_command_mode == "drop" and not self.playlist_order:
                msg = "No playlists to delete."
            elif (
                self._active_command_mode == "add"
                and self.search_query
                and self.add_sidebar_query_hits > 0
            ):
                msg = "All matches are already in this playlist."
            elif self.search_query:
                msg = "No matches."
            elif self.active_tab_id != ALL_SONGS_TAB_ID and not self._get_tab_song_basenames(self.active_tab_id):
                msg = "Playlist is empty. Use '/add' in typing mode."
            if msg is not None:
                msg_y = max(row_start, h // 2)
                msg_x = 1 + max(0, (left_w - len(msg)) // 2)
                try:
                    self.win_playlist.addstr(msg_y, msg_x, msg, curses.color_pair(3))
                except curses.error:
                    pass
        search_active = self.mode == 'typing' and (self._active_view_has_query or self._active_command_mode in ("add", "del", "drop", "sort", "autoplay"))
        for i, row in enumerate(display_data):
            basename, count, _ = row
            if self._active_command_mode == "drop":
                prefix = f"[{str(int(count)).rjust(3, ' ')} songs] "
                line = self._truncate_line(prefix + basename, text_width)
            elif self._active_command_mode == "sort":
                marker = "* " if row[2] == self.sort_mode else "  "
                line = self._truncate_line(marker + basename, text_width)
            else:
                if self.sort_mode == "time":
                    listen_seconds = self._get_listen_time_seconds(basename, count)
                    time_label = self._format_listen_time_hours(listen_seconds)
                    line = f"[{time_label}] {basename}"
                elif self.sort_mode == "trending":
                    line = f"[+{int(count)}] {basename}"
                else:
                    line = f"[{str(int(count)).rjust(3, ' ')}] {basename}"
                line = self._truncate_line(line, text_width)

            attr = curses.A_NORMAL
            index = self.scroll_pos + i
            selected = search_active and index == self.search_select_index
            if search_active and index == self.search_select_index:
                if self._active_command_mode == "add":
                    if basename in active_playlist_set:
                        attr = self.search_highlight_attr
                    else:
                        attr = self.add_search_highlight_attr
                elif self._active_command_mode in ("del", "drop"):
                    attr = self.del_search_highlight_attr
                elif self._active_command_mode == "sort":
                    attr = self.search_highlight_attr | curses.A_BOLD
                elif self._active_command_mode == "autoplay":
                    attr = self.search_highlight_attr | curses.A_BOLD
                else:
                    attr = self.search_highlight_attr
                if basename == currently_playing_basename:
                    attr |= curses.A_BOLD
            else:
                if basename == currently_playing_basename:
                    attr = curses.color_pair(2) | curses.A_BOLD
                elif self._active_command_mode == "sort" and row[2] == self.sort_mode:
                    attr = curses.A_BOLD
                elif self._active_command_mode == "add" and basename in active_playlist_set:
                    attr = self.playlist_name_attr
                elif basename in current_pool:
                    attr = curses.color_pair(5)

            try:
                if self._active_command_mode == "drop" and not selected:
                    prefix_text = self._truncate_line(prefix, text_width)
                    self.win_playlist.addstr(row_start + i, 2, prefix_text, attr)
                    remaining_width = max(0, text_width - len(prefix_text))
                    if remaining_width > 0:
                        playlist_name = self._truncate_line(basename, remaining_width)
                        self.win_playlist.addstr(row_start + i, 2 + len(prefix_text), playlist_name, attr)
                else:
                    self.win_playlist.addstr(row_start + i, 2, line, attr)
            except curses.error:
                pass # Avoid crashing if line is too long

        scrollbar_x = None
        if sidebar_enabled and sep_x is not None:
            scrollbar_x = max(1, sep_x - 1)
        self._draw_scrollbar(
            self.win_playlist,
            self.scroll_pos,
            len(self.display_data),
            max_items,
            start_row=row_start,
            x=scrollbar_x,
        )

        if sidebar_enabled and sep_x is not None:
            for i in range(max_items):
                try:
                    self.win_playlist.addch(row_start + i, sep_x, curses.ACS_VLINE)
                except curses.error:
                    pass
            right_start_x = sep_x + 2
            right_end_x = w - 2
            right_text_width = max(1, right_end_x - right_start_x)
            header_text = sidebar_header or "In Playlist"
            try:
                self.win_playlist.addstr(
                    row_start,
                    right_start_x,
                    self._truncate_line(header_text, right_text_width),
                    curses.A_BOLD,
                )
            except curses.error:
                pass
            sidebar_view_height = max(0, max_items - 1)
            for i in range(min(sidebar_view_height, len(sidebar_rows))):
                basename, _, _ = sidebar_rows[i]
                line = self._truncate_line(f"* {basename}", right_text_width)
                try:
                    self.win_playlist.addstr(row_start + 1 + i, right_start_x, line, curses.A_NORMAL)
                except curses.error:
                    pass
            remaining = len(sidebar_rows) - sidebar_view_height
            if remaining > 0 and sidebar_view_height > 0:
                more_text = self._truncate_line(f"... +{remaining} more", right_text_width)
                try:
                    self.win_playlist.addstr(
                        row_start + max_items - 1,
                        right_start_x,
                        more_text,
                        curses.A_DIM,
                    )
                except curses.error:
                    pass
