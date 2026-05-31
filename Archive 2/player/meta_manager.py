import json
import os
import time

from .constants import (
    CONFIG_KEY,
    META_FILENAME,
    PLAY_HISTORY_CONFIG_KEY,
)


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
                    data = json.load(f)
                    self._decode_play_history_mapping(data)
                    return data
        except (json.JSONDecodeError, IOError):
            pass
        return {}

    def save(self):
        try:
            with open(self.meta_path, 'w') as f:
                json.dump(self._get_serializable_data(), f, indent=2, sort_keys=True)
        except IOError:
            pass

    def _decode_play_history(self, history):
        if not isinstance(history, list):
            return []

        decoded = []
        current = None
        for index, value in enumerate(history):
            if not isinstance(value, (int, float)):
                continue
            if index == 0 or current is None:
                current = value
            else:
                current += value
            decoded.append(current)
        return decoded

    def _encode_play_history(self, history):
        if not isinstance(history, list):
            return []

        encoded = []
        previous = None
        for value in history:
            if not isinstance(value, (int, float)):
                continue
            if previous is None:
                encoded.append(value)
            else:
                encoded.append(value - previous)
            previous = value
        return encoded

    def _decode_play_history_mapping(self, data):
        config = data.get(CONFIG_KEY)
        if not isinstance(config, dict):
            return
        mapping = config.get(PLAY_HISTORY_CONFIG_KEY)
        if not isinstance(mapping, dict):
            return
        for basename, history in list(mapping.items()):
            mapping[basename] = self._decode_play_history(history)
        data[CONFIG_KEY] = config

    def _get_serializable_data(self):
        serialized = dict(self.data)
        config = self.data.get(CONFIG_KEY)
        if not isinstance(config, dict):
            return serialized

        serialized_config = dict(config)
        mapping = config.get(PLAY_HISTORY_CONFIG_KEY)
        if isinstance(mapping, dict):
            serialized_mapping = {}
            for basename, history in mapping.items():
                serialized_mapping[basename] = self._encode_play_history(history)
            serialized_config[PLAY_HISTORY_CONFIG_KEY] = serialized_mapping

        serialized[CONFIG_KEY] = serialized_config
        return serialized

    def _get_config(self):
        config = self.data.get(CONFIG_KEY)
        if not isinstance(config, dict):
            config = {}
            self.data[CONFIG_KEY] = config
        return config

    def _get_config_map(self, key):
        config = self._get_config()
        mapping = config.get(key)
        if not isinstance(mapping, dict):
            mapping = {}
            config[key] = mapping
        return mapping

    def sync_with_disk(self):
        """Ensures metadata is synced with the MP3s on disk."""
        try:
            disk_files = {os.path.splitext(f)[0] for f in os.listdir(self.folder_path) if f.lower().endswith('.mp3')}
            meta_files = {key for key in self.data.keys() if not str(key).startswith("__")}

            for basename in disk_files - meta_files:
                self.data[basename] = 0

            for basename in meta_files - disk_files:
                del self.data[basename]

            config = self.data.get(CONFIG_KEY)
            if isinstance(config, dict):
                for map_key in (PLAY_HISTORY_CONFIG_KEY,):
                    mapping = config.get(map_key)
                    if not isinstance(mapping, dict):
                        continue
                    for basename in list(mapping.keys()):
                        if basename not in disk_files:
                            del mapping[basename]
                self.data[CONFIG_KEY] = config

            self.save()
        except OSError:
            pass

    def increment_count(self, basename):
        if basename in self.data and isinstance(self.data.get(basename), int):
            self.data[basename] += 1
        else:
            self.data[basename] = 1

        now = int(time.time())
        play_history = self._get_config_map(PLAY_HISTORY_CONFIG_KEY)
        history = play_history.get(basename)
        if not isinstance(history, list):
            history = []
        history.append(now)
        play_history[basename] = history

        self.save()

    def get_all_sorted_data(self):
        return sorted(
            [(key, value) for key, value in self.data.items() if not str(key).startswith("__") and isinstance(value, int)],
            key=lambda item: item[0]
        )

    def get_play_history(self, basename):
        config = self.data.get(CONFIG_KEY)
        if not isinstance(config, dict):
            return []
        mapping = config.get(PLAY_HISTORY_CONFIG_KEY)
        if not isinstance(mapping, dict):
            return []
        history = mapping.get(basename)
        if not isinstance(history, list):
            return []
        return [ts for ts in history if isinstance(ts, (int, float))]

    def get_recent_play_count(self, basename, window_days, now=None):
        if now is None:
            now = time.time()
        try:
            window_seconds = max(0.0, float(window_days)) * 86400.0
        except (TypeError, ValueError):
            window_seconds = 0.0
        cutoff = float(now) - window_seconds
        return sum(1 for ts in self.get_play_history(basename) if ts >= cutoff)

    def get_config_value(self, key, default=None):
        config = self.data.get(CONFIG_KEY)
        if not isinstance(config, dict):
            return default
        return config.get(key, default)

    def set_config_value(self, key, value):
        config = self.data.get(CONFIG_KEY)
        if not isinstance(config, dict):
            config = {}
        if value is None:
            config.pop(key, None)
        else:
            config[key] = value
        self.data[CONFIG_KEY] = config
        self.save()
