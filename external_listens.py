from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent
EXTERNAL_PLAYS_FILE = ROOT / "external_plays.jsonl"
EXTERNAL_TRASH_FILE = ROOT / "external_plays_trash.jsonl"
EXTERNAL_MAPPINGS_FILE = ROOT / "external_track_mappings.json"
LISTEN_TIMESTAMP_FILES = {
    "mp3": ROOT / "listen_timestamps_mp3.json",
    "mid-mp3s": ROOT / "listen_timestamps_mid-mp3s.json",
}
LOCAL_LIBRARY_FOLDERS = {
    "mp3": ROOT / "static/mp3",
    "mid-mp3s": ROOT / "static/mid-mp3s",
}
EXTERNAL_STORE_LOCK = (
    Path(tempfile.gettempdir()) / f"audiotag-external-listens-{os.getuid()}.lock"
)


@dataclass(frozen=True)
class LocalTrackRef:
    library: str
    track: str

    @property
    def key(self) -> str:
        return f"{self.library}/{self.track}"


@contextmanager
def store_lock() -> Iterator[None]:
    EXTERNAL_STORE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with EXTERNAL_STORE_LOCK.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            for record in records:
                output.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                )
                output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_jsonl_objects(path: Path) -> List[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records: List[dict] = []
    for line in lines:
        try:
            value = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def external_key(
    provider: str,
    source_id: str,
    *,
    client_bundle: str = "",
    title: str = "",
    artist: str = "",
    album: str = "",
    duration: float = 0.0,
) -> str:
    provider = provider.strip().lower() or "unknown"
    source_id = source_id.strip()
    if source_id:
        return f"{provider}:{source_id}"
    identity = {
        "provider": provider,
        "client_bundle": client_bundle.strip(),
        "title": " ".join(title.split()),
        "artist": " ".join(artist.split()),
        "album": " ".join(album.split()),
        "duration_seconds": round(max(0.0, float(duration or 0.0)), 1),
    }
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return f"metadata:{hashlib.sha256(payload).hexdigest()[:24]}"


def compact_play(event: dict) -> Optional[dict]:
    """Normalize one current or legacy completed-play event."""
    if "external_key" in event:
        if not bool(event.get("qualifies", event.get("completed", False))):
            return None
        external_id = str(event.get("external_key") or "").strip()
        timestamp = str(
            event.get("timestamp")
            or event.get("started_at")
            or event.get("ended_at")
            or ""
        ).strip()
    else:
        external_id = str(event.get("id") or "").strip()
        timestamp = str(event.get("timestamp") or "").strip()
    if not external_id or not timestamp:
        return None
    try:
        duration = round(max(0.0, float(event.get("duration") or 0.0)), 3)
    except (TypeError, ValueError):
        duration = 0.0
    return {
        "id": external_id,
        "title": str(event.get("title") or "Untitled").strip() or "Untitled",
        "artist": str(event.get("artist") or "").strip(),
        "timestamp": timestamp,
        "duration": duration,
    }


def _load_track_records(path: Path) -> List[dict]:
    records: List[dict] = []
    by_id: Dict[str, dict] = {}
    try:
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                try:
                    raw = json.loads(line)
                except (TypeError, ValueError):
                    continue
                if not isinstance(raw, dict):
                    continue

                raw_timestamps = raw.get("timestamps")
                if isinstance(raw_timestamps, list):
                    external_id = str(raw.get("id") or "").strip()
                    timestamps = [
                        value.strip()
                        for value in raw_timestamps
                        if isinstance(value, str) and value.strip()
                    ]
                    if not external_id or not timestamps:
                        continue
                    try:
                        duration = round(max(0.0, float(raw.get("duration") or 0.0)), 3)
                    except (TypeError, ValueError):
                        duration = 0.0
                    metadata = {
                        "title": str(raw.get("title") or "Untitled").strip()
                        or "Untitled",
                        "artist": str(raw.get("artist") or "").strip(),
                        "duration": duration,
                    }
                else:
                    play = compact_play(raw)
                    if play is None:
                        continue
                    external_id = play["id"]
                    timestamps = [play["timestamp"]]
                    metadata = {
                        "title": play["title"],
                        "artist": play["artist"],
                        "duration": play["duration"],
                    }

                record = by_id.get(external_id)
                if record is None:
                    record = {
                        "id": external_id,
                        **metadata,
                        "timestamps": [],
                    }
                    records.append(record)
                    by_id[external_id] = record
                else:
                    record.update(metadata)
                for timestamp in timestamps:
                    if timestamp not in record["timestamps"]:
                        record["timestamps"].append(timestamp)
    except OSError:
        pass

    for record in records:
        record["timestamps"].sort()
    return records


def append_play(event: dict, path: Path = EXTERNAL_PLAYS_FILE) -> None:
    play = compact_play(event)
    if play is None:
        return
    with store_lock():
        records = _load_track_records(path)
        record = next((item for item in records if item["id"] == play["id"]), None)
        if record is None:
            record = {
                "id": play["id"],
                "title": play["title"],
                "artist": play["artist"],
                "duration": play["duration"],
                "timestamps": [],
            }
            records.append(record)
        else:
            record.update(
                title=play["title"],
                artist=play["artist"],
                duration=play["duration"],
            )
        if play["timestamp"] not in record["timestamps"]:
            record["timestamps"].append(play["timestamp"])
            record["timestamps"].sort()
        _atomic_write_jsonl(path, records)


def load_plays(path: Path = EXTERNAL_PLAYS_FILE) -> List[dict]:
    plays: List[dict] = []
    for record in _load_track_records(path):
        for timestamp in record["timestamps"]:
            plays.append(
                {
                    "id": record["id"],
                    "title": record["title"],
                    "artist": record["artist"],
                    "timestamp": timestamp,
                    "duration": record["duration"],
                }
            )
    plays.sort(key=lambda play: play["timestamp"])
    return plays


def _target_from_key(value: str) -> Optional[LocalTrackRef]:
    library, separator, track = value.partition("/")
    if not separator or not library.strip() or not track.strip():
        return None
    return LocalTrackRef(library.strip(), track.strip())


def _load_mapping_data(path: Path) -> Dict[str, List[str]]:
    try:
        with path.open("r", encoding="utf-8") as source:
            raw = json.load(source)
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}

    # Read the original external-id-first format so existing mappings migrate
    # without requiring a separate command.
    legacy = raw.get("mappings")
    if isinstance(legacy, dict):
        groups: Dict[str, List[str]] = {}
        for external_id, record in legacy.items():
            if not isinstance(record, dict):
                continue
            target = record.get("target")
            if not isinstance(target, dict):
                continue
            local = LocalTrackRef(
                str(target.get("library") or "").strip(),
                str(target.get("track") or "").strip(),
            )
            if local.library and local.track:
                groups.setdefault(local.key, []).append(str(external_id))
        return groups

    groups = {}
    for local_key, external_ids in raw.items():
        if str(local_key).startswith("_"):
            continue
        if _target_from_key(str(local_key)) is None or not isinstance(
            external_ids, list
        ):
            continue
        cleaned = []
        for external_id in external_ids:
            value = str(external_id).strip()
            if value and value not in cleaned:
                cleaned.append(value)
        if cleaned:
            groups[str(local_key)] = cleaned
    return groups


def _save_mapping_data(path: Path, groups: Dict[str, List[str]]) -> None:
    _atomic_write_json(path, groups)


def load_mappings(path: Path = EXTERNAL_MAPPINGS_FILE) -> Dict[str, dict]:
    result: Dict[str, dict] = {}
    order = 0
    groups = _load_mapping_data(path)
    for local_key, external_ids in groups.items():
        target = _target_from_key(local_key)
        if target is None:
            continue
        for external_id in external_ids:
            order += 1
            result[external_id] = {
                "target": {"library": target.library, "track": target.track},
                "order": order,
            }
    return result


def save_mapping(
    external_id: str,
    target: LocalTrackRef,
    path: Path = EXTERNAL_MAPPINGS_FILE,
) -> dict:
    external_id = external_id.strip()
    if not external_id:
        raise ValueError("The external track ID is empty.")
    if not target.library.strip() or not target.track.strip():
        raise ValueError("The local library and track are required.")
    with store_lock():
        groups = _load_mapping_data(path)
        for local_key in list(groups):
            groups[local_key] = [
                value for value in groups[local_key] if value != external_id
            ]
            if not groups[local_key]:
                del groups[local_key]
        target_ids = groups.pop(target.key, [])
        target_ids.append(external_id)
        groups[target.key] = target_ids
        _save_mapping_data(path, groups)
    return {
        "target": {"library": target.library, "track": target.track},
        "order": sum(len(values) for values in groups.values()),
    }


def delete_external_id(
    external_id: str,
    plays_path: Path = EXTERNAL_PLAYS_FILE,
    mappings_path: Path = EXTERNAL_MAPPINGS_FILE,
    trash_path: Path = EXTERNAL_TRASH_FILE,
) -> Optional[dict]:
    """Move one external track to durable trash and return its record."""
    external_id = external_id.strip()
    if not external_id:
        return None
    with store_lock():
        records = _load_track_records(plays_path)
        deleted = next(
            (dict(record) for record in records if record["id"] == external_id),
            None,
        )
        if deleted is None:
            return None
        records = [record for record in records if record["id"] != external_id]
        groups = _load_mapping_data(mappings_path)
        mapping_keys = [
            local_key
            for local_key, external_ids in groups.items()
            if external_id in external_ids
        ]
        for local_key in list(groups):
            groups[local_key] = [
                value for value in groups[local_key] if value != external_id
            ]
            if not groups[local_key]:
                del groups[local_key]
        trash = _load_jsonl_objects(trash_path)
        trash.append(
            {
                "deleted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "record": deleted,
                "mapping_keys": mapping_keys,
            }
        )
        # Persist the recovery copy first. If the process stops between writes,
        # undo remains safe because restoration merges timestamps by identity.
        _atomic_write_jsonl(trash_path, trash)
        _atomic_write_jsonl(plays_path, records)
        _save_mapping_data(mappings_path, groups)
    return deleted


def _restore_record(records: List[dict], record: dict) -> Optional[dict]:
    external_id = str(record.get("id") or "").strip()
    raw_timestamps = record.get("timestamps")
    if not external_id or not isinstance(raw_timestamps, list):
        return None
    timestamps = [
        value.strip()
        for value in raw_timestamps
        if isinstance(value, str) and value.strip()
    ]
    if not timestamps:
        return None
    restored = next(
        (item for item in records if item["id"] == external_id),
        None,
    )
    if restored is None:
        restored = {
            "id": external_id,
            "title": str(record.get("title") or "Untitled"),
            "artist": str(record.get("artist") or ""),
            "duration": float(record.get("duration") or 0.0),
            "timestamps": [],
        }
        records.append(restored)
    for timestamp in timestamps:
        if timestamp not in restored["timestamps"]:
            restored["timestamps"].append(timestamp)
    restored["timestamps"].sort()
    return restored


def restore_deleted_record(
    record: dict,
    plays_path: Path = EXTERNAL_PLAYS_FILE,
) -> None:
    """Restore one supplied external-play record without changing trash."""
    with store_lock():
        records = _load_track_records(plays_path)
        if _restore_record(records, record) is not None:
            _atomic_write_jsonl(plays_path, records)


def restore_last_deleted(
    plays_path: Path = EXTERNAL_PLAYS_FILE,
    mappings_path: Path = EXTERNAL_MAPPINGS_FILE,
    trash_path: Path = EXTERNAL_TRASH_FILE,
) -> Optional[dict]:
    """Restore and remove the newest valid entry from persistent trash."""
    with store_lock():
        trash = _load_jsonl_objects(trash_path)
        while trash:
            entry = trash.pop()
            record = entry.get("record")
            if not isinstance(record, dict):
                continue
            records = _load_track_records(plays_path)
            restored = _restore_record(records, record)
            if restored is None:
                continue

            external_id = str(restored["id"])
            groups = _load_mapping_data(mappings_path)
            already_mapped = any(
                external_id in external_ids for external_ids in groups.values()
            )
            mapping_keys = entry.get("mapping_keys")
            if not already_mapped and isinstance(mapping_keys, list):
                for local_key in mapping_keys:
                    if not isinstance(local_key, str) or not local_key.strip():
                        continue
                    external_ids = groups.setdefault(local_key, [])
                    if external_id not in external_ids:
                        external_ids.append(external_id)

            # Restore active data before removing its recovery copy. Repeating
            # undo after an interrupted write is idempotent.
            _atomic_write_jsonl(plays_path, records)
            _save_mapping_data(mappings_path, groups)
            _atomic_write_jsonl(trash_path, trash)
            return dict(restored)
        if trash_path.exists():
            _atomic_write_jsonl(trash_path, [])
    return None


def _provider_from_id(external_id: str) -> str:
    provider, separator, _value = external_id.partition(":")
    return provider if separator else "external"


def distinct_tracks(events: Iterable[dict]) -> Dict[str, dict]:
    distinct: Dict[str, dict] = {}
    for play in events:
        external_id = str(play.get("id") or "")
        timestamp = str(play.get("timestamp") or "")
        if not external_id or not timestamp:
            continue
        item = distinct.setdefault(
            external_id,
            {
                "external_key": external_id,
                "provider": _provider_from_id(external_id),
                "title": str(play.get("title") or "Untitled"),
                "artist": str(play.get("artist") or ""),
                "duration": float(play.get("duration") or 0.0),
                "plays": 0,
                "qualified_plays": 0,
                "first_played_at": timestamp,
                "last_played_at": timestamp,
            },
        )
        item["plays"] += 1
        item["qualified_plays"] += 1
        if timestamp >= str(item.get("last_played_at") or ""):
            item.update(
                {
                    "title": str(play.get("title") or item["title"]),
                    "artist": str(play.get("artist") or item["artist"]),
                    "duration": float(play.get("duration") or item["duration"]),
                    "last_played_at": timestamp,
                }
            )
    return distinct


def mapped_contributions(
    library: str,
    events: Optional[Iterable[dict]] = None,
    mappings: Optional[Dict[str, dict]] = None,
) -> Tuple[Dict[str, int], List[dict]]:
    events = list(events) if events is not None else load_plays()
    mappings = mappings if mappings is not None else load_mappings()
    counts: Counter[str] = Counter()
    history: List[dict] = []
    seen: set[Tuple[str, str]] = set()
    for play in events:
        external_id = str(play.get("id") or "")
        timestamp = str(play.get("timestamp") or "")
        identity = (external_id, timestamp)
        if not external_id or not timestamp or identity in seen:
            continue
        seen.add(identity)
        mapping = mappings.get(external_id)
        target = mapping.get("target") if isinstance(mapping, dict) else None
        if not isinstance(target, dict):
            continue
        if str(target.get("library") or "") != library:
            continue
        track = str(target.get("track") or "")
        if not track:
            continue
        counts[Path(track).stem] += 1
        history.append(
            {
                "timestamp": timestamp,
                "track": track,
                "source": "manual",
                "external_listen": True,
            }
        )
    history.sort(key=lambda item: item["timestamp"])
    return dict(counts), history


def sync_mapped_listen_timestamps(
    events: Optional[Iterable[dict]] = None,
    mappings: Optional[Dict[str, dict]] = None,
    timestamp_files: Optional[Dict[str, Path]] = None,
    library_folders: Optional[Dict[str, Path]] = None,
) -> List[Path]:
    """Materialize mapped external plays in each library's timestamp history."""
    events = list(events) if events is not None else load_plays()
    mappings = mappings if mappings is not None else load_mappings()
    timestamp_files = timestamp_files or LISTEN_TIMESTAMP_FILES
    library_folders = library_folders or LOCAL_LIBRARY_FOLDERS
    changed: List[Path] = []

    with store_lock():
        for library, path in timestamp_files.items():
            _counts, external_history = mapped_contributions(library, events, mappings)
            folder = library_folders.get(library)
            if folder is not None:
                external_history = [
                    item
                    for item in external_history
                    if (folder / str(item.get("track") or "")).is_file()
                ]
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                raw = []
            except (OSError, ValueError):
                continue
            if not isinstance(raw, list):
                continue
            local_history = [
                dict(item)
                for item in raw
                if isinstance(item, dict) and item.get("external_listen") is not True
            ]
            combined = [*local_history, *external_history]
            combined.sort(key=lambda item: str(item.get("timestamp") or ""))
            if combined != raw:
                _atomic_write_json(path, combined)
                changed.append(path)
    return changed


def store_revision(
    paths: Iterable[Path] = (EXTERNAL_PLAYS_FILE, EXTERNAL_MAPPINGS_FILE),
) -> Tuple[Tuple[str, int, int], ...]:
    revision = []
    for path in paths:
        try:
            stat = path.stat()
            revision.append((str(path), stat.st_mtime_ns, stat.st_size))
        except OSError:
            revision.append((str(path), 0, 0))
    return tuple(revision)
