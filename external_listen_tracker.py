#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import plistlib
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from external_listens import append_play, external_key


TRACKER_LOCK = (
    Path(tempfile.gettempdir()) / f"audiotag-external-listen-tracker-{os.getuid()}.lock"
)
NOW_PLAYING_KEYS = {
    "title": "kMRMediaRemoteNowPlayingInfoTitle",
    "artist": "kMRMediaRemoteNowPlayingInfoArtist",
    "album": "kMRMediaRemoteNowPlayingInfoAlbum",
    "duration": "kMRMediaRemoteNowPlayingInfoDuration",
    "elapsed": "kMRMediaRemoteNowPlayingInfoElapsedTime",
    "rate": "kMRMediaRemoteNowPlayingInfoPlaybackRate",
    "bundle": "kMRMediaRemoteNowPlayingInfoClientBundleIdentifier",
    "unique_id": "kMRMediaRemoteNowPlayingInfoUniqueIdentifier",
}
BROWSER_APPS = {
    "com.google.Chrome": "Google Chrome",
    "com.microsoft.edgemac": "Microsoft Edge",
}
CLOCK_COMPLETION_GRACE_SECONDS = 1.5
PAUSED_END_CONFIRM_SECONDS = 2.0
MIN_COMPLETION_FRACTION = 0.98
LOCK_STATE_CACHE_SECONDS = 0.25
_lock_state_checked_at = -float("inf")
_lock_state_checked_value = False


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ioreg_lock_state(payload: bytes) -> Optional[bool]:
    try:
        root = plistlib.loads(payload)
    except (TypeError, ValueError, plistlib.InvalidFileException):
        return None
    if not isinstance(root, dict):
        return None
    if "IOConsoleLocked" in root:
        return bool(root.get("IOConsoleLocked"))
    users = root.get("IOConsoleUsers")
    if not isinstance(users, list):
        return None
    for user in users:
        if isinstance(user, dict) and "CGSSessionScreenIsLocked" in user:
            return bool(user.get("CGSSessionScreenIsLocked"))
    return None


def mac_is_locked(*, force: bool = False) -> bool:
    global _lock_state_checked_at, _lock_state_checked_value
    if sys.platform != "darwin":
        return False
    checked_at = time.monotonic()
    if (
        not force
        and checked_at - _lock_state_checked_at < LOCK_STATE_CACHE_SECONDS
    ):
        return _lock_state_checked_value

    locked: Optional[bool] = None
    try:
        from Quartz import CGSessionCopyCurrentDictionary  # type: ignore

        session = CGSessionCopyCurrentDictionary() or {}
        if "CGSSessionScreenIsLocked" in session:
            locked = bool(session.get("CGSSessionScreenIsLocked"))
        elif "CGSSessionOnConsoleKey" in session:
            locked = not bool(session.get("CGSSessionOnConsoleKey"))
    except Exception:
        pass

    if locked is None:
        try:
            result = subprocess.run(
                ["/usr/sbin/ioreg", "-n", "Root", "-d1", "-a"],
                capture_output=True,
                timeout=1.0,
                check=False,
            )
            if result.returncode == 0:
                locked = ioreg_lock_state(result.stdout)
        except (OSError, subprocess.TimeoutExpired):
            pass

    _lock_state_checked_at = checked_at
    if locked is not None:
        _lock_state_checked_value = locked
    return _lock_state_checked_value


def run_osascript(source: str, timeout: float = 1.5) -> str:
    try:
        result = subprocess.run(
            ["/usr/bin/osascript", "-e", source],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def browser_tabs(app_name: str) -> List[Tuple[str, str]]:
    source = f"""
set output to ""
set tabCharacter to ASCII character 9
tell application "{app_name}"
    repeat with browserWindow in windows
        repeat with browserTab in tabs of browserWindow
            set output to output & (title of browserTab) & tabCharacter
            set output to output & (URL of browserTab) & linefeed
        end repeat
    end repeat
end tell
return output
"""
    rows: List[Tuple[str, str]] = []
    for line in run_osascript(source).splitlines():
        if "\t" not in line:
            continue
        title, url = line.split("\t", 1)
        if url.startswith(("https://", "http://")):
            rows.append((title.strip(), url.strip()))
    return rows


def clean_browser_title(value: str) -> str:
    value = value.strip()
    for suffix in (" - YouTube", " | Spotify", " on Spotify", " - Apple Music"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    return " ".join(value.split()).casefold()


def identity_from_url(url: str) -> Tuple[str, str]:
    try:
        parsed = urlparse(url)
    except ValueError:
        return "", ""
    host = parsed.netloc.casefold().removeprefix("www.")
    path_parts = [part for part in parsed.path.split("/") if part]
    if host in {"youtube.com", "music.youtube.com"}:
        video_id = (parse_qs(parsed.query).get("v") or [""])[0]
        return ("youtube", video_id)
    if host == "youtu.be" and path_parts:
        return ("youtube", path_parts[0])
    if host == "open.spotify.com" and len(path_parts) >= 2 and path_parts[0] == "track":
        return ("spotify", path_parts[1])
    if host.endswith("music.apple.com"):
        song_id = (parse_qs(parsed.query).get("i") or [""])[0]
        if not song_id and len(path_parts) >= 2 and path_parts[-2] == "song":
            song_id = path_parts[-1]
        return ("apple_music", song_id)
    return "", ""


def malformed_media_title(title: str) -> bool:
    value = title.strip().casefold()
    return (
        len(value) > 180
        or value.startswith(("http://", "https://"))
        or (value.count("&") >= 4 and value.count("=") >= 4)
    )


def same_media_duration(first: dict, second: dict) -> bool:
    first_duration = float(first.get("duration") or 0.0)
    second_duration = float(second.get("duration") or 0.0)
    return (
        first_duration > 0
        and second_duration > 0
        and abs(first_duration - second_duration)
        <= max(2.0, min(first_duration, second_duration) * 0.01)
    )


def same_media_text(first: dict, second: dict, field_name: str) -> bool:
    first_value = " ".join(str(first.get(field_name) or "").split()).casefold()
    second_value = " ".join(str(second.get(field_name) or "").split()).casefold()
    return first_value == second_value


def stabilize_transient_metadata(
    active: "PlaybackSession", metadata: dict
) -> dict:
    current = active.metadata
    if (
        metadata.get("client_bundle") != current.get("client_bundle")
        or not same_media_duration(current, metadata)
    ):
        return metadata
    current_malformed = malformed_media_title(str(current.get("title") or ""))
    incoming_malformed = malformed_media_title(str(metadata.get("title") or ""))
    identity_fields = (
        "external_key",
        "provider",
        "source_id",
        "client_bundle",
        "title",
        "artist",
        "album",
    )
    if current_malformed or incoming_malformed:
        if incoming_malformed and not current_malformed:
            stabilized = dict(metadata)
            for field_name in identity_fields:
                stabilized[field_name] = current.get(field_name)
            return stabilized
        if current_malformed and not incoming_malformed:
            # The session began while MediaRemote exposed a resource URL.
            # Promote it without throwing away the playback already observed.
            for field_name in identity_fields:
                current[field_name] = metadata.get(field_name)
        return metadata

    # Browser MediaRemote metadata can fluctuate without the media changing.
    # In particular, Edge has reported one track alternately as 239.527 and
    # 239.573 seconds. That crosses external_key()'s tenth-second hash boundary
    # and used to reset the fallback clock used when elapsed is always zero.
    # Keep the session's canonical identity while its stable human-readable
    # identity still describes the same media.
    if not (
        same_media_text(current, metadata, "title")
        and same_media_text(current, metadata, "artist")
    ):
        return metadata
    stabilized = dict(metadata)
    for field_name in (*identity_fields, "duration"):
        stabilized[field_name] = current.get(field_name)
    return stabilized


class SourceIdentityResolver:
    def resolve(
        self, bundle: str, title: str, artist: str, duration: float
    ) -> Tuple[str, str]:
        return self._resolve_uncached(bundle, title)

    def _resolve_uncached(self, bundle: str, title: str) -> Tuple[str, str]:
        if bundle == "com.spotify.client":
            spotify_url = run_osascript(
                'tell application "Spotify" to if player state is playing then '
                "return spotify url of current track"
            )
            provider, source_id = identity_from_url(spotify_url)
            return (provider or "spotify", source_id)

        if bundle == "com.apple.Music":
            persistent_id = run_osascript(
                'tell application "Music" to if player state is playing then '
                "return persistent ID of current track"
            )
            return ("apple_music", persistent_id)

        app_name = BROWSER_APPS.get(bundle)
        if app_name:
            supported: List[Tuple[str, str, str, str]] = []
            for tab_title, url in browser_tabs(app_name):
                provider, source_id = identity_from_url(url)
                if provider and source_id:
                    supported.append((tab_title, url, provider, source_id))
            exact = [
                row
                for row in supported
                if clean_browser_title(row[0]) == clean_browser_title(title)
            ]
            if len(exact) == 1:
                _, _, provider, source_id = exact[0]
                return provider, source_id
            return "web", ""

        bundle_lc = bundle.casefold()
        if "spotify" in bundle_lc:
            return "spotify", ""
        if "music" in bundle_lc:
            return "apple_music", ""
        return "unknown", ""


def read_now_playing(resolver: SourceIdentityResolver) -> Optional[dict]:
    try:
        result = subprocess.run(
            ["nowplaying-cli", "get-raw"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        raw = json.loads(result.stdout)
    except (TypeError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    title = str(raw.get(NOW_PLAYING_KEYS["title"]) or "").strip()
    artist = str(raw.get(NOW_PLAYING_KEYS["artist"]) or "").strip()
    album = str(raw.get(NOW_PLAYING_KEYS["album"]) or "").strip()
    bundle = str(raw.get(NOW_PLAYING_KEYS["bundle"]) or "").strip()
    duration = float(raw.get(NOW_PLAYING_KEYS["duration"]) or 0.0)
    elapsed = float(raw.get(NOW_PLAYING_KEYS["elapsed"]) or 0.0)
    rate = float(raw.get(NOW_PLAYING_KEYS["rate"]) or 0.0)
    if not title:
        return None
    provider, source_id = resolver.resolve(bundle, title, artist, duration)
    if not source_id:
        unique_id = str(raw.get(NOW_PLAYING_KEYS["unique_id"]) or "").strip()
        if unique_id and bundle not in BROWSER_APPS:
            source_id = unique_id
    key = external_key(
        provider,
        source_id,
        client_bundle=bundle,
        title=title,
        artist=artist,
        album=album,
        duration=duration,
    )
    return {
        "external_key": key,
        "provider": provider,
        "source_id": source_id,
        "client_bundle": bundle,
        "title": title,
        "artist": artist,
        "album": album,
        "duration": duration,
        "elapsed": max(0.0, elapsed),
        "rate": rate,
    }


@dataclass
class PlaybackSession:
    metadata: dict
    last_elapsed: float
    max_elapsed: float
    end_candidate_at: Optional[str] = None
    end_candidate_observed_at: Optional[float] = None
    estimated_elapsed: float = field(init=False)
    observed_playing_seconds: float = 0.0
    last_observed_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self.estimated_elapsed = self.last_elapsed

    def update(
        self,
        metadata: dict,
        *,
        observed_at: Optional[float] = None,
        max_clock_step: float = 5.0,
    ) -> None:
        observed_at = time.monotonic() if observed_at is None else observed_at
        previous_rate = float(self.metadata.get("rate") or 0.0)
        elapsed = float(metadata.get("elapsed") or 0.0)

        # Some browser Now Playing sessions publish a valid playing rate and
        # duration but leave elapsed time fixed at zero. Advance a local media
        # clock while that session is observed playing. The cap avoids counting
        # a long machine sleep or an extended gap in MediaRemote updates.
        wall_step = max(0.0, observed_at - self.last_observed_at)
        if previous_rate > 0:
            media_step = min(wall_step, max_clock_step) * previous_rate
            self.estimated_elapsed += media_step
            self.observed_playing_seconds += media_step
        if elapsed > self.last_elapsed:
            self.estimated_elapsed = max(self.estimated_elapsed, elapsed)

        self.metadata = metadata
        self.last_elapsed = elapsed
        self.last_observed_at = observed_at
        self.max_elapsed = max(
            self.max_elapsed,
            self.estimated_elapsed,
        )
        rate = float(metadata.get("rate") or 0.0)
        if rate > 0:
            self.end_candidate_at = None
            self.end_candidate_observed_at = None
        elif previous_rate > 0 and self.reached_end():
            # Keep the time playback stopped, but do not record it yet. A resume
            # cancels this candidate; a later advance confirms it as an ending.
            self.end_candidate_at = now_iso()
            self.end_candidate_observed_at = observed_at

    def reached_end(self) -> bool:
        duration = float(self.metadata.get("duration") or 0.0)
        return (
            duration > 0
            and self.max_elapsed >= duration * MIN_COMPLETION_FRACTION
        )

    def clock_completion_ready(self) -> bool:
        duration = float(self.metadata.get("duration") or 0.0)
        rate = float(self.metadata.get("rate") or 0.0)
        return (
            duration > 0
            and rate > 0
            and self.observed_playing_seconds
            >= duration + CLOCK_COMPLETION_GRACE_SECONDS
        )

    def paused_end_confirmed(self, observed_at: Optional[float] = None) -> bool:
        if self.end_candidate_observed_at is None:
            return False
        observed_at = time.monotonic() if observed_at is None else observed_at
        paused_seconds = observed_at - self.end_candidate_observed_at
        return paused_seconds >= PAUSED_END_CONFIRM_SECONDS

    def consume_completion(self) -> None:
        duration = float(self.metadata.get("duration") or 0.0)
        if duration <= 0:
            return
        self.observed_playing_seconds = max(
            0.0, self.observed_playing_seconds - duration
        )
        self.estimated_elapsed = max(0.0, self.estimated_elapsed - duration)
        self.max_elapsed = max(0.0, self.max_elapsed - duration)
        self.end_candidate_at = None
        self.end_candidate_observed_at = None

    def play_record(self, completed_at: Optional[str] = None) -> Optional[dict]:
        if malformed_media_title(str(self.metadata.get("title") or "")):
            return None
        return {
            "id": self.metadata["external_key"],
            "title": self.metadata.get("title", "Untitled"),
            "artist": self.metadata.get("artist", ""),
            "timestamp": completed_at or self.end_candidate_at or now_iso(),
            "duration": float(self.metadata.get("duration") or 0.0),
        }

    def completed_play(self, completed_at: Optional[str] = None) -> Optional[dict]:
        if not self.reached_end():
            return None
        return self.play_record(completed_at)


@dataclass
class LockCompletionGate:
    suppress_until_observed_boundary: bool = False

    def lock(self) -> None:
        self.suppress_until_observed_boundary = True

    def should_record(self, *, observed_boundary: bool) -> bool:
        if not self.suppress_until_observed_boundary:
            return True
        if observed_boundary:
            self.suppress_until_observed_boundary = False
        return False


def acquire_tracker_lock() -> Optional[object]:
    lock_file = TRACKER_LOCK.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        return None
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(str(os.getpid()))
    lock_file.flush()
    return lock_file


def run_tracker(
    poll_seconds: float = 1.0,
    stop_event: Optional[threading.Event] = None,
    quiet: bool = False,
) -> int:
    lock_file = acquire_tracker_lock()
    if lock_file is None:
        if not quiet:
            print("The external-listen tracker is already running.", file=sys.stderr)
        return 0
    stop_event = stop_event or threading.Event()
    resolver = SourceIdentityResolver()
    active: Optional[PlaybackSession] = None
    lock_gate = LockCompletionGate()

    def record(play: Optional[dict], *, observed_boundary: bool) -> None:
        if play is None:
            if observed_boundary:
                lock_gate.should_record(observed_boundary=True)
            return
        if lock_gate.should_record(observed_boundary=observed_boundary):
            append_play(play)

    def finalize(*, observed_boundary: bool) -> None:
        nonlocal active
        if active is None:
            return
        record(
            active.completed_play(),
            observed_boundary=observed_boundary,
        )
        active = None

    try:
        while not stop_event.is_set():
            if mac_is_locked():
                active = None
                lock_gate.lock()
                stop_event.wait(poll_seconds)
                continue
            metadata = read_now_playing(resolver)
            # Locking can happen during the metadata subprocess calls.
            if mac_is_locked(force=True):
                active = None
                lock_gate.lock()
                stop_event.wait(poll_seconds)
                continue
            if metadata is not None and active is not None:
                metadata = stabilize_transient_metadata(active, metadata)
            playing = bool(metadata and float(metadata.get("rate") or 0.0) > 0)
            if active is None:
                if metadata is not None:
                    elapsed = float(metadata.get("elapsed") or 0.0)
                    active = PlaybackSession(
                        metadata=metadata,
                        last_elapsed=elapsed,
                        max_elapsed=elapsed,
                    )
            elif metadata is None:
                # A paused browser tab and a finished tab can both temporarily
                # disappear from Now Playing. Keep the session pending until a
                # later state tells us whether it resumed or really advanced.
                pass
            elif metadata["external_key"] != active.metadata["external_key"]:
                finalize(observed_boundary=True)
                elapsed = float(metadata.get("elapsed") or 0.0)
                active = PlaybackSession(
                    metadata=metadata,
                    last_elapsed=elapsed,
                    max_elapsed=elapsed,
                )
            else:
                elapsed = float(metadata.get("elapsed") or 0.0)
                replayed = playing and elapsed + 5.0 < active.last_elapsed
                if replayed:
                    finalize(observed_boundary=True)
                    active = PlaybackSession(
                        metadata=metadata,
                        last_elapsed=elapsed,
                        max_elapsed=elapsed,
                    )
                else:
                    active.update(
                        metadata,
                        max_clock_step=max(5.0, poll_seconds * 3.0),
                    )

            if active is not None:
                while active.clock_completion_ready():
                    play = active.play_record()
                    if play is None:
                        break
                    record(play, observed_boundary=False)
                    active.consume_completion()
                if active.paused_end_confirmed():
                    play = active.completed_play(active.end_candidate_at)
                    if play is not None:
                        record(play, observed_boundary=True)
                        active.consume_completion()
            stop_event.wait(poll_seconds)
    finally:
        # Stopping the tracker is not evidence that a paused track completed.
        lock_file.close()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record macOS Now Playing sessions for manual mapping to local MP3s."
        )
    )
    parser.add_argument(
        "--poll", type=float, default=1.0, help="Polling interval in seconds."
    )
    parser.add_argument("--quiet", action="store_true", help="Only print fatal errors.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stop_event = threading.Event()

    def stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    return run_tracker(max(0.25, args.poll), stop_event, args.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
