"""Icon-free headphone-level estimates for audio decoded inside the player.

This module never captures system audio.  It measures PCM samples already owned
by pygame and reads the selected CoreAudio device's volume setting.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pygame
from scipy.signal import bilinear, lfilter


SILENCE_DB = -120.0

# These broad offsets are for orientation only.  Apple doesn't publish a
# sensitivity specification for its wired earbuds, and fit changes their output
# substantially.  They convert full-scale PCM RMS plus electrical gain to a
# rough acoustic range.
APPLE_WIRED_OFFSET_LOW_DB = 121.0
APPLE_WIRED_OFFSET_HIGH_DB = 132.0


def _fourcc(value: str) -> int:
    return int.from_bytes(value.encode("ascii"), "big")


class _AudioObjectPropertyAddress(ctypes.Structure):
    _fields_ = [
        ("selector", ctypes.c_uint32),
        ("scope", ctypes.c_uint32),
        ("element", ctypes.c_uint32),
    ]


@dataclass(frozen=True)
class OutputDevice:
    object_id: int
    name: str


@dataclass(frozen=True)
class ListeningEstimate:
    low_db_spl: float
    high_db_spl: float
    mac_volume_percent: float | None


@dataclass(frozen=True)
class ExposureSummary:
    today_played_seconds: float
    today_laeq_low_db: float | None
    today_laeq_high_db: float | None
    week_played_seconds: float
    week_laeq_low_db: float | None
    week_laeq_high_db: float | None
    week_dose_percent: float


class ExposureRecorder:
    """Persist measured playback separately from listen-completion history."""

    VERSION = 1
    CHECKPOINT_SECONDS = 5.0

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._data = self._load()
        self._active: dict | None = None
        self._last_checkpoint = time.monotonic()
        previous_active = self._data.pop("active", None)
        if isinstance(previous_active, dict):
            if float(previous_active.get("played_seconds", 0.0) or 0.0) > 0.0:
                previous_active["ended_at"] = str(
                    previous_active.get("checkpointed_at")
                    or datetime.now().astimezone().isoformat(timespec="milliseconds")
                )
                previous_active.pop("checkpointed_at", None)
                self._finalize_values(previous_active)
                self._data["events"].append(self._clean_event(previous_active))
            self._save()

    @staticmethod
    def _clean_event(event: dict) -> dict:
        keys = (
            "started_at",
            "ended_at",
            "played_seconds",
            "estimated_laeq_low_db",
            "estimated_laeq_high_db",
            "who_dose_percent",
        )
        return {key: event[key] for key in keys if key in event}

    def _load(self) -> dict:
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            loaded = {}
        if not isinstance(loaded, dict):
            loaded = {}
        events = loaded.get("events")
        clean_events = (
            [self._clean_event(event) for event in events if isinstance(event, dict)]
            if isinstance(events, list)
            else []
        )
        return {
            "version": self.VERSION,
            "events": clean_events,
            **(
                {"active": loaded["active"]}
                if isinstance(loaded.get("active"), dict)
                else {}
            ),
        }

    def _save(self) -> None:
        payload = dict(self._data)
        if self._active is not None:
            active = dict(self._active)
            active["checkpointed_at"] = datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            )
            payload["active"] = active
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
        except OSError:
            # Exposure logging must never interrupt playback.
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def begin_track(self) -> None:
        if self._active is not None:
            self.finish_track()
        self._active = {
            "started_at": datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            ),
            "played_seconds": 0.0,
            "low_energy_seconds": 0.0,
            "high_energy_seconds": 0.0,
            "who_dose_fraction": 0.0,
        }
        self._last_checkpoint = time.monotonic()
        self._save()

    @staticmethod
    def _energy(level_db: float) -> float:
        return 10.0 ** (max(-200.0, min(200.0, level_db)) / 10.0)

    def update(self, estimate: ListeningEstimate, elapsed: float) -> None:
        if self._active is None or elapsed <= 0.0:
            return
        active = self._active
        active["played_seconds"] += elapsed
        active["low_energy_seconds"] += elapsed * self._energy(
            estimate.low_db_spl
        )
        active["high_energy_seconds"] += elapsed * self._energy(
            estimate.high_db_spl
        )
        allowance_hours = 40.0 * (
            2.0 ** ((80.0 - estimate.high_db_spl) / 3.0)
        )
        active["who_dose_fraction"] += elapsed / max(
            allowance_hours * 3600.0,
            0.001,
        )
        if time.monotonic() - self._last_checkpoint >= self.CHECKPOINT_SECONDS:
            self._last_checkpoint = time.monotonic()
            self._save()

    @staticmethod
    def _finalize_values(event: dict) -> None:
        seconds = float(event.get("played_seconds", 0.0) or 0.0)
        if seconds > 0.0:
            for energy_key, output_key in (
                ("low_energy_seconds", "estimated_laeq_low_db"),
                ("high_energy_seconds", "estimated_laeq_high_db"),
            ):
                energy = float(event.get(energy_key, 0.0) or 0.0)
                event[output_key] = (
                    10.0 * math.log10(energy / seconds) if energy > 0.0 else None
                )
        for internal_key in (
            "low_energy_seconds",
            "high_energy_seconds",
        ):
            event.pop(internal_key, None)
        event["played_seconds"] = round(seconds, 3)
        event["who_dose_percent"] = round(
            float(event.pop("who_dose_fraction", 0.0) or 0.0) * 100.0,
            6,
        )

    def finish_track(self) -> None:
        if self._active is None:
            return
        event = self._active
        self._active = None
        if float(event.get("played_seconds", 0.0) or 0.0) > 0.0:
            event["ended_at"] = datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            )
            self._finalize_values(event)
            self._data["events"].append(self._clean_event(event))
        self._save()

    @staticmethod
    def _parse_time(value: object) -> datetime | None:
        try:
            parsed = datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
        return parsed

    def _totals_since(
        self,
        records: list,
        period_start: datetime,
        now: datetime,
    ) -> tuple[float, float, float, float]:
        played = low_energy = high_energy = dose = 0.0
        for event in records:
            if not isinstance(event, dict):
                continue
            started = self._parse_time(event.get("started_at"))
            ended = self._parse_time(event.get("ended_at")) or now
            if started is None or ended <= period_start or ended <= started:
                continue
            total_wall = (ended - started).total_seconds()
            overlap = (min(ended, now) - max(started, period_start)).total_seconds()
            if overlap <= 0.0 or total_wall <= 0.0:
                continue
            share = min(1.0, overlap / total_wall)
            event_seconds = float(event.get("played_seconds", 0.0) or 0.0) * share
            if event_seconds <= 0.0:
                continue
            played += event_seconds
            if "low_energy_seconds" in event:
                low_energy += float(event.get("low_energy_seconds", 0.0) or 0.0) * share
                high_energy += float(event.get("high_energy_seconds", 0.0) or 0.0) * share
                dose += float(event.get("who_dose_fraction", 0.0) or 0.0) * share
            else:
                low_level = event.get("estimated_laeq_low_db")
                high_level = event.get("estimated_laeq_high_db")
                if isinstance(low_level, (int, float)):
                    low_energy += event_seconds * self._energy(float(low_level))
                if isinstance(high_level, (int, float)):
                    high_energy += event_seconds * self._energy(float(high_level))
                dose += float(event.get("who_dose_percent", 0.0) or 0.0) * share / 100.0
        return played, low_energy, high_energy, dose

    @staticmethod
    def _laeq(energy_seconds: float, played_seconds: float) -> float | None:
        if played_seconds <= 0.0 or energy_seconds <= 0.0:
            return None
        return 10.0 * math.log10(energy_seconds / played_seconds)

    def summary(self) -> ExposureSummary:
        now = datetime.now().astimezone()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today_start - timedelta(days=today_start.weekday())
        records = list(self._data["events"])
        if self._active is not None:
            records.append(self._active)
        today = self._totals_since(records, today_start, now)
        week = self._totals_since(records, week_start, now)
        return ExposureSummary(
            today_played_seconds=today[0],
            today_laeq_low_db=self._laeq(today[1], today[0]),
            today_laeq_high_db=self._laeq(today[2], today[0]),
            week_played_seconds=week[0],
            week_laeq_low_db=self._laeq(week[1], week[0]),
            week_laeq_high_db=self._laeq(week[2], week[0]),
            week_dose_percent=week[3] * 100.0,
        )


class PcmLevelEnvelope:
    """Short-window RMS levels calculated from a pygame Sound."""

    def __init__(self, levels_dbfs: np.ndarray, seconds_per_level: float) -> None:
        self._levels_dbfs = levels_dbfs
        self._seconds_per_level = seconds_per_level

    @classmethod
    def from_sound(
        cls, sound: pygame.mixer.Sound, window_seconds: float = 0.5
    ) -> PcmLevelEnvelope | None:
        mixer_format = pygame.mixer.get_init()
        if mixer_format is None:
            return None
        sample_rate, sample_format, channels = mixer_format
        # headphones_markov initializes pygame as signed 16-bit PCM.  Refuse an
        # unexpected format instead of silently calculating the wrong level.
        if sample_format != -16 or channels < 1:
            return None

        samples = np.frombuffer(sound.get_raw(), dtype=np.int16)
        complete_samples = (samples.size // channels) * channels
        if complete_samples == 0:
            return None
        frames = samples[:complete_samples].reshape(-1, channels)
        window_frames = max(1, int(sample_rate * window_seconds))
        filter_b, filter_a = _a_weighting_filter(float(sample_rate))
        filter_state = [
            np.zeros(max(len(filter_a), len(filter_b)) - 1, dtype=np.float64)
            for _ in range(channels)
        ]
        levels: list[float] = []
        for start in range(0, frames.shape[0], window_frames):
            block = (
                frames[start : start + window_frames].astype(np.float64)
                / 32768.0
            )
            if block.size == 0:
                continue
            weighted = np.empty_like(block)
            for channel in range(channels):
                weighted[:, channel], filter_state[channel] = lfilter(
                    filter_b,
                    filter_a,
                    block[:, channel],
                    zi=filter_state[channel],
                )
            channel_rms = np.sqrt(np.mean(np.square(weighted), axis=0))
            rms = float(np.max(channel_rms))
            levels.append(
                SILENCE_DB if rms <= 0.0 else 20.0 * math.log10(rms)
            )
        if not levels:
            return None
        return cls(np.asarray(levels, dtype=np.float32), window_seconds)

    def dbfs_at(self, elapsed_seconds: float) -> float:
        index = int(max(0.0, elapsed_seconds) / self._seconds_per_level)
        index = min(index, len(self._levels_dbfs) - 1)
        return float(self._levels_dbfs[index])


def _a_weighting_filter(sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a digital IEC-style A-weighting filter for the mixer sample rate."""
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    a1000 = 1.9997
    numerator = [
        (2.0 * math.pi * f4) ** 2 * (10.0 ** (a1000 / 20.0)),
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    denominator = np.polymul(
        [1.0, 4.0 * math.pi * f4, (2.0 * math.pi * f4) ** 2],
        [1.0, 4.0 * math.pi * f1, (2.0 * math.pi * f1) ** 2],
    )
    denominator = np.polymul(
        np.polymul(denominator, [1.0, 2.0 * math.pi * f3]),
        [1.0, 2.0 * math.pi * f2],
    )
    return bilinear(numerator, denominator, sample_rate)


if sys.platform == "darwin":
    _CORE_AUDIO = ctypes.CDLL(
        "/System/Library/Frameworks/CoreAudio.framework/CoreAudio"
    )
    _CORE_FOUNDATION = ctypes.CDLL(
        "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
    )

    _CORE_AUDIO.AudioObjectHasProperty.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(_AudioObjectPropertyAddress),
    ]
    _CORE_AUDIO.AudioObjectHasProperty.restype = ctypes.c_ubyte
    _CORE_AUDIO.AudioObjectGetPropertyDataSize.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(_AudioObjectPropertyAddress),
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    _CORE_AUDIO.AudioObjectGetPropertyDataSize.restype = ctypes.c_int32
    _CORE_AUDIO.AudioObjectGetPropertyData.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(_AudioObjectPropertyAddress),
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
    ]
    _CORE_AUDIO.AudioObjectGetPropertyData.restype = ctypes.c_int32

    _CORE_FOUNDATION.CFStringGetLength.argtypes = [ctypes.c_void_p]
    _CORE_FOUNDATION.CFStringGetLength.restype = ctypes.c_long
    _CORE_FOUNDATION.CFStringGetMaximumSizeForEncoding.argtypes = [
        ctypes.c_long,
        ctypes.c_uint32,
    ]
    _CORE_FOUNDATION.CFStringGetMaximumSizeForEncoding.restype = ctypes.c_long
    _CORE_FOUNDATION.CFStringGetCString.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_long,
        ctypes.c_uint32,
    ]
    _CORE_FOUNDATION.CFStringGetCString.restype = ctypes.c_ubyte
    _CORE_FOUNDATION.CFRelease.argtypes = [ctypes.c_void_p]


_SYSTEM_OBJECT = 1
_GLOBAL_SCOPE = _fourcc("glob")
_OUTPUT_SCOPE = _fourcc("outp")
_DEVICES = _fourcc("dev#")
_OBJECT_NAME = _fourcc("lnam")
_VIRTUAL_MAIN_VOLUME = _fourcc("vmvc")
_VOLUME_SCALAR = _fourcc("volm")
_VOLUME_DECIBELS = _fourcc("vold")
_MUTE = _fourcc("mute")
_UTF8 = 0x08000100


def _read_audio_property(
    object_id: int,
    selector: int,
    scope: int,
    element: int,
    value_type: type[ctypes._SimpleCData],  # type: ignore[attr-defined]
) -> int | float | None:
    if sys.platform != "darwin":
        return None
    address = _AudioObjectPropertyAddress(selector, scope, element)
    if not _CORE_AUDIO.AudioObjectHasProperty(object_id, ctypes.byref(address)):
        return None
    value = value_type()
    size = ctypes.c_uint32(ctypes.sizeof(value))
    status = _CORE_AUDIO.AudioObjectGetPropertyData(
        object_id,
        ctypes.byref(address),
        0,
        None,
        ctypes.byref(size),
        ctypes.byref(value),
    )
    return value.value if status == 0 else None


def _read_cfstring(object_id: int, selector: int) -> str | None:
    if sys.platform != "darwin":
        return None
    pointer = _read_audio_property(
        object_id, selector, _GLOBAL_SCOPE, 0, ctypes.c_void_p
    )
    if not pointer:
        return None
    try:
        length = _CORE_FOUNDATION.CFStringGetLength(pointer)
        size = (
            _CORE_FOUNDATION.CFStringGetMaximumSizeForEncoding(length, _UTF8) + 1
        )
        buffer = ctypes.create_string_buffer(size)
        if not _CORE_FOUNDATION.CFStringGetCString(
            pointer, buffer, size, _UTF8
        ):
            return None
        return buffer.value.decode("utf-8")
    finally:
        _CORE_FOUNDATION.CFRelease(pointer)


def output_devices() -> list[OutputDevice]:
    if sys.platform != "darwin":
        return []
    address = _AudioObjectPropertyAddress(_DEVICES, _GLOBAL_SCOPE, 0)
    size = ctypes.c_uint32()
    status = _CORE_AUDIO.AudioObjectGetPropertyDataSize(
        _SYSTEM_OBJECT, ctypes.byref(address), 0, None, ctypes.byref(size)
    )
    if status != 0 or size.value == 0:
        return []
    count = size.value // ctypes.sizeof(ctypes.c_uint32)
    values = (ctypes.c_uint32 * count)()
    status = _CORE_AUDIO.AudioObjectGetPropertyData(
        _SYSTEM_OBJECT,
        ctypes.byref(address),
        0,
        None,
        ctypes.byref(size),
        values,
    )
    if status != 0:
        return []
    return [
        OutputDevice(
            int(object_id),
            _read_cfstring(int(object_id), _OBJECT_NAME) or "",
        )
        for object_id in values
    ]


def find_output_device(name: str) -> OutputDevice | None:
    devices = output_devices()
    exact = next((device for device in devices if device.name == name), None)
    if exact is not None:
        return exact
    wanted = name.casefold()
    return next(
        (device for device in devices if wanted in device.name.casefold()), None
    )


def _first_output_property(
    device: OutputDevice,
    selector: int,
    value_type: type[ctypes._SimpleCData],  # type: ignore[attr-defined]
) -> int | float | None:
    for element in (0, 1, 2):
        value = _read_audio_property(
            device.object_id, selector, _OUTPUT_SCOPE, element, value_type
        )
        if value is not None:
            return value
    return None


def device_volume(device: OutputDevice) -> tuple[float | None, float | None]:
    scalar = _first_output_property(device, _VIRTUAL_MAIN_VOLUME, ctypes.c_float)
    if scalar is None:
        scalar = _first_output_property(device, _VOLUME_SCALAR, ctypes.c_float)
    decibels = _first_output_property(device, _VOLUME_DECIBELS, ctypes.c_float)
    muted = _first_output_property(device, _MUTE, ctypes.c_uint32) == 1
    scalar_value = float(scalar) if scalar is not None else None
    percent = scalar_value * 100.0 if scalar_value is not None else None
    if muted or scalar_value == 0.0:
        return percent, SILENCE_DB
    if decibels is not None and math.isfinite(float(decibels)):
        return percent, float(decibels)
    if scalar_value is not None and scalar_value > 0.0:
        return percent, 20.0 * math.log10(scalar_value)
    return percent, None


class PlayerEarMeter:
    """Combine decoded PCM, player gain, and output volume without audio capture."""

    def __init__(
        self,
        output_device_name: str,
        exposure_log_path: str | Path | None = None,
    ) -> None:
        self._output_device_name = output_device_name
        self._device: OutputDevice | None = None
        self._cached_volume: tuple[float | None, float | None] = (None, None)
        self._last_volume_read = 0.0
        self._last_exposure_update: float | None = None
        self._cached_summary: ExposureSummary | None = None
        self._last_summary_read = 0.0
        self._recorder = (
            ExposureRecorder(exposure_log_path)
            if exposure_log_path is not None
            else None
        )

    def begin_track(self) -> None:
        """Avoid counting silence between tracks as headphone exposure."""
        self._last_exposure_update = None
        self._last_summary_read = 0.0
        if self._recorder is not None:
            self._recorder.begin_track()

    def finish_track(self) -> None:
        self._last_exposure_update = None
        self._last_summary_read = 0.0
        if self._recorder is not None:
            self._recorder.finish_track()

    def exposure_summary(self) -> ExposureSummary | None:
        if self._recorder is None:
            return None
        now = time.monotonic()
        if self._cached_summary is None or now - self._last_summary_read >= 0.5:
            self._cached_summary = self._recorder.summary()
            self._last_summary_read = now
        return self._cached_summary

    def _volume(self) -> tuple[float | None, float | None]:
        now = time.monotonic()
        if now - self._last_volume_read < 0.5:
            return self._cached_volume
        self._last_volume_read = now
        if self._device is None:
            self._device = find_output_device(self._output_device_name)
        if self._device is None:
            self._cached_volume = (None, None)
        else:
            self._cached_volume = device_volume(self._device)
        return self._cached_volume

    def estimate(
        self, pcm_dbfs: float, player_volume_scale: float
    ) -> ListeningEstimate | None:
        percent, output_attenuation_db = self._volume()
        if output_attenuation_db is None or player_volume_scale <= 0.0:
            return None
        player_gain_db = 20.0 * math.log10(player_volume_scale)
        electrical_level = pcm_dbfs + player_gain_db + output_attenuation_db
        high_db_spl = electrical_level + APPLE_WIRED_OFFSET_HIGH_DB
        now = time.monotonic()
        elapsed = 0.0
        if self._last_exposure_update is not None:
            elapsed = max(0.0, min(now - self._last_exposure_update, 1.0))
        self._last_exposure_update = now
        estimate = ListeningEstimate(
            electrical_level + APPLE_WIRED_OFFSET_LOW_DB,
            high_db_spl,
            percent,
        )
        if self._recorder is not None:
            self._recorder.update(estimate, elapsed)
        return estimate
