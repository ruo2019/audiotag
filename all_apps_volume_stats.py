#!/usr/bin/env python3
"""Live headphone-volume estimates for all audio-playing macOS apps.

This is a standalone companion to headphones_markov.py's headphone listening
stats.  It captures the system-audio mix with ScreenCaptureKit, A-weights the
PCM, applies the current CoreAudio output-volume attenuation, and maintains
today/week exposure totals in a JSON file.

The first run compiles the embedded Swift capture helper with Apple's bundled
Swift compiler.  macOS will ask for Screen & System Audio Recording permission.
No Python packages are required.
"""

from __future__ import annotations

import argparse
import ctypes
import gzip
import hashlib
import json
import math
import os
import platform
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


SILENCE_DB = -120.0
DEFAULT_OFFSET_LOW_DB = 121.0
DEFAULT_OFFSET_HIGH_DB = 132.0
DEFAULT_LOG = Path(__file__).with_name("headphone_exposure")


# ScreenCaptureKit produces the mixed app audio.  The helper deliberately emits
# only a 0.5-second A-weighted RMS number, never audio samples or a recording.
SWIFT_CAPTURE_SOURCE = r'''
import Foundation
import ScreenCaptureKit
import CoreMedia
import AudioToolbox

private let numerator: [Double] = [
    0.23430179229951348, -0.46860358459902696,
    -0.23430179229951348, 0.9372071691980539,
    -0.23430179229951348, -0.46860358459902696,
    0.23430179229951348
]
private let denominator: [Double] = [
    1.0, -4.113043408775871, 6.553121752655046,
    -4.990849294163378, 1.785737302937571,
    -0.2461905953194862, 0.011224250033231168
]

private struct IIRFilter {
    var x = [Double](repeating: 0.0, count: 6)
    var y = [Double](repeating: 0.0, count: 6)

    mutating func process(_ sample: Double) -> Double {
        var output = numerator[0] * sample
        for index in 1..<7 {
            output += numerator[index] * x[index - 1]
            output -= denominator[index] * y[index - 1]
        }
        for index in stride(from: 5, through: 1, by: -1) {
            x[index] = x[index - 1]
            y[index] = y[index - 1]
        }
        x[0] = sample
        y[0] = output
        return output
    }
}

final class AudioReceiver: NSObject, SCStreamOutput, SCStreamDelegate, @unchecked Sendable {
    private var filters = [IIRFilter(), IIRFilter()]
    private var energy = [Double](repeating: 0.0, count: 2)
    private var windowFrames = 0
    private let framesPerWindow = 24_000  // 0.5 seconds at requested 48 kHz

    func stream(_ stream: SCStream, didStopWithError error: any Error) {
        FileHandle.standardError.write(Data("capture stopped: \(error)\n".utf8))
        fflush(stderr)
        exit(2)
    }

    func stream(
        _ stream: SCStream,
        didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of outputType: SCStreamOutputType
    ) {
        guard outputType == .audio, CMSampleBufferDataIsReady(sampleBuffer) else { return }
        guard let description = CMSampleBufferGetFormatDescription(sampleBuffer),
              let formatPointer = CMAudioFormatDescriptionGetStreamBasicDescription(description)
        else { return }
        let format = formatPointer.pointee
        let frames = CMSampleBufferGetNumSamples(sampleBuffer)
        guard frames > 0 else { return }

        var needed = 0
        var retainedBlock: CMBlockBuffer?
        var status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer,
            bufferListSizeNeededOut: &needed,
            bufferListOut: nil,
            bufferListSize: 0,
            blockBufferAllocator: kCFAllocatorDefault,
            blockBufferMemoryAllocator: kCFAllocatorDefault,
            flags: UInt32(kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment),
            blockBufferOut: &retainedBlock
        )
        guard status == noErr, needed > 0 else { return }
        let storage = UnsafeMutableRawPointer.allocate(
            byteCount: needed,
            alignment: MemoryLayout<AudioBufferList>.alignment
        )
        defer { storage.deallocate() }
        let list = storage.assumingMemoryBound(to: AudioBufferList.self)
        status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer,
            bufferListSizeNeededOut: &needed,
            bufferListOut: list,
            bufferListSize: needed,
            blockBufferAllocator: kCFAllocatorDefault,
            blockBufferMemoryAllocator: kCFAllocatorDefault,
            flags: UInt32(kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment),
            blockBufferOut: &retainedBlock
        )
        guard status == noErr else { return }

        let audioBuffers = UnsafeMutableAudioBufferListPointer(list)
        let isFloat = (format.mFormatFlags & kAudioFormatFlagIsFloat) != 0
        let isSignedInteger = (format.mFormatFlags & kAudioFormatFlagIsSignedInteger) != 0
        let isNonInterleaved = (format.mFormatFlags & kAudioFormatFlagIsNonInterleaved) != 0
        guard format.mFormatID == kAudioFormatLinearPCM, isFloat || isSignedInteger else { return }

        for frame in 0..<frames {
            for channel in 0..<2 {
                let bufferIndex = isNonInterleaved ? min(channel, audioBuffers.count - 1) : 0
                guard bufferIndex >= 0, let data = audioBuffers[bufferIndex].mData else { continue }
                let channelsInBuffer = max(1, Int(audioBuffers[bufferIndex].mNumberChannels))
                let sampleIndex = isNonInterleaved ? frame : frame * channelsInBuffer + min(channel, channelsInBuffer - 1)
                let sample: Double
                if isFloat && format.mBitsPerChannel == 32 {
                    sample = Double(data.assumingMemoryBound(to: Float.self)[sampleIndex])
                } else if isFloat && format.mBitsPerChannel == 64 {
                    sample = data.assumingMemoryBound(to: Double.self)[sampleIndex]
                } else if isSignedInteger && format.mBitsPerChannel == 16 {
                    sample = Double(data.assumingMemoryBound(to: Int16.self)[sampleIndex]) / 32768.0
                } else if isSignedInteger && format.mBitsPerChannel == 32 {
                    sample = Double(data.assumingMemoryBound(to: Int32.self)[sampleIndex]) / 2147483648.0
                } else {
                    continue
                }
                let weighted = filters[channel].process(sample)
                energy[channel] += weighted * weighted
            }
            windowFrames += 1
            if windowFrames >= framesPerWindow {
                let rms = sqrt(max(energy[0], energy[1]) / Double(windowFrames))
                let db = rms > 0 ? 20.0 * log10(rms) : -120.0
                print(String(format: "%.6f", max(-120.0, db)))
                fflush(stdout)
                energy = [0.0, 0.0]
                windowFrames = 0
            }
        }
    }
}

@main
struct CaptureMain {
    static func main() async {
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(
                false,
                onScreenWindowsOnly: false
            )
            guard let display = content.displays.first else {
                throw NSError(
                    domain: "AllAppsAudioMeter",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "No display is available for ScreenCaptureKit"]
                )
            }
            let filter = SCContentFilter(
                display: display,
                excludingApplications: [],
                exceptingWindows: []
            )
            let configuration = SCStreamConfiguration()
            configuration.capturesAudio = true
            configuration.excludesCurrentProcessAudio = true
            configuration.sampleRate = 48_000
            configuration.channelCount = 2
            configuration.width = 2
            configuration.height = 2
            configuration.minimumFrameInterval = CMTime(value: 1, timescale: 1)
            configuration.queueDepth = 3

            let receiver = AudioReceiver()
            let stream = SCStream(filter: filter, configuration: configuration, delegate: receiver)
            let queue = DispatchQueue(label: "all-apps-audio-meter.samples")
            try stream.addStreamOutput(receiver, type: .audio, sampleHandlerQueue: queue)
            try await stream.startCapture()
            FileHandle.standardError.write(Data("READY\n".utf8))
            fflush(stderr)
            // Keep the async main task, stream, and receiver alive. Calling
            // dispatchMain() here is invalid when Swift resumes this task on
            // the main queue and causes libdispatch to terminate with SIGTRAP.
            while true {
                try await Task.sleep(nanoseconds: 3_600_000_000_000)
            }
        } catch {
            FileHandle.standardError.write(Data("capture setup failed: \(error)\n".utf8))
            fflush(stderr)
            exit(2)
        }
    }
}
'''


def _fourcc(value: str) -> int:
    return int.from_bytes(value.encode("ascii"), "big")


class AudioObjectPropertyAddress(ctypes.Structure):
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


class CoreAudio:
    SYSTEM_OBJECT = 1
    GLOBAL_SCOPE = _fourcc("glob")
    OUTPUT_SCOPE = _fourcc("outp")
    DEVICES = _fourcc("dev#")
    DEFAULT_OUTPUT = _fourcc("dOut")
    OBJECT_NAME = _fourcc("lnam")
    VIRTUAL_MAIN_VOLUME = _fourcc("vmvc")
    VOLUME_SCALAR = _fourcc("volm")
    VOLUME_DECIBELS = _fourcc("vold")
    MUTE = _fourcc("mute")
    UTF8 = 0x08000100

    def __init__(self) -> None:
        self.audio = ctypes.CDLL(
            "/System/Library/Frameworks/CoreAudio.framework/CoreAudio"
        )
        self.cf = ctypes.CDLL(
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
        )
        self.audio.AudioObjectHasProperty.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(AudioObjectPropertyAddress),
        ]
        self.audio.AudioObjectHasProperty.restype = ctypes.c_ubyte
        self.audio.AudioObjectGetPropertyDataSize.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(AudioObjectPropertyAddress),
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.audio.AudioObjectGetPropertyDataSize.restype = ctypes.c_int32
        self.audio.AudioObjectGetPropertyData.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(AudioObjectPropertyAddress),
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_void_p,
        ]
        self.audio.AudioObjectGetPropertyData.restype = ctypes.c_int32
        self.cf.CFStringGetLength.argtypes = [ctypes.c_void_p]
        self.cf.CFStringGetLength.restype = ctypes.c_long
        self.cf.CFStringGetMaximumSizeForEncoding.argtypes = [
            ctypes.c_long,
            ctypes.c_uint32,
        ]
        self.cf.CFStringGetMaximumSizeForEncoding.restype = ctypes.c_long
        self.cf.CFStringGetCString.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_long,
            ctypes.c_uint32,
        ]
        self.cf.CFStringGetCString.restype = ctypes.c_ubyte
        self.cf.CFRelease.argtypes = [ctypes.c_void_p]

    def _read(
        self,
        object_id: int,
        selector: int,
        scope: int,
        element: int,
        value_type: type[ctypes._SimpleCData],  # type: ignore[attr-defined]
    ) -> int | float | None:
        address = AudioObjectPropertyAddress(selector, scope, element)
        if not self.audio.AudioObjectHasProperty(object_id, ctypes.byref(address)):
            return None
        value = value_type()
        size = ctypes.c_uint32(ctypes.sizeof(value))
        status = self.audio.AudioObjectGetPropertyData(
            object_id,
            ctypes.byref(address),
            0,
            None,
            ctypes.byref(size),
            ctypes.byref(value),
        )
        return value.value if status == 0 else None

    def _name(self, object_id: int) -> str | None:
        pointer = self._read(
            object_id, self.OBJECT_NAME, self.GLOBAL_SCOPE, 0, ctypes.c_void_p
        )
        if not pointer:
            return None
        try:
            length = self.cf.CFStringGetLength(pointer)
            size = self.cf.CFStringGetMaximumSizeForEncoding(length, self.UTF8) + 1
            buffer = ctypes.create_string_buffer(size)
            if not self.cf.CFStringGetCString(pointer, buffer, size, self.UTF8):
                return None
            return buffer.value.decode("utf-8")
        finally:
            self.cf.CFRelease(pointer)

    def devices(self) -> list[OutputDevice]:
        address = AudioObjectPropertyAddress(self.DEVICES, self.GLOBAL_SCOPE, 0)
        size = ctypes.c_uint32()
        status = self.audio.AudioObjectGetPropertyDataSize(
            self.SYSTEM_OBJECT, ctypes.byref(address), 0, None, ctypes.byref(size)
        )
        if status != 0 or size.value == 0:
            return []
        count = size.value // ctypes.sizeof(ctypes.c_uint32)
        values = (ctypes.c_uint32 * count)()
        status = self.audio.AudioObjectGetPropertyData(
            self.SYSTEM_OBJECT,
            ctypes.byref(address),
            0,
            None,
            ctypes.byref(size),
            values,
        )
        if status != 0:
            return []
        return [
            OutputDevice(int(value), self._name(int(value)) or "(unnamed)")
            for value in values
        ]

    def default_output(self) -> OutputDevice | None:
        object_id = self._read(
            self.SYSTEM_OBJECT,
            self.DEFAULT_OUTPUT,
            self.GLOBAL_SCOPE,
            0,
            ctypes.c_uint32,
        )
        if object_id is None or int(object_id) == 0:
            return None
        return OutputDevice(int(object_id), self._name(int(object_id)) or "(unnamed)")

    def find_device(self, wanted: str) -> OutputDevice | None:
        devices = self.devices()
        exact = next((device for device in devices if device.name == wanted), None)
        if exact is not None:
            return exact
        folded = wanted.casefold()
        return next(
            (device for device in devices if folded in device.name.casefold()), None
        )

    def _first_property(
        self,
        device: OutputDevice,
        selector: int,
        value_type: type[ctypes._SimpleCData],  # type: ignore[attr-defined]
    ) -> int | float | None:
        for element in (0, 1, 2):
            value = self._read(
                device.object_id,
                selector,
                self.OUTPUT_SCOPE,
                element,
                value_type,
            )
            if value is not None:
                return value
        return None

    def volume(self, device: OutputDevice) -> tuple[float | None, float | None]:
        scalar = self._first_property(device, self.VIRTUAL_MAIN_VOLUME, ctypes.c_float)
        if scalar is None:
            scalar = self._first_property(device, self.VOLUME_SCALAR, ctypes.c_float)
        decibels = self._first_property(device, self.VOLUME_DECIBELS, ctypes.c_float)
        muted = self._first_property(device, self.MUTE, ctypes.c_uint32) == 1
        scalar_value = float(scalar) if scalar is not None else None
        percent = scalar_value * 100.0 if scalar_value is not None else None
        if muted or scalar_value == 0.0:
            return percent, SILENCE_DB
        if decibels is not None and math.isfinite(float(decibels)):
            return percent, float(decibels)
        if scalar_value is not None and scalar_value > 0.0:
            return percent, 20.0 * math.log10(scalar_value)
        return percent, None


def screen_capture_authorized(request_if_needed: bool = True) -> bool:
    """Check the permission ScreenCaptureKit needs before launching its helper."""
    try:
        core_graphics = ctypes.CDLL(
            "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
        )
        preflight = core_graphics.CGPreflightScreenCaptureAccess
        preflight.argtypes = []
        preflight.restype = ctypes.c_bool
        if preflight():
            return True
        if not request_if_needed:
            return False
        request = core_graphics.CGRequestScreenCaptureAccess
        request.argtypes = []
        request.restype = ctypes.c_bool
        return bool(request())
    except (AttributeError, OSError):
        # ScreenCaptureKit will still return its own useful error on systems
        # where the CoreGraphics preflight functions are unavailable.
        return True


class MacSessionLock:
    """Read the current macOS GUI session's screen-lock state."""

    def __init__(self) -> None:
        self.cg = ctypes.CDLL(
            "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
        )
        self.cf = ctypes.CDLL(
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
        )
        self.cg.CGSessionCopyCurrentDictionary.argtypes = []
        self.cg.CGSessionCopyCurrentDictionary.restype = ctypes.c_void_p
        self.cf.CFStringCreateWithCString.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        self.cf.CFStringCreateWithCString.restype = ctypes.c_void_p
        self.cf.CFDictionaryGetValueIfPresent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.cf.CFDictionaryGetValueIfPresent.restype = ctypes.c_ubyte
        self.cf.CFBooleanGetValue.argtypes = [ctypes.c_void_p]
        self.cf.CFBooleanGetValue.restype = ctypes.c_ubyte
        self.cf.CFRelease.argtypes = [ctypes.c_void_p]
        self._locked_key = self.cf.CFStringCreateWithCString(
            None, b"CGSSessionScreenIsLocked", CoreAudio.UTF8
        )

    def is_locked(self) -> bool:
        session = self.cg.CGSessionCopyCurrentDictionary()
        if not session:
            return False
        try:
            value = ctypes.c_void_p()
            found = self.cf.CFDictionaryGetValueIfPresent(
                session, self._locked_key, ctypes.byref(value)
            )
            return bool(found and value.value and self.cf.CFBooleanGetValue(value))
        finally:
            self.cf.CFRelease(session)

    def close(self) -> None:
        if self._locked_key:
            self.cf.CFRelease(self._locked_key)
            self._locked_key = None


class ExposureRecorder:
    """Append-only monthly JSONL storage with one entry per active minute."""

    def __init__(
        self,
        path: Path | None,
        device_name: str | None = None,
        offset_low_db: float = DEFAULT_OFFSET_LOW_DB,
        offset_high_db: float = DEFAULT_OFFSET_HIGH_DB,
    ) -> None:
        self.directory = path
        self.device_name = device_name or "unknown"
        self.offset_low_db = float(offset_low_db)
        self.offset_high_db = float(offset_high_db)
        self._recent_entries: list[dict[str, Any]] = []
        self._current: dict[str, Any] | None = None
        self._active = False
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._migrate_old_json()
            self._archive_completed_months()
            self._load_current_week()

    @staticmethod
    def _energy(level_db: float) -> float:
        return 10.0 ** (max(-200.0, min(200.0, level_db)) / 10.0)

    @staticmethod
    def _parse_time(value: object) -> datetime | None:
        try:
            parsed = datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
        return parsed

    @staticmethod
    def _number(value: object) -> float | None:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        return None

    def _normalize_old_entries(
        self, raw_entries: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        prepared: list[
            tuple[
                datetime,
                float,
                float | None,
                float | None,
                float | None,
                float,
            ]
        ] = []
        for raw in raw_entries:
            timestamp = self._parse_time(raw.get("timestamp"))
            average = raw.get("average_db_a")
            volume = raw.get("mac_volume_percent")
            average_high = (
                self._number(average.get("high"))
                if isinstance(average, dict)
                else self._number(average[1])
                if isinstance(average, list) and len(average) > 1
                else self._number(average)
            )
            volume_average = (
                self._number(volume.get("average"))
                if isinstance(volume, dict)
                else self._number(volume)
            )
            source_average = self._number(
                raw.get("source_average_dbfs_a", raw.get("source_laeq_dbfs_a"))
            )
            if timestamp is None or average_high is None:
                continue
            prepared.append(
                (
                    timestamp,
                    average_high,
                    volume_average,
                    source_average,
                    self._number(raw.get("weekly_dose_percent")),
                    self._number(raw.get("who_dose_percent")) or 0.0,
                )
            )

        prepared.sort(key=lambda item: item[0])
        week: tuple[int, int] | None = None
        cumulative_dose = 0.0
        entries: list[dict[str, Any]] = []
        for timestamp, average, volume, source, cumulative, increment in prepared:
            timestamp_week = (timestamp.isocalendar().year, timestamp.isocalendar().week)
            if timestamp_week != week:
                week = timestamp_week
                cumulative_dose = 0.0
            cumulative_dose = cumulative if cumulative is not None else cumulative_dose + increment
            entries.append(
                {
                    "timestamp": timestamp.isoformat(timespec="seconds"),
                    "average_db_a": round(average, 2),
                    "mac_volume_percent": round(volume, 2) if volume is not None else None,
                    "weekly_dose_percent": round(cumulative_dose, 6),
                    "source_average_dbfs_a": (
                        round(source, 2) if source is not None else None
                    ),
                }
            )
        return entries

    def _migrate_old_json(self) -> None:
        if self.directory is None or self.directory.name != DEFAULT_LOG.name:
            return
        old_path = self.directory.parent / "all_apps_headphone_exposure.json"
        if not old_path.is_file():
            return
        try:
            loaded = json.loads(old_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return
        if not isinstance(loaded, list):
            return
        entries = self._normalize_old_entries(
            [entry for entry in loaded if isinstance(entry, dict)]
        )
        existing = {
            json.dumps(entry, sort_keys=True, separators=(",", ":"))
            for entry in self._iter_entries()
        }
        for entry in entries:
            identity = json.dumps(entry, sort_keys=True, separators=(",", ":"))
            if identity not in existing:
                self._append_line(entry, update_recent=False)
                existing.add(identity)
        backup = old_path.with_suffix(old_path.suffix + ".migrated")
        counter = 1
        while backup.exists():
            backup = old_path.with_suffix(old_path.suffix + f".migrated.{counter}")
            counter += 1
        os.replace(old_path, backup)

    def _month_path(self, timestamp: datetime) -> Path:
        assert self.directory is not None
        return self.directory / f"{timestamp:%Y-%m}.jsonl"

    def _append_line(
        self, entry: dict[str, Any], *, update_recent: bool = True
    ) -> None:
        if self.directory is None:
            if update_recent:
                self._recent_entries.append(entry)
            return
        timestamp = self._parse_time(entry.get("timestamp"))
        if timestamp is None:
            return
        path = self._month_path(timestamp)
        encoded = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
        with path.open("a", encoding="utf-8") as output:
            output.write(encoded + "\n")
            output.flush()
            os.fsync(output.fileno())
        if update_recent:
            self._recent_entries.append(entry)

    def _iter_file(self, path: Path) -> list[dict[str, Any]]:
        opener = gzip.open if path.suffix == ".gz" else open
        entries: list[dict[str, Any]] = []
        try:
            with opener(path, "rt", encoding="utf-8") as source:
                for line in source:
                    try:
                        entry = json.loads(line)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(entry, dict):
                        entries.append(entry)
        except OSError:
            pass
        return entries

    def _iter_entries(self) -> list[dict[str, Any]]:
        if self.directory is None:
            return list(self._recent_entries)
        paths = sorted(
            [*self.directory.glob("????-??.jsonl"), *self.directory.glob("????-??.jsonl.gz")]
        )
        entries: list[dict[str, Any]] = []
        for path in paths:
            entries.extend(self._iter_file(path))
        entries.sort(key=lambda entry: str(entry.get("timestamp", "")))
        return entries

    def _archive_completed_months(self) -> None:
        if self.directory is None:
            return
        current_name = f"{datetime.now().astimezone():%Y-%m}.jsonl"
        for source_path in self.directory.glob("????-??.jsonl"):
            if source_path.name == current_name:
                continue
            archive_path = source_path.with_suffix(source_path.suffix + ".gz")
            if archive_path.exists():
                continue
            temporary = archive_path.with_suffix(archive_path.suffix + ".tmp")
            try:
                with source_path.open("rb") as source, gzip.open(
                    temporary, "wb", compresslevel=9
                ) as archive:
                    shutil.copyfileobj(source, archive)
                os.replace(temporary, archive_path)
                source_path.unlink()
            except OSError:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def _load_current_week(self) -> None:
        now = datetime.now().astimezone()
        week_start = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(
            days=now.weekday()
        )
        self._recent_entries = [
            entry
            for entry in self._iter_entries()
            if (self._parse_time(entry.get("timestamp")) or datetime.min.replace(tzinfo=now.tzinfo))
            >= week_start
        ]

    def _weekly_dose_before(self, timestamp: datetime) -> float:
        target = (timestamp.isocalendar().year, timestamp.isocalendar().week)
        for entry in reversed(self._recent_entries):
            previous = self._parse_time(entry.get("timestamp"))
            if previous is None:
                continue
            if (previous.isocalendar().year, previous.isocalendar().week) == target:
                return float(entry.get("weekly_dose_percent", 0.0) or 0.0)
        return 0.0

    def _public_entry(self, bucket: dict[str, Any]) -> dict[str, Any]:
        seconds = float(bucket["seconds"])

        def laeq(key: str) -> float | None:
            energy = float(bucket[key])
            return (
                10.0 * math.log10(energy / seconds)
                if seconds > 0 and energy > 0
                else None
            )

        timestamp = self._parse_time(bucket["timestamp"]) or datetime.now().astimezone()
        high_average = laeq("high_energy_seconds")
        source_average = laeq("source_energy_seconds")
        volume_seconds = float(bucket["volume_seconds"])
        weekly_dose = self._weekly_dose_before(timestamp) + float(
            bucket["dose_fraction"]
        ) * 100.0
        return {
            "timestamp": bucket["timestamp"],
            "average_db_a": round(
                high_average if high_average is not None else SILENCE_DB, 2
            ),
            "mac_volume_percent": (
                round(float(bucket["volume_sum"]) / volume_seconds, 2)
                if volume_seconds > 0
                else None
            ),
            "weekly_dose_percent": round(weekly_dose, 6),
            "source_average_dbfs_a": (
                round(source_average, 2) if source_average is not None else None
            ),
        }

    def begin(self, initial_elapsed: float = 0.0) -> None:
        del initial_elapsed
        self._active = True

    def _finish_current(self) -> None:
        if self._current is None or self._current["seconds"] <= 0:
            self._current = None
            return
        entry = self._public_entry(self._current)
        self._append_line(entry)
        self._current = None

    def update(
        self,
        estimate: ListeningEstimate,
        elapsed: float,
        pcm_dbfs: float,
        output_attenuation_db: float,
    ) -> None:
        del output_attenuation_db
        if not self._active or elapsed <= 0:
            return
        now = datetime.now().astimezone()
        minute = now.replace(second=0, microsecond=0)
        minute_key = minute.isoformat(timespec="seconds")
        if self._current is not None and self._current["minute"] != minute_key:
            self._finish_current()
            self._archive_completed_months()
        if self._current is None:
            self._current = {
                "timestamp": now.isoformat(timespec="seconds"),
                "minute": minute_key,
                "seconds": 0.0,
                "high_energy_seconds": 0.0,
                "source_energy_seconds": 0.0,
                "volume_seconds": 0.0,
                "volume_sum": 0.0,
                "dose_fraction": 0.0,
            }
        bucket = self._current
        bucket["seconds"] += elapsed
        bucket["high_energy_seconds"] += elapsed * self._energy(estimate.high_db_spl)
        bucket["source_energy_seconds"] += elapsed * self._energy(pcm_dbfs)
        if estimate.mac_volume_percent is not None:
            bucket["volume_seconds"] += elapsed
            bucket["volume_sum"] += elapsed * estimate.mac_volume_percent
        allowance_hours = 40.0 * (2.0 ** ((80.0 - estimate.high_db_spl) / 3.0))
        bucket["dose_fraction"] += elapsed / max(allowance_hours * 3600.0, 0.001)

    def finish(self, reason: str = "monitor_stopped") -> None:
        del reason
        self._active = False
        self._finish_current()

    def summary(self) -> ExposureSummary:
        now = datetime.now().astimezone()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today_start - timedelta(days=today_start.weekday())
        records: list[tuple[dict[str, Any], float]] = [
            (entry, 60.0) for entry in self._recent_entries
        ]
        if self._current is not None:
            records.append(
                (self._public_entry(self._current), float(self._current["seconds"]))
            )

        def totals(start: datetime) -> tuple[float, float, float, float]:
            played = low_energy = high_energy = dose = 0.0
            for entry, seconds in records:
                timestamp = self._parse_time(entry.get("timestamp"))
                high = self._number(entry.get("average_db_a"))
                if timestamp is None or timestamp < start or high is None:
                    continue
                low = high - (self.offset_high_db - self.offset_low_db)
                played += seconds
                low_energy += seconds * self._energy(low)
                high_energy += seconds * self._energy(high)
                dose = max(
                    dose,
                    float(entry.get("weekly_dose_percent", 0.0) or 0.0) / 100.0,
                )
            return played, low_energy, high_energy, dose

        today = totals(today_start)
        week = totals(week_start)

        def laeq(energy_seconds: float, seconds: float) -> float | None:
            return (
                10.0 * math.log10(energy_seconds / seconds)
                if seconds > 0 and energy_seconds > 0
                else None
            )

        return ExposureSummary(
            today[0],
            laeq(today[1], today[0]),
            laeq(today[2], today[0]),
            week[0],
            laeq(week[1], week[0]),
            laeq(week[2], week[0]),
            week[3] * 100.0,
        )


def helper_path() -> Path:
    digest = hashlib.sha256(SWIFT_CAPTURE_SOURCE.encode()).hexdigest()[:16]
    cache = Path.home() / "Library" / "Caches" / "all-apps-volume-stats"
    return cache / f"capture-{digest}"


def build_helper() -> Path:
    executable = helper_path()
    if executable.is_file() and os.access(executable, os.X_OK):
        return executable
    swiftc = shutil.which("swiftc") or "/usr/bin/swiftc"
    if not Path(swiftc).exists():
        raise RuntimeError("Apple's Swift compiler was not found (install Xcode Command Line Tools)")
    executable.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="all-apps-volume-stats-") as temp_dir:
        source = Path(temp_dir) / "capture.swift"
        source.write_text(SWIFT_CAPTURE_SOURCE, encoding="utf-8")
        temporary_executable = Path(temp_dir) / "capture"
        result = subprocess.run(
            [swiftc, "-O", "-parse-as-library", str(source), "-o", str(temporary_executable)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            details = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"Could not compile the embedded capture helper:\n{details}")
        os.replace(temporary_executable, executable)
        executable.chmod(0o755)
    return executable


def estimate_level(
    pcm_dbfs: float,
    volume_attenuation_db: float,
    volume_percent: float | None,
    offset_low: float,
    offset_high: float,
) -> ListeningEstimate:
    electrical = pcm_dbfs + volume_attenuation_db
    return ListeningEstimate(
        electrical + offset_low,
        electrical + offset_high,
        volume_percent,
    )


def format_duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {remaining_seconds:02d}s"


def level_range(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "not enough level data yet"
    return f"{low:.0f}–{high:.0f} dB(A)"


def render_text(
    device: OutputDevice,
    estimate: ListeningEstimate | None,
    summary: ExposureSummary,
    status: str,
    log_path: Path | None,
) -> str:
    if estimate is None:
        current = "Current: no audible system audio"
    else:
        volume = (
            f"{estimate.mac_volume_percent:.0f}%"
            if estimate.mac_volume_percent is not None
            else "unknown"
        )
        current = (
            f"Current: {estimate.low_db_spl:.1f}–{estimate.high_db_spl:.1f} dB(A) estimated"
            f" | Mac volume {volume}"
        )
    dose = summary.week_dose_percent
    dose_text = "<0.01%" if dose < 0.01 else f"{dose:.2f}%"
    allowance = f"{dose - 100:.1f}% over" if dose > 100 else f"{100 - dose:.1f}% remaining"
    margin_line = "70 dB(A) target margin: waiting for audio"
    if estimate is not None:
        difference = 70.0 - estimate.high_db_spl
        margin = f"{difference:.0f} dB below" if difference >= 0 else f"{-difference:.0f} dB above"
        margin_line = f"70 dB(A) target margin (upper estimate): {margin}"
    return "\n".join(
        [
            "All-app Headphone Listening Stats",
            f"Source: all app audio | Output: {device.name} | {status}",
            "",
            current,
            (
                f"Today: {format_duration(summary.today_played_seconds)}"
                f" | estimated average {level_range(summary.today_laeq_low_db, summary.today_laeq_high_db)}"
            ),
            (
                f"This week (Mon–now): {format_duration(summary.week_played_seconds)}"
                f" | estimated average {level_range(summary.week_laeq_low_db, summary.week_laeq_high_db)}"
            ),
            f"WHO weekly allowance: {dose_text} used | {allowance}",
            margin_line,
            "",
            f"Exposure log: {log_path if log_path is not None else 'disabled'}",
            "Press Ctrl+C to stop.",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate headphone listening levels from all macOS app audio."
    )
    parser.add_argument(
        "--device",
        help="CoreAudio output device name (default: current system output)",
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="list CoreAudio devices and exit"
    )
    parser.add_argument(
        "--log",
        "--log-dir",
        dest="log",
        type=Path,
        default=DEFAULT_LOG,
        help=f"monthly JSONL directory (default: {DEFAULT_LOG.name})",
    )
    parser.add_argument("--no-log", action="store_true", help="do not persist exposure")
    parser.add_argument(
        "--offset-low",
        type=float,
        default=DEFAULT_OFFSET_LOW_DB,
        help="low acoustic calibration offset in dB",
    )
    parser.add_argument(
        "--offset-high",
        type=float,
        default=DEFAULT_OFFSET_HIGH_DB,
        help="high acoustic calibration offset in dB",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=-80.0,
        metavar="DBFS",
        help="A-weighted level at/below which audio is idle (default: -80)",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit one JSON object per measurement"
    )
    parser.add_argument(
        "--prepare", action="store_true", help="compile the native helper and exit"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if sys.platform != "darwin":
        print("error: this script requires macOS 14.2 or newer", file=sys.stderr)
        return 2
    version = tuple(int(part) for part in platform.mac_ver()[0].split(".")[:2])
    if version and version < (14, 2):
        print("error: system-audio capture requires macOS 14.2 or newer", file=sys.stderr)
        return 2

    core_audio = CoreAudio()
    if args.list_devices:
        default = core_audio.default_output()
        for device in core_audio.devices():
            marker = " *" if default and device.object_id == default.object_id else ""
            print(f"{device.name}{marker}")
        print("* current default output")
        return 0

    try:
        helper = build_helper()
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.prepare:
        print(helper)
        return 0

    if not screen_capture_authorized():
        print(
            "error: macOS reports that screen/system-audio capture permission is not "
            "active for the app hosting this shell. Fully quit that app (Cmd+Q), reopen "
            "it, and try again after enabling Screen & System Audio Recording.",
            file=sys.stderr,
        )
        return 2

    device = core_audio.find_device(args.device) if args.device else core_audio.default_output()
    if device is None:
        requested = repr(args.device) if args.device else "the default output"
        print(f"error: could not find {requested}; use --list-devices", file=sys.stderr)
        return 2

    _initial_percent, initial_attenuation = core_audio.volume(device)
    if initial_attenuation is None:
        print(
            f"error: {device.name!r} does not expose a software volume level; "
            "choose a volume-controllable headphone output with --device",
            file=sys.stderr,
        )
        return 2

    log_path = None if args.no_log else args.log.expanduser().resolve()
    recorder = ExposureRecorder(
        log_path,
        device_name=device.name,
        offset_low_db=args.offset_low,
        offset_high_db=args.offset_high,
    )
    selector = selectors.DefaultSelector()
    session_lock = MacSessionLock()
    process: subprocess.Popen[str] | None = None
    estimate: ListeningEstimate | None = None
    last_active_at: float | None = None
    last_measurement_at: float | None = None
    ready = False
    helper_messages: list[str] = []
    interactive = sys.stdout.isatty() and not args.json

    def start_capture() -> None:
        nonlocal process, ready, helper_messages, last_measurement_at
        process = subprocess.Popen(
            [str(helper)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None and process.stderr is not None
        selector.register(process.stdout, selectors.EVENT_READ, "level")
        selector.register(process.stderr, selectors.EVENT_READ, "message")
        ready = False
        helper_messages = []
        last_measurement_at = None

    def stop_capture() -> None:
        nonlocal process, ready
        if process is None:
            return
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    selector.unregister(stream)
                except KeyError:
                    pass
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        process = None
        ready = False

    def show(status: str) -> None:
        summary = recorder.summary()
        if args.json:
            payload = {
                "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "device": device.name,
                "status": status,
                "current": asdict(estimate) if estimate is not None else None,
                "exposure": asdict(summary),
            }
            print(json.dumps(payload), flush=True)
            return
        output = render_text(device, estimate, summary, status, log_path)
        if interactive:
            print("\033[2J\033[H" + output, end="", flush=True)
        else:
            print(output, flush=True)

    screen_locked = session_lock.is_locked()
    if not screen_locked:
        start_capture()
    show("paused while Mac is locked" if screen_locked else "requesting capture permission…")
    exit_code = 0
    try:
        while True:
            events = selector.select(timeout=0.6)
            now = time.monotonic()

            locked_now = session_lock.is_locked()
            if locked_now != screen_locked:
                screen_locked = locked_now
                estimate = None
                last_active_at = None
                last_measurement_at = None
                if screen_locked:
                    recorder.finish("screen_locked")
                    stop_capture()
                    show("paused while Mac is locked")
                else:
                    start_capture()
                    show("resuming capture…")
            if screen_locked:
                continue

            for key, _ in events:
                line = key.fileobj.readline()
                if not line:
                    try:
                        selector.unregister(key.fileobj)
                    except KeyError:
                        pass
                    continue
                line = line.strip()
                if key.data == "message":
                    if line == "READY":
                        ready = True
                    elif line:
                        helper_messages.append(line)
                    continue
                try:
                    pcm_dbfs = float(line)
                except ValueError:
                    continue
                percent, attenuation = core_audio.volume(device)
                elapsed = (
                    min(1.0, max(0.0, now - last_measurement_at))
                    if last_measurement_at is not None
                    else 0.5
                )
                last_measurement_at = now
                if (
                    pcm_dbfs > args.silence_threshold
                    and attenuation is not None
                    and attenuation > SILENCE_DB
                ):
                    estimate = estimate_level(
                        pcm_dbfs,
                        attenuation,
                        percent,
                        args.offset_low,
                        args.offset_high,
                    )
                    recorder.begin(elapsed)
                    recorder.update(estimate, elapsed, pcm_dbfs, attenuation)
                    last_active_at = now
                else:
                    estimate = None
                    if last_active_at is not None and now - last_active_at >= 1.5:
                        recorder.finish("silence")
                        last_active_at = None
                show("monitoring" if ready else "starting capture…")

            if (
                last_measurement_at is not None
                and now - last_measurement_at >= 1.5
                and estimate is not None
            ):
                estimate = None
                recorder.finish("audio_stream_stalled")
                last_active_at = None
                show("waiting for audio")

            if process is None:
                continue
            return_code = process.poll()
            if return_code is not None:
                recorder.finish("capture_helper_exited")
                # A short-lived helper can close stdout before selectors reports
                # its final stderr line.  Drain both pipes after exit so the real
                # ScreenCaptureKit error is never replaced by a permission guess.
                remaining_error = process.stderr.read().strip()
                if remaining_error:
                    helper_messages.extend(remaining_error.splitlines())
                message = (
                    "\n".join(helper_messages)
                    if helper_messages
                    else f"the capture helper exited unexpectedly (status {return_code})"
                )
                print(
                    "\nerror from the macOS capture helper:\n" + message + "\n\n"
                    "If the error says permission was denied, fully quit and reopen the app "
                    "hosting this shell after enabling it under System Settings > Privacy & "
                    "Security > Screen & System Audio Recording.",
                    file=sys.stderr,
                )
                exit_code = return_code or 2
                break
    except KeyboardInterrupt:
        pass
    finally:
        recorder.finish()
        stop_capture()
        session_lock.close()
        selector.close()
        if interactive:
            print()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
