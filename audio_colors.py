"""Stable, per-track colors derived only from intrinsic audio measurements."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Callable, Mapping, Sequence

import numpy as np


ALGORITHM_VERSION = "audiotag-intrinsic-audio-oklch-v5"
CACHE_VERSION = 3
CACHE_FILENAME = ".autoplay_audio_colors.json"
AUDIO_FEATURE_SIZE = 8
SAMPLE_RATE = 22050
AUDIO_FEATURE_NAMES = (
    "brightness",
    "warmth",
    "density",
    "dynamics",
    "noisiness",
    "tonality",
    "keyX",
    "keyY",
)
CLIP_SECONDS = 10.0
CLIP_CENTERS = (0.2, 0.5, 0.8)
HEX_COLOR_RE = re.compile(r"^#[0-9a-f]{6}$")


@dataclass(frozen=True)
class AudioColorRecord:
    audio_features: np.ndarray
    color: str
    content_sha256: str


def _file_signature(path: Path) -> dict[str, int]:
    stat_result = path.stat()
    return {"mtimeNs": stat_result.st_mtime_ns, "size": stat_result.st_size}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_audio_features(value: object) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) != AUDIO_FEATURE_SIZE:
        return None
    try:
        features = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if features.shape != (AUDIO_FEATURE_SIZE,) or not np.all(np.isfinite(features)):
        return None
    return features


def _record_from_entry(entry: object) -> AudioColorRecord | None:
    if not isinstance(entry, dict):
        return None
    audio_features = _valid_audio_features(entry.get("audioFeatures"))
    color = str(entry.get("color") or "").lower()
    content_sha256 = str(entry.get("contentSha256") or "").lower()
    if audio_features is None or not HEX_COLOR_RE.fullmatch(color):
        return None
    if len(content_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in content_sha256):
        return None
    return AudioColorRecord(audio_features, color, content_sha256)


def load_audio_color_cache(cache_path: Path) -> dict:
    empty = {"version": CACHE_VERSION, "entries": {}}
    if not cache_path.is_file():
        return empty
    try:
        value = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return empty
    if not isinstance(value, dict) or not isinstance(value.get("entries"), dict):
        return empty
    if (
        value.get("version") != CACHE_VERSION
        or value.get("algorithmVersion") != ALGORITHM_VERSION
    ):
        return empty
    return value


def save_audio_color_cache(cache_path: Path, cache: dict) -> None:
    payload = {
        "version": CACHE_VERSION,
        "algorithmVersion": ALGORITHM_VERSION,
        "entries": cache.get("entries", {}),
    }
    temp_path = cache_path.with_name(f".{cache_path.name}.tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(cache_path)


AUDIO_COLOR_CACHE_LOCK = Lock()


def _oklab_to_linear_srgb(lightness: float, a_value: float, b_value: float) -> tuple[float, float, float]:
    l_value = lightness + 0.3963377774 * a_value + 0.2158037573 * b_value
    m_value = lightness - 0.1055613458 * a_value - 0.0638541728 * b_value
    s_value = lightness - 0.0894841775 * a_value - 1.2914855480 * b_value
    l_cube, m_cube, s_cube = l_value**3, m_value**3, s_value**3
    return (
        4.0767416621 * l_cube - 3.3077115913 * m_cube + 0.2309699292 * s_cube,
        -1.2684380046 * l_cube + 2.6097574011 * m_cube - 0.3413193965 * s_cube,
        -0.0041960863 * l_cube - 0.7034186147 * m_cube + 1.7076147010 * s_cube,
    )


def _linear_to_srgb(value: float) -> float:
    if value <= 0.0031308:
        return 12.92 * value
    return 1.055 * (value ** (1.0 / 2.4)) - 0.055


def _oklab_to_hex(lightness: float, a_value: float, b_value: float) -> str:
    # Preserve lightness and hue; reduce only chroma until the color is displayable.
    chroma_scale = 1.0
    for _attempt in range(32):
        rgb = _oklab_to_linear_srgb(lightness, a_value * chroma_scale, b_value * chroma_scale)
        if all(0.0 <= component <= 1.0 for component in rgb):
            break
        chroma_scale *= 0.92
    rgb = _oklab_to_linear_srgb(lightness, a_value * chroma_scale, b_value * chroma_scale)
    channels = [
        round(max(0.0, min(1.0, _linear_to_srgb(component))) * 255.0)
        for component in rgb
    ]
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def _unit_interval(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _scaled(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    normalized = _unit_interval((value - low) / (high - low))
    return normalized * normalized * (3.0 - 2.0 * normalized)


def audio_features_for_clips(
    clips: Sequence[np.ndarray],
    sample_rate: int = 48000,
) -> np.ndarray:
    """Measure absolute acoustic properties without comparing library tracks."""
    import librosa

    measurements = []
    combined_chroma = []
    for clip in clips:
        magnitude = np.abs(librosa.stft(clip, n_fft=2048, hop_length=512))
        power = magnitude**2
        frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
        centroid = float(
            np.median(librosa.feature.spectral_centroid(S=magnitude, sr=sample_rate))
        )
        rms = librosa.feature.rms(S=magnitude)
        rms_db = 20.0 * np.log10(np.maximum(rms, 1e-8))
        loudness_db = float(np.median(rms_db))
        dynamic_range_db = float(np.percentile(rms_db, 90) - np.percentile(rms_db, 10))
        flatness = float(np.median(librosa.feature.spectral_flatness(S=magnitude)))
        zero_crossing_rate = float(np.median(librosa.feature.zero_crossing_rate(clip)))

        audible = (frequencies >= 40.0) & (frequencies <= 16000.0)
        bass = (frequencies >= 40.0) & (frequencies < 300.0)
        total_energy = float(power[audible].sum()) + 1e-12
        bass_fraction = float(power[bass].sum()) / total_energy

        onset_envelope = librosa.onset.onset_strength(
            y=clip,
            sr=sample_rate,
            hop_length=512,
        )
        onset_activity = float(
            np.median(onset_envelope) + 0.35 * np.mean(onset_envelope)
        )
        chroma = librosa.feature.chroma_stft(S=power, sr=sample_rate)
        combined_chroma.append(np.mean(chroma, axis=1))
        measurements.append(
            (
                centroid,
                loudness_db,
                dynamic_range_db,
                flatness,
                zero_crossing_rate,
                bass_fraction,
                onset_activity,
            )
        )

    if not measurements or not combined_chroma:
        raise ValueError("audio feature analysis received no clips")

    (
        centroid,
        loudness_db,
        dynamic_range_db,
        flatness,
        zero_crossing_rate,
        bass_fraction,
        onset_activity,
    ) = np.median(np.asarray(measurements, dtype=np.float64), axis=0)

    brightness = _scaled(math.log2(max(centroid, 1.0)), math.log2(700.0), math.log2(4200.0))
    bass_weight = _scaled(bass_fraction, 0.42, 0.90)
    warmth = _unit_interval(0.58 * bass_weight + 0.42 * (1.0 - brightness))
    energy = _scaled(loudness_db, -32.0, -12.0)
    dynamics = _scaled(dynamic_range_db, 5.0, 22.0)
    activity = _scaled(onset_activity, 0.85, 1.60)
    density = _unit_interval(0.55 * energy + 0.30 * activity + 0.15 * (1.0 - dynamics))
    flatness_score = _scaled(math.log10(max(flatness, 1e-7)), -5.2, -2.0)
    crossing_score = _scaled(zero_crossing_rate, 0.025, 0.18)
    noisiness = _unit_interval(0.55 * flatness_score + 0.45 * crossing_score)

    chroma = np.mean(np.asarray(combined_chroma, dtype=np.float64), axis=0)
    chroma_probability = chroma / max(float(chroma.sum()), 1e-12)
    entropy = -float(
        np.sum(chroma_probability * np.log(chroma_probability + 1e-12))
    ) / math.log(12.0)
    prominence = float((chroma.max() - chroma.mean()) / max(float(chroma.max()), 1e-12))

    # Project the complete pitch-class distribution around the circle of fifths.
    # This is continuous: no key label or discrete mood/style bucket is chosen.
    fifth_positions = (np.arange(12, dtype=np.float64) * 7.0) % 12.0
    fifth_angles = fifth_positions / 12.0 * math.tau
    pitch_vector = np.sum(chroma_probability * np.exp(1j * fifth_angles))
    pitch_focus = abs(pitch_vector)
    if pitch_focus > 1e-12:
        key_x = float(pitch_vector.real / pitch_focus)
        key_y = float(pitch_vector.imag / pitch_focus)
    else:
        key_x, key_y = 1.0, 0.0
    tonality = _unit_interval(
        0.45 * _scaled(1.0 - entropy, 0.002, 0.12)
        + 0.35 * _scaled(prominence, 0.12, 0.58)
        + 0.20 * _scaled(pitch_focus, 0.015, 0.22)
    )

    return np.asarray(
        [
            brightness,
            warmth,
            density,
            dynamics,
            noisiness,
            tonality,
            key_x,
            key_y,
        ],
        dtype=np.float32,
    )


def color_for_audio_features(features: Sequence[float]) -> str:
    """Map continuous acoustic measurements directly into perceptual color."""
    vector = np.asarray(features, dtype=np.float64)
    if vector.shape != (AUDIO_FEATURE_SIZE,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"Expected {AUDIO_FEATURE_SIZE} finite audio features")
    brightness, warmth, density, dynamics, noisiness, tonality = (
        _unit_interval(value) for value in vector[:6]
    )
    key_x = max(-1.0, min(1.0, float(vector[6])))
    key_y = max(-1.0, min(1.0, float(vector[7])))

    # Hue is a continuous acoustic phase combining pitch distribution, spectral
    # brightness, density, dynamics, and warmth. There are no named regions,
    # lookup colors, exemplars, or classifier labels in this mapping.
    pitch_turn = (math.atan2(key_y, key_x) / math.tau) % 1.0
    hue_turn = (
        pitch_turn
        + 0.85 * brightness
        + 0.65 * density
        + 0.45 * dynamics
        + 0.30 * warmth
    ) % 1.0
    hue = hue_turn * math.tau

    lightness = (
        0.43
        + 0.23 * brightness
        + 0.08 * dynamics
        + 0.06 * (1.0 - density)
        - 0.025 * warmth
    )
    lightness = max(0.36, min(0.76, lightness))
    chroma = 0.035 + 0.110 * math.sqrt(tonality) + 0.015 * (1.0 - noisiness)
    chroma = max(0.020, min(0.160, chroma))

    return _oklab_to_hex(
        lightness,
        chroma * math.cos(hue),
        chroma * math.sin(hue),
    )


class IntrinsicAudioColorAnalyzer:
    @staticmethod
    def _clips_for_file(path: Path) -> list[np.ndarray]:
        import librosa

        duration = max(0.0, float(librosa.get_duration(path=str(path)) or 0.0))
        if duration <= CLIP_SECONDS:
            starts = [0.0]
        else:
            maximum_start = duration - CLIP_SECONDS
            starts = [
                max(0.0, min(maximum_start, duration * center - CLIP_SECONDS / 2.0))
                for center in CLIP_CENTERS
            ]
        clips = [
            librosa.load(
                str(path), sr=SAMPLE_RATE, mono=True, offset=start, duration=CLIP_SECONDS
            )[0].astype(np.float32, copy=False)
            for start in starts
        ]
        if not clips or any(not len(clip) for clip in clips):
            raise ValueError("decoded audio was empty")
        return clips

    def analysis_for_file(self, path: Path) -> np.ndarray:
        clips = self._clips_for_file(path)
        return audio_features_for_clips(clips, sample_rate=SAMPLE_RATE)


def _ensure_audio_color_records_unlocked(
    song_paths: Mapping[str, Path],
    cache_path: Path,
    progress: Callable[[str], None] | None = None,
) -> dict[str, AudioColorRecord]:
    cache = load_audio_color_cache(cache_path)
    entries = cache.setdefault("entries", {})
    records: dict[str, AudioColorRecord] = {}
    by_content_hash: dict[str, AudioColorRecord] = {}
    for entry in entries.values():
        record = _record_from_entry(entry)
        if record is not None:
            by_content_hash.setdefault(record.content_sha256, record)

    pending: list[tuple[str, Path, dict[str, int], str]] = []
    cache_changed = False
    for song_name, path in song_paths.items():
        cache_key = str(path.resolve())
        try:
            signature = _file_signature(path)
        except OSError:
            continue
        entry = entries.get(cache_key)
        record = _record_from_entry(entry)
        if record is not None and entry.get("signature") == signature:
            records[song_name] = record
            continue

        try:
            content_sha256 = _sha256_file(path)
        except OSError as exc:
            if progress:
                progress(f"Audio color skipped '{song_name}': {exc}")
            continue
        matching_record = by_content_hash.get(content_sha256)
        if matching_record is not None:
            entries[cache_key] = {
                "signature": signature,
                "contentSha256": matching_record.content_sha256,
                "color": matching_record.color,
                "audioFeatures": matching_record.audio_features.round(8).tolist(),
            }
            records[song_name] = matching_record
            cache_changed = True
            continue
        pending.append((song_name, path, signature, content_sha256))

    if cache_changed:
        save_audio_color_cache(cache_path, cache)
    if not pending:
        return records

    if progress:
        progress(
            f"Analyzing {len(pending)} uncached audio color"
            f"{'s' if len(pending) != 1 else ''}..."
        )
    try:
        analyzer = IntrinsicAudioColorAnalyzer()
    except Exception as exc:
        if progress:
            progress(f"Audio analysis unavailable; using fallback colors: {exc}")
        return records

    for index, (song_name, path, signature, content_sha256) in enumerate(pending, start=1):
        if progress:
            progress(f"Audio color {index}/{len(pending)}: {song_name}")
        try:
            audio_features = analyzer.analysis_for_file(path)
            color = color_for_audio_features(audio_features)
        except Exception as exc:
            if progress:
                progress(f"Audio color skipped '{song_name}': {exc}")
            continue
        record = AudioColorRecord(audio_features, color, content_sha256)
        cache_key = str(path.resolve())
        entries[cache_key] = {
            "signature": signature,
            "contentSha256": content_sha256,
            "color": color,
            "audioFeatures": audio_features.round(8).tolist(),
        }
        records[song_name] = record
        by_content_hash[content_sha256] = record
        save_audio_color_cache(cache_path, cache)

    return records


def ensure_audio_color_records(
    song_paths: Mapping[str, Path],
    cache_path: Path,
    progress: Callable[[str], None] | None = None,
) -> dict[str, AudioColorRecord]:
    with AUDIO_COLOR_CACHE_LOCK:
        return _ensure_audio_color_records_unlocked(song_paths, cache_path, progress)


def cached_audio_color_records(
    song_paths: Mapping[str, Path],
    cache_path: Path,
) -> dict[str, AudioColorRecord]:
    """Return only immediately reusable records without hashing or model work."""
    # save_audio_color_cache replaces the file atomically, so readers can take a
    # completed snapshot while a long-running analyzer owns the writer lock.
    entries = load_audio_color_cache(cache_path).get("entries", {})
    records: dict[str, AudioColorRecord] = {}
    for song_name, path in song_paths.items():
        try:
            signature = _file_signature(path)
        except OSError:
            continue
        entry = entries.get(str(path.resolve()))
        record = _record_from_entry(entry)
        if record is not None and entry.get("signature") == signature:
            records[song_name] = record
    return records


class AudioColorBackgroundWorker:
    """Analyze new or changed tracks on one daemon thread."""

    def __init__(
        self,
        cache_path: Path,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.cache_path = cache_path
        self.progress = progress
        self._lock = Lock()
        self._wake_event = Event()
        self._stop_event = Event()
        self._song_paths: dict[str, Path] = {}
        self._submitted_signatures: dict[str, tuple[int, int] | None] = {}
        self._thread = Thread(
            target=self._run,
            name="audio-color-analysis",
            daemon=True,
        )
        self._thread.start()

    def submit(self, song_paths: Mapping[str, Path]) -> None:
        changed = False
        with self._lock:
            for path in song_paths.values():
                resolved_path = path.resolve()
                cache_key = str(resolved_path)
                try:
                    stat_result = resolved_path.stat()
                    signature = (stat_result.st_mtime_ns, stat_result.st_size)
                except OSError:
                    signature = None
                if self._submitted_signatures.get(cache_key) == signature:
                    continue
                self._submitted_signatures[cache_key] = signature
                self._song_paths[cache_key] = resolved_path
                changed = True
        if changed:
            self._wake_event.set()

    def close(self) -> None:
        self._stop_event.set()
        self._wake_event.set()

    def _run(self) -> None:
        while True:
            self._wake_event.wait()
            self._wake_event.clear()
            if self._stop_event.is_set():
                return
            with self._lock:
                song_paths = dict(self._song_paths)
            try:
                ensure_audio_color_records(
                    song_paths,
                    self.cache_path,
                    progress=self.progress,
                )
            except Exception as exc:
                if self.progress:
                    self.progress(f"Background audio color analysis failed: {exc}")


def artist_colors_from_tracks(
    song_artists: Mapping[str, Sequence[str]],
    song_counts: Mapping[str, int],
    song_records: Mapping[str, AudioColorRecord],
) -> dict[str, str]:
    grouped: dict[str, list[tuple[np.ndarray, float]]] = {}
    for song_name, artists in song_artists.items():
        record = song_records.get(song_name)
        if record is None or record.audio_features is None:
            continue
        weight = max(0.0, float(song_counts.get(song_name, 0) or 0))
        for artist in artists:
            grouped.setdefault(artist, []).append((record.audio_features, weight))

    colors = {}
    for artist, rows in grouped.items():
        weights = np.asarray([weight for _features, weight in rows], dtype=np.float64)
        if float(weights.sum()) <= 0:
            weights = np.ones(len(rows), dtype=np.float64)
        matrix = np.stack([features for features, _weight in rows]).astype(np.float64)
        audio_features = np.average(matrix, axis=0, weights=weights)
        key_norm = math.hypot(float(audio_features[6]), float(audio_features[7]))
        if key_norm > 1e-12:
            audio_features[6:8] /= max(1.0, key_norm)
        colors[artist] = color_for_audio_features(audio_features)
    return colors
