import json
import math
import os

from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from .constants import ANALYSIS_CACHE_FILENAME, ANALYSIS_CACHE_VERSION


# --- Audio Helper Functions (Volume Normalization) ---

def get_audio_loudness(full_filepath):
    """Calculates the loudness in dBFS using the top 25% loudest seconds."""
    try:
        audio = AudioSegment.from_mp3(full_filepath)
        duration = audio.duration_seconds if audio.duration_seconds and audio.duration_seconds > 0 else 0.0
        if duration <= 0:
            return -100.0, duration

        # Analyze 1-second windows; use the mean of the loudest 25% seconds.
        second_loudness = []
        for start_ms in range(0, len(audio), 1000):
            segment = audio[start_ms:start_ms + 1000]
            segment_dbfs = segment.dBFS
            if segment_dbfs == -math.inf:
                segment_dbfs = -100.0
            second_loudness.append(segment_dbfs)

        if not second_loudness:
            return -100.0, duration

        second_loudness.sort(reverse=True)
        top_count = max(1, math.ceil(0.25 * len(second_loudness)))
        top_slice = second_loudness[:top_count]
        return sum(top_slice) / len(top_slice), duration
    except (CouldntDecodeError, FileNotFoundError, Exception):
        return None, None


def get_audio_duration(full_filepath):
    """Returns the audio duration in seconds."""
    try:
        audio = AudioSegment.from_mp3(full_filepath)
        duration = audio.duration_seconds if audio.duration_seconds and audio.duration_seconds > 0 else 0.0
        return duration
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

            duration = None
            if cached_data and cached_data.get('mtime') == mtime:
                loudness = cached_data.get('loudness')
                duration = cached_data.get('duration')
                if not isinstance(duration, (int, float)):
                    duration = get_audio_duration(full_path)
                    if duration is not None:
                        cached_data['duration'] = duration
                        analysis_cache[basename] = cached_data
                        needs_saving = True
            else:
                notify(f"Analyzing [{i + 1}/{len(all_files)}]: {basename[:50]}")
                loudness, duration = get_audio_loudness(full_path)
                analysis_cache[basename] = {'loudness': loudness, 'duration': duration, 'mtime': mtime}
                needs_saving = True

            audio_analysis[basename] = {'loudness': loudness, 'duration': duration}
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
            sample_loudness, _ = get_audio_loudness(sample_path)
            if isinstance(sample_loudness, (int, float)):
                target_loudness = sample_loudness

    for basename, data in audio_analysis.items():
        data['scale'] = calculate_volume_scale(target_loudness, data['loudness'])

    return audio_analysis, target_loudness
