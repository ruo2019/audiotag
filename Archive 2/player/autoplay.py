from collections import Counter, defaultdict
import random


def build_global_events(history_mapping):
    events = []
    for song_name, timestamps in history_mapping.items():
        for timestamp in timestamps:
            if isinstance(timestamp, (int, float)):
                events.append((int(timestamp), song_name))
    events.sort(key=lambda item: (item[0], item[1]))
    return events


def build_transition_counts(events, window_seconds, include_self_loops=False):
    edge_counts = Counter()
    for start_index, (current_timestamp, current_song) in enumerate(events):
        next_index = start_index + 1
        while next_index < len(events):
            next_timestamp, next_song = events[next_index]
            gap_seconds = next_timestamp - current_timestamp
            if gap_seconds > window_seconds:
                break
            if gap_seconds >= 0 and (include_self_loops or current_song != next_song):
                edge_counts[(current_song, next_song)] += 1
            next_index += 1
    return edge_counts


def build_autoplay_snapshot(song_names, history_mapping, window_seconds, fallback_probability):
    events = build_global_events(history_mapping)
    edge_counts = build_transition_counts(events, window_seconds, include_self_loops=True)
    outgoing = defaultdict(dict)
    for (source_song, target_song), count in edge_counts.items():
        if count <= 0:
            continue
        outgoing[source_song][target_song] = count
    return {
        "song_names": tuple(song_names),
        "window_seconds": int(window_seconds),
        "fallback_probability": max(0.0, min(1.0, float(fallback_probability))),
        "events": len(events),
        "edge_count": len(edge_counts),
        "outgoing": {source: dict(targets) for source, targets in outgoing.items()},
    }


def _dedupe_song_names(song_names):
    unique_names = []
    seen = set()
    for song_name in song_names:
        if song_name in seen:
            continue
        seen.add(song_name)
        unique_names.append(song_name)
    return unique_names


def _weighted_choice(candidates, weights, rng):
    total_weight = sum(max(0.0, float(weight)) for weight in weights)
    if total_weight <= 0:
        return rng.choice(candidates)
    threshold = rng.random() * total_weight
    cumulative = 0.0
    for candidate, weight in zip(candidates, weights):
        cumulative += max(0.0, float(weight))
        if threshold <= cumulative:
            return candidate
    return candidates[-1]


def choose_autoplay_song(snapshot, candidate_songs, current_song=None, rng=None):
    if rng is None:
        rng = random

    pool = _dedupe_song_names(candidate_songs)
    if not pool:
        return None
    if len(pool) == 1:
        return pool[0]

    candidates = list(pool)

    if not current_song or not isinstance(snapshot, dict):
        return rng.choice(candidates)

    outgoing = snapshot.get("outgoing", {}).get(current_song, {})
    observed = [(song_name, outgoing[song_name]) for song_name in candidates if outgoing.get(song_name, 0) > 0]
    if not observed:
        return rng.choice(candidates)

    missing = [song_name for song_name in candidates if outgoing.get(song_name, 0) <= 0]
    fallback_total = snapshot.get("fallback_probability", 0.0) if missing else 0.0
    observed_total = max(0.0, 1.0 - fallback_total)
    observed_count_total = sum(count for _, count in observed)

    weighted_candidates = []
    weights = []
    for song_name, count in observed:
        weight = observed_total * count / observed_count_total if observed_count_total > 0 else 0.0
        weighted_candidates.append(song_name)
        weights.append(weight)

    if missing:
        missing_weight = fallback_total / len(missing) if fallback_total > 0 else 0.0
        for song_name in missing:
            weighted_candidates.append(song_name)
            weights.append(missing_weight)

    if not weighted_candidates:
        return rng.choice(candidates)
    return _weighted_choice(weighted_candidates, weights, rng)
