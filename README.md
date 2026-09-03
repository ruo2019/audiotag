# Audiotag

Audiotag is a local MP3 tagging, playback, and library-maintenance toolkit. The
current repo is centered around mood-tagged playback, listen history, queue
management, Markov/autoplay experiments, artist tagging, MP3 trimming, renaming,
and YouTube-audio helper workflows.

The default music libraries are:

- `static/mp3/` with metadata in `tags.json`, `artists.json`,
  `listen_counts.json`, and `listen_timestamps_mp3.json`
- `static/mid-mp3s/` with metadata in `mid_tags.json`, `mid_artists.json`,
  `mid_listen_counts.json`, and `listen_timestamps_mid-mp3s.json`

Older copies and experiments live in files named `*copy*.py`, `*_original.py`,
`shorter.py`, and `Archive 2/`. Prefer the top-level scripts documented below
unless you are intentionally comparing old behavior.

## Setup

Use Python 3.10+ and install FFmpeg before using playback analysis, trimming,
mixing, or YouTube helpers.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install flask werkzeug pygame sentence-transformers scikit-learn torch librosa pydub tabulate numpy mutagen yt-dlp
```

On macOS:

```bash
brew install ffmpeg
# optional, used by link.py for pinned audio output
brew install mpv
```

The first mood-search run downloads a local Sentence Transformers model. Current
players use `sentence-transformers/all-MiniLM-L6-v2`; the older `player.py` uses
`sentence-transformers/sentence-t5-base`.

## Main Player

`headphones_markov.py` is the most current player. By default it opens both
`static/mp3` and `static/mid-mp3s` as TUI library tabs, supports manual mood
search, listen tracking, and an auto mode backed by play-history transitions.

```bash
python headphones_markov.py
python headphones_markov.py --mood "calm ambient study" --top 5
python headphones_markov.py --mode auto
python headphones_markov.py --folder static/mp3 --tags tags.json
python headphones_markov.py --folders static/mp3 static/mid-mp3s
```

Useful options:

- `--vol -16` sets target integrated LUFS; otherwise the sample track is used.
- `--sample "Deep Stone Crypt Theme.mp3"` selects the reference track.
- `--no-tui` disables the curses UI.
- `--youtube-cookies-from-browser chrome` or `--youtube-cookies cookies.txt`
  configures YouTube helpers used by the Markov player.

Related players:

- `gui.py` opens the queue GUI by default and has `--tui` for the legacy curses
  loop.
- `headphones.py` is the non-Markov mood TUI with listen counters and
  Similar/Most tabs.
- `player.py` is the older required-`--mood` looping player.
- `play.py` is a terminal player for direct folder playback and volume analysis.

Press **Ctrl+E** in `headphones_markov.py` for icon-free headphone listening
stats; press **Ctrl+E** or **Esc** to close them. The panel shows the current
estimated range and Mac volume, today/week duration and average, WHO weekly
allowance used/remaining, and the margin from the cautious target.

Played time, energy-averaged level, and WHO dose are checkpointed in the separate
generated file `headphone_exposure.json`; `listen_timestamps*.json` is unchanged,
and partial plays count. The estimate uses pygame's decoded PCM, player gain, Mac
volume, and a broad Apple wired-earbud profile. It excludes the YouTube side
player and other apps and does not request recording permission.

The displayed target keeps the upper estimate below 70 dB(A) for cautious
8–10-hour daily listening. That is deliberately lower than the approximately
78 dB(A) adult exposure level corresponding to 10 hours every day under the
[WHO 80 dB(A)/40-hour weekly model](https://www.who.int/publications/i/item/9789241515276),
because the analog earbuds have no calibrated profile. The estimate tracks
changes well but is not a physical measurement at the eardrum.

## Tag MP3s

The Flask tagger writes mood tags to `tags.json`.

```bash
python app.py
```

Open `http://127.0.0.1:5000`, tag tracks from `static/mp3/`, and submit tags
from the browser. `app.py` only works on the main `static/mp3` library unless you
change its constants.

For artist metadata:

```bash
python artist_tagger.py
python artist_tagger.py --mid
python artist_tagger.py --all --limit 25
python artist_tagger.py --list-output-devices
```

Artist tags are stored separately from mood tags.

## Trim MP3s

`mp3_trimmer.py` runs a browser UI for frame-copy MP3 trimming. It requires
`ffmpeg` and `ffprobe`.

```bash
python mp3_trimmer.py
```

Open `http://127.0.0.1:5111`. The trimmer reads from `static/mp3/`,
`static/mid-mp3s/`, `static/trim-uploads/`, and `static/trimmed/`; outputs go to
`static/trimmed/`. It uses FFmpeg copy mode (`-c copy`), so audio is not
re-encoded and cuts land on MP3 frame boundaries.

## Rename Tracks

Use `rename_mp3.py` when changing filenames. It updates MP3 files plus known
Audiotag metadata, listen counts, playlists, artist files, and caches.

```bash
python rename_mp3.py
python rename_mp3.py "Old Name" "New Name"
python rename_mp3.py "Old Name" "New Name" --apply
python rename_mp3.py "Old Name" "New Name" --mid --apply
```

Without `--apply`, command-line mode is a dry run.

## Visualize Play History

Generate an autoplay graph from listen history:

```bash
python visualize_autoplay_graph.py mp3
python visualize_autoplay_graph.py mid-mp3s
python visualize_autoplay_graph.py all --live
```

The graph computes stable sound-derived colors from up to three samples of each
track using only continuous spectral, pitch, energy, dynamic, and noisiness
measurements. It does not use mood labels, prompts, palettes, other songs, or
custom color overrides as analysis inputs. In `--live` mode, the dashboard starts
without waiting for audio-color analysis: new or modified tracks are analyzed on
a background thread, cached one at a time, and appear automatically on a later
refresh without blocking the page. Progress is printed in the terminal; later
runs reuse `.autoplay_audio_colors.json`.

Generate a simpler Markov transition graph:

```bash
python markov_graph.py --folder static/mp3
```

Both commands write interactive HTML files in the repo root unless `--output` is
provided.

## YouTube Helpers

Search and preview YouTube audio interactively:

```bash
python link.py "song search text" -n 10
```

Bulk-download audio from a line-based input file:

```bash
python bulk_download_mp3s.py download_list.txt --dry-run
python bulk_download_mp3s.py download_list.txt
```

Each non-empty line should be:

```text
VIDEO_ID title [m|mm]
```

Use `[m]` for `static/mp3` and `[mm]` for `static/mid-mp3s`. YouTube cookie
configuration can come from `YTDLP_COOKIES` or a cookies file passed with
`--cookies`.

## Other Utilities

```bash
python mp3_to_json.py static/mp3 -o mp3_filenames.json
python align_mix.py reference.mp3 overlay.mp3 output.mp3 --dry-run
python reverse.py
```

`align_mix.py` uses FFmpeg to align and mix two MP3s. `mp3_to_json.py` exports
MP3 stems. `reverse.py` is a small ad hoc audio utility.

## Generated Data

The players and tools write local state while you use them:

- Listen counts and timestamps: `listen_counts.json`,
  `mid_listen_counts.json`, `listen_timestamps_*.json`
- Queue state: `queue_playlists_*.json`
- Embedding/loudness caches inside MP3 folders:
  `.track_emb_cache.npz`, `.loudness_cache.json`
- Generated visualizations: `autoplay_graph_*.html`, `markov_graph_*.html`
- Intrinsic audio features and fixed song colors: `.autoplay_audio_colors.json`

These files are part of the working library state, so expect them to change after
playback, tagging, renaming, graphing, or trimming.
