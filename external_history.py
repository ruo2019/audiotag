#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
EXTERNAL_PLAYS_FILE = ROOT / "external_plays.jsonl"
EXTERNAL_TRASH_FILE = ROOT / "external_plays_trash.jsonl"
HISTORY_PURGE_LOCK = (
    Path(tempfile.gettempdir()) / f"audiotag-external-history-{os.getuid()}.lock"
)


class HistoryPurgeError(RuntimeError):
    """Raised when a safe external-play history rewrite cannot be completed."""


@dataclass(frozen=True)
class HistoryPurgeResult:
    removed_ids: int
    commits_preserved: int
    old_head: str
    new_head: str
    changed: bool


def _run(
    args: list[str],
    *,
    cwd: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    result = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise HistoryPurgeError(
            f"Command failed ({' '.join(args)}): {detail or 'unknown error'}"
        )
    return result


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(["git", *args], cwd=repo, check=check)


def load_trashed_external_ids(path: Path = EXTERNAL_TRASH_FILE) -> set[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return set()
    ids: set[str] = set()
    for line in lines:
        try:
            entry = json.loads(line)
        except (TypeError, ValueError):
            continue
        if not isinstance(entry, dict):
            continue
        record = entry.get("record")
        if not isinstance(record, dict):
            continue
        external_id = str(record.get("id") or "").strip()
        if external_id:
            ids.add(external_id)
    return ids


def filter_external_plays(contents: bytes, removed_ids: set[str]) -> bytes:
    """Remove matching JSONL records while preserving all other bytes."""
    kept: list[bytes] = []
    for line in contents.splitlines(keepends=True):
        try:
            record = json.loads(line)
        except (TypeError, ValueError, UnicodeDecodeError):
            kept.append(line)
            continue
        external_id = (
            str(record.get("id") or "").strip()
            if isinstance(record, dict)
            else ""
        )
        if external_id not in removed_ids:
            kept.append(line)
    return b"".join(kept)


def _history_ids(repo: Path, ref: str) -> set[str]:
    commits = _git(
        repo,
        "log",
        "--format=%H",
        ref,
        "--",
        EXTERNAL_PLAYS_FILE.name,
    ).stdout.splitlines()
    found: set[str] = set()
    seen_blobs: set[str] = set()
    for commit in commits:
        blob = _git(
            repo,
            "rev-parse",
            f"{commit}:{EXTERNAL_PLAYS_FILE.name}",
            check=False,
        )
        if blob.returncode != 0:
            continue
        blob_id = blob.stdout.strip()
        if not blob_id or blob_id in seen_blobs:
            continue
        seen_blobs.add(blob_id)
        raw = _git(repo, "show", f"{commit}:{EXTERNAL_PLAYS_FILE.name}").stdout
        for line in raw.splitlines():
            try:
                record = json.loads(line)
            except (TypeError, ValueError):
                continue
            if not isinstance(record, dict):
                continue
            external_id = str(record.get("id") or "").strip()
            if external_id:
                found.add(external_id)
    return found


def _callback_source(removed_ids: Iterable[str]) -> str:
    ids_literal = repr(sorted(set(removed_ids)))
    return f'''\
if filename == b"{EXTERNAL_PLAYS_FILE.name}":
    import json
    removed_ids = set({ids_literal})
    contents = value.get_contents_by_identifier(blob_id)
    kept = []
    for line in contents.splitlines(keepends=True):
        try:
            record = json.loads(line)
        except (TypeError, ValueError, UnicodeDecodeError):
            kept.append(line)
            continue
        external_id = str(record.get("id") or "").strip() if isinstance(record, dict) else ""
        if external_id not in removed_ids:
            kept.append(line)
    filtered = b"".join(kept)
    if filtered != contents:
        blob_id = value.insert_file_with_contents(filtered)
return (filename, mode, blob_id)
'''


def _remote_head(repo: Path, remote_url: str, branch: str) -> str:
    result = _git(repo, "ls-remote", remote_url, f"refs/heads/{branch}")
    fields = result.stdout.split()
    if len(fields) < 2:
        raise HistoryPurgeError(f"Remote branch {branch!r} was not found.")
    return fields[0]


def purge_trashed_external_history(
    repo: Path = ROOT,
    trash_path: Path = EXTERNAL_TRASH_FILE,
    *,
    dry_run: bool = False,
) -> HistoryPurgeResult:
    """Remove currently trashed IDs from prior commits and safely update origin."""
    repo = repo.resolve()
    removed_ids = load_trashed_external_ids(trash_path)
    old_head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    if not removed_ids:
        return HistoryPurgeResult(0, 0, old_head, old_head, False)

    branch = _git(repo, "symbolic-ref", "--short", "HEAD").stdout.strip()
    if not branch:
        raise HistoryPurgeError("History purge requires an attached Git branch.")
    if _git(repo, "diff", "--cached", "--quiet", check=False).returncode != 0:
        raise HistoryPurgeError(
            "History purge refused because the Git index has staged changes."
        )
    remote_url = _git(repo, "remote", "get-url", "origin").stdout.strip()
    _git(repo, "fetch", "origin", branch)
    remote_ref = f"origin/{branch}"
    local_head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    expected_remote = _git(repo, "rev-parse", remote_ref).stdout.strip()
    live_remote = _remote_head(repo, remote_url, branch)
    if local_head != expected_remote or expected_remote != live_remote:
        raise HistoryPurgeError(
            "History purge requires local HEAD, origin, and the live remote to match."
        )

    present_ids = _history_ids(repo, "HEAD") & removed_ids
    commit_count = int(_git(repo, "rev-list", "--count", "HEAD").stdout.strip())
    if not present_ids:
        return HistoryPurgeResult(0, commit_count, old_head, old_head, False)
    if dry_run:
        return HistoryPurgeResult(
            len(present_ids), commit_count, old_head, old_head, False
        )

    git_filter_repo = shutil.which("git-filter-repo")
    if git_filter_repo is None:
        probe = _git(repo, "filter-repo", "--version", check=False)
        if probe.returncode != 0:
            raise HistoryPurgeError(
                "git-filter-repo is required (install it with brew install git-filter-repo)."
            )

    HISTORY_PURGE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PURGE_LOCK.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        with tempfile.TemporaryDirectory(prefix="audiotag-history-purge-") as temp:
            clone = Path(temp) / "repo"
            _run(
                [
                    "git",
                    "clone",
                    "--no-local",
                    "--single-branch",
                    "--branch",
                    branch,
                    remote_url,
                    str(clone),
                ],
                cwd=repo,
            )
            callback = Path(temp) / "filter_callback.py"
            callback.write_text(_callback_source(present_ids), encoding="utf-8")
            before_count = int(
                _git(clone, "rev-list", "--count", "HEAD").stdout.strip()
            )
            _git(
                clone,
                "filter-repo",
                "--prune-empty",
                "never",
                "--prune-degenerate",
                "never",
                "--file-info-callback",
                str(callback),
            )
            after_count = int(
                _git(clone, "rev-list", "--count", "HEAD").stdout.strip()
            )
            if after_count != before_count:
                raise HistoryPurgeError(
                    f"Commit preservation check failed ({before_count} -> {after_count})."
                )
            remaining = _history_ids(clone, "HEAD") & present_ids
            if remaining:
                raise HistoryPurgeError(
                    f"Verification failed: {len(remaining)} trashed IDs remain in history."
                )
            commit_map = clone / ".git" / "filter-repo" / "commit-map"
            if commit_map.exists():
                deleted = [
                    line
                    for line in commit_map.read_text(encoding="utf-8").splitlines()[1:]
                    if line.split() and line.split()[-1] == "0" * 40
                ]
                if deleted:
                    raise HistoryPurgeError(
                        f"Commit preservation check failed: {len(deleted)} commits deleted."
                    )
            new_head = _git(clone, "rev-parse", "HEAD").stdout.strip()
            _git(clone, "remote", "add", "origin", remote_url)
            if _remote_head(clone, remote_url, branch) != expected_remote:
                raise HistoryPurgeError("Remote changed while history was being filtered.")
            _git(
                clone,
                "push",
                f"--force-with-lease={branch}:{expected_remote}",
                "origin",
                branch,
            )

        _git(repo, "fetch", "origin", branch)
        fetched_head = _git(repo, "rev-parse", remote_ref).stdout.strip()
        if fetched_head != new_head:
            raise HistoryPurgeError("The rewritten remote HEAD could not be verified.")
        _git(repo, "reset", "--mixed", remote_ref)
        return HistoryPurgeResult(
            len(present_ids), after_count, old_head, new_head, True
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Remove IDs currently in external_plays_trash.jsonl from all prior "
            "external_plays.jsonl revisions while preserving every commit."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report how many trashed IDs still occur in history without rewriting",
    )
    args = parser.parse_args()
    try:
        result = purge_trashed_external_history(dry_run=args.dry_run)
    except HistoryPurgeError as error:
        parser.exit(1, f"History purge failed: {error}\n")
    if args.dry_run:
        print(f"{result.removed_ids} trashed external IDs still occur in history.")
    elif result.changed:
        print(
            f"Removed {result.removed_ids} trashed external IDs; preserved "
            f"{result.commits_preserved} commits; pushed {result.new_head[:8]}."
        )
    else:
        print("No trashed external IDs occur in reachable history.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
