from __future__ import annotations

import fcntl
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, Iterable, Optional


DEFAULT_AUTOPLAY_GRAPH_PORT = 8787
DEFAULT_EXPOSURE_DASHBOARD_PORT = 8769
DEFAULT_LOG_PATH = (
    Path.home() / "Library" / "Logs" / "Audiotag" / "background-services.log"
)
MAX_LOG_BYTES = 2 * 1024 * 1024
MANAGER_LOCK_PATH = (
    Path(tempfile.gettempdir()) / f"audiotag-background-services-{os.getuid()}.lock"
)


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    script: str
    arguments: tuple[str, ...] = ()

    def command(self, root: Path) -> list[str]:
        return [sys.executable, str(root / self.script), *self.arguments]


@dataclass
class ServiceState:
    spec: ServiceSpec
    process: Optional[subprocess.Popen[bytes]] = None
    external_pid: Optional[int] = None
    started_at: float = 0.0
    restart_at: float = 0.0
    failure_count: int = 0


DEFAULT_SERVICES = (
    ServiceSpec("external listens", "external_listen_tracker.py", ("--quiet",)),
    ServiceSpec("headphone exposure", "all_apps_volume_stats.py", ("--quiet",)),
    ServiceSpec(
        "headphone exposure dashboard",
        "plot_headphone_exposure.py",
        (
            "--live",
            "--no-open",
            "--port",
            str(DEFAULT_EXPOSURE_DASHBOARD_PORT),
        ),
    ),
    ServiceSpec(
        "autoplay graph",
        "visualize_autoplay_graph.py",
        (
            "all",
            "--include-self-loops",
            "--live",
            "--no-open",
            "--port",
            str(DEFAULT_AUTOPLAY_GRAPH_PORT),
        ),
    ),
)


class BackgroundServiceManager:
    """Keep the player's noninteractive companion processes alive."""

    def __init__(
        self,
        root: Path,
        services: Iterable[ServiceSpec] = DEFAULT_SERVICES,
        log_path: Path = DEFAULT_LOG_PATH,
    ) -> None:
        self.root = root.resolve()
        self.states = [ServiceState(spec) for spec in services]
        self.log_path = log_path.expanduser()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._manager_lock: Optional[IO[str]] = None
        self._log_file: Optional[IO[bytes]] = None
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> bool:
        if self._active:
            return True
        if not self._acquire_manager_lock():
            return False
        try:
            self._open_log()
        except OSError:
            self._close_manager_lock()
            return False
        self._stop_event.clear()
        self._active = True
        self._write_log(f"player service manager started (pid {os.getpid()})")
        for state in self.states:
            self._ensure_running(state)
        self._thread = threading.Thread(
            target=self._monitor,
            name="audiotag-background-services",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        if not self._active:
            self._close_manager_lock()
            return
        self._stop_event.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)

        owned = [
            state.process
            for state in self.states
            if state.process is not None and state.process.poll() is None
        ]
        for process in owned:
            self._signal_process_group(process, signal.SIGINT)
        self._wait_for_processes(owned, 4.0)
        for process in owned:
            if process.poll() is None:
                self._signal_process_group(process, signal.SIGTERM)
        self._wait_for_processes(owned, 2.0)
        for process in owned:
            if process.poll() is None:
                self._signal_process_group(process, signal.SIGKILL)
        self._wait_for_processes(owned, 1.0)

        self._write_log("player service manager stopped")
        self._active = False
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        self._close_manager_lock()

    def _acquire_manager_lock(self) -> bool:
        lock_file = MANAGER_LOCK_PATH.open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_file.close()
            return False
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(str(os.getpid()))
        lock_file.flush()
        self._manager_lock = lock_file
        return True

    def _close_manager_lock(self) -> None:
        if self._manager_lock is None:
            return
        self._manager_lock.close()
        self._manager_lock = None

    def _open_log(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if self.log_path.stat().st_size >= MAX_LOG_BYTES:
                rotated = self.log_path.with_suffix(self.log_path.suffix + ".1")
                rotated.unlink(missing_ok=True)
                self.log_path.replace(rotated)
        except FileNotFoundError:
            pass
        self._log_file = self.log_path.open("ab", buffering=0)

    def _write_log(self, message: str) -> None:
        if self._log_file is None:
            return
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        self._log_file.write(f"\n[{timestamp}] {message}\n".encode("utf-8"))

    def _monitor(self) -> None:
        while not self._stop_event.wait(2.0):
            for state in self.states:
                self._ensure_running(state)

    def _ensure_running(self, state: ServiceState) -> None:
        process = state.process
        now = time.monotonic()
        if process is not None:
            return_code = process.poll()
            if return_code is None:
                return
            lifetime = max(0.0, now - state.started_at)
            if lifetime >= 60.0:
                state.failure_count = 0
            else:
                state.failure_count += 1
            delay = min(60.0, float(2 ** min(state.failure_count, 6)))
            state.restart_at = now + delay
            state.process = None
            self._write_log(
                f"{state.spec.name} exited with status {return_code}; "
                f"retrying in {delay:.0f}s"
            )

        external_pid = self._find_external_process(state.spec.script)
        if external_pid is not None:
            if state.external_pid != external_pid:
                self._write_log(
                    f"{state.spec.name} is already running outside the player "
                    f"(pid {external_pid}); it will be adopted after that process exits"
                )
            state.external_pid = external_pid
            return
        if state.external_pid is not None:
            self._write_log(
                f"outside {state.spec.name} process ended; starting a managed copy"
            )
            state.external_pid = None
            state.restart_at = 0.0
        if now < state.restart_at or self._stop_event.is_set():
            return

        script_path = self.root / state.spec.script
        if not script_path.is_file():
            state.restart_at = now + 60.0
            self._write_log(f"cannot start {state.spec.name}: missing {script_path}")
            return
        assert self._log_file is not None
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        try:
            state.process = subprocess.Popen(
                state.spec.command(self.root),
                cwd=self.root,
                stdin=subprocess.DEVNULL,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
        except OSError as error:
            state.failure_count += 1
            state.restart_at = now + min(60.0, float(2 ** min(state.failure_count, 6)))
            self._write_log(f"could not start {state.spec.name}: {error}")
            return
        state.started_at = now
        state.restart_at = 0.0
        self._write_log(
            f"started {state.spec.name} (pid {state.process.pid}): "
            f"{shlex.join(state.spec.command(self.root))}"
        )

    @staticmethod
    def _find_external_process(script_name: str) -> Optional[int]:
        try:
            listing = subprocess.run(
                ["ps", "-axo", "pid=,command="],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            return None
        own_pid = os.getpid()
        for line in listing.splitlines():
            fields = line.strip().split(maxsplit=1)
            if len(fields) != 2:
                continue
            try:
                pid = int(fields[0])
                command = shlex.split(fields[1])
            except (ValueError, TypeError):
                continue
            if pid == own_pid:
                continue
            if any(Path(token).name == script_name for token in command[1:]):
                return pid
        return None

    @staticmethod
    def _signal_process_group(
        process: subprocess.Popen[bytes], requested_signal: signal.Signals
    ) -> None:
        try:
            os.killpg(process.pid, requested_signal)
        except ProcessLookupError:
            pass

    @staticmethod
    def _wait_for_processes(
        processes: Iterable[subprocess.Popen[bytes]], timeout: float
    ) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if all(process.poll() is not None for process in processes):
                return
            time.sleep(0.05)

    def __enter__(self) -> "BackgroundServiceManager":
        self.start()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.stop()
