#!/usr/bin/env python3
"""Restart a resumable satellite batch after bounded, checkpoint-aware failures."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TextIO


AUTH_FAILURE = re.compile(
    r"401 Client Error|403 Client Error|Authentication error \(401\)|"
    r"Unauthorised \(403\)|Unauthorized \(403\)|Forbidden for url: .*?/token",
    re.IGNORECASE,
)
INVALID_ARGUMENT_EXIT_CODES = {2, 64}


def utc_now() -> datetime:
    return datetime.now(UTC)


def read_checkpoint(manifest: Path) -> str | None:
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        value = payload.get("collection_run", {}).get("discovery_checkpoint_utc")
        return str(value) if value else None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / 1024**3


def attempt_has_auth_failure(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return any(AUTH_FAILURE.search(line) is not None for line in handle)
    except OSError:
        return False


class SatelliteBatchSupervisor:
    def __init__(
        self,
        *,
        run_dir: Path,
        manifest: Path,
        disk_path: Path,
        command: list[str],
        min_free_gib: float = 10.0,
        max_stalled_failures: int = 12,
        max_auth_failures: int = 3,
        base_backoff_seconds: float = 60.0,
        max_backoff_seconds: float = 900.0,
        output: TextIO | None = None,
        install_signal_handlers: bool = True,
    ) -> None:
        if not command:
            raise ValueError("a child command is required")
        if min_free_gib < 0:
            raise ValueError("min_free_gib must be non-negative")
        if max_stalled_failures < 1 or max_auth_failures < 1:
            raise ValueError("failure limits must be positive")
        if base_backoff_seconds < 0 or max_backoff_seconds < 0:
            raise ValueError("backoff values must be non-negative")

        self.run_dir = run_dir.resolve()
        self.manifest = manifest.resolve()
        self.disk_path = disk_path.resolve()
        self.command = command
        self.min_free_gib = min_free_gib
        self.max_stalled_failures = max_stalled_failures
        self.max_auth_failures = max_auth_failures
        self.base_backoff_seconds = base_backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds
        self.output = output
        self.install_signal_handlers = install_signal_handlers
        self.state_path = self.run_dir / "supervisor-state.json"
        self.combined_log = self.run_dir / "satellite-batch.log"
        self.stop_event = threading.Event()
        self.stop_signal: int | None = None
        self.child: subprocess.Popen[str] | None = None

    def _emit(self, message: str, *logs: TextIO) -> None:
        text = message if message.endswith("\n") else message + "\n"
        for handle in logs:
            handle.write(text)
            handle.flush()
        if self.output is not None:
            self.output.write(text)
            self.output.flush()

    def _state(self, state: str, **values: Any) -> None:
        payload = {
            "state": state,
            "updated_at_utc": utc_now().isoformat(),
            "supervisor_pid": os.getpid(),
            **values,
        }
        write_state(self.state_path, payload)

    def _update_state(self, **values: Any) -> None:
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            payload = {}
        payload.update(values)
        payload["updated_at_utc"] = utc_now().isoformat()
        write_state(self.state_path, payload)

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        self.stop_signal = signum
        self.stop_event.set()
        child = self.child
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    def _run_child(self, attempt_log: Path, combined: TextIO) -> int:
        with attempt_log.open("a", encoding="utf-8") as attempt_handle:
            self.child = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                bufsize=1,
                start_new_session=True,
            )
            self._update_state(child_pid=self.child.pid)
            assert self.child.stdout is not None
            for line in self.child.stdout:
                attempt_handle.write(line)
                attempt_handle.flush()
                combined.write(line)
                combined.flush()
                if self.output is not None:
                    self.output.write(line)
                    self.output.flush()
            return self.child.wait()

    def _backoff(self, stalled_failures: int) -> float:
        exponent = max(0, stalled_failures - 1)
        return min(
            self.base_backoff_seconds * 2**exponent,
            self.max_backoff_seconds,
        )

    def run(self) -> int:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.install_signal_handlers:
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)

        attempt = 0
        stalled_failures = 0
        auth_failures = 0
        last_exit_code: int | None = None
        last_checkpoint = read_checkpoint(self.manifest)

        with self.combined_log.open("a", encoding="utf-8") as combined:
            while not self.stop_event.is_set():
                available_gib = free_gib(self.disk_path)
                if available_gib < self.min_free_gib:
                    self._state(
                        "failed",
                        reason="low_disk_space",
                        attempt=attempt,
                        last_exit_code=last_exit_code,
                        checkpoint_after=last_checkpoint,
                        free_gib=available_gib,
                        required_free_gib=self.min_free_gib,
                        consecutive_stalled_failures=stalled_failures,
                        consecutive_auth_failures=auth_failures,
                        next_retry_at_utc=None,
                    )
                    self._emit(
                        f"Supervisor stopped: only {available_gib:.2f} GiB free; "
                        f"{self.min_free_gib:.2f} GiB required.",
                        combined,
                    )
                    return 70

                attempt += 1
                checkpoint_before = read_checkpoint(self.manifest)
                attempt_log = self.run_dir / f"attempt-{attempt:03d}.log"
                started_at = utc_now()
                self._state(
                    "running",
                    attempt=attempt,
                    child_pid=None,
                    checkpoint_before=checkpoint_before,
                    checkpoint_after=checkpoint_before,
                    last_exit_code=last_exit_code,
                    consecutive_stalled_failures=stalled_failures,
                    consecutive_auth_failures=auth_failures,
                    free_gib=available_gib,
                    next_retry_at_utc=None,
                )
                self._emit(
                    f"Supervisor attempt {attempt:03d} started at "
                    f"{started_at.isoformat()} from checkpoint {checkpoint_before}.",
                    combined,
                )

                try:
                    exit_code = self._run_child(attempt_log, combined)
                except OSError as exc:
                    self._state(
                        "failed",
                        reason="child_spawn_error",
                        attempt=attempt,
                        last_exit_code=127,
                        checkpoint_before=checkpoint_before,
                        checkpoint_after=read_checkpoint(self.manifest),
                        error=str(exc),
                        consecutive_stalled_failures=stalled_failures,
                        consecutive_auth_failures=auth_failures,
                        next_retry_at_utc=None,
                    )
                    self._emit(f"Supervisor could not start the batch: {exc}", combined)
                    return 127
                finally:
                    self.child = None

                last_exit_code = exit_code
                checkpoint_after = read_checkpoint(self.manifest)
                progressed = bool(
                    checkpoint_after and checkpoint_after != checkpoint_before
                )
                if progressed:
                    stalled_failures = 0
                    auth_failures = 0
                elif exit_code != 0:
                    stalled_failures += 1
                last_checkpoint = checkpoint_after or last_checkpoint

                if exit_code == 0:
                    self._state(
                        "completed",
                        attempt=attempt,
                        last_exit_code=0,
                        checkpoint_before=checkpoint_before,
                        checkpoint_after=checkpoint_after,
                        consecutive_stalled_failures=stalled_failures,
                        consecutive_auth_failures=auth_failures,
                        next_retry_at_utc=None,
                    )
                    self._emit("Supervisor: satellite batch completed successfully.", combined)
                    return 0

                if self.stop_event.is_set():
                    signum = self.stop_signal or signal.SIGTERM
                    status = 128 + signum
                    self._state(
                        "stopped",
                        reason="user_signal",
                        signal=signum,
                        attempt=attempt,
                        last_exit_code=exit_code,
                        checkpoint_before=checkpoint_before,
                        checkpoint_after=checkpoint_after,
                        consecutive_stalled_failures=stalled_failures,
                        consecutive_auth_failures=auth_failures,
                        next_retry_at_utc=None,
                    )
                    return status

                if exit_code in INVALID_ARGUMENT_EXIT_CODES:
                    self._state(
                        "failed",
                        reason="invalid_arguments",
                        attempt=attempt,
                        last_exit_code=exit_code,
                        checkpoint_before=checkpoint_before,
                        checkpoint_after=checkpoint_after,
                        consecutive_stalled_failures=stalled_failures,
                        consecutive_auth_failures=auth_failures,
                        next_retry_at_utc=None,
                    )
                    self._emit(
                        f"Supervisor stopped after non-retryable exit code {exit_code}.",
                        combined,
                    )
                    return exit_code

                auth_failure = attempt_has_auth_failure(attempt_log)
                auth_failures = auth_failures + 1 if auth_failure else 0
                if auth_failures >= self.max_auth_failures:
                    self._state(
                        "failed",
                        reason="persistent_authentication_failure",
                        attempt=attempt,
                        last_exit_code=exit_code,
                        checkpoint_before=checkpoint_before,
                        checkpoint_after=checkpoint_after,
                        consecutive_stalled_failures=stalled_failures,
                        consecutive_auth_failures=auth_failures,
                        next_retry_at_utc=None,
                    )
                    self._emit(
                        "Supervisor stopped after persistent authentication failures.",
                        combined,
                    )
                    return exit_code

                if stalled_failures >= self.max_stalled_failures:
                    self._state(
                        "failed",
                        reason="repeated_failure_without_checkpoint_progress",
                        attempt=attempt,
                        last_exit_code=exit_code,
                        checkpoint_before=checkpoint_before,
                        checkpoint_after=checkpoint_after,
                        consecutive_stalled_failures=stalled_failures,
                        consecutive_auth_failures=auth_failures,
                        next_retry_at_utc=None,
                    )
                    self._emit(
                        "Supervisor stopped after repeated failures at the same checkpoint.",
                        combined,
                    )
                    return exit_code

                delay = self._backoff(stalled_failures)
                retry_at = utc_now() + timedelta(seconds=delay)
                self._state(
                    "backing_off",
                    reason=("authentication_failure" if auth_failure else "child_failure"),
                    attempt=attempt,
                    last_exit_code=exit_code,
                    checkpoint_before=checkpoint_before,
                    checkpoint_after=checkpoint_after,
                    made_checkpoint_progress=progressed,
                    consecutive_stalled_failures=stalled_failures,
                    consecutive_auth_failures=auth_failures,
                    free_gib=free_gib(self.disk_path),
                    next_retry_at_utc=retry_at.isoformat(),
                )
                self._emit(
                    f"Supervisor: batch exited {exit_code}; retrying in "
                    f"{delay:.0f}s from checkpoint {checkpoint_after}.",
                    combined,
                )
                self.stop_event.wait(delay)

        signum = self.stop_signal or signal.SIGTERM
        return 128 + signum


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Supervise and restart a resumable AURORA satellite batch."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--disk-path", required=True, type=Path)
    parser.add_argument("--min-free-gib", type=float, default=10.0)
    parser.add_argument("--max-stalled-failures", type=int, default=12)
    parser.add_argument("--max-auth-failures", type=int, default=3)
    parser.add_argument("--base-backoff-seconds", type=float, default=60.0)
    parser.add_argument("--max-backoff-seconds", type=float, default=900.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        build_parser().error("a child command must follow --")
    supervisor = SatelliteBatchSupervisor(
        run_dir=args.run_dir,
        manifest=args.manifest,
        disk_path=args.disk_path,
        command=command,
        min_free_gib=args.min_free_gib,
        max_stalled_failures=args.max_stalled_failures,
        max_auth_failures=args.max_auth_failures,
        base_backoff_seconds=args.base_backoff_seconds,
        max_backoff_seconds=args.max_backoff_seconds,
        output=sys.stdout,
    )
    return supervisor.run()


if __name__ == "__main__":
    raise SystemExit(main())
