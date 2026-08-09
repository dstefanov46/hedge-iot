import io
import importlib.util
import json
from pathlib import Path
import sys


SUPERVISOR_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "supervise_satellite_batch.py"
)
SUPERVISOR_SPEC = importlib.util.spec_from_file_location(
    "aurora_satellite_supervisor", SUPERVISOR_PATH
)
assert SUPERVISOR_SPEC is not None and SUPERVISOR_SPEC.loader is not None
SUPERVISOR_MODULE = importlib.util.module_from_spec(SUPERVISOR_SPEC)
SUPERVISOR_SPEC.loader.exec_module(SUPERVISOR_MODULE)
SatelliteBatchSupervisor = SUPERVISOR_MODULE.SatelliteBatchSupervisor


def _write_manifest(path: Path, checkpoint: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "collection_run": {
                    "discovery_checkpoint_utc": checkpoint,
                    "discovery_coverage_utc": [],
                },
                "products": [],
            }
        ),
        encoding="utf-8",
    )


def _supervisor(
    tmp_path: Path,
    command: list[str],
    **overrides,
) -> SatelliteBatchSupervisor:
    defaults = {
        "run_dir": tmp_path / "run",
        "manifest": tmp_path / "manifest.json",
        "disk_path": tmp_path,
        "command": command,
        "min_free_gib": 0,
        "max_stalled_failures": 3,
        "max_auth_failures": 3,
        "base_backoff_seconds": 0,
        "max_backoff_seconds": 0,
        "output": io.StringIO(),
        "install_signal_handlers": False,
    }
    defaults.update(overrides)
    return SatelliteBatchSupervisor(**defaults)


def test_supervisor_restarts_after_progress_and_completes(tmp_path):
    manifest = tmp_path / "manifest.json"
    counter = tmp_path / "attempt-count"
    _write_manifest(manifest, "2025-08-31T12:00:00+00:00")
    code = f"""
import json
from pathlib import Path
import sys
counter = Path({str(counter)!r})
attempt = int(counter.read_text() if counter.exists() else "0") + 1
counter.write_text(str(attempt))
if attempt == 1:
    manifest = Path({str(manifest)!r})
    payload = json.loads(manifest.read_text())
    payload["collection_run"]["discovery_checkpoint_utc"] = "2025-09-01T00:00:00+00:00"
    manifest.write_text(json.dumps(payload))
    print("temporary provider failure after progress")
    raise SystemExit(1)
print("resumed successfully")
"""
    supervisor = _supervisor(tmp_path, [sys.executable, "-c", code])

    assert supervisor.run() == 0

    state = json.loads(supervisor.state_path.read_text())
    assert state["state"] == "completed"
    assert state["attempt"] == 2
    assert state["checkpoint_after"] == "2025-09-01T00:00:00+00:00"
    assert state["consecutive_stalled_failures"] == 0
    assert (supervisor.run_dir / "attempt-001.log").exists()
    assert (supervisor.run_dir / "attempt-002.log").exists()
    combined = supervisor.combined_log.read_text()
    assert "temporary provider failure after progress" in combined
    assert "resumed successfully" in combined


def test_supervisor_stops_after_repeated_failure_without_progress(tmp_path):
    _write_manifest(tmp_path / "manifest.json", "2025-08-31T12:00:00+00:00")
    supervisor = _supervisor(
        tmp_path,
        [sys.executable, "-c", "print('unexpected crash'); raise SystemExit(1)"],
        max_stalled_failures=2,
    )

    assert supervisor.run() == 1

    state = json.loads(supervisor.state_path.read_text())
    assert state["state"] == "failed"
    assert state["reason"] == "repeated_failure_without_checkpoint_progress"
    assert state["attempt"] == 2
    assert state["consecutive_stalled_failures"] == 2


def test_supervisor_stops_after_persistent_authentication_failure(tmp_path):
    _write_manifest(tmp_path / "manifest.json", "2025-08-31T12:00:00+00:00")
    supervisor = _supervisor(
        tmp_path,
        [
            sys.executable,
            "-c",
            "print('HTTPError: 403 Client Error: Forbidden for url: https://api.eumetsat.int/token'); raise SystemExit(1)",
        ],
        max_auth_failures=2,
    )

    assert supervisor.run() == 1

    state = json.loads(supervisor.state_path.read_text())
    assert state["state"] == "failed"
    assert state["reason"] == "persistent_authentication_failure"
    assert state["attempt"] == 2
    assert state["consecutive_auth_failures"] == 2


def test_supervisor_does_not_restart_invalid_arguments(tmp_path):
    _write_manifest(tmp_path / "manifest.json", "2025-08-31T12:00:00+00:00")
    supervisor = _supervisor(
        tmp_path,
        [sys.executable, "-c", "raise SystemExit(2)"],
    )

    assert supervisor.run() == 2

    state = json.loads(supervisor.state_path.read_text())
    assert state["reason"] == "invalid_arguments"
    assert state["attempt"] == 1


def test_supervisor_refuses_to_start_below_disk_threshold(tmp_path):
    _write_manifest(tmp_path / "manifest.json", "2025-08-31T12:00:00+00:00")
    supervisor = _supervisor(
        tmp_path,
        ["command-must-not-run"],
        min_free_gib=10**9,
    )

    assert supervisor.run() == 70

    state = json.loads(supervisor.state_path.read_text())
    assert state["reason"] == "low_disk_space"
    assert state["attempt"] == 0
