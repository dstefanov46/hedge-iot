#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUNS_ROOT="$REPO_ROOT/outputs/satellite-runs"
AURORA="$REPO_ROOT/.venv/bin/aurora"
PYTHON="$REPO_ROOT/.venv/bin/python"
SUPERVISOR="$REPO_ROOT/scripts/supervise_satellite_batch.py"
CONFIG="$REPO_ROOT/configs/satellite.toml"
SITE_CONFIG="$REPO_ROOT/configs/site_hirvensalmi.json"
MANIFEST="$REPO_ROOT/data/raw/satellite/native/manifest.json"

START_UTC="2025-01-01T00:00:00Z"
END_UTC="2026-01-01T00:00:00Z"
CHUNK_HOURS=4
DOWNLOAD_WORKERS=4
PROCESSING_WORKERS=16
PIPELINE_QUEUE_SIZE=32
MIN_FREE_GIB=10
MAX_STALLED_FAILURES=12
MAX_AUTH_FAILURES=3
BASE_BACKOFF_SECONDS=60
MAX_BACKOFF_SECONDS=900

for required_executable in "$AURORA" "$PYTHON"; do
  if [[ ! -x "$required_executable" ]]; then
    echo "Missing executable: $required_executable" >&2
    exit 1
  fi
done

for required_file in "$CONFIG" "$SITE_CONFIG" "$SUPERVISOR"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing configuration file: $required_file" >&2
    exit 1
  fi
done

for required_command in mpstat nohup setsid; do
  if ! command -v "$required_command" >/dev/null 2>&1; then
    echo "Missing required command: $required_command" >&2
    exit 1
  fi
done

if pgrep -f '[a]urora satellite-batch' >/dev/null 2>&1; then
  echo "A satellite-batch process is already running; refusing to start another." >&2
  pgrep -af '[a]urora satellite-batch' >&2 || true
  exit 1
fi

RUN_ID="full-year-2025-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$RUNS_ROOT/$RUN_ID"

mkdir -p "$RUN_DIR"
ln -sfn "$RUN_ID" "$RUNS_ROOT/latest"

nohup setsid bash -c '
  repo_root="$1"
  run_dir="$2"
  python="$3"
  supervisor="$4"
  aurora="$5"
  config="$6"
  site_config="$7"
  manifest="$8"
  start_utc="$9"
  end_utc="${10}"
  chunk_hours="${11}"
  download_workers="${12}"
  processing_workers="${13}"
  pipeline_queue_size="${14}"
  min_free_gib="${15}"
  max_stalled_failures="${16}"
  max_auth_failures="${17}"
  base_backoff_seconds="${18}"
  max_backoff_seconds="${19}"

  mpstat -P ALL 60 > "$run_dir/mpstat.log" 2>&1 &
  mpstat_pid=$!

  cleanup() {
    kill "$mpstat_pid" 2>/dev/null || true
    wait "$mpstat_pid" 2>/dev/null || true
  }
  trap cleanup EXIT

  cd "$repo_root"
  env \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    BLIS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    "$python" "$supervisor" \
      --run-dir "$run_dir" \
      --manifest "$manifest" \
      --disk-path "$repo_root" \
      --min-free-gib "$min_free_gib" \
      --max-stalled-failures "$max_stalled_failures" \
      --max-auth-failures "$max_auth_failures" \
      --base-backoff-seconds "$base_backoff_seconds" \
      --max-backoff-seconds "$max_backoff_seconds" \
      -- \
      "$aurora" satellite-batch \
      --config "$config" \
      --site-config "$site_config" \
      --start "$start_utc" \
      --end "$end_utc" \
      --chunk-hours "$chunk_hours" \
      --download-workers "$download_workers" \
      --processing-workers "$processing_workers" \
      --pipeline-queue-size "$pipeline_queue_size" \
      --delete-native

  run_status=$?
  printf "%s\n" "$run_status" > "$run_dir/exit-status"
  exit "$run_status"
' bash "$REPO_ROOT" "$RUN_DIR" "$PYTHON" "$SUPERVISOR" "$AURORA" \
  "$CONFIG" "$SITE_CONFIG" "$MANIFEST" "$START_UTC" "$END_UTC" \
  "$CHUNK_HOURS" "$DOWNLOAD_WORKERS" "$PROCESSING_WORKERS" \
  "$PIPELINE_QUEUE_SIZE" "$MIN_FREE_GIB" "$MAX_STALLED_FAILURES" \
  "$MAX_AUTH_FAILURES" "$BASE_BACKOFF_SECONDS" "$MAX_BACKOFF_SECONDS" \
  > "$RUN_DIR/launcher.log" 2>&1 < /dev/null &

RUN_PID=$!
printf '%s\n' "$RUN_PID" > "$RUN_DIR/pid"

cat <<EOF
Full-year satellite batch started.

Process group: $RUN_PID
Run directory: $RUN_DIR
Interval: $START_UTC to $END_UTC
Workers: download=$DOWNLOAD_WORKERS, processing=$PROCESSING_WORKERS
Chunk/queue: ${CHUNK_HOURS}h / $PIPELINE_QUEUE_SIZE

Follow progress:
  tail -f "$RUNS_ROOT/latest/satellite-batch.log"

Check the launcher process:
  ps -fp "\$(cat "$RUNS_ROOT/latest/pid")"

Check completion status (0 means success):
  cat "$RUNS_ROOT/latest/exit-status"

Check supervisor state and retry schedule:
  cat "$RUNS_ROOT/latest/supervisor-state.json"

Per-attempt logs:
  ls -1 "$RUNS_ROOT/latest"/attempt-*.log

Stop the complete process group if necessary:
  kill -- -"\$(cat "$RUNS_ROOT/latest/pid")"
EOF
