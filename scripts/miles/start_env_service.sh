#!/usr/bin/env bash
# Start seta_env env_service for miles RL training.
#
# Usage:
#   bash scripts/miles/start_env_service.sh
#
# Environment variables (with defaults):
#   DATASET_ROOT    Path to dataset folder (default: dataset)
#   MAX_SLOTS       Max concurrent trials (default: 8)
#   PORT            Service port (default: 8002)
#   HOST            Bind address (default: 0.0.0.0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8002}"
MAX_SLOTS="${MAX_SLOTS:-8}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/dataset}"

CONFIG="${SCRIPT_DIR}/seta_env_config.yaml"

echo "Starting seta_env env_service"
echo "  config:       $CONFIG"
echo "  dataset_root: $DATASET_ROOT"
echo "  max_slots:    $MAX_SLOTS"
echo "  host:port:    $HOST:$PORT"
echo ""

cd "$REPO_ROOT"

exec env \
    ENV_SERVICE_CONFIG="$CONFIG" \
    DATASET_ROOT="$DATASET_ROOT" \
    MAX_SLOTS="$MAX_SLOTS" \
    python -m uvicorn seta_env.services.env_service:app \
        --host "$HOST" \
        --port "$PORT"
