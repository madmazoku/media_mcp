#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
exec python3 "$DIR/src/cli.py" "$@"
