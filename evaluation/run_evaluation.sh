#!/bin/bash
set -euo pipefail

python evaluate_midge.py --repeat 3 "$@"
