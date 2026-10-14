#!/usr/bin/env bash
# Move code and results between this laptop and the rented machine ("Host vast" in ~/.ssh/config).
# Code is edited and committed on the laptop; the machine only runs it.
#
#   scripts/sync.sh        push the code to the machine
#   scripts/sync.sh pull   copy the computed tables (.npz, .csv) back into ./processed
set -e
HOST="${VAST_HOST:-vast}"
REMOTE_REPO=/workspace/perturbation-baselines
REMOTE_PROCESSED=/workspace/processed
cd "$(dirname "$0")/.."
SSH="ssh -o ClearAllForwardings=yes"      # the ssh config forwards port 8080, which clashes with an open session

if [ "${1:-push}" = "pull" ]; then
    mkdir -p processed
    rsync -av -e "$SSH" --include '*/' --include '*.npz' --include '*.csv' --exclude '*' \
        "$HOST:$REMOTE_PROCESSED/" processed/
else
    rsync -av --delete -e "$SSH" \
        --exclude .git/ --exclude .venv/ --exclude __pycache__/ --exclude .DS_Store \
        --exclude data/ --exclude processed/ --exclude controls/ --exclude notes/ \
        ./ "$HOST:$REMOTE_REPO/"
fi
