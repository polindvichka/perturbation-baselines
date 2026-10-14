#!/usr/bin/env bash
# Download public perturbation datasets to $VCC_DATA/data (see src/paths.py)
set -e
DEST="${VCC_DATA:-$(cd "$(dirname "$0")/.." && pwd)}/data"
mkdir -p "$DEST" && cd "$DEST"
B=https://scverse-exampledata.s3.eu-west-1.amazonaws.com/pertpy
wget -c $B/replogle_2022_rpe1.h5ad
wget -c $B/replogle_2022_k562_gwps.h5ad
