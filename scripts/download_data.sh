#!/usr/bin/env bash
# Download public perturbation datasets to /workspace/data
set -e
mkdir -p /workspace/data && cd /workspace/data
B=https://scverse-exampledata.s3.eu-west-1.amazonaws.com/pertpy
wget -c $B/adamson_2016_upr_perturb_seq.h5ad
wget -c $B/replogle_2022_rpe1.h5ad
wget -c $B/replogle_2022_k562_gwps.h5ad