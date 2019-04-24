#!/usr/bin/env bash
echo "Running..."
CURRENT_LOCATION=$(pwd)

OUTPUT_PATH=plots/

mkdir -p $OUTPUT_PATH
python make_plots.py --out_path $OUTPUT_PATH --label test

