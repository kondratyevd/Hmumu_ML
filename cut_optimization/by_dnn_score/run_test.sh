#!/usr/bin/env bash

CURRENT_LOCATION=$(pwd)
SIG_INPUT_PATH="/home/dkondra/tmp/Run_2019-03-29_10-49-20/Keras_multi/model_50_D2_25_D2_25_D2/root/output_t*.root" # both train and test
DATA_INPUT_PATH="/home/dkondra/tmp/Run_2019-03-29_10-49-20/Keras_multi/model_50_D2_25_D2_25_D2/root/output_Data.root"
# SIG_TREE="tree"
DATA_TREE="tree_Data"
# LUMI=35866
OUTPUT_PATH=output/test/
mkdir $OUTPUT_PATH
python categorize.py --sig_in_path "$SIG_INPUT_PATH" --data_in_path "$DATA_INPUT_PATH" --out_path "$OUTPUT_PATH"  --data_tree "$DATA_TREE"
cd $OUTPUT_PATH
for filename in *.txt; do
    # this is to retrieve whatever there is between "datacard" and ".txt" and use as a suffix for combine output. 
    # I stole that from some stackexchange topic, don't really know how it works
    SUFF=$(echo "$filename" | sed "s|datacard\(.*\)\.txt|\1|");
    echo $filename
    combine -M Significance --expectSignal=1 -t -1 -n "$SUFF" -d $filename
done
cd $CURRENT_LOCATION