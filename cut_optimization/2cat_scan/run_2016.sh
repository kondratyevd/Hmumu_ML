#!/usr/bin/env bash

CURRENT_LOCATION=$(pwd)
# SIG_INPUT_PATH=../../output/Run_2018-12-19_14-25-02/Keras_multi/model_50_D2_25_D2_25_D2/root/
SIG_INPUT_PATH="/mnt/hadoop/store/user/dkondrat/skim/2016/H2Mu_gg/H2Mu_gg.root"
DATA_INPUT_PATH="/mnt/hadoop/store/user/dkondrat/skim/2016/SingleMu_2016/*root"
SIG_TREE="dimuons/tree"
DATA_TREE="dimuons/tree"
LUMI=35866
OUTPUT_PATH=output/
python categorize.py --suff _2016 --sig_in_path "$SIG_INPUT_PATH" --data_in_path "$DATA_INPUT_PATH" --out_path "$OUTPUT_PATH" --sig_tree "$SIG_TREE" --data_tree "$DATA_TREE" --lumi "$LUMI"
cd $OUTPUT_PATH
for filename in *.txt; do
    # this is to retrieve whatever there is between "datacard" and ".txt" and use as a suffix for combine output. 
    # I stole that from some stackexchange topic, don't really know how it works
    SUFF=$(echo "$filename" | sed "s|datacard\(.*\)\.txt|\1|");
    echo $filename
    combine -M Significance --expectSignal=1 -t -1 -n "$SUFF" -d $filename
done
cd $CURRENT_LOCATION