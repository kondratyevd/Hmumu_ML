#!/usr/bin/env bash
# root -l -q load.C
CURRENT_LOCATION=$(pwd)
SIG_INPUT_PATH="/mnt/hadoop/store/user/dkondrat/skim/2016/H2Mu_gg/H2Mu_gg.root"
DATA_INPUT_PATH="/mnt/hadoop/store/user/dkondrat/skim/2016/SingleMu_2016/*root"
SIG_TREE="tree"
DATA_TREE="dimuons/tree"
LUMI=35866
OUTPUT_PATH=output/ggh_2016_dcb_nuis/
mkdir $OUTPUT_PATH
python categorize.py --nuis --nuis_val 0.1 --smodel dcb --sig_in_path "$SIG_INPUT_PATH" --data_in_path "$DATA_INPUT_PATH" --out_path "$OUTPUT_PATH" --sig_tree "$SIG_TREE" --data_tree "$DATA_TREE" --lumi "$LUMI"
cd $OUTPUT_PATH
for filename in *.txt; do
    # this is to retrieve whatever there is between "datacard" and ".txt" and use as a suffix for combine output.
    # I stole that from some stackexchange topic, don't really know how it works
    SUFF=$(echo "$filename" | sed "s|datacard\(.*\)\.txt|\1|");
    echo $filename
    combine -M Significance --expectSignal=1 -t -1 -n "$SUFF" -d $filename --LoadLibrary /home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/RooDCBShape_cxx.so
done
cd $CURRENT_LOCATION