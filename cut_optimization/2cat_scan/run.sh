#!/usr/bin/env bash

CURRENT_LOCATION=$(pwd)
INPUT_PATH=../../output/Run_2018-12-19_14-25-02/Keras_multi/model_50_D2_25_D2_25_D2/root/
OUTPUT_PATH=output/
python categorize.py --in_path $INPUT_PATH --out_path $OUTPUT_PATH
cd $OUTPUT_PATH
for filename in *.txt; do
    # this is to retrieve whatever there is between "datacard" and ".txt" and use as a suffix for combine output. 
    # I stole that from some stacexchange topic, don't really know how it works
    SUFF=$(echo "$filename" | sed "s|datacard\(.*\)\.txt|\1|");
    echo $filename
    combine -M Significance --expectSignal=1 -t -1 -n "$SUFF" -d $filename
done
cd $CURRENT_LOCATION