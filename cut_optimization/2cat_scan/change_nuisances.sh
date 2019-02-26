#!/usr/bin/env bash

CURRENT_LOCATION=$(pwd)
SOURCE_DIR=/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/ggh_2017_nuis/
TARGET_DIR=/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/ggh_2017_nuis_0p001/
mkdir $TARGET_DIR
cp $SOURCE_DIR/datacard* $TARGET_DIR
cp $SOURCE_DIR/workspace* $TARGET_DIR

cd $TARGET_DIR
for filename in *.txt; do
	sed -i 's/param    0    1./param    0    0.01/g' $filename
	SUFF=$(echo "$filename" | sed "s|datacard\(.*\)\.txt|\1|");
    echo $filename
    combine -M Significance --expectSignal=1 -t -1 -n "$SUFF" -d $filename
done

cd $CURRENT_LOCATION