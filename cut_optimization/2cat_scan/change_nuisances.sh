#!/usr/bin/env bash

CURRENT_LOCATION=$(pwd)
SOURCE_DIR=/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/ggh_2017_nuis/
TARGET_DIR=/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/ggh_2017_nuis_0p01/
mkdir $TARGET_DIR
cp $SOURCE_DIR/* $TARGET_DIR

cd $TARGET_DIR
for filename in *.txt; do
	sed -i 's/param    0    1./param    0    0.1/g' $filename.txt
done

cd $CURRENT_LOCATION