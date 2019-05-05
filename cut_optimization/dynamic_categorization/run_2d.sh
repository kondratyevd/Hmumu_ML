#!/usr/bin/env bash
echo "Running..."
CURRENT_LOCATION=$(pwd)

method=""


case $1 in


	1)
		echo 'Running option 1: BDT'
		SIG_INPUT_PATH="/home/dkondra/tmp/BDTG_UCSD_hiStat_cs/all_signal/*.root" # both train and test
		DATA_INPUT_PATH="/home/dkondra/tmp/BDTG_UCSD_hiStat_cs/all_singleMu/*.root"
		DATA_TREE="tree"
		method="BDTmva"
		min_var1=-1
		max_var1=1
		nSteps1=2
		min_var2=0
		max_var2=2.4
		nSteps2=5
		penalty=2 # in %
		;;

	2)
		echo 'Running option 2: BDT'
		SIG_INPUT_PATH="/home/dkondra/tmp/BDTG_UCSD_hiStat_cs/all_signal/*.root" # both train and test
		DATA_INPUT_PATH="/home/dkondra/tmp/BDTG_UCSD_hiStat_cs/all_singleMu/*.root"
		DATA_TREE="tree"
		method="BDTmva"
		min_var1=-1
		max_var1=1
		nSteps1=20
		min_var2=0
		max_var2=2.4
		nSteps2=24
		penalty=2 # in %
		;;


	*)
		echo 'Wrong option ' $1
		;;	
esac


LUMI=41394.221
TMP_PATH=/tmp/dkondra/categorization/output/scan_$1/
OUTPUT_PATH=output/test_run_$1/
mkdir -p $TMP_PATH
mkdir -p $OUTPUT_PATH

cd $TMP_PATH

python $CURRENT_LOCATION/categorize_dynamic_2d.py --option $1 --smodel '3gaus' --sig_in_path "$SIG_INPUT_PATH" --data_in_path "$DATA_INPUT_PATH" --out_path "$TMP_PATH"  --data_tree "$DATA_TREE" --method $method --min_var1 $min_var1 --max_var1 $max_var1 --nSteps1 $nSteps1 --min_var2 $min_var2 --max_var2 $max_var2 --nSteps2 $nSteps2 --lumi $LUMI --penalty $penalty

rm datacard*
rm workspace*
cd $CURRENT_LOCATION
cp "$TMP_PATH"/* "$OUTPUT_PATH"