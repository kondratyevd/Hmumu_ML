#!/usr/bin/env bash
echo "Running..."
CURRENT_LOCATION=$(pwd)

method=""

case $1 in

	0)
		echo 'Running option 0: inclusive'
		SIG_INPUT_PATH="/home/dkondra/tmp/Run_2019-04-27_13-20-29/Keras_multi/model_50_D2_25_D2_25_D2/root/output_t*.root" # both train and test
		DATA_INPUT_PATH="/home/dkondra/tmp/Run_2019-04-27_13-20-29/Keras_multi/model_50_D2_25_D2_25_D2/root/output_Data.root"
		DATA_TREE="tree_Data"
		method="DNNmulti"
		min_var=1
		max_var=3
		nSteps=0 
		;;

	1)
		echo 'Running option 1: BDT'
		SIG_INPUT_PATH="/home/dkondra/tmp/BDTG_UCSD_hiStat_cs/all_signal/*.root" # both train and test
		DATA_INPUT_PATH="/home/dkondra/tmp/BDTG_UCSD_hiStat_cs/all_singleMu/*.root"
		DATA_TREE="tree"
		method="BDTmva"
		min_var=-1
		max_var=1
		nSteps=2
		nIter=1
		penalty=2
		;;


	2)
		echo 'Running option 2: DNN'
		SIG_INPUT_PATH="/home/dkondra/tmp/Run_2019-04-27_13-20-29/Keras_multi/model_50_D2_25_D2_25_D2/root/output_t*.root" # both train and test
		DATA_INPUT_PATH="/home/dkondra/tmp/Run_2019-04-27_13-20-29/Keras_multi/model_50_D2_25_D2_25_D2/root/output_Data.root"
		DATA_TREE="tree_Data"
		method="DNNmulti"
		min_var=1
		max_var=3
		nSteps=3
		nIter=2

		;;	


	3)
		echo 'Running option 3: Rapidity categorization'
		SIG_INPUT_PATH="/home/dkondra/tmp/Run_2019-04-27_13-20-29/Keras_multi/model_50_D2_25_D2_25_D2/root/output_t*.root" # both train and test
		DATA_INPUT_PATH="/home/dkondra/tmp/Run_2019-04-27_13-20-29/Keras_multi/model_50_D2_25_D2_25_D2/root/output_Data.root"
		DATA_TREE="tree_Data"
		method="Rapidity_multi"
		min_var=0
		max_var=2.4
		nSteps=5
		;;	

	3.2)
		echo 'Running option 3.2: Rapidity categorization (Roch corrected)'
		SIG_INPUT_PATH="/home/dkondra/tmp/BDTG_UCSD_hiStat_cs/all_signal/*.root" # both train and test
		DATA_INPUT_PATH="/home/dkondra/tmp/BDTG_UCSD_hiStat_cs/all_singleMu/*.root"
		DATA_TREE="tree"
		method="Rapidity_BDT"
		min_var=0
		max_var=2.4
		nSteps=5
		penalty=1
		nIter=2
		;;	

	4)
		echo 'Running option 4: DNN 20 categories'
		SIG_INPUT_PATH="/home/dkondra/tmp/Run_2019-04-27_13-20-29/Keras_multi/model_50_D2_25_D2_25_D2/root/output_t*.root" # both train and test
		DATA_INPUT_PATH="/home/dkondra/tmp/Run_2019-04-27_13-20-29/Keras_multi/model_50_D2_25_D2_25_D2/root/output_Data.root"
		DATA_TREE="tree_Data"
		method="DNNmulti"
		min_var=1
		max_var=3
		nSteps=20
		penalty=1
		nIter=10		
		;;	

	5)
		echo 'Running option 5: BDT 20 categories'
		SIG_INPUT_PATH="/home/dkondra/tmp/BDTG_UCSD_hiStat_cs/all_signal/*.root" # both train and test
		DATA_INPUT_PATH="/home/dkondra/tmp/BDTG_UCSD_hiStat_cs/all_singleMu/*.root"
		DATA_TREE="tree"
		method="BDTmva"
		min_var=-1
		max_var=1
		nSteps=20
		penalty=1
		nIter=10
		;;	

	*)
		echo 'Wrong option ' $1
		;;	
esac


# SIG_TREE="tree"
LUMI=41394.221
# LUMI=35866
TMP_PATH=/tmp/dkondra/categorization/output/scan_greedy_$1/
OUTPUT_PATH=output/test_greedy_$1/
mkdir -p $TMP_PATH
mkdir -p $OUTPUT_PATH

cd $TMP_PATH
python $CURRENT_LOCATION/categorize_greedy.py --option $1 --smodel '3gaus' --sig_in_path "$SIG_INPUT_PATH" --data_in_path "$DATA_INPUT_PATH" --out_path "$TMP_PATH"  --data_tree "$DATA_TREE" --method $method --min_var $min_var --max_var $max_var --nSteps $nSteps --lumi $LUMI --nIter $nIter --penalty $penalty

# rm datacard*
# rm workspace*
cd $CURRENT_LOCATION
cp "$TMP_PATH"/* "$OUTPUT_PATH"