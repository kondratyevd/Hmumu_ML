#!/usr/bin/env bash
echo "Running..."
CURRENT_LOCATION=$(pwd)

method=""

case $1 in

	0)
		echo 'Running option 0: inclusive'
		SIG_INPUT_PATH="/scratch/gilbreth/dkondra/ML_output/Run_2019-04-21_11-06-40/Keras_multi/model_50_D2_25_D2_25_D2/root/output_t*.root" # both train and test
		DATA_INPUT_PATH="/scratch/gilbreth/dkondra/ML_output/Run_2019-04-21_11-06-40/Keras_multi/model_50_D2_25_D2_25_D2/root/output_Data.root"
		DATA_TREE="tree_Data"
		method="DNNmulti"
		min_mva=1
		max_mva=3
		nSteps=0 
		;;

	0.1.1)
		echo 'Running option 0.1.1: BDT, cut 1, scan 1'
		SIG_INPUT_PATH="/home/dkondra/tmp/BDTG_UCSD_hiStat_cs/all_signal/*.root" # both train and test
		DATA_INPUT_PATH="/home/dkondra/tmp/BDTG_UCSD_hiStat_cs/all_singleMu/*.root"
		DATA_TREE="tree"
		method="BDT"
		min_mva=-1
		max_mva=1
		nSteps=10 
		;;

	0.1.2)
		echo 'Running option 0.1.2: BDT, cut 1, scan 2'
		SIG_INPUT_PATH="/home/dkondra/tmp/BDTG_UCSD_hiStat_cs/all_signal/*.root" # both train and test
		DATA_INPUT_PATH="/home/dkondra/tmp/BDTG_UCSD_hiStat_cs/all_singleMu/*.root"
		DATA_TREE="tree"
		method="BDT"
		min_mva=-0.4	
		max_mva=0
		nSteps=20 
		;;

	0.1.3)
		echo 'Running option 0.1.3: BDT, cut 1, scan 3'
		SIG_INPUT_PATH="/home/dkondra/tmp/BDTG_UCSD_hiStat_cs/all_signal/*.root" # both train and test
		DATA_INPUT_PATH="/home/dkondra/tmp/BDTG_UCSD_hiStat_cs/all_singleMu/*.root"
		DATA_TREE="tree"
		method="BDT"
		min_mva=0.2	
		max_mva=0.8
		nSteps=30 
		;;

	1.1.1)
		echo 'Running option 1.1.1: DNN, cut 1, scan 1'
		SIG_INPUT_PATH="/scratch/gilbreth/dkondra/ML_output/Run_2019-04-21_11-06-40/Keras_multi/model_50_D2_25_D2_25_D2/root/output_t*.root" # both train and test
		DATA_INPUT_PATH="/scratch/gilbreth/dkondra/ML_output/Run_2019-04-21_11-06-40/Keras_multi/model_50_D2_25_D2_25_D2/root/output_Data.root"
		DATA_TREE="tree_Data"
		method="DNNmulti"
		min_mva=1
		max_mva=3
		nSteps=10 
		;;	

	1.1.2)
		echo 'Running option 1.1.2: DNN, cut 1, scan 2'
		SIG_INPUT_PATH="/scratch/gilbreth/dkondra/ML_output/Run_2019-04-21_11-06-40/Keras_multi/model_50_D2_25_D2_25_D2/root/output_t*.root" # both train and test
		DATA_INPUT_PATH="/scratch/gilbreth/dkondra/ML_output/Run_2019-04-21_11-06-40/Keras_multi/model_50_D2_25_D2_25_D2/root/output_Data.root"
		DATA_TREE="tree_Data"
		method="DNNmulti"
		min_mva=1.2
		max_mva=1.8
		nSteps=30 
		;;	





	*)
		echo 'Wrong option ' $1
		;;	
esac


# SIG_TREE="tree"

# LUMI=35866
OUTPUT_PATH=output/scan_$1/
mkdir -p $OUTPUT_PATH
python categorize_scan.py --option $1 --smodel '3gaus' --sig_in_path "$SIG_INPUT_PATH" --data_in_path "$DATA_INPUT_PATH" --out_path "$OUTPUT_PATH"  --data_tree "$DATA_TREE" --method $method --min_mva $min_mva --max_mva $max_mva --nSteps $nSteps
cd $OUTPUT_PATH
for filename in *.txt; do
    # this is to retrieve whatever there is between "datacard" and ".txt" and use as a suffix for combine output. 
    # I stole that from some stackexchange topic, don't really know how it works
    SUFF=$(echo "$filename" | sed "s|datacard\(.*\)\.txt|\1|");
    echo $filename
    combine -M Significance --expectSignal=1 -t -1 -n "$SUFF" -d $filename
done
rm datacard*
rm workspace*
cd $CURRENT_LOCATION