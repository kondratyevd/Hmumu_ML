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
		min_var=-1
		max_var=1
		nSteps=4
		penalty=5 # in %
		;;


	2)
		echo 'Running option 2: DNN'
		SIG_INPUT_PATH="/home/dkondra/tmp/Run_2019-04-27_13-20-29/Keras_multi/model_50_D2_25_D2_25_D2/root/output_t*.root" # both train and test
		DATA_INPUT_PATH="/home/dkondra/tmp/Run_2019-04-27_13-20-29/Keras_multi/model_50_D2_25_D2_25_D2/root/output_Data.root"
		DATA_TREE="tree_Data"
		method="DNNmulti"
		min_var=1
		max_var=3
		nSteps=5
		;;	

	2.1)
		echo 'Running option 2.1: DNN'
		SIG_INPUT_PATH="/home/dkondra/tmp/Run_2019-04-27_13-20-29/Keras_multi/model_50_D2_25_D2_25_D2/root/output_t*.root" # both train and test
		DATA_INPUT_PATH="/home/dkondra/tmp/Run_2019-04-27_13-20-29/Keras_multi/model_50_D2_25_D2_25_D2/root/output_Data.root"
		DATA_TREE="tree_Data"
		method="DNNmulti"
		min_var=1
		max_var=3
		nSteps=3
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
		penalty=2
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
		penalty=2
		;;	

	4)
		echo 'Running option 4: DNN 10 categories'
		SIG_INPUT_PATH="/home/dkondra/tmp/Run_2019-04-27_13-20-29/Keras_multi/model_50_D2_25_D2_25_D2/root/output_t*.root" # both train and test
		DATA_INPUT_PATH="/home/dkondra/tmp/Run_2019-04-27_13-20-29/Keras_multi/model_50_D2_25_D2_25_D2/root/output_Data.root"
		DATA_TREE="tree_Data"
		method="DNNmulti"
		min_var=1
		max_var=3
		nSteps=10
		penalty=2
		;;	

	4.1)
		echo 'Running option 4.1: DNN 20 categories'
		SIG_INPUT_PATH="/home/dkondra/tmp/Run_2019-04-27_13-20-29/Keras_multi/model_50_D2_25_D2_25_D2/root/output_t*.root" # both train and test
		DATA_INPUT_PATH="/home/dkondra/tmp/Run_2019-04-27_13-20-29/Keras_multi/model_50_D2_25_D2_25_D2/root/output_Data.root"
		DATA_TREE="tree_Data"
		method="DNNmulti"
		min_var=1
		max_var=3
		nSteps=20
		penalty=1
		;;	
	4.2)
		echo 'Running option 4.2: DNN 20 categories'
		SIG_INPUT_PATH="/home/dkondra/tmp/Run_2019-04-27_13-20-29/Keras_multi/model_50_D2_25_D2_25_D2/root/output_t*.root" # both train and test
		DATA_INPUT_PATH="/home/dkondra/tmp/Run_2019-04-27_13-20-29/Keras_multi/model_50_D2_25_D2_25_D2/root/output_Data.root"
		DATA_TREE="tree_Data"
		method="DNNmulti"
		min_var=1
		max_var=3
		nSteps=20
		penalty=2
		;;	
	4.3)
		echo 'Running option 4.3: DNN 20 categories'
		SIG_INPUT_PATH="/home/dkondra/tmp/Run_2019-04-27_13-20-29/Keras_multi/model_50_D2_25_D2_25_D2/root/output_t*.root" # both train and test
		DATA_INPUT_PATH="/home/dkondra/tmp/Run_2019-04-27_13-20-29/Keras_multi/model_50_D2_25_D2_25_D2/root/output_Data.root"
		DATA_TREE="tree_Data"
		method="DNNmulti"
		min_var=1
		max_var=3
		nSteps=20
		penalty=3
		;;	
	4.4)
		echo 'Running option 4.4: DNN 20 categories'
		SIG_INPUT_PATH="/home/dkondra/tmp/Run_2019-04-27_13-20-29/Keras_multi/model_50_D2_25_D2_25_D2/root/output_t*.root" # both train and test
		DATA_INPUT_PATH="/home/dkondra/tmp/Run_2019-04-27_13-20-29/Keras_multi/model_50_D2_25_D2_25_D2/root/output_Data.root"
		DATA_TREE="tree_Data"
		method="DNNmulti"
		min_var=1
		max_var=3
		nSteps=20
		penalty=4
		;;	


	5)
		echo 'Running option 5: BDT 10 categories'
		SIG_INPUT_PATH="/home/dkondra/tmp/BDTG_UCSD_hiStat_cs/all_signal/*.root" # both train and test
		DATA_INPUT_PATH="/home/dkondra/tmp/BDTG_UCSD_hiStat_cs/all_singleMu/*.root"
		DATA_TREE="tree"
		method="BDTmva"
		min_var=-1
		max_var=1
		nSteps=10
		penalty=2
		;;	

	5.1)
		echo 'Running option 5.1: BDT 20 categories'
		SIG_INPUT_PATH="/home/dkondra/tmp/BDTG_UCSD_hiStat_cs/all_signal/*.root" # both train and test
		DATA_INPUT_PATH="/home/dkondra/tmp/BDTG_UCSD_hiStat_cs/all_singleMu/*.root"
		DATA_TREE="tree"
		method="BDTmva"
		min_var=-1
		max_var=1
		nSteps=20
		penalty=1
		;;	

	5.2)
		echo 'Running option 5.2: BDT 20 categories'
		SIG_INPUT_PATH="/home/dkondra/tmp/BDTG_UCSD_hiStat_cs/all_signal/*.root" # both train and test
		DATA_INPUT_PATH="/home/dkondra/tmp/BDTG_UCSD_hiStat_cs/all_singleMu/*.root"
		DATA_TREE="tree"
		method="BDTmva"
		min_var=-1
		max_var=1
		nSteps=20
		penalty=2
		;;	
	5.3)
		echo 'Running option 5.3: BDT 20 categories'
		SIG_INPUT_PATH="/home/dkondra/tmp/BDTG_UCSD_hiStat_cs/all_signal/*.root" # both train and test
		DATA_INPUT_PATH="/home/dkondra/tmp/BDTG_UCSD_hiStat_cs/all_singleMu/*.root"
		DATA_TREE="tree"
		method="BDTmva"
		min_var=-1
		max_var=1
		nSteps=20
		penalty=3
		;;	
	5.4)
		echo 'Running option 5.4: BDT 20 categories'
		SIG_INPUT_PATH="/home/dkondra/tmp/BDTG_UCSD_hiStat_cs/all_signal/*.root" # both train and test
		DATA_INPUT_PATH="/home/dkondra/tmp/BDTG_UCSD_hiStat_cs/all_singleMu/*.root"
		DATA_TREE="tree"
		method="BDTmva"
		min_var=-1
		max_var=1
		nSteps=20
		penalty=4
		;;	


	6.1)
		echo 'Running option 6.1: UCSD samples, bdtuf'
		SIG_INPUT_PATH=""
		DATA_INPUT_PATH=""
		DATA_TREE=""
		method="UCSD_bdtuf"
		min_var=-1
		max_var=1
		nSteps=200
		penalty=1
		;;	

	6.2)
		echo 'Running option 6.2: UCSD samples, bdtucsd_inclusive'
		SIG_INPUT_PATH=""
		DATA_INPUT_PATH=""
		DATA_TREE=""
		method="UCSD_bdtucsd_inclusive"
		min_var=-1
		max_var=1
		nSteps=200
		penalty=3
		;;	

	6.3)
		echo 'Running option 6.3: UCSD samples, bdtucsd_01jet'
		SIG_INPUT_PATH=""
		DATA_INPUT_PATH=""
		DATA_TREE=""
		method="UCSD_bdtucsd_01jet"
		min_var=-1
		max_var=1
		nSteps=200
		penalty=3
		;;	

	6.4)
		echo 'Running option 6.4: UCSD samples, bdtucsd_2jet'
		SIG_INPUT_PATH=""
		DATA_INPUT_PATH=""
		DATA_TREE=""
		method="UCSD_bdtucsd_2jet"
		min_var=-1
		max_var=1
		nSteps=200
		penalty=3
		;;	


	0)
		echo 'Running option 0: test run'
		SIG_INPUT_PATH=""
		DATA_INPUT_PATH=""
		DATA_TREE=""
		method="UCSD_bdtucsd_inclusive"
		min_var=-1
		max_var=1
		nSteps=100
		penalty=1
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

python $CURRENT_LOCATION/categorize_dynamic.py --option $1 --smodel '3gaus' --sig_in_path "$SIG_INPUT_PATH" --data_in_path "$DATA_INPUT_PATH" --out_path "$TMP_PATH"  --data_tree "$DATA_TREE" --method $method --min_var $min_var --max_var $max_var --nSteps $nSteps --lumi $LUMI --penalty $penalty

rm datacard*
rm workspace*
cd $CURRENT_LOCATION
cp "$TMP_PATH"/* "$OUTPUT_PATH"