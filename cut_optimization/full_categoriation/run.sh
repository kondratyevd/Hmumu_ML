#!/usr/bin/env bash
echo "Running..."
CURRENT_LOCATION=$(pwd)

case $1 in
	0)
		echo 'Running option 0'
		# SIG_INPUT_PATH="/scratch/gilbreth/dkondra/ML_output/Run_2019-03-29_10-49-20/Keras_multi/model_50_D2_25_D2_25_D2/root/output_t*.root" # both train and test
		# DATA_INPUT_PATH="/scratch/gilbreth/dkondra/ML_output/Run_2019-03-29_10-49-20/Keras_multi/model_50_D2_25_D2_25_D2/root/output_Data.root"
		exit 1;;
	1)
		echo 'Running option 1'
		# SIG_INPUT_PATH="/scratch/gilbreth/dkondra/ML_output/Run_2019-03-29_10-49-20/Keras_multi/model_50_D2_25_D2_25_D2/root/output_t*.root" # both train and test
		# DATA_INPUT_PATH="/scratch/gilbreth/dkondra/ML_output/Run_2019-03-29_10-49-20/Keras_multi/model_50_D2_25_D2_25_D2/root/output_Data.root"
		exit 1;;
	2)
		echo 'Running option 2'
		# SIG_INPUT_PATH="/scratch/gilbreth/dkondra/ML_output/Run_2019-03-29_10-49-20/Keras_multi/model_50_D2_25_D2_25_D2/root/output_t*.root" # both train and test
		# DATA_INPUT_PATH="/scratch/gilbreth/dkondra/ML_output/Run_2019-03-29_10-49-20/Keras_multi/model_50_D2_25_D2_25_D2/root/output_Data.root"
		exit 1;;
	3)
		echo 'Running option 3'
		# SIG_INPUT_PATH="/scratch/gilbreth/dkondra/ML_output/Run_2019-03-29_10-49-20/Keras_multi/model_50_D2_25_D2_25_D2/root/output_t*.root" # both train and test
		# DATA_INPUT_PATH="/scratch/gilbreth/dkondra/ML_output/Run_2019-03-29_10-49-20/Keras_multi/model_50_D2_25_D2_25_D2/root/output_Data.root"
		exit 1;;
	4)
		echo 'Running option 4'
		# SIG_INPUT_PATH="/scratch/gilbreth/dkondra/ML_output/Run_2019-03-29_10-49-20/Keras_multi/model_50_D2_25_D2_25_D2/root/output_t*.root" # both train and test
		# DATA_INPUT_PATH="/scratch/gilbreth/dkondra/ML_output/Run_2019-03-29_10-49-20/Keras_multi/model_50_D2_25_D2_25_D2/root/output_Data.root"
		exit 1;;
	5)
		echo 'Running option 5'
		# SIG_INPUT_PATH="/scratch/gilbreth/dkondra/ML_output/Run_2019-03-29_10-49-20/Keras_multi/model_50_D2_25_D2_25_D2/root/output_t*.root" # both train and test
		# DATA_INPUT_PATH="/scratch/gilbreth/dkondra/ML_output/Run_2019-03-29_10-49-20/Keras_multi/model_50_D2_25_D2_25_D2/root/output_Data.root"
		exit 1;;
	6)
		echo 'Running option 6'
		SIG_INPUT_PATH="/scratch/gilbreth/dkondra/ML_output/Run_2019-04-08_11-37-05/Keras_multi/model_50_D2_25_D2_25_D2/root/output_t*.root" # both train and test
		DATA_INPUT_PATH="/scratch/gilbreth/dkondra/ML_output/Run_2019-04-08_11-37-05/Keras_multi/model_50_D2_25_D2_25_D2/root/output_Data.root"
		;;
	7)
		echo 'Running option 7'
		SIG_INPUT_PATH="/scratch/gilbreth/dkondra/ML_output/Run_2019-04-08_11-51-21/Keras_multi/model_50_D2_25_D2_25_D2/root/output_t*.root" # both train and test
		DATA_INPUT_PATH="/scratch/gilbreth/dkondra/ML_output/Run_2019-04-08_11-51-21/Keras_multi/model_50_D2_25_D2_25_D2/root/output_Data.root"
		;;
	8)
		echo 'Running option 8'
		SIG_INPUT_PATH="/scratch/gilbreth/dkondra/ML_output/Run_2019-04-08_11-37-09/Keras_multi/model_50_D2_25_D2_25_D2/root/output_t*.root" # both train and test
		DATA_INPUT_PATH="/scratch/gilbreth/dkondra/ML_output/Run_2019-04-08_11-37-09/Keras_multi/model_50_D2_25_D2_25_D2/root/output_Data.root"
		;;
	9)
		echo 'Running option 9'
		SIG_INPUT_PATH="/scratch/gilbreth/dkondra/ML_output/Run_2019-04-08_11-37-12/Keras_multi/model_50_D2_25_D2_25_D2/root/output_t*.root" # both train and test
		DATA_INPUT_PATH="/scratch/gilbreth/dkondra/ML_output/Run_2019-04-08_11-37-12/Keras_multi/model_50_D2_25_D2_25_D2/root/output_Data.root"
		;;
	10)
		echo 'Running option 10'
		SIG_INPUT_PATH="/scratch/gilbreth/dkondra/ML_output/Run_2019-04-08_11-37-16/Keras_multi/model_50_D2_25_D2_25_D2/root/output_t*.root" # both train and test
		DATA_INPUT_PATH="/scratch/gilbreth/dkondra/ML_output/Run_2019-04-08_11-37-16/Keras_multi/model_50_D2_25_D2_25_D2/root/output_Data.root"
		;;

	*)
		echo 'Wrong option ' $1
		exit 1;;	



# SIG_TREE="tree"
DATA_TREE="tree_Data"
# LUMI=35866
OUTPUT_PATH=output/test_$1/
mkdir $OUTPUT_PATH
python categorize.py --option $1 --smodel '3gaus' --sig_in_path "$SIG_INPUT_PATH" --data_in_path "$DATA_INPUT_PATH" --out_path "$OUTPUT_PATH"  --data_tree "$DATA_TREE"
cd $OUTPUT_PATH
for filename in *.txt; do
    # this is to retrieve whatever there is between "datacard" and ".txt" and use as a suffix for combine output. 
    # I stole that from some stackexchange topic, don't really know how it works
    SUFF=$(echo "$filename" | sed "s|datacard\(.*\)\.txt|\1|");
    echo $filename
    combine -M Significance --expectSignal=1 -t -1 -n "$SUFF" -d $filename
done
cd $CURRENT_LOCATION