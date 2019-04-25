#!/bin/sh

python src/prepare.py
RunID=$(cat "output/CURRENT_RUN_ID")

case $1 in
	# 0)
	# 	echo 'Running option 0'
	# 	python test_new/test_dnn.py
	# 	;;
	3.1)
		echo 'Running option 3.1'
		python test_new3/test_bdt_ucsd_hiStat.py
		;;
	3.2)
		echo 'Running option 3.2'
		python test_new3/test_bdt_ucsd_hiStat_ebe.py
		;;
	3.3)
		echo 'Running option 3.3'
		python test_new3/test_dnn_hiStat_m120To130_CS.py
		;;
	3.4)
		echo 'Running option 3.4'
		python test_new3/test_dnn_hiStat_m120To130_noSingleMu.py
		;;

	3.5)
		echo 'Running option 3.5'
		python test_new3/test_dnn_hiStat_m120To130_options.py
		;;


	*)
		echo 'Wrong option ' $1;;		

esac

