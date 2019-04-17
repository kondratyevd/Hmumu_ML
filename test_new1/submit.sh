#!/bin/sh

python src/prepare.py
RunID=$(cat "output/CURRENT_RUN_ID")

case $1 in
	# 0)
	# 	echo 'Running option 0'
	# 	python test_new/test_dnn.py
	# 	;;
	1.1)
		echo 'Running option 1.1'
		python test_new1/test_dnn_hiStat_m110To150.py
		;;
	1.2)
		echo 'Running option 1.2'
		python test_new1/test_dnn_hiStat_m110To150_CS.py
		;;
	1.3)
		echo 'Running option 1.3'
		python test_new1/test_dnn_hiStat_m120To130.py
		;;
	1.4)
		echo 'Running option 1.4'
		python test_new1/test_dnn_hiStat_m120To130_CS.py
		;;

	*)
		echo 'Wrong option ' $1;;		

esac

