#!/bin/sh

python src/prepare.py
RunID=$(cat "output/CURRENT_RUN_ID")

case $1 in
	# 0)
	# 	echo 'Running option 0'
	# 	python test_new/test_dnn.py
	# 	;;
	2.1)
		echo 'Running option 2.1'
		python test_new2/test_dnn_hiStat_m110To150.py
		;;
	2.2)
		echo 'Running option 2.2'
		python test_new2/test_dnn_hiStat_m110To150_CS.py
		;;
	2.3)
		echo 'Running option 2.3'
		python test_new2/test_dnn_hiStat_m120To130.py
		;;
	2.4)
		echo 'Running option 2.4'
		python test_new2/test_dnn_hiStat_m120To130_CS.py
		;;

	*)
		echo 'Wrong option ' $1;;		

esac

