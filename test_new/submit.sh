#!/bin/sh

python src/prepare.py
RunID=$(cat "output/CURRENT_RUN_ID")

case $1 in
	0)
		echo 'Running option 0'
		python test_new/test_bdt_uf.py
		;;
	1)
		echo 'Running option 1'
		python test_new/test_bdt_uf_hiStat.py
		;;
	2)
		echo 'Running option 2'
		python test_new/test_bdt_uf_hiStat_ebe.py
		;;
	3)
		echo 'Running option 3'
		python test_new/test_bdt_ucsd.py
		;;
	4)
		echo 'Running option 4'
		python test_new/test_bdt_ucsd_hiStat.py
		;;
	5)
		echo 'Running option 5'
		python test_new/test_bdt_ucsd_hiStat_ebe.py
		;;
	6)
		echo 'Running option 6'
		python test_new/test_dnn_multi.py
		;;
	7)
		echo 'Running option 7'
		python test_new/test_dnn_multi_hiStat.py
		;;
	8)
		echo 'Running option 8'
		python test_new/test_dnn_multi_hiStat_ebe.py
		;;
	9)
		echo 'Running option 9'
		python test_new/test_dnn_binary_hiStat.py
		;;
	10)
		echo 'Running option 10'
		python test_new/test_dnn_binary_hiStat_ebe.py
		;;
	11)
		echo 'Running option 11'
		python test_new/test_dnn_binary.py
		;;
	12)
		echo 'Running option 12'
		python test_new/test_dnn_multi_hiStat_m120To130.py
		;;
	13)
		echo 'Running option 13'
		python test_new/test_dnn_multi_hiStat_m120To130_repeated.py
		;;	

	*)
		echo 'Wrong option ' $1;;		

esac

