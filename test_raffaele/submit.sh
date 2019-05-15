#!/bin/sh

python src/prepare.py
RunID=$(cat "output/CURRENT_RUN_ID")

case $1 in
	0)
		echo 'Running option 0'
		python test_raffaele/test_dnn.py
		;;

	*)
		echo 'Wrong option ' $1;;		

esac

