#!/bin/sh

python src/prepare.py
RunID=$(cat "output/CURRENT_RUN_ID")
python test/test_hadoop_2016_MIT.py 
#&
#tensorboard --logdir=output/$RunID/keras_logs/ --port=6007
#wait
# mv dataset output/$RunID/TMVA/