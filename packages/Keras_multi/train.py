#!/usr/bin/env python
import os, sys
sys.path.append( os.path.dirname(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) ))

from kerasMultiTrainer import KerasMultiTrainer

def train(framework, package):
	with KerasMultiTrainer(framework, package) as t:
		t.convert_to_pandas()
		t.train_models()	

