#!/usr/bin/env python
from ROOT import TMVA, TFile, TCut, gROOT
import os, sys
sys.path.append( os.path.dirname(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) ))

from tmvaTrainer import TMVATrainer

def train(framework, package):
	with TMVATrainer(framework, package) as t:
		t.train_methods()