#!/usr/bin/env python
from ROOT import TMVA, TFile, TCut, gROOT
import os, sys
sys.path.append( os.path.dirname(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) ))

from tmvaApplicator import TMVAApplicator

def apply(framework, package):
	with TMVAApplicator(framework) as t:
		pass