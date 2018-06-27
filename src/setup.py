#!/usr/bin/env python
###############################################  Imports  #####################################################

from datetime import datetime
import sys, os, errno
from config import *
from info_writer import Writer
###########################################  Run description  #################################################



with open ("output/CURRENT_RUN_ID", "r") as IDfile:
    RunID=IDfile.read()


description = "test"

writer = Writer("output/"+RunID)
writer.Write(description)


writer.Write("\n\nFiles:\n\n")
for f in files:
	writer.Write("%80s.root		weight=%f\n"%(inputDir+f.name,f.weight))
writer.Write("\nTree path: %s"%treePath)

###############################################  Variables  ###################################################

input_variables = {}

input_variables["muPairs.pt"] = 1 	
input_variables["muPairs.dEta"] = 1
input_variables["muPairs.dPhi"] = 1
input_variables["muons.pt"] = 2	
input_variables["muons.eta"] = 2	
input_variables["muons.phi"] = 2		
# input_variables["nJets"] = 1
# input_variables["jets.pt"] = 1
# input_variables["jets.eta"] = 2
# input_variables["jets.phi"] = 2

numOfVar = 0

for i in input_variables:
	if i not in [v.name for v in variables]:
		sys.exit("\n\nERROR: Variable %s not found in the list. Check this file: %s\n\n"%(i,config.__file__))

writer.Write("\n\n\nVariables:\n\n")

for var in variables:
	if var.name in input_variables.keys():
		if input_variables[var.name] > var.itemsExpected:
			sys.exit("Too many items required for %s"%var.name)
		else:
			setattr(var, 'use', True)
			setattr(var, 'nItems', input_variables[var.name])
			writer.Write("%s	x%i\n"%(var.name, var.nItems))
			numOfVar = numOfVar+var.nItems
	
	else:
		setattr(var, 'use', False)
writer.Write("\n\n")

###############################################################################################################




usePackage = []
usePackage.append("TMVA")
usePackage.append("Keras")

############################################### Transformations ###############################################

transf_list = []

transf_list.append('I')
transf_list.append('N,G,D')
transf_list.append('N,G,P')









