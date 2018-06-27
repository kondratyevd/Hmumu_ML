import sys

import os, sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) ))


import setup
from setup import input_variables, description, variables, files, transf_list, inputDir, treePath, writer

import mva_methods
from mva_methods import methods

from locations import outputDir
from ROOT import TMVA, TFile


def PrepareDataloader():

	dataloader=TMVA.DataLoader("dataset")

	for var in [v for v in variables if v.use]:
		if var.isMultiDim:								# multidimensional variables (for example, two muons)
			for i in range(var.nItems):
				dataloader.AddVariable(var.name+"[%i]"%i, var.title+"[%i]"%i, var.units, var.type)

		else:											# flat variables (for example, number of jets)
			dataloader.AddVariable(var.name, var.title, var.units, var.type)
		
	for f in files:
		path = inputDir+f.name+'.root'
		thefile = TFile.Open(path)
		tree = thefile.Get(treePath)
		setattr(f, 'events', tree.GetEntries())
		tree.SetDirectory(0)
		thefile.Close()

		if f.isSignal:
			dataloader.AddSignalTree(tree,f.weight)
		else:
			dataloader.AddBackgroundTree(tree,f.weight)

	return dataloader



def BookMethods(factory, dataloader):
	writer.Write("\nMethods:\n")
	for method in methods:
		writer.Write(method.name+"		("+method.options+")\n\n")
		factory.BookMethod(dataloader, method.type, method.name, method.options)


