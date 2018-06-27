#!/usr/bin/env python
from ROOT import TMVA, TFile, TCut, gROOT
import os, sys
sys.path.append( os.path.dirname(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) ))

from setup import transf_list, writer

from locations import outRootFilePath, PrepareRun
PrepareRun()

from helpers import PrepareDataloader, BookMethods

def Train():

	writer.Write("-"*60)
	writer.Write("\nUsing package: TMVA\n")

	TMVA.Tools.Instance()
	TMVA.PyMethodBase.PyInitialize()
	
	outputFile = TFile.Open( outRootFilePath, "RECREATE" )
	
	transformations = ';'.join(transf_list)
	factory = TMVA.Factory( "TMVAClassification", outputFile, "!V:!Silent:Color:DrawProgressBar:Transformations=%s:AnalysisType=Classification"%transformations)
	
	dataloader = PrepareDataloader() # Adding files and variables 
	
	dataloader.PrepareTrainingAndTestTree(TCut(''), 'nTrain_Signal=0:nTrain_Background=0:SplitMode=Random:NormMode=NumEvents:!V')
	
	BookMethods(factory, dataloader) 
	
	factory.TrainAllMethods()
	factory.TestAllMethods()
	factory.EvaluateAllMethods()
	
	outputFile.Close()