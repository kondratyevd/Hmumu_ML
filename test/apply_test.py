import os, sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from src.classifier import Framework

import ROOT 
from ROOT import gInterpreter, gSystem
gInterpreter.ProcessLine('#include "interface/JetInfo.h"')
gInterpreter.ProcessLine('#include "interface/EventInfo.h"')
gInterpreter.ProcessLine('#include "interface/VertexInfo.h"')
gInterpreter.ProcessLine('#include "interface/JetPairInfo.h"')
gInterpreter.ProcessLine('#include "interface/MuonInfo.h"')
gInterpreter.ProcessLine('#include "interface/MuPairInfo.h"')
gInterpreter.ProcessLine('#include "interface/EleInfo.h"')
gInterpreter.ProcessLine('#include "interface/MetInfo.h"')
gInterpreter.ProcessLine('#include "interface/MhtInfo.h"')
gInterpreter.ProcessLine('#include "interface/GenParentInfo.h"')
gInterpreter.ProcessLine('#include "interface/GenMuonInfo.h"')
gInterpreter.ProcessLine('#include "interface/GenMuPairInfo.h"')
gInterpreter.ProcessLine('#include "interface/GenJetInfo.h"')

c = Framework()
treePath = 'dimuons/tree'

# c.add_data_to_evaluate('SingleMu2017B', '/mnt/hadoop/store/user/dkondrat/SingleMuon/SingleMu_2017B/190327_172730/0000/*root')
c.add_mc_to_evaluate('ZJets_AMC', '/mnt/hadoop/store/user/dkondrat//DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/ZJets_AMC/190323_195010/0000//tuple_1*root', 5765.4)


c.set_tree_path(treePath)

c.add_variable("muPairs.pt", 				1) 
c.add_variable("muPairs.eta", 				1)
c.add_variable("muPairs.dEta", 				1) 
c.add_variable("muPairs.dPhi", 				1)
c.add_variable("met.pt", 					1)
c.add_variable("nJetsCent", 				1)
c.add_variable("nJetsFwd",					1)
c.add_variable("nBMed",						1)
c.add_variable("jets.eta",					2)
c.add_variable("jetPairs.dEta",				1)
c.add_variable("jetPairs.mass",				1)


c.set_year("2017")
c.weigh_by_event(True)

c.add_package("TMVA")
c.add_method("BDTG_UF_v1")

c.setApplication(outPath="/home/dkondra/tmp/BDTG_UF_v1/", xmlPath="/home/dkondra/Hmumu_analysis/Hmumu_ML/dataset/weights/BDTG_UF_v1/TMVAClassification_BDTG_UF_v1.weights.xml")
c.apply_methods()

