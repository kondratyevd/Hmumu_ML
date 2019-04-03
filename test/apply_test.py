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

c.add_data_to_evaluate('SingleMu2017B', '/mnt/hadoop/store/user/dkondrat/SingleMuon/SingleMu_2017B/190327_172730/0000/*root')
# c.add_data_to_evaluate('SingleMu2016C', '/mnt/hadoop/store/user/dkondrat/SingleMuon/SingleMu_2017C/190327_172856/0000/*.root')
# c.add_data_to_evaluate('SingleMu2016D', '/mnt/hadoop/store/user/dkondrat/SingleMuon/SingleMu_2017D/190327_172928/0000/*.root')
# c.add_data_to_evaluate('SingleMu2016E', '/mnt/hadoop/store/user/dkondrat/SingleMuon/SingleMu_2017E/190327_173006/0000/*.root')
# c.add_data_to_evaluate('SingleMu2016F', '/mnt/hadoop/store/user/dkondrat/SingleMuon/SingleMu_2017F/190327_173052/0000/*.root')
c.add_mc_to_evaluate('ZJets_AMC', '/mnt/hadoop/store/user/dkondrat//DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/ZJets_AMC/190323_195010/0000/*root', 5765.4)
c.add_mc_to_evaluate('tt_ll_POW', '/mnt/hadoop/store/user/dkondrat/TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8/tt_ll_POW/190322_195819/0000/*root', 85.656)
c.add_mc_to_evaluate('H2Mu_gg', '/mnt/hadoop/store/user/dkondrat/GluGluHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/H2Mu_gg_powheg/190323_193526/0000/*root', 0.009618)
c.add_mc_to_evaluate('H2Mu_VBF', '/mnt/hadoop/store/user/dkondrat/VBFHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/H2Mu_VBF_powheg/190322_195711/0000/*root', 0.0008208)


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

# c.add_method("BDTG_UF_v1")
# c.setApplication(outPath="/home/dkondra/tmp/BDTG_UF_v1/", xmlPath="/home/dkondra/Hmumu_analysis/Hmumu_ML/dataset/weights/BDTG_UF_v1/TMVAClassification_BDTG_UF_v1.weights.xml")

c.add_method("BDTG_MIT")
c.setApplication(outPath="/home/dkondra/tmp/BDTG_MIT/", xmlPath="/home/dkondra/Hmumu_analysis/Hmumu_ML/dataset/weights/BDTG_MIT/TMVAClassification_BDTG_MIT.weights.xml")


c.apply_methods()

