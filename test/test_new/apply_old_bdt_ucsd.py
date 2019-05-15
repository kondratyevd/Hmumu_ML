import os, sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from src.classifier import Framework

from samples import *

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

c.add_data_to_evaluate(SingleMu2017B.name, SingleMu2017B.path+"/*root")
c.add_data_to_evaluate(SingleMu2017C.name, SingleMu2017C.path+"/*root")
c.add_data_to_evaluate(SingleMu2017D.name, SingleMu2017D.path+"/*root")
c.add_data_to_evaluate(SingleMu2017E.name, SingleMu2017E.path+"/*root")
c.add_data_to_evaluate(SingleMu2017F.name, SingleMu2017F.path+"/*root")

c.add_mc_to_evaluate(ZJets_aMC_2017.name, ZJets_aMC_2017.path+"/*root", ZJets_aMC_2017.xSec)
c.add_mc_to_evaluate(tt_ll_POW_2017.name, tt_ll_POW_2017.path+"/*root", tt_ll_POW_2017.xSec)
c.add_mc_to_evaluate(ggH_2017_powheg.name, ggH_2017_powheg.path+"/*root", ggH_2017_powheg.xSec)
c.add_mc_to_evaluate(VBF_2017_powheg.name, VBF_2017_powheg.path+"/*root", VBF_2017_powheg.xSec)



c.set_tree_path(treePath)

c.add_variable("muPairs.pt", 				1) 
c.add_variable("muPairs.eta", 				1)
c.add_variable("muPairs.dEta", 				1) 
c.add_variable("muPairs.dPhi", 				1)
c.add_variable("met.pt", 					1)

c.add_variable("mu1_pt_Roch_over_mass", 	1)
c.add_variable("mu2_pt_Roch_over_mass", 	1)
c.add_variable('muons.eta',					2)
c.add_variable("min_dR_mu_jet", 		 	1)
c.add_variable("nJets", 					1)
c.add_variable("nBMed",						1)
c.add_variable("zeppenfeld",				1)

c.add_variable("jets.pt",					2)
c.add_variable("jetPairs.mass",				1)
c.add_variable("jetPairs.dEta",				1)
c.add_variable("jetPairs.dPhi",				1)



c.set_year("2017")
c.weigh_by_event(True)

c.add_package("TMVA")

c.add_method("BDTG_UCSD")
c.setApplication(outPath="/home/dkondra/tmp/BDTG_UCSD/", xmlPath="/home/dkondra/Hmumu_analysis/Hmumu_ML/dataset/weights/TMVAClassification_bdt_ucsd_BDTG_UCSD.weights.xml")
c.hasMass = True


c.apply_methods()

