import os, sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from src.classifier import Framework

from samples import *

c = Framework()
c.label = "bdt_uf_hiStat"
comment = "Option 2: BDTG_UF_v1 (as in HIG-17-019), w/o even-by-event mass res., ZJets_AMC_hiStat ( mass 105-160 )"	
				# change this line for each run
c.add_comment(comment)
print comment

c.outDir = '/tmp/dkondrat/ML_output/'
treePath = 'dimuons/tree'
metadataPath = 'dimuons/metadata'

c.set_tree_path(treePath)
c.set_metadata_path(metadataPath)

c.set_year("2017")
c.ebe_weights = False

##################### Input samples #######################

c.add_signal_dir(ggH_2017_powheg.name, ggH_2017_powheg.path, ggH_2017_powheg.xSec)
c.add_signal_dir(VBF_2017_powheg.name, VBF_2017_powheg.path, VBF_2017_powheg.xSec)

c.add_background_dir(ZJets_aMC_2017_hiStat.name, ZJets_aMC_2017_hiStat.path, ZJets_aMC_2017_hiStat.xSec)
c.add_background_dir(tt_ll_POW_2017.name, tt_ll_POW_2017.path, tt_ll_POW_2017.xSec)

##########################################################


###  ------   Raffaele's variables   ------ ###
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
###############################################


c.add_data_spectator('muons.pt',			2)
c.add_data_spectator('muPairs.mass',	    1)
c.add_data_spectator('muPairs.phi',			1)
c.add_data_spectator('muons.isMediumID',	2)
c.add_data_spectator('jets.phi',			2)
c.add_data_spectator('nJets',				1)

c.add_spectator('muons.pt',					2)
c.add_spectator('muPairs.mass',		        1)
c.add_spectator('muPairs.phi',				1)
c.add_spectator('muons.isMediumID',			2)
c.add_spectator('jets.phi',					2)
c.add_spectator('nJets',					1)

c.add_spectator('PU_wgt',					1)
c.add_spectator('GEN_wgt', 					1)
c.add_spectator('IsoMu_SF_3',				1)
c.add_spectator('MuID_SF_3', 				1)
c.add_spectator('MuIso_SF_3',				1)

c.weigh_by_event(True)
c.add_package("TMVA")
c.add_method("BDTG_UF_v1")
# c.add_method("BDTG_MIT")

c.weigh_by_event(True)

c.train_methods()

print "Training is done: "
print comment
print "Output saved to:"
print c.outPath
