import os, sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from src.classifier import Framework

from samples import *

c = Framework()
c.label = "bdt_dnn_multi_hiStat"
comment = "Option 1.1: 4-class DNN, mass window [110, 150], w/o even-by-event mass res., ZJets_AMC_hiStat ( mass 105-160 ), w/o CS variables"	
				# change this line for each run
c.add_comment(comment)
print comment

c.outDir = '/tmp/dkondrat/ML_output/'
treePath = 'dimuons/tree'
metadataPath = 'dimuons/metadata'

c.set_tree_path(treePath)
c.set_metadata_path(metadataPath)

c.set_year("2017")
c.massWindow = [110,150]
c.multiclass = True
c.dy_label = "ZJets_aMC"
c.tt_label = "tt_ll_POW"
c.ggh_label = "H2Mu_gg"
c.vbf_label = "H2Mu_VBF"


##################### Input samples #######################

c.add_data(SingleMu2017B.name, SingleMu2017B.path,  SingleMu2017B.lumi)
c.add_data(SingleMu2017C.name, SingleMu2017C.path,  SingleMu2017C.lumi)
c.add_data(SingleMu2017D.name, SingleMu2017D.path,  SingleMu2017D.lumi)
c.add_data(SingleMu2017E.name, SingleMu2017E.path,  SingleMu2017E.lumi)
c.add_data(SingleMu2017F.name, SingleMu2017F.path,  SingleMu2017F.lumi)

c.add_category(c.ggh_label, True)
c.add_dir_to_category(ggH_2017_powheg.name, ggH_2017_powheg.path, ggH_2017_powheg.xSec, c.ggh_label)

c.add_category(c.vbf_label, True)
c.add_dir_to_category(VBF_2017_powheg.name, VBF_2017_powheg.path, VBF_2017_powheg.xSec, c.vbf_label)

c.add_category(c.dy_label, False)
c.add_dir_to_category(ZJets_aMC_2017_hiStat.name, ZJets_aMC_2017_hiStat.path, ZJets_aMC_2017_hiStat.xSec, c.dy_label)

c.add_category(c.tt_label, False)
c.add_dir_to_category(tt_ll_POW_2017.name, tt_ll_POW_2017.path, tt_ll_POW_2017.xSec, c.tt_label)

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

c.add_package("Keras_multi")
c.add_method("model_50_D2_25_D2_25_D2") # Dropout 0.2



c.train_methods()

print "Training is done: "
print comment
print "Output saved to:"
print c.outPath

