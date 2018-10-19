import os, sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from src.classifier import Framework

c = Framework()
comment = "Multiclassification: 2 classes, V3"	
				# change this line for each run
c.add_comment(comment)
print comment
treePath = 'dimuons/tree'

mc_path = '/tmp/Hmumu_ntuples/updated/'

# c.add_category('H2Mu_gg', True)
# c.add_dir_to_category('H2Mu_gg', mc_path+'/H2Mu_ggH/', 0.009618, 'H2Mu_gg')

# c.add_category('H2Mu_VBF', True)
# c.add_dir_to_category('H2Mu_VBF', mc_path+'/H2Mu_VBF/', 0.0008208, 'H2Mu_VBF')

# c.add_category('ZJets_MG', False)
# c.add_dir_to_category('ZJets_MG', mc_path+'/ZJets_MG/', 5765.4, 'ZJets_MG')

# c.add_category('tt_ll_AMC', False)
# c.add_dir_to_category('tt_ll_AMC', mc_path+'/tt_ll_AMC/', 85.656*0.9, 'tt_ll_AMC')

c.add_category('signal', True)
c.add_dir_to_category('H2Mu_gg', mc_path+'/H2Mu_ggH/', 0.009618, 'signal')

# c.add_category('signal', True)
c.add_dir_to_category('H2Mu_VBF', mc_path+'/H2Mu_VBF/', 0.0008208, 'signal')

c.add_category('background', False)
c.add_dir_to_category('ZJets_MG', mc_path+'/ZJets_MG/', 5765.4, 'background')

# c.add_category('background', False)
c.add_dir_to_category('tt_ll_AMC', mc_path+'/tt_ll_AMC/', 85.656*0.9, 'background')

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
c.add_variable("min_dR_mu_jet"	,			1)
c.add_variable("max_dR_mu_jet"	,			1)
c.add_variable("min_dR_mumu_jet",			1)
c.add_variable("max_dR_mumu_jet",			1)
c.add_variable("zepenfeld",					1)

c.add_variable('muons.eta',					2)
c.add_variable('muons.phi',					2)

decorrelate = False

if(decorrelate):
	c.add_variable("mu1_pt_by_mass",			1)
	c.add_variable("mu2_pt_by_mass",			1)
	c.add_spectator('muons.pt',					2)
else:
	c.add_variable('muons.pt',					2)

c.add_spectator('muPairs.mass',				1)
c.add_spectator('muPairs.phi',				1)

# c.add_spectator('muons.eta',				2)
# c.add_spectator('muons.phi',				2)
c.add_spectator('muons.isMediumID',			2)
c.add_spectator('jets.phi',					2)

c.add_spectator('nJets',					1)
c.add_spectator('PU_wgt',					1)
c.add_spectator('GEN_wgt', 					1)
c.add_spectator('IsoMu_SF_3',				1)
c.add_spectator('MuID_SF_3', 				1)
c.add_spectator('MuIso_SF_3',				1)
c.add_spectator('IsoMu_SF_4',				1)
c.add_spectator('MuID_SF_4', 				1)
c.add_spectator('MuIso_SF_4',				1)


c.weigh_by_event(True)
c.set_year("2016")
# c.add_package("TMVA")
# c.add_transf("N,G,P")
# c.add_method("BDTG_UF_v1")
c.add_package("Keras_multi")
# c.add_method("UCSD_model")	# 50_D2
# c.add_method("model_50_25") # no Dropout
# c.add_method("model_50_D1_25_D1") # Dropout 0.1
# c.add_method("model_50_D2_25_D2") # Dropout 0.2

c.custom_loss = True
c.add_method("model_50_D2_25_D2_kldiv0")
c.add_method("model_50_D2_25_D2_kldiv1")
c.add_method("model_50_D2_25_D2_kldiv2")
# c.add_method("model_50_D2_25_D2_kldiv3")
# c.add_method("model_50_D2_25_D2_kldiv4")
# c.add_method("model_50_D2_25_D2_kldiv5")


# c.add_method("model_50_D3_25_D3") # Dropout 0.3
# c.add_method("model_50_D1_25_D1_10_D1") # Dropout 0.1
# c.add_method("model_50_25_25") # no Dropout
# c.add_method("model_50_D1_25_D1_25_D1") # Dropout 0.1
# c.add_method("model_50_D1_50_D1_50_D1") # Dropout 0.1
# c.add_method("model_50_D2_25_D2_10_D2") # Dropout 0.2
# c.add_method("model_50_D2_25_D2_25_D2") # Dropout 0.2
# c.add_method("model_50_D3_25_D3_25_D3") # Dropout 0.3
# c.add_method("model_50_D2_50_D2_50_D2") # Dropout 0.2

# c.add_method("model_50_D1_25_D1_25_D1_25_D1") # Dropout 0.1
# c.add_method("model_50_D1_25_D1_25_D1_25_D1_25_D1") # Dropout 0.1
# c.add_method("model_50_D2_25_D2_25_D2_25_D2") # Dropout 0.2
# c.add_method("model_50_D2_25_D2_25_D2_25_D2_25_D2") # Dropout 0.2
# c.add_method("model_50_D2_25_D2") # Dropout 0.2
# c.add_method("model_50_D3_25_D3") # Dropout 0.3




c.train_methods()


## HIG-17-019 variables:

# c.add_variable("muPairs.pt", 				1) 
# c.add_variable("muPairs.eta", 				1)
# c.add_variable("muPairs.dEta", 				1) 
# c.add_variable("muPairs.dPhi", 				1)
# c.add_variable("met.pt", 					1)
# c.add_variable("nJetsCent", 				1)
# c.add_variable("nJetsFwd",					1)
# c.add_variable("nBMed",						1)
# c.add_variable("jets.eta",					2)
# c.add_variable("jetPairs.dEta",				2)
# c.add_variable("jetPairs.mass",				2)