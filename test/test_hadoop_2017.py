import os, sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from src.classifier import Framework

c = Framework()
comment = "Multiclassification: 4 classes, V2, +2017 data, jet variable initialization: -5"	
				# change this line for each run
c.add_comment(comment)
print comment
treePath = 'dimuons/tree'

# path = '/tmp/Hmumu_ntuples/updated/'
path = "/mnt/hadoop/store/user/dkondrat/2017_ntuples/"
# path = '/scratch/browngpu/dkondra/2016_ntuples_updated/'

c.add_data('SingleMu_2017B'		, path+'SingleMuon/SingleMu_2017B/180802_163835/0000/'	, 4823		)
c.add_data('SingleMu_2017C'		, path+'SingleMuon/SingleMu_2017C/180802_163916/0000/'	, 9664		)
c.add_data('SingleMu_2017D'		, path+'SingleMuon/SingleMu_2017D/180802_163956/0000/'	, 4252		)
c.add_data('SingleMu_2017E'		, path+'SingleMuon/SingleMu_2017E/180802_164036/0000/'	, 9278		)
c.add_data('SingleMu_2017F'		, path+'SingleMuon/SingleMu_2017F/180802_164117/0000/'	, (13540-916))


c.add_category('H2Mu_gg', True)
c.add_dir_to_category('H2Mu_gg', path+'/GluGlu_HToMuMu_M125_13TeV_amcatnloFXFX_pythia8/180802_164158/0000/', 0.009618, 'H2Mu_gg')

c.add_category('H2Mu_VBF', True)
c.add_dir_to_category('H2Mu_VBF', path+'/VBFH_HToMuMu_M125_13TeV_amcatnloFXFX_pythia8/180802_164241/0000/', 0.0008208, 'H2Mu_VBF')

c.add_category('ZJets_aMC', False)
c.add_dir_to_category('ZJets_aMC', path+'/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/180802_165055/0000/', 5765.4, 'ZJets_aMC')

c.add_category('tt_ll_AMC', False)
c.add_dir_to_category('tt_ll_AMC', path+'/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/180802_165355/0000/', 815.960, 'tt_ll_AMC')

c.set_tree_path(treePath)

c.add_variable("muPairs.pt", 				1) 
c.add_variable("muPairs.eta", 				1)
c.add_variable("muPairs.dEta", 				1) 
c.add_variable("muPairs.dPhi", 				1)
c.add_variable("met.pt", 					1)


# c.set_year("2016-2orMoreJets")
# var_set = "V2-2orMoreJets"
var_set = "V1"
c.set_year("2017")


if var_set == "V1":
	c.add_variable("nJetsCent", 				1)
	c.add_variable("nJetsFwd",					1)
	c.add_variable("nBMed",						1)
	c.add_variable("jets.eta",					2)
	c.add_variable("jetPairs.dEta",				1)
	c.add_variable("jetPairs.mass",				1)
	c.add_spectator('muons.pt',					2)
	c.add_data_spectator('muons.pt',			2)
elif var_set == "V2":
	c.add_variable("nJetsCent", 				1)
	c.add_variable("nJetsFwd",					1)
	c.add_variable("nBMed",						1)
	c.add_variable("jets.eta",					2)
	c.add_variable("jetPairs.dEta",				1)
	c.add_variable("jetPairs.mass",				1)
	c.add_spectator('muons.pt',					2)
	c.add_data_spectator('muons.pt',			2)
	c.add_variable("min_dR_mu_jet"	,			1)
	c.add_variable("max_dR_mu_jet"	,			1)
	c.add_variable("min_dR_mumu_jet",			1)
	c.add_variable("max_dR_mumu_jet",			1)
	c.add_variable("zepenfeld",					1)
elif var_set == "V3":
	c.add_variable("nJetsCent", 				1)
	c.add_variable("nJetsFwd",					1)
	c.add_variable("nBMed",						1)
	c.add_variable("jets.eta",					2)
	c.add_variable("jetPairs.dEta",				1)
	c.add_variable("jetPairs.mass",				1)
	c.add_variable('muons.pt',					2)
	c.add_variable('muons.eta',					2)
	c.add_variable('muons.phi',					2)
	c.add_variable("min_dR_mu_jet"	,			1)
	c.add_variable("max_dR_mu_jet"	,			1)
	c.add_variable("min_dR_mumu_jet",			1)
	c.add_variable("max_dR_mumu_jet",			1)
	c.add_variable("zepenfeld",					1)
elif var_set == "V4":
	c.add_variable("nJetsCent", 				1)
	c.add_variable("nJetsFwd",					1)
	c.add_variable("nBMed",						1)
	c.add_variable("jets.eta",					2)
	c.add_variable("jetPairs.dEta",				1)
	c.add_variable("jetPairs.mass",				1)
	c.add_spectator('muons.pt',					2)
	c.add_data_spectator('muons.pt',			2)
	c.add_variable("mu1_pt_by_mass",			1)
	c.add_variable("mu2_pt_by_mass",			1)	
	c.add_variable('muons.eta',					2)
	c.add_variable('muons.phi',					2)
	c.add_variable("min_dR_mu_jet"	,			1)
	c.add_variable("max_dR_mu_jet"	,			1)
	c.add_variable("min_dR_mumu_jet",			1)
	c.add_variable("max_dR_mumu_jet",			1)
	c.add_variable("zepenfeld",					1)
elif var_set == "V2-noJets":
	c.add_spectator('muons.pt',					2)
	c.add_data_spectator('muons.pt',			2)
elif var_set == "V2-1jet":
	c.add_variable("nJetsCent", 				1)
	c.add_variable("nJetsFwd",					1)
	c.add_variable("nBMed",						1)
	c.add_variable("jets.eta",					1)
	c.add_spectator('muons.pt',					2)
	c.add_data_spectator('muons.pt',			2)
	c.add_variable("min_dR_mu_jet"	,			1)
	c.add_variable("max_dR_mu_jet"	,			1)
	c.add_variable("min_dR_mumu_jet",			1)
	c.add_variable("max_dR_mumu_jet",			1)
elif var_set == "V2-2orMoreJets":
	c.add_variable("nJetsCent", 				1)
	c.add_variable("nJetsFwd",					1)
	c.add_variable("nBMed",						1)
	c.add_variable("jets.eta",					2)
	c.add_variable("jetPairs.dEta",				1)
	c.add_variable("jetPairs.mass",				1)
	c.add_spectator('muons.pt',					2)
	c.add_data_spectator('muons.pt',			2)
	c.add_variable("min_dR_mu_jet"	,			1)
	c.add_variable("max_dR_mu_jet"	,			1)
	c.add_variable("min_dR_mumu_jet",			1)
	c.add_variable("max_dR_mumu_jet",			1)
	c.add_variable("zepenfeld",					1)

c.add_spectator('muPairs.mass',		        1)
c.add_spectator('muPairs.phi',				1)
c.add_spectator('muons.isMediumID',			2)
c.add_spectator('jets.phi',					2)

c.add_spectator('muons.eta',				2)
c.add_spectator('nJets',					1)
c.add_spectator('PU_wgt',					1)
c.add_spectator('GEN_wgt', 					1)
c.add_spectator('IsoMu_SF_3',				1)
c.add_spectator('MuID_SF_3', 				1)
c.add_spectator('MuIso_SF_3',				1)
# c.add_spectator('IsoMu_SF_4',				1)
# c.add_spectator('MuID_SF_4', 				1)
# c.add_spectator('MuIso_SF_4',				1)

c.add_data_spectator('muons.eta',			2)
c.add_data_spectator('muPairs.mass',	    1)
c.add_data_spectator('muPairs.phi',			1)
c.add_data_spectator('muons.isMediumID',	2)
c.add_data_spectator('jets.phi',			2)
c.add_data_spectator('nJets',				1)

c.weigh_by_event(True)
# c.add_package("TMVA")
# c.add_transf("N,G,P")
# c.add_method("BDTG_UF_v1")
c.add_package("Keras_multi")
# c.add_method("UCSD_model")	# 50_D2
# c.add_method("model_50_25") # no Dropout
# c.add_method("model_50_D1_25_D1") # Dropout 0.1
c.add_method("model_50_D2_25_D2_25_D2") # Dropout 0.2
# c.add_method("model_50_25_25") # no dropout
# c.add_method("andrea_model_3") # Andrea's model #3




c.train_methods()

