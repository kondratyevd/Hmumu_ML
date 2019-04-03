import os, sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from src.classifier import Framework

c = Framework()
# comment = "Multiclassification: 4 classes, V2, +2016 data, jet variable initialization: -5"	
				# change this line for each run
# c.add_comment(comment)
# print comment
treePath = 'dimuons/tree'

# path = '/tmp/Hmumu_ntuples/updated/'
# path = '/scratch/browngpu/dkondra/2016_ntuples_updated/'
# path = "/mnt/hadoop/store/user/dkondrat/"

# c.add_data('SingleMu_2016B', path+'SingleMuon/SingleMu_2017B/190327_172730/0000/',   		4823				) # lumi [/pb]
# c.add_data('SingleMu_2016B'		, path+'/SingleMu_2016B/'	, 5788)
# c.add_data('SingleMu_2016C'		, path+'/SingleMu_2016C/'	, 2573)
# c.add_data('SingleMu_2016D'		, path+'/SingleMu_2016D/'	, 4248)
# c.add_data('SingleMu_2016E'		, path+'/SingleMu_2016E/'	, 4009)
# c.add_data('SingleMu_2016F_1'	, path+'/SingleMu_2016F_1/'	, 3102)
# c.add_data('SingleMu_2016G'		, path+'/SingleMu_2016G/'	, 7540)
# c.add_data('SingleMu_2016H_1'	, path+'/SingleMu_2016H_1/'	, 8392)
# c.add_data('SingleMu_2016H_2'	, path+'/SingleMu_2016H_2/'	, 214)

# c.add_category('H2Mu_gg', True)
# c.add_dir_to_category('H2Mu_gg', path+'/GluGluHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/H2Mu_gg_powheg/190323_193526/0000/', 0.009618, 'H2Mu_gg')

# c.add_category('H2Mu_VBF', True)
# c.add_dir_to_category('H2Mu_VBF', path+'/VBFHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/H2Mu_VBF_powheg/190322_195711/0000/', 0.0008208, 'H2Mu_VBF')

# c.add_category('ZJets_aMC', False)
# c.add_dir_to_category('ZJets_aMC', path+'/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/ZJets_AMC/190323_195010/0000/', 5765.4, 'ZJets_aMC')

# c.add_category('tt_ll_POW', False)
# c.add_dir_to_category('tt_ll_POW', path+'/TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8/tt_ll_POW/190322_195819/0000/', 85.656, 'tt_ll_POW')

# c.add_signal_dir('H2Mu_gg', path+'/GluGluHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/H2Mu_gg_powheg/190323_193526/0000/', 0.009618)
# c.add_signal_dir('H2Mu_VBF', path+'/VBFHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/H2Mu_VBF_powheg/190322_195711/0000/', 0.0008208)

# c.add_background_dir('ZJets_aMC', path+'/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/ZJets_AMC/190323_195010/0000/', 5765.4)
# c.add_background_dir('tt_ll_POW', path+'/TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8/tt_ll_POW/190322_195819/0000/', 85.656)

path = path = "/tmp/dkondrat/updated_2017_mc/"

c.add_signal_dir('H2Mu_gg', path+'/H2Mu_ggH/', 0.009618)
c.add_signal_dir('H2Mu_VBF', path+'/H2Mu_VBF/', 0.0008208)

c.add_background_dir('ZJets_aMC', path+'/ZJets_AMC/', 5765.4)
c.add_background_dir('tt_ll_POW', path+'/tt_ll_POW/', 85.656)

# c.add_signal_file('H2Mu_gg', path+'/GluGluHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/H2Mu_gg_powheg/190323_193526/0000/tuple_1.root', 0.009618)
# c.add_signal_file('H2Mu_VBF', path+'/VBFHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/H2Mu_VBF_powheg/190322_195711/0000/tuple_1.root', 0.0008208)

# c.add_background_file('ZJets_aMC', path+'/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/ZJets_AMC/190323_195010/0000/tuple_1.root', 5765.4)
# c.add_background_file('tt_ll_POW', path+'/TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8/tt_ll_POW/190322_195819/0000/tuple_1.root', 85.656)

c.set_tree_path(treePath)

c.add_variable("muPairs.pt", 				1) 
c.add_variable("muPairs.eta", 				1)
c.add_variable("muPairs.dEta", 				1) 
c.add_variable("muPairs.dPhi", 				1)
c.add_variable("met.pt", 					1)


# c.set_year("2016-2orMoreJets")
# var_set = "V2-2orMoreJets"
var_set = "V2"
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
	c.add_variable("zeppenfeld",					1)
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
	c.add_variable("zeppenfeld",					1)
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
	c.add_variable("zeppenfeld",					1)
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
	c.add_variable("zeppenfeld",					1)

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
c.add_package("TMVA")
# c.add_transf("N,G,P")
# c.add_method("BDTG_UF_v1")
c.add_method("BDTG_MIT")
# c.add_method("BDTG_MIT_lite")
# c.add_package("Keras_multi")
# c.add_method("UCSD_model")	# 50_D2
# c.add_method("model_50_25") # no Dropout
# c.add_method("model_50_D1_25_D1") # Dropout 0.1
# c.add_method("model_50_D2_25_D2_25_D2") # Dropout 0.2
# c.add_method("model_50_25_25") # no dropout
# c.add_method("andrea_model_3") # Andrea's model #3

# c.custom_loss = True
# c.add_method("model_50_D2_25_D2_25_D2_mass_control_bkg_0p5")
# c.add_method("model_50_D2_25_D2_25_D2_mass_control_bkg_1")
# c.add_method("model_50_D2_25_D2_25_D2_mass_control_bkg_1p5")
# c.add_method("model_50_D2_25_D2_25_D2_mass_control_bkg_2")
# c.add_method("model_50_D2_25_D2_25_D2_mass_control_bkg_2p5")
# c.add_method("model_50_D2_25_D2_25_D2_mass_control_bkg_3")
# c.add_method("model_50_D2_25_D2_25_D2_mass_control_bkg_3p5")
# c.add_method("model_50_D2_25_D2_25_D2_mass_control_bkg_4")

# c.add_method("model_50_D2_25_D2_mutual_mass_control_5")
# c.add_method("model_50_D2_25_D2_mutual_mass_control_sym_5")
# c.add_method("model_50_D2_25_D2_mass_control_0")
# c.add_method("model_50_D2_25_D2_mass_control_0p1")
# c.add_method("model_50_D2_25_D2_mass_control_0p2")
# c.add_method("model_50_D2_25_D2_mass_control_0p3")
# c.add_method("model_50_D2_25_D2_mass_control_0p4")
# c.add_method("model_50_D2_25_D2_mass_control_0p5")
# c.add_method("model_50_D2_25_D2_mass_control_1")
# c.add_method("model_50_D2_25_D2_mass_control_2")
# c.add_method("model_50_D2_25_D2_mass_control_3")
# c.add_method("model_50_D2_25_D2_mass_control_4")
# c.add_method("model_50_D2_25_D2_mass_control_5")
# c.add_method("model_50_D2_25_D2_kldiv0")
# c.add_method("model_50_D2_25_D2_kldiv1")
# c.add_method("model_50_D2_25_D2_kldiv2")
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
