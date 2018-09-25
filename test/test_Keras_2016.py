import os, sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from src.classifier import Framework

c = Framework()
c.add_comment("Keras test with selected backgrounds") # change this line for each run!
treePath = 'dimuons/tree'

mc_path = "/mnt/hadoop/store/user/dkondrat/"
signal = [
		['H2Mu_VBF',	"/VBF_HToMuMu_M125_13TeV_powheg_pythia8/H2Mu_VBF/180827_202716/0000/",								0.0008208	],
		['H2Mu_gg',		"/GluGlu_HToMuMu_M125_13TeV_powheg_pythia8/H2Mu_gg/180827_202700/0000/",							0.009618	],
		# ['H2Mu_WH_neg',	"/WMinusH_HToMuMu_M125_13TeV_powheg_pythia8/H2Mu_WH_neg/180827_202757/0000/",					0.0001164	],
		# ['H2Mu_WH_pos',	"/WPlusH_HToMuMu_M125_13TeV_powheg_pythia8/H2Mu_WH_pos/180827_202738/0000/",						0.0001858	],
		# ['H2Mu_ZH',		"/ZH_HToMuMu_M125_13TeV_powheg_pythia8/H2Mu_ZH/180827_202818/0000/",								0.0002136	]
]

bkg_path = "/mnt/hadoop/store/user/dkondrat/"#"/tmp/Hmumu_ntuples"
background = [

	# ['WW',						"/WWTo2L2Nu_13TeV-powheg/WW/180827_203218/0000/",													12.46		],
	# ['WWW',					"/WWW_4F_TuneCUETP8M1_13TeV-amcatnlo-pythia8/WWW/180827_203402/0000/",							0.2086		],
	# ['WWZ',					"/WWZ_TuneCUETP8M1_13TeV-amcatnlo-pythia8/WWZ/180827_203422/0000/", 								0.1651		],
	# ['WZZ',					"/WZZ_TuneCUETP8M1_13TeV-amcatnlo-pythia8/WZZ/180827_203439/0000/", 								0.05565		],
	# ['WZ_2l',					"/WZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8/WZ_2l/180827_203235/0000/", 							4.409		],
	# ['WZ_3l',					"/WZTo3LNu_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/WZ_3l_AMC/180827_203253/0000/", 					2.113		],
	# ['ZJets_AMC',				"/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/ZJets_AMC/180827_202835/0000/", 			5765.4		],
	# ['ZZZ',						"/ZZZ_TuneCUETP8M1_13TeV-amcatnlo-pythia8/ZZZ/180827_203458/0000/" ,									0.01398		],
	# ['ZZ_2l_2q',				"/ZZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8/ZZ_2l_2q/180827_203327/0000/",							3.22 		],
	# ['ZZ_2l_2v',				"/ZZTo2L2Nu_13TeV_powheg_pythia8/ZZ_2l_2v/180827_203311/0000/",										0.564 		],
	# ['ZZ_4l',					"/ZZTo4L_13TeV-amcatnloFXFX-pythia8/ZZ_4l_AMC/180827_203344/0000/",									1.212 		],
	# ['tW_neg_1',				"/ST_tW_antitop_5f_NoFullyHadronicDecays_13TeV-powheg_TuneCUETP8M1/tW_neg_1/180827_203047/0000/",	35.85		],
	# ['tW_neg_2',				"/ST_tW_antitop_5f_NoFullyHadronicDecays_13TeV-powheg_TuneCUETP8M1/tW_neg_2/180827_203103/0000/",	35.85		],
	# ['tW_pos_1',				"/ST_tW_top_5f_NoFullyHadronicDecays_13TeV-powheg_TuneCUETP8M1/tW_pos_1/180827_203006/0000/",		35.85		],
	# ['tW_pos_2',				"/ST_tW_top_5f_NoFullyHadronicDecays_13TeV-powheg_TuneCUETP8M1/tW_pos_2/180827_203024/0000/",		35.85		],
	# ['tZq',						"/tZq_ll_4f_13TeV-amcatnlo-pythia8/tZq/180827_203516/0000/",											0.0758		],
	# ['ttW_1',					"/TTWJetsToLNu_TuneCUETP8M1_13TeV-amcatnloFXFX-madspin-pythia8/ttW_1/180827_203536/0000/",			0.2043		],
	# ['ttW_2',					"/TTWJetsToLNu_TuneCUETP8M1_13TeV-amcatnloFXFX-madspin-pythia8/ttW_2/180827_203553/0000/",			0.2043		],
	# ['ttZ',						"/TTZToLLNuNu_M-10_TuneCUETP8M1_13TeV-amcatnlo-pythia8/ttZ/180827_203612/0000/",						0.2529		],
	['tt_ll_AMC',				"/TTJets_Dilept_TuneCUETP8M2T4_13TeV-amcatnloFXFX-pythia8/tt_ll_AMC/180827_203154/0000/",			85.656*0.9	],
	# ['tt_ll_MG_1',				"/TTJets_DiLept_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/tt_ll_MG_1/180827_203121/0000/",				85.656		],
	# ['tt_ll_MG_2',				"/TTJets_DiLept_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/tt_ll_MG_2/180827_203138/0000/",				85.656		],
	# ['DY_0J',					"/DYToLL_0J_13TeV-amcatnloFXFX-pythia8/DY_0J/180827_202852/0000/",									4754*0.96	],
	# ['DY_1J',					"/DYToLL_1J_13TeV-amcatnloFXFX-pythia8/DY_1J/180827_202911/0000/",									888.9*0.86*0.985*0.995	],
	# ['DY_2J_1',					"/DYToLL_2J_13TeV-amcatnloFXFX-pythia8/DY_2J_1/180827_202929/0000/",									348.8*0.88*0.975*0.992	],
	# ['DY_2J_2',					"/DYToLL_2J_13TeV-amcatnloFXFX-pythia8/DY_2J_2/180827_202948/0000/",									348.8*0.88*0.975*0.992	],
	['ZJets_MG',				"/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/ZJets_MG/180913_191722/0000/",		5765.4		],
	# ['ZJets_MG_HT_70_100',		"/DYJetsToLL_M-50_HT-70to100_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/ZJets_MG_HT_70_100/180913_191823/0000/",			0.98*178.952		],
	# ['ZJets_MG_HT_100_200_A',	"/DYJetsToLL_M-50_HT-100to200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/ZJets_MG_HT_100_200_A/180913_191844/0000/",		0.96*181.302		],
	# ['ZJets_MG_HT_100_200_B',	"/DYJetsToLL_M-50_HT-100to200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/ZJets_MG_HT_100_200_B/180913_191907/0000/",		0.96*181.302		],
	# ['ZJets_MG_HT_200_400_A',	"/DYJetsToLL_M-50_HT-200to400_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/ZJets_MG_HT_200_400_A/180913_191940/0000/",		0.96*50.4177		],
	# ['ZJets_MG_HT_200_400_B',	"/DYJetsToLL_M-50_HT-200to400_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/ZJets_MG_HT_200_400_B/180913_192001/0000/",		0.96*50.4177		],
	# ['ZJets_MG_HT_400_600_A',	"/DYJetsToLL_M-50_HT-400to600_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/ZJets_MG_HT_400_600_A/180913_192022/0000/",		0.96*6.98394		],
	# ['ZJets_MG_HT_600_800',		"/DYJetsToLL_M-50_HT-600to800_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/ZJets_MG_HT_600_800/180913_192118/0000/",		0.96*1.68141		],
	# ['ZJets_MG_HT_800_1200',	"/DYJetsToLL_M-50_HT-800to1200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/ZJets_MG_HT_800_1200/180913_192145/0000/",		0.96*0.775392		],
	# ['ZJets_MG_HT_1200_2500',	"/DYJetsToLL_M-50_HT-1200to2500_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/ZJets_MG_HT_1200_2500/180913_192208/0000/",	0.96*0.186222		],	
	# ['ZJets_MG_HT_2500_inf',	"/DYJetsToLL_M-50_HT-2500toInf_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/ZJets_MG_HT_2500_inf/180913_192230/0000/",		0.96*0.004385		],
	# ['ZJets_hiM',				"/DYJetsToLL_M-100to200_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/ZJets_hiM/180913_191802/0000/",						5765.4*1.235		]

]


for s in signal:
	c.add_signal_dir(s[0], mc_path+s[1], s[2])

for b in background:
	c.add_background_dir(b[0], bkg_path+b[1], b[2])

c.set_tree_path(treePath)

c.add_variable("muPairs.pt", 				1) #second argument is the number of objects considered
c.add_variable("muPairs.eta", 				1)
c.add_variable("muPairs.dEta", 				1) 
c.add_variable("muPairs.dPhi", 				1)
c.add_variable("met.pt", 					1)
c.add_variable("nJetsCent", 				1)
c.add_variable("nJetsFwd",					1)
c.add_variable("nBMed",						1)
c.add_variable("jets.eta",					2)
c.add_variable("jetPairs.dEta",				2)
c.add_variable("jetPairs.mass",				2)

c.add_spectator('muPairs.mass',				1)
c.add_spectator('muons.pt',					2)
c.add_spectator('muons.isMediumID',			2)
c.add_spectator('PU_wgt',					1)
c.add_spectator('GEN_wgt', 					1)
c.add_spectator('IsoMu_SF_3',				1)
c.add_spectator('MuID_SF_3', 				1)
c.add_spectator('MuIso_SF_3',				1)

c.weigh_by_event(True)
c.set_year("2016")
# c.add_package("TMVA")
# c.add_transf("N,G,P")
# c.add_method("BDTG_UF_v1")
c.add_package("Keras")
c.add_method("model_3x20")

c.train_methods()

