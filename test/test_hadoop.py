import os, sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from src.classifier import Framework

c = Framework()

treePath = 'dimuons/tree'

mc_path = "/mnt/hadoop/store/user/dkondrat/"
signal = [
		['H2Mu_VBF',	"/VBF_HToMuMu_M125_13TeV_powheg_pythia8/H2Mu_VBF/180827_202716/0000/tuple*.root",							0.0008208	,		18.835700	],
		['H2Mu_gg',		"/GluGlu_HToMuMu_M125_13TeV_powheg_pythia8/H2Mu_gg/180827_202700/0000/tuple*.root",							0.009618	,		232.557886	],
		['H2Mu_WH_neg',	"/WMinusH_HToMuMu_M125_13TeV_powheg_pythia8/H2Mu_WH_neg/180827_202757/0000/tuple*.root",					0.0001164	,		0.304798	],
		['H2Mu_WH_pos',	"/WPlusH_HToMuMu_M125_13TeV_powheg_pythia8/H2Mu_WH_pos/180827_202738/0000/tuple*.root",						0.0001858	,		0.437732	],
		['H2Mu_ZH',		"/ZH_HToMuMu_M125_13TeV_powheg_pythia8/H2Mu_ZH/180827_202818/0000/tuple*.root",								0.0002136	,		0.837788	]
]

bkg_path = "/mnt/hadoop/store/user/dkondrat/"#"/tmp/Hmumu_ntuples"
background = [

	['WW',				"/WWTo2L2Nu_13TeV-powheg/WW/180827_203218/0000/tuple*.root",													12.46	,		10524.477792	],
	# ['WWW',				"/WWW_4F_TuneCUETP8M1_13TeV-amcatnlo-pythia8/WWW/180827_203402/0000/tuple*.root",							0.2086	,		79.037588		],
	# ['WWZ',				"/WWZ_TuneCUETP8M1_13TeV-amcatnlo-pythia8/WWZ/180827_203422/0000/tuple*.root", 								0.1651	,		63.230267		],
	# ['WZZ',				"/WZZ_TuneCUETP8M1_13TeV-amcatnlo-pythia8/WZZ/180827_203439/0000/tuple*.root", 								0.05565	,		17.997588		],
	['WZ_2l',			"/WZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8/WZ_2l/180827_203235/0000/tuple*.root", 						4.409	, 		909.986306		],
	['WZ_3l',			"/WZTo3LNu_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/WZ_3l_AMC/180827_203253/0000/tuple*.root", 				2.113	,		1853.675094		],
	['ZJets_AMC',		"/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/ZJets_AMC/180827_202835/0000/tuple*.root", 		5765.4	, 		1137600.914372	],
	['ZZZ',				"/ZZZ_TuneCUETP8M1_13TeV-amcatnlo-pythia8/ZZZ/180827_203458/0000/tuple*.root" ,								0.01398	, 		3.428108		],
	['ZZ_2l_2q',		"/ZZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8/ZZ_2l_2q/180827_203327/0000/tuple*.root",						3.22	, 		722.244018		],
	['ZZ_2l_2v',		"/ZZTo2L2Nu_13TeV_powheg_pythia8/ZZ_2l_2v/180827_203311/0000/tuple*.root",									0.564	, 		212.112867		],
	['ZZ_4l',			"/ZZTo4L_13TeV-amcatnloFXFX-pythia8/ZZ_4l_AMC/180827_203344/0000/tuple*.root",								1.212	, 		946.472171		],
	['tW_neg_1',		"/ST_tW_antitop_5f_NoFullyHadronicDecays_13TeV-powheg_TuneCUETP8M1/tW_neg_1/180827_203047/0000/tuple*.root",	35.85	,		9563.665061		],
	['tW_neg_2',		"/ST_tW_antitop_5f_NoFullyHadronicDecays_13TeV-powheg_TuneCUETP8M1/tW_neg_2/180827_203103/0000/tuple*.root",	35.85	,		9548.841351		],
	['tW_pos_1',		"/ST_tW_top_5f_NoFullyHadronicDecays_13TeV-powheg_TuneCUETP8M1/tW_pos_1/180827_203006/0000/tuple*.root",		35.85	,		9502.605773		],
	['tW_pos_2',		"/ST_tW_top_5f_NoFullyHadronicDecays_13TeV-powheg_TuneCUETP8M1/tW_pos_2/180827_203024/0000/tuple*.root",		35.85	,		9546.889180		],
	['tZq',				"/tZq_ll_4f_13TeV-amcatnlo-pythia8/tZq/180827_203516/0000/tuple*.root",										0.0758	, 		85.976629		],
	['ttW_1',			"/TTWJetsToLNu_TuneCUETP8M1_13TeV-amcatnloFXFX-madspin-pythia8/ttW_1/180827_203536/0000/tuple*.root",		0.2043	, 		196.426646		],
	['ttW_2',			"/TTWJetsToLNu_TuneCUETP8M1_13TeV-amcatnloFXFX-madspin-pythia8/ttW_2/180827_203553/0000/tuple*.root",		0.2043	, 		195.472703		],
	['ttZ',				"/TTZToLLNuNu_M-10_TuneCUETP8M1_13TeV-amcatnlo-pythia8/ttZ/180827_203612/0000/tuple*.root",					0.2529	,		246.881972		],
	['tt_ll_AMC',		"/TTJets_Dilept_TuneCUETP8M2T4_13TeV-amcatnloFXFX-pythia8/tt_ll_AMC/180827_203154/0000/tuple*.root",		85.656*0.9	, 		97340.321489	],
	['tt_ll_MG_1',		"/TTJets_DiLept_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/tt_ll_MG_1/180827_203121/0000/tuple*.root",			85.656	, 		98109.617801	],
	['tt_ll_MG_2',		"/TTJets_DiLept_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/tt_ll_MG_2/180827_203138/0000/tuple*.root",			85.656	, 		97747.879187	],
	['DY_0J',			"/DYToLL_0J_13TeV-amcatnloFXFX-pythia8/DY_0J/180827_202852/0000/tuple*root",								4754*0.96,		1				],
	['DY_1J',			"/DYToLL_1J_13TeV-amcatnloFXFX-pythia8/DY_1J/180827_202911/0000/tuple*root",								888.9*0.86*0.985*0.995, 1		],
	['DY_2J_1',			"/DYToLL_2J_13TeV-amcatnloFXFX-pythia8/DY_2J_1/180827_202929/0000/tuple*root",								348.8*0.88*0.975*0.992,	1		],
	['DY_2J_2',			"/DYToLL_2J_13TeV-amcatnloFXFX-pythia8/DY_2J_2/180827_202948/0000/tuple*root",								348.8*0.88*0.975*0.992,	1		],

]


for s in signal:
	c.add_signal(s[0], mc_path+s[1], s[2], s[3])

for b in background:
	c.add_background(b[0], bkg_path+b[1], b[2], b[3])

c.set_tree_path(treePath)

c.add_variable("muPairs.pt", 				1) #second argument is the number of objects considered
c.add_variable("muPairs.eta", 				1)
c.add_variable("muPairs.dEta", 				1) 
c.add_variable("muPairs.dPhi", 				1)
# c.add_variable("met.pt", 					1)
c.add_variable("nJetsCent", 				1)
c.add_variable("nJetsFwd",					1)
c.add_variable("nBMed",						1)
c.add_variable("jets.eta",					2)
c.add_variable("jetPairs.dEta",				2)
c.add_variable("jetPairs.mass",				2)

# c.add_variable("muons.pt", 	2) #two muons
# c.add_variable("muons.eta", 	2)
# c.add_variable("muons.phi", 	2)

c.add_package("TMVA")
c.add_transf("N,G,P")

# c.add_package("Keras")

c.train_methods()