import os, sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from src.classifier import Framework

c = Framework()

# inFileDir = "/data/dmitry/Hmumu/ntuples/"
treePath = 'dimuons/tree'

mc_path = "/mnt/hadoop/store/user/dkondrat/"
signal = [
		["/VBF_HToMuMu_M125_13TeV_powheg_pythia8/H2Mu_VBF/180827_202716/0000/*.root",18.835700],
		["/GluGlu_HToMuMu_M125_13TeV_powheg_pythia8/H2Mu_gg/180827_202700/0000/*.root",232.557886],
		["/WMinusH_HToMuMu_M125_13TeV_powheg_pythia8/H2Mu_WH_neg/180827_202757/0000/*.root",0.304798],
		["/WPlusH_HToMuMu_M125_13TeV_powheg_pythia8/H2Mu_WH_pos/180827_202738/0000/*.root",0.437732],
		["/ZH_HToMuMu_M125_13TeV_powheg_pythia8/H2Mu_ZH/180827_202818/0000/*.root",0.837788]
]

bkg_path = "/mnt/hadoop/store/user/dkondrat/"#"/tmp/Hmumu_ntuples"
background = [

	[	"/WWTo2L2Nu_13TeV-powheg/WW/180827_203218/0000/*.root",	10524.477792		],
	[	"/WWW_4F_TuneCUETP8M1_13TeV-amcatnlo-pythia8/WWW/180827_203402/0000/*.root",	79.037588		],
	[	"/WWZ_TuneCUETP8M1_13TeV-amcatnlo-pythia8/WWZ/180827_203422/0000/*.root",	63.230267		],
	[	"/WZZ_TuneCUETP8M1_13TeV-amcatnlo-pythia8/WZZ/180827_203439/0000/*.root",	17.997588		],
	[	"/WZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8/WZ_2l/180827_203235/0000/*.root",909.986306			],
	[	"/WZTo3LNu_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/WZ_3l_AMC/180827_203253/0000/*.root",	1853.675094		],
	[	"/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/ZJets_AMC/180827_202835/0000/*.root", 1137600.914372			],
	[	"/ZZZ_TuneCUETP8M1_13TeV-amcatnlo-pythia8/ZZZ/180827_203458/0000/*.root" ,	3.428108		],
	[	"/ZZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8/ZZ_2l_2q/180827_203327/0000/*.root",	722.244018		],
	[	"/ZZTo2L2Nu_13TeV_powheg_pythia8/ZZ_2l_2v/180827_203311/0000/*.root",	212.112867		],
	[	"/ZZTo4L_13TeV-amcatnloFXFX-pythia8/ZZ_4l_AMC/180827_203344/0000/*.root",	946.472171		],
	[	"/ST_tW_antitop_5f_NoFullyHadronicDecays_13TeV-powheg_TuneCUETP8M1/tW_neg_1/180827_203047/0000/*.root",	9563.665061		],
	[	"/ST_tW_antitop_5f_NoFullyHadronicDecays_13TeV-powheg_TuneCUETP8M1/tW_neg_2/180827_203103/0000/*.root",	9548.841351		],
	[	"/ST_tW_top_5f_NoFullyHadronicDecays_13TeV-powheg_TuneCUETP8M1/tW_pos_1/180827_203006/0000/*.root",	9502.605773		],
	[	"/ST_tW_top_5f_NoFullyHadronicDecays_13TeV-powheg_TuneCUETP8M1/tW_pos_2/180827_203024/0000/*.root",	9546.889180		],
	[	"/tZq_ll_4f_13TeV-amcatnlo-pythia8/tZq/180827_203516/0000/*.root",	85.976629		],
	[	"/TTWJetsToLNu_TuneCUETP8M1_13TeV-amcatnloFXFX-madspin-pythia8/ttW_1/180827_203536/0000/*.root",	196.426646		],
	[	"/TTWJetsToLNu_TuneCUETP8M1_13TeV-amcatnloFXFX-madspin-pythia8/ttW_2/180827_203553/0000/*.root",	195.472703		],
	[	"/TTZToLLNuNu_M-10_TuneCUETP8M1_13TeV-amcatnlo-pythia8/ttZ/180827_203612/0000/*.root",	246.881972		],
	[	"/TTJets_Dilept_TuneCUETP8M2T4_13TeV-amcatnloFXFX-pythia8/tt_ll_AMC/180827_203154/0000/*.root",	97340.321489		],
	[	"/TTJets_DiLept_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/tt_ll_MG_1/180827_203121/0000/*.root",	98109.617801		],
	[	"/TTJets_DiLept_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/tt_ll_MG_2/180827_203138/0000/*.root",	97747.879187		]

]


for s in signal:
	c.add_signal(s[0], mc_path+s[0], s[1])

for b in background:
	c.add_background(b[0], bkg_path+b[0], b[1])
# c.add_signal('H2Mu_gg', 	inFileDir+"H2Mu_gg.root", 	0.006343) # label, path, weight
# c.add_signal('H2Mu_VBF', 	inFileDir+"H2Mu_VBF.root", 	0.000495)

# c.add_background('dy', inFileDir+"ZJets_AMC.root", 29.853717)

c.set_tree_path(treePath)

c.add_variable("muPairs.pt", 	1) #second argument is the number of objects considered
c.add_variable("muPairs.eta", 	1)
c.add_variable("muPairs.dEta", 	1) #for example, one muon pair
c.add_variable("muPairs.dPhi", 	1)
c.add_variable("met.pt", 1)
c.add_variable("nJetsCent", 1)
c.add_variable("nJetsFwd",1)
c.add_variable("nBMed",1)
# c.add_variable("muons.pt", 		2) #two muons
# c.add_variable("muons.eta", 	2)
# c.add_variable("muons.phi", 	2)

c.add_package("TMVA")
# c.add_transf("N,G,P")

# c.add_package("Keras")

c.train_methods()