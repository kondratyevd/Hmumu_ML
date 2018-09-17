import os, sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from src.classifier import Framework

c = Framework()

treePath = 'dimuons/tree'

mc_path = "/mnt/hadoop/store/user/dkondrat/2017_ntuples/"

signal = [
		["H2Mu_VBF", 		"/VBFH_HToMuMu_M125_13TeV_amcatnloFXFX_pythia8/180802_164241/0000/tuple_10.root"    , 				0.0008208		],
		["H2Mu_gg", 		"/GluGlu_HToMuMu_M125_13TeV_amcatnloFXFX_pythia8/180802_164158/0000/tuple_1.root"     , 			0.009618		],
		["H2Mu_WH_pos", 	"/WPlusH_HToMuMu_M125_13TeV_powheg_pythia8/180802_164326/0000/tuple_*.root" , 						0.0001858		],
		["H2Mu_WH_neg", 	"/WMinusH_HToMuMu_M125_13TeV_powheg_pythia8/180802_164407/0000/tuple_*.root" , 						0.0001164		],
		["H2Mu_ZH", 		"/ZH_HToMuMu_M125_13TeV_powheg_pythia8/180802_164450/0000/tuple_*.root"     , 						0.00003865		]
]

bkg_path = "/mnt/hadoop/store/user/dkondrat/2017_ntuples/"
background = [

	["WW", 	  				"/WWTo2L2Nu_NNPDF31_TuneCP5_13TeV-powheg-pythia8/180802_165439/0000/tuple_1.root"          , 		12.46			],
	["WZ_3l", 				"/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/180802_165652/0000/tuple_1.root"       , 				4.42965			],
	["ZJets_AMC", 			"/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/180802_165055/0000/tuple_1.root", 				5765.4			],
	["ZZ_2l_2v", 			"/ZZTo2L2Nu_13TeV_powheg_pythia8/180802_165732/0000/tuple_1.root"    , 								0.564			],
	["ZZ_4l", 				"/ZZTo4L_13TeV_powheg_pythia8/180802_165816/0000/tuple_1.root"       , 								1.256			],
	["tZq", 				"/tZq_ll_4f_ckm_NLO_TuneCP5_PSweights_13TeV-amcatnlo-pythia8/180802_170157/0000/tuple_1.root" ,		0.0758			],
	["ttZ", 				"/ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8/180802_170325/0000/tuple_1.root"         , 				0.2529			],
	["tt_ll_AMC", 			"/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/180802_165355/0000/tuple_1.root"   , 					85.656*0.9		]

]


for s in signal:
	c.add_signal(s[0], mc_path+s[1], s[2])

for b in background:
	c.add_background(b[0], bkg_path+b[1], b[2])


c.set_tree_path(treePath)

c.add_variable("muPairs.pt", 				1) #second argument is the number of objects considered
c.add_variable("muPairs.eta", 				1)
c.add_variable("muPairs.phi", 				1)
c.add_variable("muPairs.dEta", 				1) 
c.add_variable("muPairs.dPhi", 				1)
c.add_variable("muons.eta", 				2)
c.add_variable("met.pt", 					1)
c.add_variable("nJets",		 				1)
c.add_variable("nJetsCent", 				1)
# c.add_variable("nJetsFwd",					1)
c.add_variable("nBMed",						1)
# c.add_variable("jets.eta",					2)
c.add_variable("jetPairs.dEta",				2)
c.add_variable("jetPairs.mass",				2)

# c.add_variable("muons.pt", 	2) #two muons
# c.add_variable("muons.phi", 	2)

c.set_year("2017")
c.add_package("TMVA")
c.add_transf("N,G,P")
c.add_method("BDTG_MIT")

# c.add_package("Keras")

c.train_methods()