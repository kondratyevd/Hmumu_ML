import os, sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from src.classifier import Framework

c = Framework()

inFileDir = "/data/dmitry/Hmumu/ntuples/"
treePath = 'dimuons/tree'

signal = [
		['H2Mu_VBF',18.835700],
		['H2Mu_gg',232.557886],
		['H2Mu_WH_neg',0.304798],
		['H2Mu_WH_pos',0.437732],
		['H2Mu_ZH',0.837788]
]

background = [

	[	'WW',	10524.477792		],
	[	'WWW',	79.037588		],
	[	'WWZ',	63.230267		],
	[	'WZZ',	17.997588		],
	[	'WZ_2l',909.986306			],
	[	'WZ_3l',	1853.675094		],
	[	'ZJets_AMC', 1137600.914372			],
	[	'ZZZ',	3.428108		],
	[	'ZZ_2l_2q',	722.244018		],
	[	'ZZ_2l_2v',	212.112867		],
	[	'ZZ_4l',	946.472171		],
	[	'tW_neg_1',	9563.665061		],
	[	'tW_neg_2',	9548.841351		],
	[	'tW_pos_1',	9502.605773		],
	[	'tW_pos_2',	9546.889180		],
	[	'tZq',	85.976629		],
	[	'ttW_1',	196.426646		],
	[	'ttW_2',	195.472703		],
	[	'ttZ',	246.881972		],
	[	'tt_ll_AMC',	97340.321489		],
	[	'tt_ll_MG_1',	98109.617801		],
	[	'tt_ll_MG_2',	97747.879187		]

]


for s in signal:
	c.add_signal(s[0], inFileDir+s[0]+".root", s[1])

for b in background:
	c.add_background(b[0], inFileDir+b[0]+".root", b[1])
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