import os, sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from src.classifier import Framework

c = Framework()

inFileDir = "/data/dmitry/Hmumu/ntuples/"
treePath = 'dimuons/tree'

c.add_signal('gluglu', inFileDir+"H2Mu_gg.root", 0.006343) # label, path, weight
c.add_signal('vbf', inFileDir+"H2Mu_VBF.root", 0.000495)

c.add_background('dy', inFileDir+"ZJets_AMC.root", 29.853717)

c.set_tree_path(treePath)

c.add_variable("muPairs.pt", 	1) #second argument is the number of objects considered
c.add_variable("muPairs.dEta", 	1) #for example, one muon pair
c.add_variable("muPairs.dPhi", 	1)
c.add_variable("muons.pt", 		2) #two muons
c.add_variable("muons.eta", 	2)
c.add_variable("muons.phi", 	2)

c.add_package("TMVA")
# c.add_transf("N,G,P")

# c.add_package("Keras")

c.train_methods()