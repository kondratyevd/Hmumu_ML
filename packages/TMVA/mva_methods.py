from ROOT import TMVA

import os, sys, errno
################################ Methods ################################


class mva_method(object):
	def __init__(self, type, name, options):
		self.type = type
		self.name = name
		self.options = options

def compile_method_list(framework, package):
	methods = []

		
	# methods.append(mva_method(
	# 	type = TMVA.Types.kMLP,
	# 	name = 'MLP_2x20_NGD',
	# 	options = "H:!V:NeuronType=tanh:NCycles=600:VarTransform=N,G,D:HiddenLayers=20,20:TestRate=5:LearningRate=0.05:DecayRate=0.00001!UseRegulator"
	# 	))
	
	methods.append(mva_method(
		type = TMVA.Types.kMLP,
		name = 'MLP_20,20,20_NGP',
		options = "H:!V:NeuronType=tanh:NCycles=200:VarTransform=N,G,P:HiddenLayers=20,20,20:TestRate=5:LearningRate=0.05:DecayRate=0.00001!UseRegulator"
		))	

	# methods.append(mva_method(
	# 	type = TMVA.Types.kMLP,
	# 	name = 'test_method',
	# 	# options = "H:!V:NeuronType=tanh:NCycles=10:VarTransform=N,G,P:HiddenLayers=20,20:TestRate=5:LearningRate=0.05:DecayRate=0.00001!UseRegulator"
	# 	))
			  
	methods.append(mva_method(
		type = TMVA.Types.kBDT,
		name = 'BDTG_UF_v1',
		options = "!H:!V:NTrees=500::BoostType=Grad:Shrinkage=0.1:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=20:MaxDepth=5"
		))	

	methods.append(mva_method(
		type = TMVA.Types.kBDT,
		name = 'BDTG_UCSD',
		options = "!H:!V:CreateMVAPdfs:NTrees=1200:MinNodeSize=3%:BoostType=Grad:Shrinkage=0.1:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=40:MaxDepth=4:VarTransform=I,N"
		))	

	methods.append(mva_method(
		type = TMVA.Types.kBDT,
		name = 'BDTG_MIT',
		options = "!H:!V:NTrees=5000:MinNodeSize=3%:BoostType=Grad:Shrinkage=0.10:nCuts=40:MaxDepth=5:NodePurityLimit=0.95:Pray"
		))	

	methods.append(mva_method(
		type = TMVA.Types.kBDT,
		name = 'BDTG_MIT_lite',
		options = "!H:!V:NTrees=500:MinNodeSize=3%:BoostType=Grad:Shrinkage=0.10:nCuts=40:MaxDepth=5:NodePurityLimit=0.95:Pray"
		))	

	return methods




