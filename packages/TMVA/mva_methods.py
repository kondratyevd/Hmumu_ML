from ROOT import TMVA

import os, sys, errno
from locations import modelsDir, logsDir
# import keras_models
from keras_models import GetListOfModels
from setup import numOfVar
################################ Methods ################################

methods = []

class mva_method(object):
	def __init__(self, type, name, options):
		self.type = type
		self.name = name
		self.options = options
		
# methods.append(mva_method(
# 	type = TMVA.Types.kMLP,
# 	name = 'MLP_2x20_NGD',
# 	options = "H:!V:NeuronType=tanh:NCycles=600:VarTransform=N,G,D:HiddenLayers=20,20:TestRate=5:LearningRate=0.05:DecayRate=0.00001!UseRegulator"
# 	))
	
methods.append(mva_method(
	type = TMVA.Types.kMLP,
	name = 'MLP_2x20_NGP',
	options = "H:!V:NeuronType=tanh:NCycles=600:VarTransform=N,G,P:HiddenLayers=20,20:TestRate=5:LearningRate=0.05:DecayRate=0.00001!UseRegulator"
	))	


# Keras models should be defined and selected in keras_models.py

list_of_models = GetListOfModels(numOfVar) #the argument is for the input dimensions
for obj in list_of_models:
	obj.CompileModel()
	methods.append(mva_method(
	type = TMVA.Types.kPyKeras,
	name = obj.name,												
	options = 'H:!V:VarTransform=N,G,P:FilenameModel=%s_init.h5:NumEpochs=%i:BatchSize=%i:Tensorboard=%s'%(modelsDir+obj.name, obj.epochs, obj.batchSize, logsDir+obj.name)
		))





