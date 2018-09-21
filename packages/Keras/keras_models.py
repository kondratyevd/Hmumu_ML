from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
from keras.regularizers import l2
from keras.optimizers import SGD
import os, sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
# from locations import *



class model_init(object):
	def __init__(self, name, input_dim, batchSize, epochs, loss, optimizer):
		self.name = name
		self.batchSize = batchSize
		self.epochs = epochs
		self.inputs = Input(shape=(input_dim,), name = name+'_input') 
		self.loss = loss
		self.optimizer = optimizer
		

	def CompileModel(self, modelDir):
		self.model = Model(inputs=self.inputs, outputs=self.outputs)
		self.model.compile(loss=self.loss,									# This may be transferred into input parameters in the future
		              optimizer=self.optimizer, metrics=['accuracy', ])					# if we need to optimize any of these parameters
		self.model.save(modelDir+self.name+'_init.h5')
		self.model.summary()	



def GetListOfModels(nVar):

	list_of_models = []

	test_model = model_init('test_model', nVar, 2048, 20, 'binary_crossentropy', 'adam')
	x = Dense(1, name = test_model.name+'_layer_1', activation='sigmoid')(test_model.inputs)
	# x = Dense(2, name = test_model.name+'_layer_2', activation='sigmoid')(x)
	test_model.outputs = Dense(2, name = test_model.name+'_output',  activation='softmax')(x)


	model_2x20 = model_init('model_2x20', nVar, 2048, 200, 'binary_crossentropy', 'adam')
	x = Dense(20, name = model_2x20.name+'_layer_1', activation='sigmoid')(model_2x20.inputs)
	x = Dense(20, name = model_2x20.name+'_layer_2', activation='sigmoid')(x)
	model_2x20.outputs = Dense(2, name = model_2x20.name+'_output',  activation='softmax')(x)


	model_3x100 = model_init('model_3x100', nVar, 2048, 200, 'binary_crossentropy', 'adam')
	x = Dense(20, name = model_3x100.name+'_layer_1', kernel_initializer='normal', activation='sigmoid')(model_3x100.inputs)
	x = Dense(20, name = model_3x100.name+'_layer_2', kernel_initializer='normal', activation='sigmoid')(x)
	x = Dense(20, name = model_3x100.name+'_layer_3', kernel_initializer='normal', activation='sigmoid')(x)
	model_3x100.outputs = Dense(2, name = model_3x100.name+'_output', activation='softmax')(x)

	list_of_models.append(model_3x100)
	list_of_models.append(model_2x20)
	list_of_models.append(test_model)
	
	return list_of_models





