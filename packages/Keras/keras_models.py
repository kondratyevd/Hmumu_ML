from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Dropout, Concatenate, Lambda
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


	UCSD_model = model_init('UCSD_model', nVar, 2048, 200, 'categorical_crossentropy', 'adam')
	x = Dense(50, name = UCSD_model.name+'_layer_1', activation='relu')(UCSD_model.inputs)
	x = Dropout(0.2)(x)
	UCSD_model.outputs = Dense(2, name = UCSD_model.name+'_output',  activation='softmax')(x)


	model_50_D2_25_D2 = model_init('model_50_D2_25_D2', nVar, 2048, 200, 'categorical_crossentropy', 'adam')
	x = Dense(50, name = model_50_D2_25_D2.name+'_layer_1', activation='relu')(model_50_D2_25_D2.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	model_50_D2_25_D2.outputs = Dense(2, name = model_50_D2_25_D2.name+'_output',  activation='softmax')(x)


	model_50_D1 = model_init('model_50_D1', nVar, 2048, 200, 'categorical_crossentropy', 'adam')
	x = Dense(50, name = model_50_D1.name+'_layer_1', activation='relu')(model_50_D1.inputs)
	x = Dropout(0.1)(x)
	model_50_D1.outputs = Dense(2, name = model_50_D1.name+'_output',  activation='softmax')(x)


	model_50_D2_25_D2_25_D2 = model_init('model_50_D2_25_D2_25_D2', nVar, 2048, 200, 'categorical_crossentropy', 'adam')
	x = Dense(50, name = model_50_D2_25_D2_25_D2.name+'_layer_1', activation='relu')(model_50_D2_25_D2_25_D2.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_25_D2.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_25_D2.name+'_layer_3', activation='relu')(x)
	x = Dropout(0.2)(x)
	model_50_D2_25_D2_25_D2.outputs = Dense(2, name = model_50_D2_25_D2_25_D2.name+'_output',  activation='softmax')(x)



	list_of_models.append(UCSD_model) # 50_D2
	list_of_models.append(model_50_D1)
	list_of_models.append(model_50_D2_25_D2)
	list_of_models.append(model_50_D2_25_D2_25_D2)


	from keras import backend as K
	from tensorflow import where, greater, abs, zeros_like, exp
	import tensorflow as tf
	from keras.losses import kullback_leibler_divergence, categorical_crossentropy

	NBINS=10 # number of bins for loss function
	MMAX = 110. # max value
	MMIN = 150. # min value
	LAMBDA = 0.1 # lambda for penalty

	def loss_kldiv(y_in,x_in):
	    """
	    mass sculpting penlaty term usking kullback_leibler_divergence
	    y_in: truth [h, y]
	    x: predicted NN output for y
	    h: the truth mass histogram vector "one-hot encoded" (length NBINS=40)
	    y: the truth categorical labels  "one-hot encoded" (length NClasses=2)
	    """
	    h = y_in[:,0:NBINS]
	    y = y_in[:,NBINS:NBINS+2]
	    x = x_in[:,NBINS:NBINS+2]	    
	    h_all = K.dot(K.transpose(h), y)
	    h_all_q = h_all[:,0]
	    h_all_h = h_all[:,1]
	    h_all_q = h_all_q / K.sum(h_all_q,axis=0)
	    h_all_h = h_all_h / K.sum(h_all_h,axis=0)
	    h_btag_anti_q = K.dot(K.transpose(h), K.dot(tf.diag(y[:,0]),x))
	    h_btag_anti_h = K.dot(K.transpose(h), K.dot(tf.diag(y[:,1]),x))
	    h_btag_q = h_btag_anti_q[:,1]
	    h_btag_q = h_btag_q / K.sum(h_btag_q,axis=0)
	    h_anti_q = h_btag_anti_q[:,0]
	    h_anti_q = h_anti_q / K.sum(h_anti_q,axis=0)
	    h_btag_h = h_btag_anti_h[:,1]
	    h_btag_h = h_btag_h / K.sum(h_btag_h,axis=0)
	    h_anti_h = h_btag_anti_q[:,0]
	    h_anti_h = h_anti_h / K.sum(h_anti_h,axis=0)
	
	    return categorical_crossentropy(y, x) + \
	        LAMBDA*kullback_leibler_divergence(h_btag_q, h_anti_q) + \
	        LAMBDA*kullback_leibler_divergence(h_btag_h, h_anti_h)   


	def my_loss(y_truth, x_pred):
		return categorical_crossentropy(y_pred[:,10:12], y_truth[:,10:12])


	model_50_D2_25_D2_kldiv = model_init('model_50_D2_25_D2_kldiv', nVar, 2048, 200, [loss_kldiv], 'adam')
	x = Dense(50, name = model_50_D2_25_D2_kldiv.name+'_layer_1', activation='relu')(model_50_D2_25_D2_kldiv.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_kldiv.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	# model_50_D2_25_D2_kldiv.outputs = Dense(12, name = model_50_D2_25_D2_kldiv.name+'_output',  activation='softmax')(x)
	out1 = Dense(2, name = model_50_D2_25_D2_kldiv.name+'_output',  activation='softmax')(x)
	
	lambdaLayer = Lambda(lambda x: 0*x, name='lambda')(model_50_D2_25_D2_kldiv.inputs)
	def slicer(x):
	    return x[:,0:10]    
	lambdaLayer = Lambda(slicer)(lambdaLayer)

	model_50_D2_25_D2_kldiv.outputs = Concatenate()([lambdaLayer, out1]) # order is important


	list_of_models.append(model_50_D2_25_D2_kldiv)





	return list_of_models





