from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Dropout, Concatenate, Lambda, BatchNormalization
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



def GetListOfModels(trainer):
	input_dim = len(trainer.labels)
	output_dim = len(trainer.truth_labels)
	n_categories = len(trainer.category_labels)
	list_of_models = []


	UCSD_model = model_init('UCSD_model', input_dim, 2048, 200, 'categorical_crossentropy', 'adam')
	x = Dense(50, name = UCSD_model.name+'_layer_1', activation='relu')(UCSD_model.inputs)
	x = Dropout(0.2)(x)
	UCSD_model.outputs = Dense(output_dim, name = UCSD_model.name+'_output',  activation='softmax')(x)


	model_50_D2_25_D2 = model_init('model_50_D2_25_D2', input_dim, 2048, 200, 'categorical_crossentropy', 'adam')
	x = Dense(50, name = model_50_D2_25_D2.name+'_layer_1', activation='relu')(model_50_D2_25_D2.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	model_50_D2_25_D2.outputs = Dense(output_dim, name = model_50_D2_25_D2.name+'_output',  activation='softmax')(x)


	model_50_D1 = model_init('model_50_D1', input_dim, 2048, 200, 'categorical_crossentropy', 'adam')
	x = Dense(50, name = model_50_D1.name+'_layer_1', activation='relu')(model_50_D1.inputs)
	x = Dropout(0.1)(x)
	model_50_D1.outputs = Dense(output_dim, name = model_50_D1.name+'_output',  activation='softmax')(x)


	model_50_D2_25_D2_25_D2 = model_init('model_50_D2_25_D2_25_D2', input_dim, 2048, 200, 'categorical_crossentropy', 'adam')
	x = Dense(50, name = model_50_D2_25_D2_25_D2.name+'_layer_1', activation='relu')(model_50_D2_25_D2_25_D2.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_25_D2.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_25_D2.name+'_layer_3', activation='relu')(x)
	x = Dropout(0.2)(x)
	model_50_D2_25_D2_25_D2.outputs = Dense(output_dim, name = model_50_D2_25_D2_25_D2.name+'_output',  activation='softmax')(x)


	model_50_25_25 = model_init('model_50_25_25', input_dim, 2048, 200, 'categorical_crossentropy', 'adam')
	x = Dense(50, name = model_50_25_25.name+'_layer_1', activation='relu')(model_50_25_25.inputs)
	x = Dense(25, name = model_50_25_25.name+'_layer_2', activation='relu')(x)
	x = Dense(25, name = model_50_25_25.name+'_layer_3', activation='relu')(x)
	model_50_25_25.outputs = Dense(output_dim, name = model_50_25_25.name+'_output',  activation='softmax')(x)


#    model = Sequential()
#    model.add(Dense(40, input_dim=len(features),activation='sigmoid',use_bias=True,kernel_initializer='glorot_normal',bias_initializer='zeros',kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01)) )
#    model.add(BatchNormalization() )
#    model.add(Dropout(0.25))
#    model.add(Dense(25,activation='sigmoid',use_bias=True,kernel_regularizer=l2(0.01),kernel_initializer='glorot_normal',bias_initializer='zeros'))
#    model.add(BatchNormalization() )
#    model.add(Dense(10,activation='sigmoid',use_bias=True,kernel_initializer='glorot_normal',bias_initializer='zeros'))
#    model.add(Dense(1,activation='sigmoid'))

	andrea_model_3 = model_init('andrea_model_3', input_dim, 2048, 200, 'categorical_crossentropy', 'adam')
	x = Dense(40, name = andrea_model_3.name+'_layer_1', activation='sigmoid', use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(andrea_model_3.inputs)
	x = BatchNormalization()(x)
	x = Dropout(0.25)(x)
	x = Dense(25, name = andrea_model_3.name+'_layer_2', activation='sigmoid',use_bias=True, kernel_regularizer=l2(0.01), kernel_initializer='glorot_normal', bias_initializer='zeros')(x)
	x = BatchNormalization()(x)
	x = Dense(10, name = andrea_model_3.name+'_layer_3', activation='sigmoid',use_bias=True,kernel_initializer='glorot_normal',bias_initializer='zeros')(x)
	andrea_model_3.outputs = Dense(output_dim, name = andrea_model_3.name+'_output',  activation='softmax')(x)

	list_of_models.append(UCSD_model) # 50_D2
	list_of_models.append(model_50_D1)
	list_of_models.append(model_50_D2_25_D2)
	list_of_models.append(model_50_D2_25_D2_25_D2)
	list_of_models.append(model_50_25_25)
	list_of_models.append(andrea_model_3)

	from keras import backend as K
	from tensorflow import where, greater, abs, zeros_like, exp
	import tensorflow as tf
	from keras.losses import kullback_leibler_divergence, categorical_crossentropy

	NBINS=10 # number of bins for loss function
	MMAX = 110. # max value
	MMIN = 150. # min value


	# def loss_kldiv(y_in,x_in):
	#     """
	#     mass sculpting penlaty term usking kullback_leibler_divergence
	#     y_in: truth [h, y]
	#     x: predicted NN output for y
	#     h: the truth mass histogram vector "one-hot encoded" (length NBINS=40)
	#     y: the truth categorical labels  "one-hot encoded" (length NClasses=2)
	#     """
	#     h = y_in[:,0:NBINS]
	#     y = y_in[:,NBINS:NBINS+n_categories]
	#     x = x_in[:,NBINS:NBINS+n_categories]	    
	#     h_blike_slike_s = K.dot(K.transpose(h), K.dot(tf.diag(y[:,0]),x))
	#     h_blike_slike_b = K.dot(K.transpose(h), K.dot(tf.diag(y[:,1]),x))
	#     h_blike_s = h_blike_slike_s[:,1]
	#     h_blike_s = h_blike_s / K.sum(h_blike_s,axis=0)
	#     h_slike_s = h_blike_slike_s[:,0]
	#     h_slike_s = h_slike_s / K.sum(h_slike_s,axis=0)
	#     h_blike_b = h_blike_slike_b[:,1]
	#     h_blike_b = h_blike_b / K.sum(h_blike_b,axis=0)
	#     h_slike_b = h_blike_slike_s[:,0]
	#     h_slike_b = h_slike_b / K.sum(h_slike_b,axis=0)
	
	#     return categorical_crossentropy(y, x) + \
	#         LAMBDA*kullback_leibler_divergence(h_blike_s, h_slike_s) + \
	#         LAMBDA*kullback_leibler_divergence(h_blike_b, h_slike_b)  



	def loss_kldiv0(y_in,x_in):
	    h = y_in[:,0:NBINS]
	    y = y_in[:,NBINS:NBINS+n_categories]
	    x = x_in[:,NBINS:NBINS+n_categories]	    
	    h_blike_slike_s = K.dot(K.transpose(h), K.dot(tf.diag(y[:,0]),x))
	    h_blike_slike_b = K.dot(K.transpose(h), K.dot(tf.diag(y[:,1]),x))
	    h_blike_s = h_blike_slike_s[:,1]
	    h_blike_s = h_blike_s / K.sum(h_blike_s,axis=0)
	    h_slike_s = h_blike_slike_s[:,0]
	    h_slike_s = h_slike_s / K.sum(h_slike_s,axis=0)
	    h_blike_b = h_blike_slike_b[:,1]
	    h_blike_b = h_blike_b / K.sum(h_blike_b,axis=0)
	    h_slike_b = h_blike_slike_s[:,0]
	    h_slike_b = h_slike_b / K.sum(h_slike_b,axis=0)
	
	    return categorical_crossentropy(y, x) 


	model_50_D2_25_D2_kldiv0 = model_init('model_50_D2_25_D2_kldiv0', input_dim, 2048, 100, [loss_kldiv0], 'adam')
	x = Dense(50, name = model_50_D2_25_D2_kldiv0.name+'_layer_1', activation='relu')(model_50_D2_25_D2_kldiv0.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_kldiv0.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	out1 = Dense(n_categories , name = model_50_D2_25_D2_kldiv0.name+'_output',  activation='softmax')(x)
	
	lambdaLayer = Lambda(lambda x: 0*x, name='lambda')(model_50_D2_25_D2_kldiv0.inputs)
	def slicer(x):
	    return x[:,0:NBINS]    
	lambdaLayer = Lambda(slicer)(lambdaLayer)

	model_50_D2_25_D2_kldiv0.outputs = Concatenate()([lambdaLayer, out1]) # order is important

	list_of_models.append(model_50_D2_25_D2_kldiv0)

	def loss_kldiv1(y_in,x_in):
	    h = y_in[:,0:NBINS]
	    y = y_in[:,NBINS:NBINS+n_categories] # order of categories like in category_labels
	    x = x_in[:,NBINS:NBINS+n_categories]	    
	    h_blike_slike_s = K.dot(K.transpose(h), K.dot(tf.diag(y[:,0]),x))
	    h_blike_slike_b = K.dot(K.transpose(h), K.dot(tf.diag(y[:,1]),x))
	    h_blike_s = h_blike_slike_s[:,1]
	    h_blike_s = h_blike_s / K.sum(h_blike_s,axis=0)
	    h_slike_s = h_blike_slike_s[:,0]
	    h_slike_s = h_slike_s / K.sum(h_slike_s,axis=0)
	    h_blike_b = h_blike_slike_b[:,1]
	    h_blike_b = h_blike_b / K.sum(h_blike_b,axis=0)
	    h_slike_b = h_blike_slike_s[:,0]
	    h_slike_b = h_slike_b / K.sum(h_slike_b,axis=0)
	
	    return categorical_crossentropy(y, x) + \
	        0.1*kullback_leibler_divergence(h_blike_s, h_slike_s) + \
	        0.1*kullback_leibler_divergence(h_blike_b, h_slike_b)    


	model_50_D2_25_D2_kldiv1 = model_init('model_50_D2_25_D2_kldiv1', input_dim, 2048, 100, [loss_kldiv1], 'adam')
	x = Dense(50, name = model_50_D2_25_D2_kldiv1.name+'_layer_1', activation='relu')(model_50_D2_25_D2_kldiv1.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_kldiv1.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	out1 = Dense(n_categories , name = model_50_D2_25_D2_kldiv1.name+'_output',  activation='softmax')(x)
	
	lambdaLayer = Lambda(lambda x: 0*x, name='lambda')(model_50_D2_25_D2_kldiv1.inputs)
	def slicer(x):
	    return x[:,0:NBINS]    
	lambdaLayer = Lambda(slicer)(lambdaLayer)

	model_50_D2_25_D2_kldiv1.outputs = Concatenate()([lambdaLayer, out1]) # order is important

	list_of_models.append(model_50_D2_25_D2_kldiv1)


	def loss_kldiv2(y_in,x_in):
	    h = y_in[:,0:NBINS]
	    y = y_in[:,NBINS:NBINS+n_categories]
	    x = x_in[:,NBINS:NBINS+n_categories]	    
	    h_blike_slike_s = K.dot(K.transpose(h), K.dot(tf.diag(y[:,0]),x))
	    h_blike_slike_b = K.dot(K.transpose(h), K.dot(tf.diag(y[:,1]),x))
	    h_blike_s = h_blike_slike_s[:,1]
	    h_blike_s = h_blike_s / K.sum(h_blike_s,axis=0)
	    h_slike_s = h_blike_slike_s[:,0]
	    h_slike_s = h_slike_s / K.sum(h_slike_s,axis=0)
	    h_blike_b = h_blike_slike_b[:,1]
	    h_blike_b = h_blike_b / K.sum(h_blike_b,axis=0)
	    h_slike_b = h_blike_slike_s[:,0]
	    h_slike_b = h_slike_b / K.sum(h_slike_b,axis=0)
	
	    return categorical_crossentropy(y, x) + \
	        0.2*kullback_leibler_divergence(h_blike_s, h_slike_s) + \
	        0.2*kullback_leibler_divergence(h_blike_b, h_slike_b)  


	model_50_D2_25_D2_kldiv2 = model_init('model_50_D2_25_D2_kldiv2', input_dim, 2048, 100, [loss_kldiv2], 'adam')
	x = Dense(50, name = model_50_D2_25_D2_kldiv2.name+'_layer_1', activation='relu')(model_50_D2_25_D2_kldiv2.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_kldiv2.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	out1 = Dense(n_categories , name = model_50_D2_25_D2_kldiv2.name+'_output',  activation='softmax')(x)
	
	lambdaLayer = Lambda(lambda x: 0*x, name='lambda')(model_50_D2_25_D2_kldiv2.inputs)
	def slicer(x):
	    return x[:,0:NBINS]    
	lambdaLayer = Lambda(slicer)(lambdaLayer)

	model_50_D2_25_D2_kldiv2.outputs = Concatenate()([lambdaLayer, out1]) # order is important

	list_of_models.append(model_50_D2_25_D2_kldiv2)







	def loss_kldiv3(y_in,x_in):
	    h = y_in[:,0:NBINS]
	    y = y_in[:,NBINS:NBINS+n_categories]
	    x = x_in[:,NBINS:NBINS+n_categories]	    
	    h_blike_slike_s = K.dot(K.transpose(h), K.dot(tf.diag(y[:,0]),x))
	    h_blike_slike_b = K.dot(K.transpose(h), K.dot(tf.diag(y[:,1]),x))
	    h_blike_s = h_blike_slike_s[:,1]
	    h_blike_s = h_blike_s / K.sum(h_blike_s,axis=0)
	    h_slike_s = h_blike_slike_s[:,0]
	    h_slike_s = h_slike_s / K.sum(h_slike_s,axis=0)
	    h_blike_b = h_blike_slike_b[:,1]
	    h_blike_b = h_blike_b / K.sum(h_blike_b,axis=0)
	    h_slike_b = h_blike_slike_s[:,0]
	    h_slike_b = h_slike_b / K.sum(h_slike_b,axis=0)
	
	    return categorical_crossentropy(y, x) + \
	        0.3*kullback_leibler_divergence(h_blike_s, h_slike_s) + \
	        0.3*kullback_leibler_divergence(h_blike_b, h_slike_b)    


	model_50_D2_25_D2_kldiv3 = model_init('model_50_D2_25_D2_kldiv3', input_dim, 2048, 100, [loss_kldiv3], 'adam')
	x = Dense(50, name = model_50_D2_25_D2_kldiv3.name+'_layer_1', activation='relu')(model_50_D2_25_D2_kldiv3.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_kldiv3.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	out1 = Dense(n_categories , name = model_50_D2_25_D2_kldiv3.name+'_output',  activation='softmax')(x)
	
	lambdaLayer = Lambda(lambda x: 0*x, name='lambda')(model_50_D2_25_D2_kldiv3.inputs)
	def slicer(x):
	    return x[:,0:NBINS]    
	lambdaLayer = Lambda(slicer)(lambdaLayer)

	model_50_D2_25_D2_kldiv3.outputs = Concatenate()([lambdaLayer, out1]) # order is important

	list_of_models.append(model_50_D2_25_D2_kldiv3)





	def loss_kldiv4(y_in,x_in):
	    h = y_in[:,0:NBINS]
	    y = y_in[:,NBINS:NBINS+n_categories]
	    x = x_in[:,NBINS:NBINS+n_categories]	    
	    h_blike_slike_s = K.dot(K.transpose(h), K.dot(tf.diag(y[:,0]),x))
	    h_blike_slike_b = K.dot(K.transpose(h), K.dot(tf.diag(y[:,1]),x))
	    h_blike_s = h_blike_slike_s[:,1]
	    h_blike_s = h_blike_s / K.sum(h_blike_s,axis=0)
	    h_slike_s = h_blike_slike_s[:,0]
	    h_slike_s = h_slike_s / K.sum(h_slike_s,axis=0)
	    h_blike_b = h_blike_slike_b[:,1]
	    h_blike_b = h_blike_b / K.sum(h_blike_b,axis=0)
	    h_slike_b = h_blike_slike_s[:,0]
	    h_slike_b = h_slike_b / K.sum(h_slike_b,axis=0)
	
	    return categorical_crossentropy(y, x) + \
	        0.4*kullback_leibler_divergence(h_blike_s, h_slike_s) + \
	        0.4*kullback_leibler_divergence(h_blike_b, h_slike_b)    


	model_50_D2_25_D2_kldiv4 = model_init('model_50_D2_25_D2_kldiv4', input_dim, 2048, 100, [loss_kldiv4], 'adam')
	x = Dense(50, name = model_50_D2_25_D2_kldiv4.name+'_layer_1', activation='relu')(model_50_D2_25_D2_kldiv4.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_kldiv4.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	out1 = Dense(n_categories , name = model_50_D2_25_D2_kldiv4.name+'_output',  activation='softmax')(x)
	
	lambdaLayer = Lambda(lambda x: 0*x, name='lambda')(model_50_D2_25_D2_kldiv4.inputs)
	def slicer(x):
	    return x[:,0:NBINS]    
	lambdaLayer = Lambda(slicer)(lambdaLayer)

	model_50_D2_25_D2_kldiv4.outputs = Concatenate()([lambdaLayer, out1]) # order is important

	list_of_models.append(model_50_D2_25_D2_kldiv4)





	def loss_kldiv5(y_in,x_in):
	    h = y_in[:,0:NBINS]
	    y = y_in[:,NBINS:NBINS+n_categories]
	    x = x_in[:,NBINS:NBINS+n_categories]	    
	    h_blike_slike_s = K.dot(K.transpose(h), K.dot(tf.diag(y[:,0]),x))
	    h_blike_slike_b = K.dot(K.transpose(h), K.dot(tf.diag(y[:,1]),x))
	    h_blike_s = h_blike_slike_s[:,1]
	    h_blike_s = h_blike_s / K.sum(h_blike_s,axis=0)
	    h_slike_s = h_blike_slike_s[:,0]
	    h_slike_s = h_slike_s / K.sum(h_slike_s,axis=0)
	    h_blike_b = h_blike_slike_b[:,1]
	    h_blike_b = h_blike_b / K.sum(h_blike_b,axis=0)
	    h_slike_b = h_blike_slike_s[:,0]
	    h_slike_b = h_slike_b / K.sum(h_slike_b,axis=0)
	
	    return categorical_crossentropy(y, x) + \
	        0.5*kullback_leibler_divergence(h_blike_s, h_slike_s) + \
	        0.5*kullback_leibler_divergence(h_blike_b, h_slike_b)  


	model_50_D2_25_D2_kldiv5 = model_init('model_50_D2_25_D2_kldiv5', input_dim, 2048, 100, [loss_kldiv5], 'adam')
	x = Dense(50, name = model_50_D2_25_D2_kldiv5.name+'_layer_1', activation='relu')(model_50_D2_25_D2_kldiv5.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_kldiv5.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	out1 = Dense(n_categories , name = model_50_D2_25_D2_kldiv5.name+'_output',  activation='softmax')(x)
	
	lambdaLayer = Lambda(lambda x: 0*x, name='lambda')(model_50_D2_25_D2_kldiv5.inputs)
	def slicer(x):
	    return x[:,0:NBINS]    
	lambdaLayer = Lambda(slicer)(lambdaLayer)

	model_50_D2_25_D2_kldiv5.outputs = Concatenate()([lambdaLayer, out1]) # order is important

	list_of_models.append(model_50_D2_25_D2_kldiv5)





	def loss_multiclass_mass_control_0(y_in,x_in):
		LAMBDA = 0
		h = y_in[:,0:NBINS]
		y = y_in[:,NBINS:NBINS+n_categories] # order of categories like in category_labels
		x = x_in[:,NBINS:NBINS+n_categories]

		loss = categorical_crossentropy(y, x)  

		for icat in range(n_categories):
			mass_split_by_prediction = K.dot(K.transpose(h), K.dot(tf.diag(y[:,icat]),x))
			_mass_shape_correct_id = mass_split_by_prediction[:,icat]
			_mass_shape_correct_id = _mass_shape_correct_id / K.sum(_mass_shape_correct_id,axis=0)
			loss += LAMBDA*kullback_leibler_divergence(K.transpose(trainer.mass_histograms[icat]), _mass_shape_correct_id)
		return loss



	model_50_D2_25_D2_mass_control_0 = model_init('model_50_D2_25_D2_mass_control_0', input_dim, 2048, 100, [loss_multiclass_mass_control_0], 'adam')
	x = Dense(50, name = model_50_D2_25_D2_mass_control_0.name+'_layer_1', activation='relu')(model_50_D2_25_D2_mass_control_0.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_mass_control_0.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	out1 = Dense(n_categories , name = model_50_D2_25_D2_mass_control_0.name+'_output',  activation='softmax')(x)
	
	lambdaLayer = Lambda(lambda x: 0*x, name='lambda')(model_50_D2_25_D2_mass_control_0.inputs)
	def slicer(x):
	    return x[:,0:NBINS]    
	lambdaLayer = Lambda(slicer)(lambdaLayer)

	model_50_D2_25_D2_mass_control_0.outputs = Concatenate()([lambdaLayer, out1]) # order is important

	list_of_models.append(model_50_D2_25_D2_mass_control_0)


	def loss_multiclass_mass_control_0p1(y_in,x_in):
		LAMBDA = 0.1
		h = y_in[:,0:NBINS]
		y = y_in[:,NBINS:NBINS+n_categories] # order of categories like in category_labels
		x = x_in[:,NBINS:NBINS+n_categories]

		loss = categorical_crossentropy(y, x)  

		for icat in range(n_categories):
			mass_split_by_prediction = K.dot(K.transpose(h), K.dot(tf.diag(y[:,icat]),x))
			_mass_shape_correct_id = mass_split_by_prediction[:,icat]
			_mass_shape_correct_id = _mass_shape_correct_id / K.sum(_mass_shape_correct_id,axis=0)
			loss += LAMBDA*kullback_leibler_divergence(K.transpose(trainer.mass_histograms[icat]), _mass_shape_correct_id)
		return loss



	model_50_D2_25_D2_mass_control_0p1 = model_init('model_50_D2_25_D2_mass_control_0p1', input_dim, 2048, 100, [loss_multiclass_mass_control_0p1], 'adam')
	x = Dense(50, name = model_50_D2_25_D2_mass_control_0p1.name+'_layer_1', activation='relu')(model_50_D2_25_D2_mass_control_0p1.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_mass_control_0p1.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	out1 = Dense(n_categories , name = model_50_D2_25_D2_mass_control_0p1.name+'_output',  activation='softmax')(x)
	
	lambdaLayer = Lambda(lambda x: 0*x, name='lambda')(model_50_D2_25_D2_mass_control_0p1.inputs)
	def slicer(x):
	    return x[:,0:NBINS]    
	lambdaLayer = Lambda(slicer)(lambdaLayer)

	model_50_D2_25_D2_mass_control_0p1.outputs = Concatenate()([lambdaLayer, out1]) # order is important

	list_of_models.append(model_50_D2_25_D2_mass_control_0p1)


	def loss_multiclass_mass_control_0p2(y_in,x_in):
		LAMBDA = 0.2
		h = y_in[:,0:NBINS]
		y = y_in[:,NBINS:NBINS+n_categories] # order of categories like in category_labels
		x = x_in[:,NBINS:NBINS+n_categories]

		loss = categorical_crossentropy(y, x)  

		for icat in range(n_categories):
			mass_split_by_prediction = K.dot(K.transpose(h), K.dot(tf.diag(y[:,icat]),x))
			_mass_shape_correct_id = mass_split_by_prediction[:,icat]
			_mass_shape_correct_id = _mass_shape_correct_id / K.sum(_mass_shape_correct_id,axis=0)
			loss += LAMBDA*kullback_leibler_divergence(K.transpose(trainer.mass_histograms[icat]), _mass_shape_correct_id)
		return loss



	model_50_D2_25_D2_mass_control_0p2 = model_init('model_50_D2_25_D2_mass_control_0p2', input_dim, 2048, 100, [loss_multiclass_mass_control_0p2], 'adam')
	x = Dense(50, name = model_50_D2_25_D2_mass_control_0p2.name+'_layer_1', activation='relu')(model_50_D2_25_D2_mass_control_0p2.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_mass_control_0p2.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	out1 = Dense(n_categories , name = model_50_D2_25_D2_mass_control_0p2.name+'_output',  activation='softmax')(x)
	
	lambdaLayer = Lambda(lambda x: 0*x, name='lambda')(model_50_D2_25_D2_mass_control_0p2.inputs)
	def slicer(x):
	    return x[:,0:NBINS]    
	lambdaLayer = Lambda(slicer)(lambdaLayer)

	model_50_D2_25_D2_mass_control_0p2.outputs = Concatenate()([lambdaLayer, out1]) # order is important

	list_of_models.append(model_50_D2_25_D2_mass_control_0p2)



	def loss_multiclass_mass_control_0p3(y_in,x_in):
		LAMBDA = 0.3
		h = y_in[:,0:NBINS]
		y = y_in[:,NBINS:NBINS+n_categories] # order of categories like in category_labels
		x = x_in[:,NBINS:NBINS+n_categories]

		loss = categorical_crossentropy(y, x)  

		for icat in range(n_categories):
			mass_split_by_prediction = K.dot(K.transpose(h), K.dot(tf.diag(y[:,icat]),x))
			_mass_shape_correct_id = mass_split_by_prediction[:,icat]
			_mass_shape_correct_id = _mass_shape_correct_id / K.sum(_mass_shape_correct_id,axis=0)
			loss += LAMBDA*kullback_leibler_divergence(K.transpose(trainer.mass_histograms[icat]), _mass_shape_correct_id)
		return loss



	model_50_D2_25_D2_mass_control_0p3 = model_init('model_50_D2_25_D2_mass_control_0p3', input_dim, 2048, 100, [loss_multiclass_mass_control_0p3], 'adam')
	x = Dense(50, name = model_50_D2_25_D2_mass_control_0p3.name+'_layer_1', activation='relu')(model_50_D2_25_D2_mass_control_0p3.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_mass_control_0p3.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	out1 = Dense(n_categories , name = model_50_D2_25_D2_mass_control_0p3.name+'_output',  activation='softmax')(x)
	
	lambdaLayer = Lambda(lambda x: 0*x, name='lambda')(model_50_D2_25_D2_mass_control_0p3.inputs)
	def slicer(x):
	    return x[:,0:NBINS]    
	lambdaLayer = Lambda(slicer)(lambdaLayer)

	model_50_D2_25_D2_mass_control_0p3.outputs = Concatenate()([lambdaLayer, out1]) # order is important

	list_of_models.append(model_50_D2_25_D2_mass_control_0p3)



	def loss_multiclass_mass_control_0p4(y_in,x_in):
		LAMBDA = 0.4
		h = y_in[:,0:NBINS]
		y = y_in[:,NBINS:NBINS+n_categories] # order of categories like in category_labels
		x = x_in[:,NBINS:NBINS+n_categories]

		loss = categorical_crossentropy(y, x)  

		for icat in range(n_categories):
			mass_split_by_prediction = K.dot(K.transpose(h), K.dot(tf.diag(y[:,icat]),x))
			_mass_shape_correct_id = mass_split_by_prediction[:,icat]
			_mass_shape_correct_id = _mass_shape_correct_id / K.sum(_mass_shape_correct_id,axis=0)
			loss += LAMBDA*kullback_leibler_divergence(K.transpose(trainer.mass_histograms[icat]), _mass_shape_correct_id)
		return loss



	model_50_D2_25_D2_mass_control_0p4 = model_init('model_50_D2_25_D2_mass_control_0p4', input_dim, 2048, 100, [loss_multiclass_mass_control_0p4], 'adam')
	x = Dense(50, name = model_50_D2_25_D2_mass_control_0p4.name+'_layer_1', activation='relu')(model_50_D2_25_D2_mass_control_0p4.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_mass_control_0p4.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	out1 = Dense(n_categories , name = model_50_D2_25_D2_mass_control_0p4.name+'_output',  activation='softmax')(x)
	
	lambdaLayer = Lambda(lambda x: 0*x, name='lambda')(model_50_D2_25_D2_mass_control_0p4.inputs)
	def slicer(x):
	    return x[:,0:NBINS]    
	lambdaLayer = Lambda(slicer)(lambdaLayer)

	model_50_D2_25_D2_mass_control_0p4.outputs = Concatenate()([lambdaLayer, out1]) # order is important

	list_of_models.append(model_50_D2_25_D2_mass_control_0p4)



	def loss_multiclass_mass_control_0p5(y_in,x_in):
		LAMBDA = 0.5
		h = y_in[:,0:NBINS]
		y = y_in[:,NBINS:NBINS+n_categories] # order of categories like in category_labels
		x = x_in[:,NBINS:NBINS+n_categories]

		loss = categorical_crossentropy(y, x)  

		for icat in range(n_categories):
			mass_split_by_prediction = K.dot(K.transpose(h), K.dot(tf.diag(y[:,icat]),x))
			_mass_shape_correct_id = mass_split_by_prediction[:,icat]
			_mass_shape_correct_id = _mass_shape_correct_id / K.sum(_mass_shape_correct_id,axis=0)
			loss += LAMBDA*kullback_leibler_divergence(K.transpose(trainer.mass_histograms[icat]), _mass_shape_correct_id)
		return loss



	model_50_D2_25_D2_mass_control_0p5 = model_init('model_50_D2_25_D2_mass_control_0p5', input_dim, 2048, 100, [loss_multiclass_mass_control_0p5], 'adam')
	x = Dense(50, name = model_50_D2_25_D2_mass_control_0p5.name+'_layer_1', activation='relu')(model_50_D2_25_D2_mass_control_0p5.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_mass_control_0p5.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	out1 = Dense(n_categories , name = model_50_D2_25_D2_mass_control_0p5.name+'_output',  activation='softmax')(x)
	
	lambdaLayer = Lambda(lambda x: 0*x, name='lambda')(model_50_D2_25_D2_mass_control_0p5.inputs)
	def slicer(x):
	    return x[:,0:NBINS]    
	lambdaLayer = Lambda(slicer)(lambdaLayer)

	model_50_D2_25_D2_mass_control_0p5.outputs = Concatenate()([lambdaLayer, out1]) # order is important

	list_of_models.append(model_50_D2_25_D2_mass_control_0p5)


	def loss_multiclass_mass_control_1(y_in,x_in):
		LAMBDA = 1
		h = y_in[:,0:NBINS]
		y = y_in[:,NBINS:NBINS+n_categories] # order of categories like in category_labels
		x = x_in[:,NBINS:NBINS+n_categories]

		loss = categorical_crossentropy(y, x)  

		for icat in range(n_categories):
			mass_split_by_prediction = K.dot(K.transpose(h), K.dot(tf.diag(y[:,icat]),x))
			_mass_shape_correct_id = mass_split_by_prediction[:,icat]
			_mass_shape_correct_id = _mass_shape_correct_id / K.sum(_mass_shape_correct_id,axis=0)
			loss += LAMBDA*kullback_leibler_divergence(K.transpose(trainer.mass_histograms[icat]), _mass_shape_correct_id)
		return loss



	model_50_D2_25_D2_mass_control_1 = model_init('model_50_D2_25_D2_mass_control_1', input_dim, 2048, 100, [loss_multiclass_mass_control_1], 'adam')
	x = Dense(50, name = model_50_D2_25_D2_mass_control_1.name+'_layer_1', activation='relu')(model_50_D2_25_D2_mass_control_1.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_mass_control_1.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	out1 = Dense(n_categories , name = model_50_D2_25_D2_mass_control_1.name+'_output',  activation='softmax')(x)
	
	lambdaLayer = Lambda(lambda x: 0*x, name='lambda')(model_50_D2_25_D2_mass_control_1.inputs)
	def slicer(x):
	    return x[:,0:NBINS]    
	lambdaLayer = Lambda(slicer)(lambdaLayer)

	model_50_D2_25_D2_mass_control_1.outputs = Concatenate()([lambdaLayer, out1]) # order is important

	list_of_models.append(model_50_D2_25_D2_mass_control_1)



	def loss_multiclass_mass_control_2(y_in,x_in):
		LAMBDA = 2
		h = y_in[:,0:NBINS]
		y = y_in[:,NBINS:NBINS+n_categories] # order of categories like in category_labels
		x = x_in[:,NBINS:NBINS+n_categories]

		loss = categorical_crossentropy(y, x)  

		for icat in range(n_categories):
			mass_split_by_prediction = K.dot(K.transpose(h), K.dot(tf.diag(y[:,icat]),x))
			_mass_shape_correct_id = mass_split_by_prediction[:,icat]
			_mass_shape_correct_id = _mass_shape_correct_id / K.sum(_mass_shape_correct_id,axis=0)
			loss += LAMBDA*kullback_leibler_divergence(K.transpose(trainer.mass_histograms[icat]), _mass_shape_correct_id)
		return loss



	model_50_D2_25_D2_mass_control_2 = model_init('model_50_D2_25_D2_mass_control_2', input_dim, 2048, 100, [loss_multiclass_mass_control_2], 'adam')
	x = Dense(50, name = model_50_D2_25_D2_mass_control_2.name+'_layer_1', activation='relu')(model_50_D2_25_D2_mass_control_2.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_mass_control_2.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	out1 = Dense(n_categories , name = model_50_D2_25_D2_mass_control_2.name+'_output',  activation='softmax')(x)
	
	lambdaLayer = Lambda(lambda x: 0*x, name='lambda')(model_50_D2_25_D2_mass_control_2.inputs)
	def slicer(x):
	    return x[:,0:NBINS]    
	lambdaLayer = Lambda(slicer)(lambdaLayer)

	model_50_D2_25_D2_mass_control_2.outputs = Concatenate()([lambdaLayer, out1]) # order is important

	list_of_models.append(model_50_D2_25_D2_mass_control_2)


	def loss_multiclass_mass_control_3(y_in,x_in):
		LAMBDA = 3
		h = y_in[:,0:NBINS]
		y = y_in[:,NBINS:NBINS+n_categories] # order of categories like in category_labels
		x = x_in[:,NBINS:NBINS+n_categories]

		loss = categorical_crossentropy(y, x)  

		for icat in range(n_categories):
			mass_split_by_prediction = K.dot(K.transpose(h), K.dot(tf.diag(y[:,icat]),x))
			_mass_shape_correct_id = mass_split_by_prediction[:,icat]
			_mass_shape_correct_id = _mass_shape_correct_id / K.sum(_mass_shape_correct_id,axis=0)
			loss += LAMBDA*kullback_leibler_divergence(K.transpose(trainer.mass_histograms[icat]), _mass_shape_correct_id)
		return loss



	model_50_D2_25_D2_mass_control_3 = model_init('model_50_D2_25_D2_mass_control_3', input_dim, 2048, 100, [loss_multiclass_mass_control_3], 'adam')
	x = Dense(50, name = model_50_D2_25_D2_mass_control_3.name+'_layer_1', activation='relu')(model_50_D2_25_D2_mass_control_3.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_mass_control_3.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	out1 = Dense(n_categories , name = model_50_D2_25_D2_mass_control_3.name+'_output',  activation='softmax')(x)
	
	lambdaLayer = Lambda(lambda x: 0*x, name='lambda')(model_50_D2_25_D2_mass_control_3.inputs)
	def slicer(x):
	    return x[:,0:NBINS]    
	lambdaLayer = Lambda(slicer)(lambdaLayer)

	model_50_D2_25_D2_mass_control_3.outputs = Concatenate()([lambdaLayer, out1]) # order is important

	list_of_models.append(model_50_D2_25_D2_mass_control_3)


	def loss_multiclass_mass_control_5(y_in,x_in):
		LAMBDA = 5
		h = y_in[:,0:NBINS]
		y = y_in[:,NBINS:NBINS+n_categories] # order of categories like in category_labels
		x = x_in[:,NBINS:NBINS+n_categories]

		loss = categorical_crossentropy(y, x)  

		for icat in range(n_categories):
			mass_split_by_prediction = K.dot(K.transpose(h), K.dot(tf.diag(y[:,icat]),x))
			_mass_shape_correct_id = mass_split_by_prediction[:,icat]
			_mass_shape_correct_id = _mass_shape_correct_id / K.sum(_mass_shape_correct_id,axis=0)
			loss += LAMBDA*kullback_leibler_divergence(K.transpose(trainer.mass_histograms[icat]), _mass_shape_correct_id)
		return loss



	model_50_D2_25_D2_mass_control_5 = model_init('model_50_D2_25_D2_mass_control_5', input_dim, 2048, 100, [loss_multiclass_mass_control_5], 'adam')
	x = Dense(50, name = model_50_D2_25_D2_mass_control_5.name+'_layer_1', activation='relu')(model_50_D2_25_D2_mass_control_5.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_mass_control_5.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	out1 = Dense(n_categories , name = model_50_D2_25_D2_mass_control_5.name+'_output',  activation='softmax')(x)
	
	lambdaLayer = Lambda(lambda x: 0*x, name='lambda')(model_50_D2_25_D2_mass_control_5.inputs)
	def slicer(x):
	    return x[:,0:NBINS]    
	lambdaLayer = Lambda(slicer)(lambdaLayer)

	model_50_D2_25_D2_mass_control_5.outputs = Concatenate()([lambdaLayer, out1]) # order is important

	list_of_models.append(model_50_D2_25_D2_mass_control_5)




	def loss_cross_cat_mass_control_3(y_in,x_in):
		LAMBDA = 3
		h = y_in[:,0:NBINS]
		y = y_in[:,NBINS:NBINS+n_categories] # order of categories like in category_labels
		x = x_in[:,NBINS:NBINS+n_categories]

		loss = categorical_crossentropy(y, x)  

		for icat in range(n_categories):			# true category
			for jcat in range(n_categories):		# predicted category
				mass_split_by_prediction = K.dot(K.transpose(h), K.dot(tf.diag(y[:,icat]),x))
				mass_shape_in_jth_node = mass_split_by_prediction[:,jcat]
				mass_shape_in_jth_node = mass_shape_in_jth_node / K.sum(mass_shape_in_jth_node,axis=0)
				loss += LAMBDA*kullback_leibler_divergence(K.transpose(trainer.mass_histograms[icat]), mass_shape_in_jth_node)
		return loss



	model_50_D2_25_D2_cross_cat_mass_control_3 = model_init('model_50_D2_25_D2_cross_cat_mass_control_3', input_dim, 2048, 100, [loss_cross_cat_mass_control_3], 'adam')
	x = Dense(50, name = model_50_D2_25_D2_cross_cat_mass_control_3.name+'_layer_1', activation='relu')(model_50_D2_25_D2_cross_cat_mass_control_3.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_cross_cat_mass_control_3.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	out1 = Dense(n_categories , name = model_50_D2_25_D2_cross_cat_mass_control_3.name+'_output',  activation='softmax')(x)
	
	lambdaLayer = Lambda(lambda x: 0*x, name='lambda')(model_50_D2_25_D2_cross_cat_mass_control_3.inputs)
	def slicer(x):
	    return x[:,0:NBINS]    
	lambdaLayer = Lambda(slicer)(lambdaLayer)

	model_50_D2_25_D2_cross_cat_mass_control_3.outputs = Concatenate()([lambdaLayer, out1]) # order is important

	list_of_models.append(model_50_D2_25_D2_cross_cat_mass_control_3)



	def loss_mutual_mass_control_5(y_in,x_in):
		LAMBDA = 5
		h = y_in[:,0:NBINS]
		y = y_in[:,NBINS:NBINS+n_categories] # order of categories like in category_labels
		x = x_in[:,NBINS:NBINS+n_categories]

		loss = categorical_crossentropy(y, x)  

		for icat in range(n_categories):			
			for jcat in range(icat):		
				mass_split_by_prediction = K.dot(K.transpose(h), K.dot(tf.diag(y[:,icat]),x))
				mass_shape_in_ith_node = mass_split_by_prediction[:,icat]
				mass_shape_in_ith_node = mass_shape_in_ith_node / K.sum(mass_shape_in_ith_node,axis=0)
				mass_shape_in_jth_node = mass_split_by_prediction[:,jcat]
				mass_shape_in_jth_node = mass_shape_in_jth_node / K.sum(mass_shape_in_jth_node,axis=0)
				loss += LAMBDA*kullback_leibler_divergence(mass_shape_in_ith_node, mass_shape_in_jth_node)
		return loss



	model_50_D2_25_D2_mutual_mass_control_5 = model_init('model_50_D2_25_D2_mutual_mass_control_5', input_dim, 2048, 100, [loss_mutual_mass_control_5], 'adam')
	x = Dense(50, name = model_50_D2_25_D2_mutual_mass_control_5.name+'_layer_1', activation='relu')(model_50_D2_25_D2_mutual_mass_control_5.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_mutual_mass_control_5.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	out1 = Dense(n_categories , name = model_50_D2_25_D2_mutual_mass_control_5.name+'_output',  activation='softmax')(x)
	
	lambdaLayer = Lambda(lambda x: 0*x, name='lambda')(model_50_D2_25_D2_mutual_mass_control_5.inputs)
	def slicer(x):
	    return x[:,0:NBINS]    
	lambdaLayer = Lambda(slicer)(lambdaLayer)

	model_50_D2_25_D2_mutual_mass_control_5.outputs = Concatenate()([lambdaLayer, out1]) # order is important

	list_of_models.append(model_50_D2_25_D2_mutual_mass_control_5)


	def loss_mutual_mass_control_sym_5(y_in,x_in):
		LAMBDA = 5
		h = y_in[:,0:NBINS]
		y = y_in[:,NBINS:NBINS+n_categories] # order of categories like in category_labels
		x = x_in[:,NBINS:NBINS+n_categories]

		loss = categorical_crossentropy(y, x)  

		for icat in range(n_categories):			
			for jcat in range(icat):		
				mass_split_by_prediction = K.dot(K.transpose(h), K.dot(tf.diag(y[:,icat]),x))
				mass_shape_in_ith_node = mass_split_by_prediction[:,icat]
				mass_shape_in_ith_node = mass_shape_in_ith_node / K.sum(mass_shape_in_ith_node,axis=0)
				mass_shape_in_jth_node = mass_split_by_prediction[:,jcat]
				mass_shape_in_jth_node = mass_shape_in_jth_node / K.sum(mass_shape_in_jth_node,axis=0)
				loss += LAMBDA/2.0*kullback_leibler_divergence(mass_shape_in_ith_node, mass_shape_in_jth_node) +  LAMBDA/2.0*kullback_leibler_divergence(mass_shape_in_jth_node, mass_shape_in_ith_node)
		return loss



	model_50_D2_25_D2_mutual_mass_control_sym_5 = model_init('model_50_D2_25_D2_mutual_mass_control_sym_5', input_dim, 2048, 100, [loss_mutual_mass_control_sym_5], 'adam')
	x = Dense(50, name = model_50_D2_25_D2_mutual_mass_control_sym_5.name+'_layer_1', activation='relu')(model_50_D2_25_D2_mutual_mass_control_sym_5.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_mutual_mass_control_sym_5.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	out1 = Dense(n_categories , name = model_50_D2_25_D2_mutual_mass_control_sym_5.name+'_output',  activation='softmax')(x)
	
	lambdaLayer = Lambda(lambda x: 0*x, name='lambda')(model_50_D2_25_D2_mutual_mass_control_sym_5.inputs)
	def slicer(x):
	    return x[:,0:NBINS]    
	lambdaLayer = Lambda(slicer)(lambdaLayer)

	model_50_D2_25_D2_mutual_mass_control_sym_5.outputs = Concatenate()([lambdaLayer, out1]) # order is important

	list_of_models.append(model_50_D2_25_D2_mutual_mass_control_sym_5)



	def loss_mass_control_bkg_4(y_in,x_in):
		LAMBDA = 4
		h = y_in[:,0:NBINS]
		y = y_in[:,NBINS:NBINS+n_categories] # order of categories like in category_labels
		x = x_in[:,NBINS:NBINS+n_categories]

		loss = categorical_crossentropy(y, x)  

		for icat, true_cat in zip(range(n_categories), trainer.category_labels):			# true category
			if true_cat not in trainer.framework.bkg_categories:							# only control the bkg categories
				continue
			for jcat, pred_cat in zip(range(n_categories), trainer.category_labels):		# predicted category
				if pred_cat not in trainer.framework.bkg_categories:						# only control bkg output nodes
					continue
				mass_split_by_prediction = K.dot(K.transpose(h), K.dot(tf.diag(y[:,icat]),x))
				mass_shape_in_jth_node = mass_split_by_prediction[:,jcat]
				mass_shape_in_jth_node = mass_shape_in_jth_node / K.sum(mass_shape_in_jth_node,axis=0)
				loss += LAMBDA*kullback_leibler_divergence(K.transpose(trainer.mass_histograms[icat]), mass_shape_in_jth_node)
		return loss



	model_50_D2_25_D2_mass_control_bkg_4 = model_init('model_50_D2_25_D2_mass_control_bkg_4', input_dim, 2048, 100, [loss_mass_control_bkg_4], 'adam')
	x = Dense(50, name = model_50_D2_25_D2_mass_control_bkg_4.name+'_layer_1', activation='relu')(model_50_D2_25_D2_mass_control_bkg_4.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_mass_control_bkg_4.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	out1 = Dense(n_categories , name = model_50_D2_25_D2_mass_control_bkg_4.name+'_output',  activation='softmax')(x)
	
	lambdaLayer = Lambda(lambda x: 0*x, name='lambda')(model_50_D2_25_D2_mass_control_bkg_4.inputs)
	def slicer(x):
	    return x[:,0:NBINS]    
	lambdaLayer = Lambda(slicer)(lambdaLayer)

	model_50_D2_25_D2_mass_control_bkg_4.outputs = Concatenate()([lambdaLayer, out1]) # order is important

	list_of_models.append(model_50_D2_25_D2_mass_control_bkg_4)



	model_50_D2_25_D2_25_D2_mass_control_bkg_4 = model_init('model_50_D2_25_D2_25_D2_mass_control_bkg_4', input_dim, 2048, 100, [loss_mass_control_bkg_4], 'adam')
	x = Dense(50, name = model_50_D2_25_D2_25_D2_mass_control_bkg_4.name+'_layer_1', activation='relu')(model_50_D2_25_D2_25_D2_mass_control_bkg_4.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_25_D2_mass_control_bkg_4.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_25_D2_mass_control_bkg_4.name+'_layer_3', activation='relu')(x)
	x = Dropout(0.2)(x)
	out1 = Dense(n_categories , name = model_50_D2_25_D2_25_D2_mass_control_bkg_4.name+'_output',  activation='softmax')(x)
	
	lambdaLayer = Lambda(lambda x: 0*x, name='lambda')(model_50_D2_25_D2_25_D2_mass_control_bkg_4.inputs)
	def slicer(x):
	    return x[:,0:NBINS]    
	lambdaLayer = Lambda(slicer)(lambdaLayer)

	model_50_D2_25_D2_25_D2_mass_control_bkg_4.outputs = Concatenate()([lambdaLayer, out1]) # order is important

	list_of_models.append(model_50_D2_25_D2_25_D2_mass_control_bkg_4)





	def loss_mass_control_bkg_3p5(y_in,x_in):
		LAMBDA = 3.5
		h = y_in[:,0:NBINS]
		y = y_in[:,NBINS:NBINS+n_categories] # order of categories like in category_labels
		x = x_in[:,NBINS:NBINS+n_categories]

		loss = categorical_crossentropy(y, x)  

		for icat, true_cat in zip(range(n_categories), trainer.category_labels):			# true category
			if true_cat not in trainer.framework.bkg_categories:							# only control the bkg categories
				continue
			for jcat, pred_cat in zip(range(n_categories), trainer.category_labels):		# predicted category
				if pred_cat not in trainer.framework.bkg_categories:						# only control bkg output nodes
					continue
				mass_split_by_prediction = K.dot(K.transpose(h), K.dot(tf.diag(y[:,icat]),x))
				mass_shape_in_jth_node = mass_split_by_prediction[:,jcat]
				mass_shape_in_jth_node = mass_shape_in_jth_node / K.sum(mass_shape_in_jth_node,axis=0)
				loss += LAMBDA*kullback_leibler_divergence(K.transpose(trainer.mass_histograms[icat]), mass_shape_in_jth_node)
		return loss


	model_50_D2_25_D2_25_D2_mass_control_bkg_3p5 = model_init('model_50_D2_25_D2_25_D2_mass_control_bkg_3p5', input_dim, 2048, 100, [loss_mass_control_bkg_3p5], 'adam')
	x = Dense(50, name = model_50_D2_25_D2_25_D2_mass_control_bkg_3p5.name+'_layer_1', activation='relu')(model_50_D2_25_D2_25_D2_mass_control_bkg_3p5.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_25_D2_mass_control_bkg_3p5.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_25_D2_mass_control_bkg_3p5.name+'_layer_3', activation='relu')(x)
	x = Dropout(0.2)(x)
	out1 = Dense(n_categories , name = model_50_D2_25_D2_25_D2_mass_control_bkg_3p5.name+'_output',  activation='softmax')(x)
	
	lambdaLayer = Lambda(lambda x: 0*x, name='lambda')(model_50_D2_25_D2_25_D2_mass_control_bkg_3p5.inputs)
	def slicer(x):
	    return x[:,0:NBINS]    
	lambdaLayer = Lambda(slicer)(lambdaLayer)

	model_50_D2_25_D2_25_D2_mass_control_bkg_3p5.outputs = Concatenate()([lambdaLayer, out1]) # order is important

	list_of_models.append(model_50_D2_25_D2_25_D2_mass_control_bkg_3p5)



	def loss_mass_control_bkg_3(y_in,x_in):
		LAMBDA = 3
		h = y_in[:,0:NBINS]
		y = y_in[:,NBINS:NBINS+n_categories] # order of categories like in category_labels
		x = x_in[:,NBINS:NBINS+n_categories]

		loss = categorical_crossentropy(y, x)  

		for icat, true_cat in zip(range(n_categories), trainer.category_labels):			# true category
			if true_cat not in trainer.framework.bkg_categories:							# only control the bkg categories
				continue
			for jcat, pred_cat in zip(range(n_categories), trainer.category_labels):		# predicted category
				if pred_cat not in trainer.framework.bkg_categories:						# only control bkg output nodes
					continue
				mass_split_by_prediction = K.dot(K.transpose(h), K.dot(tf.diag(y[:,icat]),x))
				mass_shape_in_jth_node = mass_split_by_prediction[:,jcat]
				mass_shape_in_jth_node = mass_shape_in_jth_node / K.sum(mass_shape_in_jth_node,axis=0)
				loss += LAMBDA*kullback_leibler_divergence(K.transpose(trainer.mass_histograms[icat]), mass_shape_in_jth_node)
		return loss


	model_50_D2_25_D2_25_D2_mass_control_bkg_3 = model_init('model_50_D2_25_D2_25_D2_mass_control_bkg_3', input_dim, 2048, 100, [loss_mass_control_bkg_3], 'adam')
	x = Dense(50, name = model_50_D2_25_D2_25_D2_mass_control_bkg_3.name+'_layer_1', activation='relu')(model_50_D2_25_D2_25_D2_mass_control_bkg_3.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_25_D2_mass_control_bkg_3.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_25_D2_mass_control_bkg_3.name+'_layer_3', activation='relu')(x)
	x = Dropout(0.2)(x)
	out1 = Dense(n_categories , name = model_50_D2_25_D2_25_D2_mass_control_bkg_3.name+'_output',  activation='softmax')(x)
	
	lambdaLayer = Lambda(lambda x: 0*x, name='lambda')(model_50_D2_25_D2_25_D2_mass_control_bkg_3.inputs)
	def slicer(x):
	    return x[:,0:NBINS]    
	lambdaLayer = Lambda(slicer)(lambdaLayer)

	model_50_D2_25_D2_25_D2_mass_control_bkg_3.outputs = Concatenate()([lambdaLayer, out1]) # order is important

	list_of_models.append(model_50_D2_25_D2_25_D2_mass_control_bkg_3)



	def loss_mass_control_bkg_2p5(y_in,x_in):
		LAMBDA = 2.5
		h = y_in[:,0:NBINS]
		y = y_in[:,NBINS:NBINS+n_categories] # order of categories like in category_labels
		x = x_in[:,NBINS:NBINS+n_categories]

		loss = categorical_crossentropy(y, x)  

		for icat, true_cat in zip(range(n_categories), trainer.category_labels):			# true category
			if true_cat not in trainer.framework.bkg_categories:							# only control the bkg categories
				continue
			for jcat, pred_cat in zip(range(n_categories), trainer.category_labels):		# predicted category
				if pred_cat not in trainer.framework.bkg_categories:						# only control bkg output nodes
					continue
				mass_split_by_prediction = K.dot(K.transpose(h), K.dot(tf.diag(y[:,icat]),x))
				mass_shape_in_jth_node = mass_split_by_prediction[:,jcat]
				mass_shape_in_jth_node = mass_shape_in_jth_node / K.sum(mass_shape_in_jth_node,axis=0)
				loss += LAMBDA*kullback_leibler_divergence(K.transpose(trainer.mass_histograms[icat]), mass_shape_in_jth_node)
		return loss


	model_50_D2_25_D2_25_D2_mass_control_bkg_2p5 = model_init('model_50_D2_25_D2_25_D2_mass_control_bkg_2p5', input_dim, 2048, 100, [loss_mass_control_bkg_2p5], 'adam')
	x = Dense(50, name = model_50_D2_25_D2_25_D2_mass_control_bkg_2p5.name+'_layer_1', activation='relu')(model_50_D2_25_D2_25_D2_mass_control_bkg_2p5.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_25_D2_mass_control_bkg_2p5.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_25_D2_mass_control_bkg_2p5.name+'_layer_3', activation='relu')(x)
	x = Dropout(0.2)(x)
	out1 = Dense(n_categories , name = model_50_D2_25_D2_25_D2_mass_control_bkg_2p5.name+'_output',  activation='softmax')(x)
	
	lambdaLayer = Lambda(lambda x: 0*x, name='lambda')(model_50_D2_25_D2_25_D2_mass_control_bkg_2p5.inputs)
	def slicer(x):
	    return x[:,0:NBINS]    
	lambdaLayer = Lambda(slicer)(lambdaLayer)

	model_50_D2_25_D2_25_D2_mass_control_bkg_2p5.outputs = Concatenate()([lambdaLayer, out1]) # order is important

	list_of_models.append(model_50_D2_25_D2_25_D2_mass_control_bkg_2p5)





	def loss_mass_control_bkg_2(y_in,x_in):
		LAMBDA = 2
		h = y_in[:,0:NBINS]
		y = y_in[:,NBINS:NBINS+n_categories] # order of categories like in category_labels
		x = x_in[:,NBINS:NBINS+n_categories]

		loss = categorical_crossentropy(y, x)  

		for icat, true_cat in zip(range(n_categories), trainer.category_labels):			# true category
			if true_cat not in trainer.framework.bkg_categories:							# only control the bkg categories
				continue
			for jcat, pred_cat in zip(range(n_categories), trainer.category_labels):		# predicted category
				if pred_cat not in trainer.framework.bkg_categories:						# only control bkg output nodes
					continue
				mass_split_by_prediction = K.dot(K.transpose(h), K.dot(tf.diag(y[:,icat]),x))
				mass_shape_in_jth_node = mass_split_by_prediction[:,jcat]
				mass_shape_in_jth_node = mass_shape_in_jth_node / K.sum(mass_shape_in_jth_node,axis=0)
				loss += LAMBDA*kullback_leibler_divergence(K.transpose(trainer.mass_histograms[icat]), mass_shape_in_jth_node)
		return loss


	model_50_D2_25_D2_25_D2_mass_control_bkg_2 = model_init('model_50_D2_25_D2_25_D2_mass_control_bkg_2', input_dim, 2048, 100, [loss_mass_control_bkg_2], 'adam')
	x = Dense(50, name = model_50_D2_25_D2_25_D2_mass_control_bkg_2.name+'_layer_1', activation='relu')(model_50_D2_25_D2_25_D2_mass_control_bkg_2.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_25_D2_mass_control_bkg_2.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_25_D2_mass_control_bkg_2.name+'_layer_3', activation='relu')(x)
	x = Dropout(0.2)(x)
	out1 = Dense(n_categories , name = model_50_D2_25_D2_25_D2_mass_control_bkg_2.name+'_output',  activation='softmax')(x)
	
	lambdaLayer = Lambda(lambda x: 0*x, name='lambda')(model_50_D2_25_D2_25_D2_mass_control_bkg_2.inputs)
	def slicer(x):
	    return x[:,0:NBINS]    
	lambdaLayer = Lambda(slicer)(lambdaLayer)

	model_50_D2_25_D2_25_D2_mass_control_bkg_2.outputs = Concatenate()([lambdaLayer, out1]) # order is important

	list_of_models.append(model_50_D2_25_D2_25_D2_mass_control_bkg_2)




	def loss_mass_control_bkg_1p5(y_in,x_in):
		LAMBDA = 1.5
		h = y_in[:,0:NBINS]
		y = y_in[:,NBINS:NBINS+n_categories] # order of categories like in category_labels
		x = x_in[:,NBINS:NBINS+n_categories]

		loss = categorical_crossentropy(y, x)  

		for icat, true_cat in zip(range(n_categories), trainer.category_labels):			# true category
			if true_cat not in trainer.framework.bkg_categories:							# only control the bkg categories
				continue
			for jcat, pred_cat in zip(range(n_categories), trainer.category_labels):		# predicted category
				if pred_cat not in trainer.framework.bkg_categories:						# only control bkg output nodes
					continue
				mass_split_by_prediction = K.dot(K.transpose(h), K.dot(tf.diag(y[:,icat]),x))
				mass_shape_in_jth_node = mass_split_by_prediction[:,jcat]
				mass_shape_in_jth_node = mass_shape_in_jth_node / K.sum(mass_shape_in_jth_node,axis=0)
				loss += LAMBDA*kullback_leibler_divergence(K.transpose(trainer.mass_histograms[icat]), mass_shape_in_jth_node)
		return loss


	model_50_D2_25_D2_25_D2_mass_control_bkg_1p5 = model_init('model_50_D2_25_D2_25_D2_mass_control_bkg_1p5', input_dim, 2048, 100, [loss_mass_control_bkg_1p5], 'adam')
	x = Dense(50, name = model_50_D2_25_D2_25_D2_mass_control_bkg_1p5.name+'_layer_1', activation='relu')(model_50_D2_25_D2_25_D2_mass_control_bkg_1p5.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_25_D2_mass_control_bkg_1p5.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_25_D2_mass_control_bkg_1p5.name+'_layer_3', activation='relu')(x)
	x = Dropout(0.2)(x)
	out1 = Dense(n_categories , name = model_50_D2_25_D2_25_D2_mass_control_bkg_1p5.name+'_output',  activation='softmax')(x)
	
	lambdaLayer = Lambda(lambda x: 0*x, name='lambda')(model_50_D2_25_D2_25_D2_mass_control_bkg_1p5.inputs)
	def slicer(x):
	    return x[:,0:NBINS]    
	lambdaLayer = Lambda(slicer)(lambdaLayer)

	model_50_D2_25_D2_25_D2_mass_control_bkg_1p5.outputs = Concatenate()([lambdaLayer, out1]) # order is important

	list_of_models.append(model_50_D2_25_D2_25_D2_mass_control_bkg_1p5)




	def loss_mass_control_bkg_1(y_in,x_in):
		LAMBDA = 1
		h = y_in[:,0:NBINS]
		y = y_in[:,NBINS:NBINS+n_categories] # order of categories like in category_labels
		x = x_in[:,NBINS:NBINS+n_categories]

		loss = categorical_crossentropy(y, x)  

		for icat, true_cat in zip(range(n_categories), trainer.category_labels):			# true category
			if true_cat not in trainer.framework.bkg_categories:							# only control the bkg categories
				continue
			for jcat, pred_cat in zip(range(n_categories), trainer.category_labels):		# predicted category
				if pred_cat not in trainer.framework.bkg_categories:						# only control bkg output nodes
					continue
				mass_split_by_prediction = K.dot(K.transpose(h), K.dot(tf.diag(y[:,icat]),x))
				mass_shape_in_jth_node = mass_split_by_prediction[:,jcat]
				mass_shape_in_jth_node = mass_shape_in_jth_node / K.sum(mass_shape_in_jth_node,axis=0)
				loss += LAMBDA*kullback_leibler_divergence(K.transpose(trainer.mass_histograms[icat]), mass_shape_in_jth_node)
		return loss


	model_50_D2_25_D2_25_D2_mass_control_bkg_1 = model_init('model_50_D2_25_D2_25_D2_mass_control_bkg_1', input_dim, 2048, 100, [loss_mass_control_bkg_1], 'adam')
	x = Dense(50, name = model_50_D2_25_D2_25_D2_mass_control_bkg_1.name+'_layer_1', activation='relu')(model_50_D2_25_D2_25_D2_mass_control_bkg_1.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_25_D2_mass_control_bkg_1.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_25_D2_mass_control_bkg_1.name+'_layer_3', activation='relu')(x)
	x = Dropout(0.2)(x)
	out1 = Dense(n_categories , name = model_50_D2_25_D2_25_D2_mass_control_bkg_1.name+'_output',  activation='softmax')(x)
	
	lambdaLayer = Lambda(lambda x: 0*x, name='lambda')(model_50_D2_25_D2_25_D2_mass_control_bkg_1.inputs)
	def slicer(x):
	    return x[:,0:NBINS]    
	lambdaLayer = Lambda(slicer)(lambdaLayer)

	model_50_D2_25_D2_25_D2_mass_control_bkg_1.outputs = Concatenate()([lambdaLayer, out1]) # order is important

	list_of_models.append(model_50_D2_25_D2_25_D2_mass_control_bkg_1)



	def loss_mass_control_bkg_0p5(y_in,x_in):
		LAMBDA = 0.5
		h = y_in[:,0:NBINS]
		y = y_in[:,NBINS:NBINS+n_categories] # order of categories like in category_labels
		x = x_in[:,NBINS:NBINS+n_categories]

		loss = categorical_crossentropy(y, x)  

		for icat, true_cat in zip(range(n_categories), trainer.category_labels):			# true category
			if true_cat not in trainer.framework.bkg_categories:							# only control the bkg categories
				continue
			for jcat, pred_cat in zip(range(n_categories), trainer.category_labels):		# predicted category
				if pred_cat not in trainer.framework.bkg_categories:						# only control bkg output nodes
					continue
				mass_split_by_prediction = K.dot(K.transpose(h), K.dot(tf.diag(y[:,icat]),x))
				mass_shape_in_jth_node = mass_split_by_prediction[:,jcat]
				mass_shape_in_jth_node = mass_shape_in_jth_node / K.sum(mass_shape_in_jth_node,axis=0)
				loss += LAMBDA*kullback_leibler_divergence(K.transpose(trainer.mass_histograms[icat]), mass_shape_in_jth_node)
		return loss


	model_50_D2_25_D2_25_D2_mass_control_bkg_0p5 = model_init('model_50_D2_25_D2_25_D2_mass_control_bkg_0p5', input_dim, 2048, 100, [loss_mass_control_bkg_0p5], 'adam')
	x = Dense(50, name = model_50_D2_25_D2_25_D2_mass_control_bkg_0p5.name+'_layer_1', activation='relu')(model_50_D2_25_D2_25_D2_mass_control_bkg_0p5.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_25_D2_mass_control_bkg_0p5.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_25_D2_mass_control_bkg_0p5.name+'_layer_3', activation='relu')(x)
	x = Dropout(0.2)(x)
	out1 = Dense(n_categories , name = model_50_D2_25_D2_25_D2_mass_control_bkg_0p5.name+'_output',  activation='softmax')(x)
	
	lambdaLayer = Lambda(lambda x: 0*x, name='lambda')(model_50_D2_25_D2_25_D2_mass_control_bkg_0p5.inputs)
	def slicer(x):
	    return x[:,0:NBINS]    
	lambdaLayer = Lambda(slicer)(lambdaLayer)

	model_50_D2_25_D2_25_D2_mass_control_bkg_0p5.outputs = Concatenate()([lambdaLayer, out1]) # order is important

	list_of_models.append(model_50_D2_25_D2_25_D2_mass_control_bkg_0p5)



	def loss_kldiv_binary_wgtd_1(y_in,x_in):
		LAMBDA = 1
		h = y_in[:,0:NBINS]
		y = y_in[:,NBINS:NBINS+n_categories] # truth: order of categories like in category_labels
		x = x_in[:,NBINS:NBINS+n_categories] # prediction


		loss = categorical_crossentropy(y, x)  
		for icat, category in zip(range(n_categories), trainer.category_labels):
			mass_split_by_prediction = K.dot(K.transpose(h), K.dot(tf.diag(y[:,icat]),x))
			mass_bkg = K.dot(mass_split_by_prediction, tf.diag(trainer.bkg_mask)) # remove signal mass histograms
			bkg_shape = K.dot(mass_bkg, K.transpose(trainer.category_wgts))
			true_bkg_shape = trainer.bkg_histogram
			loss += LAMBDA*kullback_leibler_divergence()

		return loss



	model_50_D2_25_D2_kldiv_binary_wgtd_1 = model_init('model_50_D2_25_D2_kldiv_binary_wgtd_1', input_dim, 2048, 100, [loss_kldiv_binary_wgtd_1], 'adam')
	x = Dense(50, name = model_50_D2_25_D2_kldiv_binary_wgtd_1.name+'_layer_1', activation='relu')(model_50_D2_25_D2_kldiv_binary_wgtd_1.inputs)
	x = Dropout(0.2)(x)
	x = Dense(25, name = model_50_D2_25_D2_kldiv_binary_wgtd_1.name+'_layer_2', activation='relu')(x)
	x = Dropout(0.2)(x)
	out1 = Dense(n_categories , name = model_50_D2_25_D2_kldiv_binary_wgtd_1.name+'_output',  activation='softmax')(x)
	
	lambdaLayer = Lambda(lambda x: 0*x, name='lambda')(model_50_D2_25_D2_kldiv_binary_wgtd_1.inputs)
	def slicer(x):
	    return x[:,0:NBINS]    
	lambdaLayer = Lambda(slicer)(lambdaLayer)

	model_50_D2_25_D2_kldiv_binary_wgtd_1.outputs = Concatenate()([lambdaLayer, out1]) # order is important

	list_of_models.append(model_50_D2_25_D2_kldiv_binary_wgtd_1)


	return list_of_models