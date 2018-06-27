#!/usr/bin/env python
import os, sys
sys.path.append( os.path.dirname(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) ))
import numpy as np
import pandas

from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from locations import *

from convert import files, GetDataFrame
from src.setup import writer

from helpers import SaveToHDF, Scale

from keras_models import GetListOfModels#list_of_models

def Train():

	writer.Write("-"*60)
	writer.Write("\nUsing package: Keras\n")

	df = GetDataFrame()
	lables = list(df.drop(['sample_weight', 'category'], axis=1))

	df_train, df_test = train_test_split(df,test_size=0.2, random_state=7, shuffle = True)

	SaveToHDF(df_train, df_test, 'input')

	df_train_scaled, df_test_scaled = Scale(df_train, df_test, lables)

	SaveToHDF(df_train_scaled, df_test_scaled, 'scaled')

	early_stopping = EarlyStopping(monitor='val_loss', patience=10)
	list_of_models = GetListOfModels(df.shape[1]-2) #the argument is for the input dimensions


	writer.Write("\nModels:\n")

	for obj in list_of_models:
		writer.Write("%s, batch_size=%i, %s epochs, loss=%s, optimizer=%s\n"%(obj.name,obj.batchSize,obj.epochs,obj.loss,obj.optimizer))

		obj.CompileModel()
		tensorboard = TensorBoard(log_dir=logsDir+obj.name)
		model_checkpoint = ModelCheckpoint(modelsDir+obj.name+'_trained_lwstValLoss.h5', monitor='val_loss', 
                                   verbose=0, save_best_only=True, 
                                   save_weights_only=False, mode='auto', 
                                   period=1)

		history = obj.model.fit(			
											df_train_scaled[lables].values,
                    						df_train_scaled['category'].values,
                    						epochs=obj.epochs, 
                    						batch_size=obj.batchSize, 
                    						# sample_weight = weights_train.flatten(),			
                    						# sample_weight = df_train['sample_weight'].values,
                    						verbose=0,
                    						callbacks=[
                    							# early_stopping, 
                    							model_checkpoint,
                    							tensorboard
                    							], 
                    						validation_split=0.25,
                    						steps_per_epoch = None,
                    						shuffle=True)
	
		df_train_scaled["prediction_"+obj.name] = obj.model.predict(df_train_scaled[lables].values)
		df_test_scaled["prediction_"+obj.name] = obj.model.predict(df_test_scaled[lables].values)

		df_history = pandas.DataFrame(history.history)
		df_history.to_hdf('%shistory.hdf5'%outputDir, obj.name)


	SaveToHDF(df_train_scaled, df_test_scaled, 'scaled_w_predictions')
	

