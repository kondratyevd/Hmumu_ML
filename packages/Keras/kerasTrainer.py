import os, sys, errno
import pandas
import uproot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras_models import GetListOfModels
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

class KerasTrainer(object):
	def __init__(self, framework, package):
		self.framework = framework
		self.package = package

	def __enter__(self):
		self.df = pandas.DataFrame()

		return self

	def __exit__(self, *args):
		del self.df	


	def convert_to_pandas(self):
		for file in self.framework.file_list_s + self.framework.file_list_b:
			with uproot.open(file.path) as f: 
				uproot_tree = f[self.framework.treePath]
	
				single_file_df = pandas.DataFrame()
		
				for var in self.framework.variable_list:
					up_var = uproot_tree[var.name].array()
		
					if var.isMultiDim:
						
						# splitting the multidimensional input variables so each column corresponds to a one-dimensional variable
						# Only <itemsAdded> objects are kept
		
						single_var_df = pandas.DataFrame(data = up_var.tolist())
						single_var_df.drop(single_var_df.columns[var.itemsAdded:],axis=1,inplace=True)
						single_var_df.columns = [var.name+"[%i]"%i for i in range(var.itemsAdded)]
		
						single_file_df = pandas.concat([single_file_df, single_var_df], axis=1)
		
					else:
						single_file_df[var.name] = up_var
				
				single_file_df['sample_weight'] = file.weight
				if file in self.framework.file_list_s:
					single_file_df['category'] = 1
				else:
					single_file_df['category'] = 0

				self.df = pandas.concat([self.df,single_file_df])
	
		self.df.dropna(axis=0, how='any', inplace=True)
		self.lables = list(self.df.drop(['sample_weight', 'category'], axis=1))
		self.df_train, self.df_test = train_test_split(self.df,test_size=0.2, random_state=7, shuffle = True)
		# self.save_to_hdf(self.df_train, self.df_test, 'input')

	def train_models(self):
		self.df_train_scaled, self.df_test_scaled = self.scale(self.df_train, self.df_test, self.lables)
		# self.save_to_hdf(self.df_train_scaled, self.df_test_scaled, 'scaled')
		
		self.list_of_models = GetListOfModels(self.df.shape[1]-2) #the argument is for the input dimensions
		for obj in self.list_of_models:
			
			obj.CompileModel(self.package.dirs['modelDir'])
			# early_stopping = EarlyStopping(monitor='val_loss', patience=10)
			tensorboard = TensorBoard(log_dir=self.package.dirs['logDir']+obj.name)
			model_checkpoint = ModelCheckpoint(self.package.dirs['modelDir']+obj.name+'_trained_lwstValLoss.h5', monitor='val_loss', 
       		                           verbose=0, save_best_only=True, 
       		                           save_weights_only=False, mode='auto', 
       		                           period=1)

			history = obj.model.fit(			
									self.df_train_scaled[self.lables].values,
       		            			self.df_train_scaled['category'].values,
       		            			epochs=obj.epochs, 
       		            			batch_size=obj.batchSize, 
       		            			# sample_weight = weights_train.flatten(),			
       		            			# sample_weight = self.df_train['sample_weight'].values,
       		            			verbose=1,
       		            			callbacks=[
       		            				# early_stopping, 
       		            				model_checkpoint,
       		            				tensorboard
       		            				], 
       		            			validation_split=0.25,
       		            			steps_per_epoch = None,
       		            			shuffle=True)
	
			self.df_train_scaled["prediction_"+obj.name] = obj.model.predict(self.df_train_scaled[self.lables].values)
			self.df_test_scaled["prediction_"+obj.name] = obj.model.predict(self.df_test_scaled[self.lables].values)
			# self.save_to_hdf(self.df_train_scaled, self.df_test_scaled, 'scaled_w_predictions')
			# print self.df_train_scaled
			self.df_history = pandas.DataFrame(history.history)
			self.df_history.to_hdf('%shistory.hdf5'%self.package.mainDir, obj.name)

	def scale(self, train, test, lables):
		data_train = train.loc[:,lables]
		data_test = test.loc[:,lables]
		scaler = StandardScaler().fit(data_train.values)
		data_train = scaler.transform(data_train.values)
		data_test = scaler.transform(data_test.values)	
		train[lables] = data_train
		test[lables] = data_test
		return train, test

	def save_to_hdf(self, train, test, filename):
		train.to_hdf('%s%s.hdf5'%(self.package.dirs['dataDir'],filename), 'train')
		test.to_hdf('%s%s.hdf5'%(self.package.dirs['dataDir'], filename), 'test')


