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

					if  "met.pt" in var.name:	# quick fix for met
						up_var =  uproot_tree["met"]["pt"].array()
					else:
						up_var = uproot_tree[var.name].array()
					if var.isMultiDim:
						
						# splitting the multidimensional input variables so each column corresponds to a one-dimensional variable
						# Only <itemsAdded> objects are kept
		
						single_var_df = pandas.DataFrame(data = up_var.tolist())
						single_var_df.drop(single_var_df.columns[var.itemsAdded:],axis=1,inplace=True)
						single_var_df.columns = [var.name+"[%i]"%i for i in range(var.itemsAdded)]
						single_file_df = pandas.concat([single_file_df, single_var_df], axis=1)
						single_file_df.fillna(var.replacement, axis=0, inplace=True) # if there are not enough jets
		
					else:
						single_file_df[var.name] = up_var
						# single_file_df[var.name].fillna(var.replacement, axis=0, inplace=True)


				single_file_df['sample_weight'] = 1#file.weight
				if file in self.framework.file_list_s:
					single_file_df['signal'] = 1
					single_file_df['background'] = 0
				else:
					single_file_df['signal'] = 0
					single_file_df['background'] = 1
					# single_file_df = single_file_df.iloc[0:500]

				self.df = pandas.concat([self.df,single_file_df])
			
		# self.df.dropna(axis=0, how='any', inplace=True)
		print self.df
		self.lables = list(self.df.drop(['sample_weight', 'signal', 'background'], axis=1))
		self.df_train, self.df_test = train_test_split(self.df,test_size=0.2, random_state=7, shuffle = True)

		# print self.df_train
		# self.save_to_hdf(self.df_train, self.df_test, 'input')

	def train_models(self):
		self.df_train_scaled, self.df_test_scaled = self.scale(self.df_train, self.df_test, self.lables)
		# self.save_to_hdf(self.df_train_scaled, self.df_test_scaled, 'scaled')
		
		self.list_of_models = GetListOfModels(self.df.shape[1]-3) #the argument is for the input dimensions
		for obj in self.list_of_models:
			if obj.name not in self.framework.method_list:
				continue
			obj.CompileModel(self.package.dirs['modelDir'])
			# early_stopping = EarlyStopping(monitor='val_loss', patience=10)
			# tensorboard = TensorBoard(log_dir=self.package.dirs['logDir']+obj.name)
			# model_checkpoint = ModelCheckpoint(self.package.dirs['modelDir']+obj.name+'_trained_lwstValLoss.h5', monitor='val_loss', 
   #     		                           verbose=0, save_best_only=True, 
   #     		                           save_weights_only=False, mode='auto', 
   #     		                           period=1)

			history = obj.model.fit(			
									self.df_train_scaled[self.lables].values,
       		            			# self.df_train_scaled['category'].values,
       		            			self.df_train_scaled.loc[:,['signal', 'background']].values,
       		            			epochs=obj.epochs, 
       		            			batch_size=obj.batchSize, 
       		            			# sample_weight = weights_train.flatten(),			
       		            			# sample_weight = self.df_train['sample_weight'].values,
       		            			verbose=1,
       		            			# callbacks=[
       		            				# early_stopping, 
       		            				# model_checkpoint,
       		            				# tensorboard
       		            				# ], 
       		            			validation_split=0.25,
       		            			steps_per_epoch = None,
       		            			shuffle=True)
	

			self.df_train_scaled["predict_s_"+obj.name] =  obj.model.predict(self.df_train_scaled[self.lables].values)[:,0]
			self.df_train_scaled["predict_b_"+obj.name] =  obj.model.predict(self.df_train_scaled[self.lables].values)[:,1]

			self.df_test_scaled["predict_s_"+obj.name] =  obj.model.predict(self.df_test_scaled[self.lables].values)[:,0]
			self.df_test_scaled["predict_b_"+obj.name] =  obj.model.predict(self.df_test_scaled[self.lables].values)[:,1]

			self.save_to_hdf(self.df_train_scaled, self.df_test_scaled, 'scaled_w_predictions')
			# print self.df_train_scaled
			self.df_history = pandas.DataFrame(history.history)
			self.df_history.to_hdf('%shistory.hdf5'%self.package.mainDir, obj.name)

			self.make_ROC(self.df_train_scaled.loc[:,['signal', 'background']], self.df_train_scaled.loc[:,["predict_s_"+obj.name, "predict_b_"+obj.name]])

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


	def make_ROC(self, category_df, prediction_df):
		import ROOT

		print "Making ROC"

		df = pandas.concat([category_df, prediction_df], axis=1)	# [s, b, s_pred, b_pred]
		print df

		sig = category_df.iloc[:,0]
		bkg = category_df.iloc[:,1]
		sig_predict = prediction_df.iloc[:,0]
		bkg_predict = prediction_df.iloc[:,0]

		roc = ROOT.TGraph()
		roc.GetXaxis().SetTitle("Signal eff.")
		roc.GetYaxis().SetTitle("Background rej.")
		for i in range(200):
			cut = i / 100.0
			sig_eff = float(df.loc[  (df.iloc[:,0]==1) & (df.iloc[:,2] + (1 - df.iloc[:,3]) > cut )  ].shape[0]) / df.loc[df.iloc[:,0]==1].shape[0]
			bkg_rej = float(df.loc[  (df.iloc[:,1]==1) & (df.iloc[:,2] + (1 - df.iloc[:,3]) < cut )  ].shape[0]) / df.loc[df.iloc[:,1]==1].shape[0]
			roc.SetPoint(i, sig_eff, bkg_rej)
			print "%f:	%f 	%f"%(cut, sig_eff, bkg_rej)
		canv = ROOT.TCanvas("canv", "canv", 800, 800)
		canv.cd()
		roc.Draw("apl")
		canv.SaveAs("roc.root")



