import matplotlib
matplotlib.use('Agg')
import ROOT
import os, sys, errno
import pandas
import uproot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from keras_models import GetListOfModels
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt

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
		for file in self.framework.dir_list_s + self.framework.dir_list_b:

			for filename in os.listdir(file.path):
			    if filename.endswith(".root"): 
			        # print(os.path.join(directory, filename))

				with uproot.open(file.path+filename) as f: 
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
					if file in self.framework.dir_list_s:
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
		self.df = shuffle(self.df)
		self.df_train, self.df_test = train_test_split(self.df,test_size=0.2, random_state=7)

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
       		            			#steps_per_epoch = None,
       		            			shuffle=True)
	

			self.df_train_scaled["predict_s_"+obj.name] =  obj.model.predict(self.df_train_scaled[self.lables].values)[:,0]
			self.df_train_scaled["predict_b_"+obj.name] =  obj.model.predict(self.df_train_scaled[self.lables].values)[:,1]

			self.df_test_scaled["predict_s_"+obj.name] =  obj.model.predict(self.df_test_scaled[self.lables].values)[:,0]
			self.df_test_scaled["predict_b_"+obj.name] =  obj.model.predict(self.df_test_scaled[self.lables].values)[:,1]

			# self.save_to_hdf(self.df_train_scaled, self.df_test_scaled, 'scaled_w_predictions')
			# print self.df_train_scaled
			self.df_history = pandas.DataFrame(history.history)
			# self.df_history.to_hdf('%shistory.hdf5'%self.package.mainDir, obj.name)
			self.plot_history(history.history)
			self.plot_ROC("train", self.df_train_scaled.loc[:,['signal', 'background']], self.df_train_scaled.loc[:,["predict_s_"+obj.name, "predict_b_"+obj.name]])
			self.plot_ROC("test", self.df_test_scaled.loc[:,['signal', 'background']], self.df_test_scaled.loc[:,["predict_s_"+obj.name, "predict_b_"+obj.name]])
			self.plot_score("train", self.df_test_scaled.loc[:,['signal', 'background']], self.df_test_scaled.loc[:,["predict_s_"+obj.name, "predict_b_"+obj.name]])
			self.plot_score("test", self.df_test_scaled.loc[:,['signal', 'background']], self.df_test_scaled.loc[:,["predict_s_"+obj.name, "predict_b_"+obj.name]])


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


	def plot_ROC(self, output_name, category_df, prediction_df):
		print "Making ROC: "+output_name

		df = pandas.concat([category_df, prediction_df], axis=1)	# [s, b, s_pred, b_pred]
		# print df

		sig = category_df.iloc[:,0]
		bkg = category_df.iloc[:,1]
		sig_predict = prediction_df.iloc[:,0]
		bkg_predict = prediction_df.iloc[:,1]

		roc = ROOT.TGraph()
		roc.GetXaxis().SetTitle("Signal eff.")
		roc.GetYaxis().SetTitle("Background rej.")
		for i in range(200):
			cut = i / 100.0
			score = df.iloc[:,2] + (1 - df.iloc[:,3]) # s_pred + (1 - b_pred)
			sig_eff = float(df.loc[  (df.iloc[:,0]==1) & (score > cut )  ].shape[0]) / df.loc[df.iloc[:,0]==1].shape[0]
			bkg_rej = float(df.loc[  (df.iloc[:,1]==1) & (score < cut )  ].shape[0]) / df.loc[df.iloc[:,1]==1].shape[0]
			roc.SetPoint(i, sig_eff, bkg_rej)
		canv = ROOT.TCanvas("canv", "canv", 800, 800)
		canv.cd()
		roc.Draw("apl")
		canv.Print(self.package.mainDir+output_name+"_roc.png")
		canv.SaveAs(self.package.mainDir+output_name+"_roc.root")
		canv.Close()

	def plot_history(self, history):
		# summarize history for accuracy
		plt.plot(history['acc'])
		plt.plot(history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig(self.package.mainDir+"acc.png")
		plt.clf()
		print "Accuracy plot saved as "+self.package.mainDir+"acc.png"
		# summarize history for loss
		plt.plot(history['loss'])
		plt.plot(history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig(self.package.mainDir+"loss.png")
		print "Loss plot saved as "+self.package.mainDir+"loss.png"		

	def plot_score(self, output_name, category_df, prediction_df):
		df = pandas.concat([category_df, prediction_df], axis=1)	# [s, b, s_pred, b_pred]
		print df

		sig = category_df.iloc[:,0]
		bkg = category_df.iloc[:,1]
		sig_predict = prediction_df.iloc[:,0]
		bkg_predict = prediction_df.iloc[:,1]

		hist_s = ROOT.TH1D()
		hist_s.SetLineColor(ROOT.kRed)
		hist_b = ROOT.TH1D()
		hist_b.SetLineColor(ROOT.kBlue)


		for index, row in df.iterrows():
			if row[0]==1:
				hist_s.Fill(row[2])
			elif row[1]==1:
				hist_b.Fill(row[3])

		hist_s.Scale(1/hist_s.Integral())
		hist_b.Scale(1/hist_b.Integral())

		canv = ROOT.TCanvas("canv1", "canv1", 800, 800)
		canv.cd()
		hist_s.Draw("hist")
		hist_b.Draw("histsame")
		canv.Print(self.package.mainDir+output_name+"_score.png")
		canv.Close()


