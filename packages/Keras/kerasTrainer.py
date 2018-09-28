import matplotlib
matplotlib.use('Agg')
import ROOT
import os, sys, errno
import math
import pandas
import numpy
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
		self.sum_weight_s = 1
		self.sum_weight_b = 1


	def __enter__(self):
		self.df = pandas.DataFrame()
		return self


	def __exit__(self, *args):
		del self.df	





	def convert_to_pandas(self):
		self.calc_sum_wgts()
		for file in self.framework.dir_list_s + self.framework.dir_list_b:
			for filename in os.listdir(file.path):
			    if filename.endswith(".root"): 

				with uproot.open(file.path+filename) as f: 
					uproot_tree = f[self.framework.treePath]
					single_file_df = pandas.DataFrame()
					spect_labels = []
					for var in self.framework.variable_list + self.framework.spectator_list:
						# print "Adding variable:", var.name
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
							if var in self.framework.spectator_list:
								spect_labels.extend(single_var_df.columns)
							single_file_df = pandas.concat([single_file_df, single_var_df], axis=1)
							single_file_df.fillna(var.replacement, axis=0, inplace=True) # if there are not enough jets
			
						else:
							single_file_df[var.name] = up_var
							if var in self.framework.spectator_list:
								spect_labels.append(var.name)

					
					if self.framework.year is "2016":
						SF = (0.5*(single_file_df['IsoMu_SF_3'] + single_file_df['IsoMu_SF_4'])*0.5*(single_file_df['MuID_SF_3'] + single_file_df['MuID_SF_4'])*0.5*(single_file_df['MuIso_SF_3'] + single_file_df['MuIso_SF_4']))
					elif self.framework.year is "2017":
						SF = single_file_df['IsoMu_SF_3'] * single_file_df['MuID_SF_3'] * single_file_df['MuIso_SF_3']
					else:
						SF = 1

					weight = SF * single_file_df['GEN_wgt'] * single_file_df['PU_wgt']

					if file in self.framework.dir_list_s:
						single_file_df['signal'] = 1
						single_file_df['background'] = 0
						single_file_df['weight'] = file.weight / self.sum_weight_s * weight
					else:
						single_file_df['signal'] = 0
						single_file_df['background'] = 1
						single_file_df['weight'] = file.weight / self.sum_weight_b * weight
	
					self.df = pandas.concat([self.df,single_file_df])
		
		self.df.reset_index(inplace=True, drop=True)
		# self.add_more_variables(self.df)
		self.df = self.apply_cuts(self.df, self.framework.year)
		print self.df	
		self.labels = list(self.df.drop(['weight', 'signal', 'background']+spect_labels, axis=1))
		self.df.drop(spect_labels, axis=1, inplace=True)
		print self.labels
		self.df = shuffle(self.df)
		self.df_train, self.df_test = train_test_split(self.df,test_size=0.2, random_state=7)

	

	def train_models(self):
		self.df_train_scaled, self.df_test_scaled = self.scale(self.df_train, self.df_test, self.labels)		
		self.list_of_models = GetListOfModels(self.df.shape[1]-3) #the argument is for the input dimensions
		for obj in self.list_of_models:
			if obj.name not in self.framework.method_list:
				continue

			try:
				os.makedirs(self.package.mainDir+'/'+obj.name+'/')
			except OSError as e:
				if e.errno != errno.EEXIST:
					raise
	
			obj.CompileModel(self.package.dirs['modelDir'])
			# early_stopping = EarlyStopping(monitor='val_loss', patience=10)
			# tensorboard = TensorBoard(log_dir=self.package.dirs['logDir']+obj.name)
			# model_checkpoint = ModelCheckpoint(self.package.dirs['modelDir']+obj.name+'_trained_lwstValLoss.h5', monitor='val_loss', 
   #     		                           verbose=0, save_best_only=True, 
   #     		                           save_weights_only=False, mode='auto', 
   #     		                           period=1)

			history = obj.model.fit(			
									self.df_train_scaled[self.labels].values,
       		            			# self.df_train_scaled['category'].values,
       		            			self.df_train_scaled.loc[:,['signal', 'background']].values,
       		            			epochs=obj.epochs, 
       		            			batch_size=obj.batchSize, 
       		            			# sample_weight = weights_train.flatten(),			
       		            			# class_weight = self.df_train['sample_weight'].values,
       		            			verbose=1,
       		            			# callbacks=[
       		            				# early_stopping, 
       		            				# model_checkpoint,
       		            				# tensorboard
       		            				# ], 
       		            			validation_split=0.25,
       		            			#steps_per_epoch = None,
       		            			shuffle=True)
	

			self.df_train_scaled["predict_s_"+obj.name] =  obj.model.predict(self.df_train_scaled[self.labels].values)[:,0]
			self.df_train_scaled["predict_b_"+obj.name] =  obj.model.predict(self.df_train_scaled[self.labels].values)[:,1]

			self.df_test_scaled["predict_s_"+obj.name] =  obj.model.predict(self.df_test_scaled[self.labels].values)[:,0]
			self.df_test_scaled["predict_b_"+obj.name] =  obj.model.predict(self.df_test_scaled[self.labels].values)[:,1]


			self.df_history = pandas.DataFrame(history.history)
			self.plot_history(history.history, obj.name)

			self.plot_ROC("train", self.df_train_scaled, obj.name)
			self.plot_ROC("test", self.df_train_scaled, obj.name)
			self.plot_score("train", self.df_train_scaled, obj.name)
			self.plot_score("test", self.df_train_scaled, obj.name)



	def scale(self, train, test, labels):
		data_train = train.loc[:,labels]
		data_test = test.loc[:,labels]
		scaler = StandardScaler().fit(data_train.values)
		data_train = scaler.transform(data_train.values)
		data_test = scaler.transform(data_test.values)	
		train[labels] = data_train
		test[labels] = data_test
		return train, test

	def save_to_hdf(self, train, test, filename):
		train.to_hdf('%s%s.hdf5'%(self.package.dirs['dataDir'],filename), 'train')
		test.to_hdf('%s%s.hdf5'%(self.package.dirs['dataDir'], filename), 'test')


	def plot_ROC(self, output_name, df, method_name):
		roc = ROOT.TGraph()
		roc.SetName('roc')
		roc.GetXaxis().SetTitle("Signal eff.")
		roc.GetYaxis().SetTitle("Background rej.")
		score = df['predict_s_'+method_name] + (1 - df['predict_b_'+method_name]) # s_pred + (1 - b_pred)
		for i in range(200):
			cut = i / 100.0
			score = df['predict_s_'+method_name] + (1 - df['predict_b_'+method_name]) # s_pred + (1 - b_pred)
			sig_eff = float(df.loc[  (df['signal']==1) & (score > cut ) , ['weight'] ].sum(axis=0)) / df.loc[df['signal']==1, ['weight']].sum(axis=0)
			bkg_rej = float(df.loc[  (df['background']==1) & (score < cut ) , ['weight']  ].sum(axis=0)) / df.loc[df['background']==1, ['weight']].sum(axis=0)
			roc.SetPoint(i, sig_eff, bkg_rej)

		f = ROOT.TFile.Open(self.package.mainDir+'/'+method_name+'/'+output_name+"_roc.root", "recreate")
		roc.Write()
		f.Close()

		canv = ROOT.TCanvas("canv", "canv", 800, 800)
		canv.cd()
		roc.Draw("apl")
		canv.Print(self.package.mainDir+'/'+method_name+'/'+output_name+"_roc.png")
		canv.Close()



	def plot_history(self, history, method_name):
		plt.plot(history['acc'])
		plt.plot(history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig(self.package.mainDir+'/'+method_name+'/'+"acc.png")
		plt.clf()
		print "Accuracy plot saved as "+self.package.mainDir+'/'+method_name+'/'+"acc.png"
		plt.plot(history['loss'])
		plt.plot(history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig(self.package.mainDir+'/'+method_name+'/'+"loss.png")
		print "Loss plot saved as "+self.package.mainDir+'/'+method_name+'/'+"loss.png"		

	def plot_score(self, output_name, df, method_name):

		hist_s = ROOT.TH1D("s", "s", 100, 0, 2)
		hist_s.SetLineColor(ROOT.kRed)
		hist_s.SetFillColor(ROOT.kRed)
		hist_s.SetFillStyle(3003)
		hist_b = ROOT.TH1D("b", "b", 100, 0, 2)
		hist_b.SetLineColor(ROOT.kBlue)
		hist_b.SetFillColor(ROOT.kBlue)
		hist_b.SetFillStyle(3003)

		for index, row in df.iterrows():

			if row['signal']==1:
				hist_s.Fill( (row['predict_s_'+method_name]+(1-row['predict_b_'+method_name])), row['weight'] )
			elif row['background']==1:
				hist_b.Fill( (row['predict_s_'+method_name]+(1-row['predict_b_'+method_name])), row['weight'] )

		hist_s.Scale(1/hist_s.Integral())
		hist_b.Scale(1/hist_b.Integral())

		f = ROOT.TFile.Open(self.package.mainDir+'/'+method_name+'/'+output_name+"_score.root", "recreate")
		hist_b.Write()
		hist_s.Write()
		f.Close()

		canv = ROOT.TCanvas("canv1", "canv1", 800, 800)
		canv.cd()
		hist_b.Draw("hist")
		hist_s.Draw("histsame")
		canv.Print(self.package.mainDir+'/'+method_name+'/'+output_name+"_score.png")
		canv.Close()


	def calc_sum_wgts(self):
		self.sum_weight_s = 0
		self.sum_weight_b = 0

		for file in self.framework.dir_list_s:
			self.sum_weight_s += file.weight
		for file in self.framework.dir_list_b:
			self.sum_weight_b += file.weight

	def apply_cuts(self, df, year):
		muon1_pt 	= df['muons.pt[0]']
		muon2_pt 	= df['muons.pt[1]']
		muon1_ID 	= df['muons.isMediumID[0]']
		muon2_ID 	= df['muons.isMediumID[1]']
		muPair_mass = df['muPairs.mass[0]']

		if year is "2016":
			flag = 	((muPair_mass>113.8)&
				(muPair_mass<147.8)&
				(muon1_ID>0)&
				(muon2_ID>0)&
				(muon1_pt>26)&
				(muon2_pt>20))

		elif year is "2017":
			flag = 	((muPair_mass>110)&
				(muPair_mass<150)&
				(muon1_ID>0)&
				(muon2_ID>0)&
				(muon1_pt>30)&
				(muon2_pt>20))

		return df.loc[flag]

