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
		self.sum_weight_s = 1
		self.sum_weight_b = 1
		if self.framework.year is "2016":
			self.additional_vars=['muPairs.mass','PU_wgt','GEN_wgt', 'IsoMu_SF_3', 'IsoMu_SF_4', 'MuID_SF_3', 'MuID_SF_4', 'MuIso_SF_3', 'MuIso_SF_4']
		elif self.framework.year is "2017":
			self.additional_vars=['muPairs.mass','PU_wgt','GEN_wgt', 'IsoMu_SF_3', 'MuID_SF_3', 'MuIso_SF_3']

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
					
					for av in self.additional_vars:
						single_file_df = self.append_new_var(uproot_tree, single_file_df, av)
					
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
			
		# self.df.dropna(axis=0, how='any', inplace=True)
		
		self.df = self.apply_cuts(self.df)

		print self.df

		self.lables = list(self.df.drop(['weight', 'signal', 'background']+self.additional_vars, axis=1))
		self.df.drop(self.additional_vars, axis=1, inplace=True)
		self.df = shuffle(self.df)
		self.df_train, self.df_test = train_test_split(self.df,test_size=0.2, random_state=7)

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
	

			self.df_train_scaled["predict_s_"+obj.name] =  obj.model.predict(self.df_train_scaled[self.lables].values)[:,0]
			self.df_train_scaled["predict_b_"+obj.name] =  obj.model.predict(self.df_train_scaled[self.lables].values)[:,1]

			self.df_test_scaled["predict_s_"+obj.name] =  obj.model.predict(self.df_test_scaled[self.lables].values)[:,0]
			self.df_test_scaled["predict_b_"+obj.name] =  obj.model.predict(self.df_test_scaled[self.lables].values)[:,1]

			# self.save_to_hdf(self.df_train_scaled, self.df_test_scaled, 'scaled_w_predictions')
			# print self.df_train_scaled
			self.df_history = pandas.DataFrame(history.history)
			# self.df_history.to_hdf('%shistory.hdf5'%self.package.mainDir, obj.name)
			self.plot_history(history.history)

			self.plot_ROC("train", self.df_train_scaled, obj.name)
			self.plot_ROC("test", self.df_train_scaled, obj.name)
			self.plot_score("train", self.df_train_scaled, obj.name)
			self.plot_score("test", self.df_train_scaled, obj.name)



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


	def plot_ROC(self, output_name, df, method_name):
		roc = ROOT.TGraph()
		roc.SetName('roc')
		roc.GetXaxis().SetTitle("Signal eff.")
		roc.GetYaxis().SetTitle("Background rej.")
		score = df['predict_s_'+method_name] + (1 - df['predict_b_'+method_name]) # s_pred + (1 - b_pred)
		# print df.loc[  (df['signal']==1) & (score > 0.5 ) , ['weight'] ].sum(axis=0)
		for i in range(200):
			cut = i / 100.0
			score = df['predict_s_'+method_name] + (1 - df['predict_b_'+method_name]) # s_pred + (1 - b_pred)
			sig_eff = float(df.loc[  (df['signal']==1) & (score > cut ) , ['weight'] ].sum(axis=0)) / df.loc[df['signal']==1, ['weight']].sum(axis=0)
			bkg_rej = float(df.loc[  (df['background']==1) & (score < cut ) , ['weight']  ].sum(axis=0)) / df.loc[df['background']==1, ['weight']].sum(axis=0)
			roc.SetPoint(i, sig_eff, bkg_rej)

		f = ROOT.TFile.Open(self.package.mainDir+output_name+"_roc.root", "recreate")
		roc.Write()
		f.Close()

		canv = ROOT.TCanvas("canv", "canv", 800, 800)
		canv.cd()
		roc.Draw("apl")
		canv.Print(self.package.mainDir+output_name+"_roc.png")
		# canv.SaveAs(self.package.mainDir+output_name+"_roc.root")
		canv.Close()



	def plot_history(self, history):
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

		f = ROOT.TFile.Open(self.package.mainDir+output_name+"_score.root", "recreate")
		hist_b.Write()
		hist_s.Write()
		f.Close()

		canv = ROOT.TCanvas("canv1", "canv1", 800, 800)
		canv.cd()
		hist_b.Draw("hist")
		hist_s.Draw("histsame")
		canv.Print(self.package.mainDir+output_name+"_score.png")
		# canv.SaveAs(self.package.mainDir+output_name+"_score.root")
		canv.Close()


	def calc_sum_wgts(self):
		self.sum_weight_s = 0
		self.sum_weight_b = 0

		for file in self.framework.dir_list_s:
			self.sum_weight_s += file.weight
		for file in self.framework.dir_list_b:
			self.sum_weight_b += file.weight

		print "Sum of signal weights: %s"%self.sum_weight_s
		print "Sum of background weights: %s"%self.sum_weight_b


	def append_new_var(self, tree, df, var_name):
		column = pandas.DataFrame(data=tree[var_name].array().tolist())
		column = column.iloc[:,0]
		new_df = pandas.DataFrame()
		new_df[var_name] = column
		df = pandas.concat([df, new_df], axis=1)
		return df



	def apply_cuts(self, df):
		return df.loc[((df['muPairs.mass']>113.8)&(df['muPairs.mass']<147.8))]

