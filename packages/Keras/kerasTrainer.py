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
ROOT.gStyle.SetOptStat(0)

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
		# ttbar_flag = True
		for file in self.framework.dir_list_s + self.framework.dir_list_b:
			for filename in os.listdir(file.path):
			    if filename.endswith(".root"): 
			    	# if (file.name is "tt_ll_AMC") and (not (filename.endswith("_9.root"))):
			    	# 	ttbar_flag = False
			    	# else:
			    	# 	ttbar_flag = True
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
	
						print "Added %s with %i events"%(file.name, single_file_df.shape[0])
						# if ttbar_flag:
						self.df = pandas.concat([self.df,single_file_df])
		
		self.df.reset_index(inplace=True, drop=True)
		evts_before_cuts = self.df.shape[0]
		self.df = self.apply_cuts(self.df, self.framework.year)
		print "Applying cuts: selected %i events out of %i"%(self.df.shape[0], evts_before_cuts)
		self.labels = list(self.df.drop(['weight', 'signal', 'background']+spect_labels, axis=1))
		self.truth_labels = []
		if self.framework.custom_loss:
			self.df = self.make_mass_bins(self.df, 10, 110, 150)
		self.truth_labels.extend(['signal', 'background'])
		print self.truth_labels
		self.df = shuffle(self.df)
		self.df_train, self.df_test = train_test_split(self.df,test_size=0.2, random_state=7)

	

	def train_models(self):
		self.df_train_scaled, self.df_test_scaled = self.scale(self.df_train, self.df_test, self.labels)
		self.list_of_models = GetListOfModels(self.df[self.labels].shape[1]) #the argument is for the input dimensions
		for obj in self.list_of_models:
			if obj.name not in self.framework.method_list:
				continue

			try:
				os.makedirs(self.package.mainDir+'/'+obj.name+'/')
			except OSError as e:
				if e.errno != errno.EEXIST:
					raise

			try:
				os.makedirs(self.package.mainDir+'/'+obj.name+'/png/')
			except OSError as e:
				if e.errno != errno.EEXIST:
					raise

			try:
				os.makedirs(self.package.mainDir+'/'+obj.name+'/root/')
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
       		            			# self.df_train_scaled.loc[:,['signal', 'background']].values,
       		            			self.df_train_scaled[self.truth_labels].values,
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
	
			# self.df_train_scaled["predict_s_"+obj.name] =  obj.model.predict(self.df_train_scaled[self.labels].values)[:,0]
			# self.df_train_scaled["predict_b_"+obj.name] =  obj.model.predict(self.df_train_scaled[self.labels].values)[:,1]

			# self.df_test_scaled["predict_s_"+obj.name] =  obj.model.predict(self.df_test_scaled[self.labels].values)[:,0]
			# self.df_test_scaled["predict_b_"+obj.name] =  obj.model.predict(self.df_test_scaled[self.labels].values)[:,1]
			if self.framework.custom_loss:
				self.df_train_scaled["predict_s_"+obj.name] =  obj.model.predict(self.df_train_scaled[self.labels].values)[:,10]
				self.df_train_scaled["predict_b_"+obj.name] =  obj.model.predict(self.df_train_scaled[self.labels].values)[:,11]
	
				self.df_test_scaled["predict_s_"+obj.name] =  obj.model.predict(self.df_test_scaled[self.labels].values)[:,10]
				self.df_test_scaled["predict_b_"+obj.name] =  obj.model.predict(self.df_test_scaled[self.labels].values)[:,11]
			else:
				self.df_train_scaled["predict_s_"+obj.name] =  obj.model.predict(self.df_train_scaled[self.labels].values)[:,0]
				self.df_train_scaled["predict_b_"+obj.name] =  obj.model.predict(self.df_train_scaled[self.labels].values)[:,1]
	
				self.df_test_scaled["predict_s_"+obj.name] =  obj.model.predict(self.df_test_scaled[self.labels].values)[:,0]
				self.df_test_scaled["predict_b_"+obj.name] =  obj.model.predict(self.df_test_scaled[self.labels].values)[:,1]


			self.plot_mass_histograms(self.df_test_scaled, obj.name)

			self.df_history = pandas.DataFrame(history.history)
			self.plot_history(history.history, obj.name)

			self.plot_ROC("train", self.df_train_scaled, obj.name)
			self.plot_ROC("test", self.df_train_scaled, obj.name)
			# self.plot_score("train", self.df_train_scaled, obj.name)
			# self.plot_score("test", self.df_train_scaled, obj.name)
			self.check_overfitting(self.df_train_scaled, self.df_test_scaled, obj.name)
			self.plot_bkg_shapes(self.df_test_scaled, obj.name)



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
		# score = df['predict_s_'+method_name] + (1 - df['predict_b_'+method_name]) # s_pred + (1 - b_pred)
		for i in range(200):
			cut = i / 100.0
			score = df['predict_s_'+method_name] + (1 - df['predict_b_'+method_name]) # s_pred + (1 - b_pred)
			sig_eff = float(df.loc[  (df['signal']==1) & (score > cut ) , ['weight'] ].sum(axis=0)) / df.loc[df['signal']==1, ['weight']].sum(axis=0)
			bkg_rej = float(df.loc[  (df['background']==1) & (score < cut ) , ['weight']  ].sum(axis=0)) / df.loc[df['background']==1, ['weight']].sum(axis=0)
			roc.SetPoint(i, sig_eff, bkg_rej)
			# print (cut, sig_eff)

		f = ROOT.TFile.Open(self.package.mainDir+'/'+method_name+'/root/'+output_name+"_roc.root", "recreate")
		roc.Write()
		f.Close()

		canv = ROOT.TCanvas("canv", "canv", 800, 800)
		canv.cd()
		roc.Draw("apl")
		canv.Print(self.package.mainDir+'/'+method_name+'/png/'+output_name+"_roc.png")
		canv.Close()


	def plot_history(self, history, method_name):
		plt.plot(history['acc'])
		plt.plot(history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig(self.package.mainDir+'/'+method_name+'/png/'+"acc.png")
		plt.clf()
		print "Accuracy plot saved as "+self.package.mainDir+'/'+method_name+'/png/'+"acc.png"
		plt.plot(history['loss'])
		plt.plot(history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig(self.package.mainDir+'/'+method_name+'/png/'+"loss.png")
		print "Loss plot saved as "+self.package.mainDir+'/'+method_name+'/png/'+"loss.png"	
		plt.clf()	


	def check_overfitting(self, train, test, name):
		hist_s_train, hist_b_train = self.get_score("train", train, name)
		hist_s_test, hist_b_test = self.get_score("test", test, name)

		canv = ROOT.TCanvas("canv1", "canv1", 800, 800)
		canv.cd()
		hist_b_train.Draw("hist")
		hist_s_train.Draw("histsame")
		hist_b_test.SetMarkerStyle(20)
		hist_s_test.SetMarkerStyle(20)
		hist_b_test.SetMarkerSize(0.8)
		hist_s_test.SetMarkerSize(0.8)
		hist_b_test.SetLineWidth(1)
		hist_s_test.SetLineWidth(1)
		hist_b_test.Draw("pesame")
		hist_s_test.Draw("pesame")

		canv.Print(self.package.mainDir+'/'+name+"/png/overfit.png")
		canv.Close()


	def get_score(self, label, df, method_name):
		hist_s = ROOT.TH1D("s"+label, "", 100, 0, 2)
		hist_s.SetLineColor(ROOT.kRed)
		hist_s.SetLineWidth(2)
		hist_s.SetFillColor(ROOT.kRed)
		hist_s.SetMarkerColor(ROOT.kRed)
		hist_s.SetFillStyle(3003)
		hist_b = ROOT.TH1D("b"+label, "", 100, 0, 2)
		hist_b.SetLineColor(ROOT.kBlue)
		hist_b.SetLineWidth(2)
		hist_b.SetFillColor(ROOT.kBlue)
		hist_b.SetMarkerColor(ROOT.kBlue)
		hist_b.SetFillStyle(3003)

		for index, row in df.iterrows():
			dnn_score = row['predict_s_'+method_name]+(1-row['predict_b_'+method_name])
			if row['signal']==1:
				hist_s.Fill(dnn_score, row['weight'] )
			elif row['background']==1:
				hist_b.Fill(dnn_score, row['weight'] )

		hist_s.Scale(1/hist_s.Integral())
		hist_b.Scale(1/hist_b.Integral())

		return hist_s, hist_b



	def plot_bkg_shapes(self, df, method_name):
		ROOT.gStyle.SetOptStat(0)
		score_bins = {}
		score_bins["0-0p5"] = [0, 0.5]
		score_bins["0p5-1"] = [0.5, 1]
		score_bins["1-1p5"] = [1, 1.5]
		score_bins["1p5-2"] = [1.5, 2]
		colors = {	
			"0-0p5": ROOT.kBlue,
			"0p5-1": ROOT.kRed,
			"1-1p5": ROOT.kGreen,
			"1p5-2": ROOT.kOrange-3
			}
		hist_dict = {}
		legend = ROOT.TLegend(.6,.7,.89,.89)
		canv = ROOT.TCanvas("canv1", "canv1", 800, 800)
		canv.cd()
		for key, value in score_bins.iteritems():
			hist_dict[key] = self.bkg_shape_for_score_bin(df, method_name, value[0], value[1], colors[key])
			if hist_dict[key].Integral():
				hist_dict[key].Scale(1/hist_dict[key].Integral())
			legend.AddEntry(hist_dict[key], "%.2f%% < DNN score < %.2f%%"%(value[0]*100/2, value[1]*100/2), 'f')
			hist_dict[key].Draw("histsame")
		legend.Draw()
		canv.Print(self.package.mainDir+'/'+method_name+'/png/bkg_shapes.png')
		canv.Close()


	def bkg_shape_for_score_bin(self, df, method_name, xmin, xmax, color):

		hist = ROOT.TH1D("b%.1f"%xmin, "", 20, 110, 150)
		hist.SetLineColor(color)
		hist.SetLineWidth(2)
		hist.SetFillColor(color)
		hist.SetFillStyle(3003)

		for index, row in df.iterrows():
			dnn_score = row['predict_s_'+method_name]+(1-row['predict_b_'+method_name])
			if (row['background']==1)&(dnn_score>xmin)&(dnn_score<xmax):
				hist.Fill(row['muPairs.mass[0]'], row['weight'] )

		return hist


	def plot_mass_histograms(self, df, model_name):
		legend = ROOT.TLegend(.6,.7,.89,.89)
		hist1 =	self.plot_mass_histogram(self.df_test_scaled, 0, 0, model_name)
		hist2 =	self.plot_mass_histogram(self.df_test_scaled, 0, 1, model_name)
		hist3 =	self.plot_mass_histogram(self.df_test_scaled, 1, 0, model_name)
		hist4 =	self.plot_mass_histogram(self.df_test_scaled, 1, 1, model_name)
		hist_list = [hist1, hist2, hist3, hist4]

		hist1.SetLineColor(ROOT.kBlue)			
		hist2.SetLineColor(ROOT.kRed)			
		hist3.SetLineColor(ROOT.kGreen)			
		hist4.SetLineColor(ROOT.kOrange-3)		

		hist1.SetFillColor(ROOT.kBlue)
		hist2.SetFillColor(ROOT.kRed)	
		hist3.SetFillColor(ROOT.kGreen)	
		hist4.SetFillColor(ROOT.kOrange-3)

		legend.AddEntry(hist1, "S, predicted S", "f")
		legend.AddEntry(hist2, "S, predicted B", "f")
		legend.AddEntry(hist3, "B, predicted S", "f")
		legend.AddEntry(hist4, "B, predicted B", "f")

		canv = ROOT.TCanvas("canv", "canv", 800, 800)
		canv.cd()
		for h in hist_list:
			h.SetLineWidth(2)
			h.SetFillStyle(3003)
			h.Draw("histsame")
		legend.Draw()
		canv.Print(self.package.mainDir+'/'+model_name+"/png/mass_histograms.png")
		canv.Close()

	def plot_mass_histogram(self, df, original_category, predicted_category, model_name):
		original_categories = ['signal', 'background']
		predicted_categories = ['predict_s_'+model_name, 'predict_b_'+model_name]
		mass_bins = ['mass_bin_%i'%i for i in range(10)]

		hist = ROOT.TH1D("hist", "", 10, 110, 150)

		for index, row in df.iterrows():
			if row[original_categories[original_category]]==1:
				hist.Fill(row['muPairs.mass[0]'], row[predicted_categories[predicted_category]] )

		hist.Scale(1/hist.Integral())
		return hist



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
			# flag = 	((muPair_mass>113.8)&
			# 	(muPair_mass<147.8)&
			flag = 	((muPair_mass>110)&
				(muPair_mass<150)&
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

	def make_mass_bins(self, df, nbins, min, max):

		if "muPairs.mass[0]" not in df.columns:
			print "Add muPairs.mass[0] to spectators!"
			return

		bin_width = float((max-min)/nbins)
		# print bin_width

		for i in range(nbins):
			df["mass_bin_%i"%i] = 0
			print min+i*bin_width, min+(i+1)*bin_width
			df.loc[(df["muPairs.mass[0]"]>min+i*bin_width) & (df["muPairs.mass[0]"]<min+(i+1)*bin_width), "mass_bin_%i"%i] = 1
			self.truth_labels.append("mass_bin_%i"%i)

		return df









