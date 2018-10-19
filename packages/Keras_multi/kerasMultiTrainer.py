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

class KerasMultiTrainer(object):
	def __init__(self, framework, package):
		self.framework = framework
		self.package = package
		self.sum_weights = {}
		self.spect_labels = []
		self.mass_bin_labels = []
		self.category_labels = self.framework.signal_categories+self.framework.bkg_categories
		self.mass_histograms = []
		self.mass_histograms_th1d = {}

	def __enter__(self):
		self.df = pandas.DataFrame()
		return self


	def __exit__(self, *args):
		del self.df	


	def convert_to_pandas(self):
		self.calc_sum_wgts()
		for file in self.framework.files:
			for filename in os.listdir(file.path):
			    if filename.endswith(".root"): 
					with uproot.open(file.path+filename) as f: 
						uproot_tree = f[self.framework.treePath]
						single_file_df = pandas.DataFrame()

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
									self.spect_labels.extend(single_var_df.columns)
								single_file_df = pandas.concat([single_file_df, single_var_df], axis=1)
								single_file_df.fillna(var.replacement, axis=0, inplace=True) # if there are not enough jets
				
							else:
								single_file_df[var.name] = up_var
								if var in self.framework.spectator_list:
									self.spect_labels.append(var.name)
	
						
						if self.framework.year is "2016":
							SF = (0.5*(single_file_df['IsoMu_SF_3'] + single_file_df['IsoMu_SF_4'])*0.5*(single_file_df['MuID_SF_3'] + single_file_df['MuID_SF_4'])*0.5*(single_file_df['MuIso_SF_3'] + single_file_df['MuIso_SF_4']))
						elif self.framework.year is "2017":
							SF = single_file_df['IsoMu_SF_3'] * single_file_df['MuID_SF_3'] * single_file_df['MuIso_SF_3']
						else:
							SF = 1
	
						weight = SF * single_file_df['GEN_wgt'] * single_file_df['PU_wgt']

						for category in self.category_labels:
							single_file_df[category] = 0	
							if file.category is category:
								single_file_df[category] = 1
							single_file_df['weight'] = file.weight / self.sum_weights[category] * weight

	
						print "Added %s with %i events"%(file.name, single_file_df.shape[0])
						self.df = pandas.concat([self.df,single_file_df])
		
		self.df.reset_index(inplace=True, drop=True)
		evts_before_cuts = self.df.shape[0]
		self.df = self.apply_cuts(self.df, self.framework.year)
		print "Applying cuts: selected %i events out of %i"%(self.df.shape[0], evts_before_cuts)
		self.labels = list(self.df.drop(['weight']+self.spect_labels+self.category_labels, axis=1))
		self.truth_labels = []
		if self.framework.custom_loss:
			self.df = self.make_mass_bins(self.df, 10, 110, 150)
		self.truth_labels.extend(self.category_labels)
		print self.df
		self.df = shuffle(self.df)
		self.df_train, self.df_test = train_test_split(self.df,test_size=0.2, random_state=7)

	

	def train_models(self):
		self.df_train_scaled, self.df_test_scaled = self.scale(self.df_train, self.df_test, self.labels)
		self.list_of_models = GetListOfModels(self)
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
       		            			self.df_train_scaled[self.truth_labels].values,
       		            			epochs=obj.epochs, 
       		            			batch_size=obj.batchSize, 
       		            			verbose=1,
       		            			# callbacks=[
       		            				# early_stopping, 
       		            				# model_checkpoint,
       		            				# tensorboard
       		            				# ], 
       		            			validation_split=0.25,
       		            			#steps_per_epoch = None,
       		            			shuffle=True)
	

			train_prediction = pandas.DataFrame(data=obj.model.predict(self.df_train_scaled[self.labels].values), columns=["pred_%s_%s"%(n,obj.name) for n in self.truth_labels], index=self.df_train_scaled.index)
			test_prediction = pandas.DataFrame(data=obj.model.predict(self.df_test_scaled[self.labels].values), columns=["pred_%s_%s"%(n,obj.name) for n in self.truth_labels], index=self.df_test_scaled.index)


			self.df_train_scaled = pandas.concat([self.df_train_scaled, train_prediction], axis=1)
			self.df_test_scaled = pandas.concat([self.df_test_scaled, test_prediction], axis=1)


			self.plot_mass_histograms(self.df_test_scaled, obj.name)
			self.plot_masses_by_input_category(self.df_test_scaled, obj.name)

			self.df_history = pandas.DataFrame(history.history)
			self.plot_history(history.history, obj.name)

			self.plot_ROC("train", self.df_train_scaled, obj.name)
			self.plot_ROC("test", self.df_train_scaled, obj.name)
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
		score = pandas.DataFrame()
		df['isSignal'] = 0
		df['isBackground'] = 0
		df['score'] = 0
		for category in self.framework.signal_categories:
			df['score'] += df["pred_%s_%s"%(category,method_name)]
			df['isSignal'] += df[category]
		for category in self.framework.bkg_categories:
			df['score'] += 1 - df["pred_%s_%s"%(category,method_name)]
			df['isBackground'] += df[category]

		min_score = len(self.framework.bkg_categories) - 1
		for i in range(100*len(self.category_labels)):
			cut = min_score + i / 50.0
			sig_eff = float(df.loc[  (df['isSignal']>0) & (df['score'] > cut ) , ['weight'] ].sum(axis=0)) / df.loc[df['isSignal']>0, ['weight']].sum(axis=0)
			bkg_rej = float(df.loc[  (df['isBackground']>0) & (df['score'] < cut ) , ['weight']  ].sum(axis=0)) / df.loc[df['isBackground']>0, ['weight']].sum(axis=0)
			roc.SetPoint(i, sig_eff, bkg_rej)
			
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
		legend = ROOT.TLegend(.6,.7,.89,.89)
		train_dict = self.get_score("train", train, name)
		test_dict = self.get_score("test", test, name)

		canv = ROOT.TCanvas("canv1", "canv1", 800, 800)
		canv.cd()

		count = 0
		color_pool = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen, ROOT.kOrange-3, ROOT.kViolet, ROOT.kCyan]

		for category in self.category_labels:
			legend.AddEntry(train_dict[category], category+" train", "f")
			legend.AddEntry(test_dict[category], category+" test", "pe")

			train_dict[category].Draw("histsame")
			test_dict[category].Draw("pesame")

			train_dict[category].SetLineColor(color_pool[count])
			train_dict[category].SetFillColor(color_pool[count])
			train_dict[category].SetMarkerColor(color_pool[count])

			test_dict[category].SetLineColor(color_pool[count])
			test_dict[category].SetFillColor(color_pool[count])
			test_dict[category].SetMarkerColor(color_pool[count])

			test_dict[category].SetMarkerStyle(20)
			test_dict[category].SetMarkerSize(0.8)
			test_dict[category].SetLineWidth(1)

			count +=1

		legend.Draw()
		canv.Print(self.package.mainDir+'/'+name+"/png/overfit.png")
		canv.SaveAs(self.package.mainDir+'/'+name+"/root/overfit.root")
		canv.Close()


	def get_score(self, label, df, method_name):
		hist = {}
		for category in self.category_labels:
			hist[category] = ROOT.TH1D(category+"_"+label, "", 100, len(self.framework.bkg_categories)-1, len(self.framework.bkg_categories)+1)
			hist[category].SetLineWidth(2)
			hist[category].SetFillStyle(3003)

		for index, row in df.iterrows():
			dnn_score = 0

			for category in self.framework.signal_categories:
				dnn_score += row["pred_%s_%s"%(category,method_name)]
			for category in self.framework.bkg_categories:
				dnn_score += 1 - row["pred_%s_%s"%(category,method_name)]

			for category in self.category_labels:
				if row[category] == 1:
					hist[category].Fill(dnn_score, row['weight'])

		for category in self.category_labels:
			hist[category].Scale(1/hist[category].Integral())

		return hist



	def plot_bkg_shapes(self, df, method_name):
		ROOT.gStyle.SetOptStat(0)
		score_bins = {}
		min_bin = len(self.framework.bkg_categories) - 1
		score_bins["0-25"] = [min_bin, min_bin+0.5]
		score_bins["25-50"] = [min_bin+0.5, min_bin+1]
		score_bins["50-75"] = [min_bin+1, min_bin+1.5]
		score_bins["75-100"] = [min_bin+1.5, min_bin+2]
		colors = {	
			"0-25": ROOT.kBlue,
			"25-50": ROOT.kRed,
			"50-75": ROOT.kGreen,
			"75-100": ROOT.kOrange-3
			}
		hist_dict = {}
		legend = ROOT.TLegend(.6,.7,.89,.89)
		canv = ROOT.TCanvas("canv1", "canv1", 800, 800)
		canv.cd()
		for key, value in score_bins.iteritems():
			hist_dict[key] = self.bkg_shape_for_score_bin(df, method_name, value[0], value[1], colors[key])
			if hist_dict[key].Integral():
				hist_dict[key].Scale(1/hist_dict[key].Integral())
			legend.AddEntry(hist_dict[key], "%.2f%% < DNN score < %.2f%%"%((value[0]-min_bin)*50, (value[1]-min_bin)*50), 'f')
			hist_dict[key].Draw("histsame")
		legend.Draw()
		canv.Print(self.package.mainDir+'/'+method_name+'/png/bkg_shapes.png')
		canv.SaveAs(self.package.mainDir+'/'+method_name+'/root/bkg_shapes.root')
		canv.Close()


	def bkg_shape_for_score_bin(self, df, method_name, xmin, xmax, color):
		hist = ROOT.TH1D("b%.1f"%xmin, "", 20, 110, 150)
		hist.SetLineColor(color)
		hist.SetLineWidth(2)
		hist.SetFillColor(color)
		hist.SetFillStyle(3003)

		for index, row in df.iterrows():
			dnn_score = 0

			for category in self.framework.signal_categories:
				dnn_score += row["pred_%s_%s"%(category,method_name)]
			for category in self.framework.bkg_categories:
				dnn_score += 1 - row["pred_%s_%s"%(category,method_name)]

			for category in self.framework.bkg_categories:
				if (row[category] == 1)&(dnn_score>xmin)&(dnn_score<xmax):
					hist.Fill(row['muPairs.mass[0]'], row['weight'] )
		return hist


	def plot_mass_histograms(self, df, model_name):
		legend = ROOT.TLegend(.6,.7,.89,.89)
		class CategoryMassHist(object):
			def __init__(self, framework, df, category, model_name, color):
				self.framework = framework
				self.df = df
				self.category = category
				self.model_name = model_name
				self.color = color
				# self.get_histos()
				self.hist_correct, self.hist_incorrect = self.framework.plot_mass_histogram(self.df, self.category, self.model_name, self.color)
							
		mass_hists = []
		color_pool = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen, ROOT.kOrange-3, ROOT.kViolet, ROOT.kCyan]
		count = 0

		canv = ROOT.TCanvas("canv", "canv", 800, 800)
		canv.cd()

		for category in self.category_labels:
			new_mass_hist = CategoryMassHist(self, df, category, model_name, color_pool[count])
			mass_hists.append(new_mass_hist)
			count += 1
			legend.AddEntry(new_mass_hist.hist_correct, category+" correct", "f")
			legend.AddEntry(new_mass_hist.hist_incorrect, category+" incorrect", "l")
			new_mass_hist.hist_correct.Draw("histsame")
			new_mass_hist.hist_incorrect.Draw("histsame")

		legend.Draw()
		canv.Print(self.package.mainDir+'/'+model_name+"/png/mass_histograms.png")
		canv.SaveAs(self.package.mainDir+'/'+model_name+"/root/mass_histograms.root")
		canv.Close()

	def plot_mass_histogram(self, df, category, model_name, color):

		hist_correct = ROOT.TH1D("hist_c_"+category, "", 10, 110, 150)
		hist_incorrect = ROOT.TH1D("hist_i_"+category, "", 10, 110, 150)

		for index, row in df.iterrows():
			for cat in self.category_labels:
				if row[category]==1:
					if cat in category:
						hist_correct.Fill(row['muPairs.mass[0]'], row["pred_%s_%s"%(cat, model_name)] )
					else:
						hist_incorrect.Fill(row['muPairs.mass[0]'], row["pred_%s_%s"%(cat, model_name)] )
		


		hist_correct.Scale(1/hist_correct.Integral())
		hist_correct.SetLineColor(color)
		hist_correct.SetFillColor(color)
		hist_correct.SetLineWidth(2)
		hist_correct.SetFillStyle(3003)

		hist_incorrect.Scale(1/hist_incorrect.Integral())
		hist_incorrect.SetLineWidth(2)
		hist_incorrect.SetLineStyle(2)
		hist_incorrect.SetLineColor(color)

		return hist_correct, hist_incorrect


	def plot_masses_by_input_category(self, df, model_name):
		for category in self.category_labels:
			self.plot_mass_from_output_nodes(df, category, model_name)

	def plot_mass_from_output_nodes(self, df, category, model_name):
		legend = ROOT.TLegend(.6,.7,.89,.89)
		hists = {}
		for cat in self.category_labels:
			hists[category+"_"+cat] = ROOT.TH1D(category+"_"+cat, "", 10, 110, 150)

		for index, row in df.iterrows():
			if row[category]==1:
				for cat in self.category_labels:			
					hists[category+"_"+cat].Fill(row['muPairs.mass[0]'], row["pred_%s_%s"%(cat, model_name)] )
		
		canv = ROOT.TCanvas("canv", "canv", 800, 800)
		canv.cd()

		color_pool = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen, ROOT.kOrange-3, ROOT.kViolet, ROOT.kCyan]
		count = 0
		for cat in self.category_labels:
			legend.AddEntry(hists[category+"_"+cat], "true: %s, pred.: %s"%(category, cat), "l")
			hists[category+"_"+cat].Scale(1/hists[category+"_"+cat].Integral())
			hists[category+"_"+cat].SetLineColor(color_pool[count])
			hists[category+"_"+cat].SetLineWidth(2)
			hists[category+"_"+cat].Draw("histsame")
			count += 1

		self.mass_histograms_th1d[category].SetLineColor(ROOT.kBlack)
		self.mass_histograms_th1d[category].SetLineWidth(2)
		self.mass_histograms_th1d[category].Draw("histsame")
		legend.AddEntry(self.mass_histograms_th1d[category], "%s: input distribution"%category, "l")

		legend.Draw()
		canv.Print(self.package.mainDir+'/'+model_name+"/png/mass_%s.png"%category)
		canv.SaveAs(self.package.mainDir+'/'+model_name+"/root/mass_%s.root"%category)
		canv.Close()
		


	def calc_sum_wgts(self):
		self.sum_weights = {}
		for category in self.category_labels:
			self.sum_weights[category] = 0

		for file in self.framework.files:
			self.sum_weights[file.category] += file.weight	


	def apply_cuts(self, df, year):
		muon1_pt 	= df['muons.pt[0]']
		muon2_pt 	= df['muons.pt[1]']
		muon1_ID 	= df['muons.isMediumID[0]']
		muon2_ID 	= df['muons.isMediumID[1]']
		muPair_mass = df['muPairs.mass[0]']

		if year is "2016":
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

		for category in self.category_labels:
			self.mass_histograms_th1d[category] = ROOT.TH1D("input_"+category, "", nbins, min, max)

		for i in range(nbins):
			df["mass_bin_%i"%i] = 0

			df.loc[(df["muPairs.mass[0]"]>min+i*bin_width) & (df["muPairs.mass[0]"]<min+(i+1)*bin_width), "mass_bin_%i"%i] = 1
			self.mass_bin_labels.append("mass_bin_%i"%i)

		self.truth_labels.extend(self.mass_bin_labels)

		for category in self.category_labels:

			mass_hist = df.loc[(df[category]>0),self.mass_bin_labels].sum(axis=0)
			print "mass hist shape for %s: "%category, mass_hist.shape
			mass_hist = mass_hist / mass_hist.sum()
			self.mass_histograms.append(mass_hist.values.tolist())	

			for i in range(nbins):
				self.mass_histograms_th1d[category].SetBinContent(i+1, mass_hist.values.tolist()[i])
		print self.mass_histograms

		return df









