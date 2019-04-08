import ROOT
import os, sys, errno
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(1)
from math import *	
from array import array	


class Analyzer(object):

	def __init__(self):
		self.out_path = ""
		self.MVASources = []
		
	class MVASource(object):
		def __init__(self, framework, name, title, path):
			self.framework = framework
			self.name = name
			self.title = title 
			self.path = path
			self.samples = []
			self.data_hist = None
			self.signal_hists = []
			self.mc_stack = ROOT.THStack()
			self.lumi = 0

		class Sample(object):
			def __init__(self, name, title, filename, treename, isData, isStacked, color, isWeightOverLumi, additional_cut):
				self.name = name
				self.title = title
				self.filename = filename
				self.treename = treename
				self.isData = isData
				self.isStacked = isStacked
				self.color = color			
				self.isWeightOverLumi = isWeightOverLumi
				self.additional_cut = additional_cut

		def set_lumi(self, lumi):
			self.lumi = lumi

		def add_sample(self, name, title, filename, treename, isData, isStacked, color, isWeightOverLumi=True, additional_cut="1"):
			new_sample = self.Sample(name, title, filename, treename, isData, isStacked, color, isWeightOverLumi, additional_cut)
			self.samples.append(new_sample)
			
		def plot(self, var_name, nBins, xmin, xmax, label=""):
			trees = {}
			# hists = {}
			self.signal_hists = []
			self.data_hist = None
			self.mc_stack = ROOT.THStack()
			legend = ROOT.TLegend(0.7, 0.8, 0.895, 0.895)

			for smp in self.samples:
				trees[smp.name] = ROOT.TChain(smp.treename)
				trees[smp.name].Add("%s/%s"%(self.path, smp.filename))
				print "Tree for %s added with %i entries"%(smp.name, trees[smp.name].GetEntries())
				hist_name = "hist_%s_%s_%s_%s"%(self.name, smp.name, var_name, label)
				hist = ROOT.TH1D(hist_name, hist_name, nBins, xmin, xmax)

				dummy = ROOT.TCanvas("dummy_"+hist_name, "dummy_"+hist_name, 100, 100)
				dummy.cd()
				if smp.isData:
					trees[smp.name].Draw("%s>>%s"%(var_name, hist_name))
					self.data_hist = hist
					self.data_hist.SetMarkerColor(smp.color)
					self.data_hist.SetLineColor(smp.color)
					self.data_hist.SetMarkerStyle(20)
					self.data_hist.SetMarkerSize(0.8)
					legend.AddEntry(hist, "Data %i /pb"%self.lumi, "pe")
				else:
					if smp.isWeightOverLumi:
						trees[smp.name].Draw("%s>>%s"%(var_name, hist_name), "weight_over_lumi*%f*(%s)"%(self.lumi, smp.additional_cut))
					else:
						trees[smp.name].Draw("%s>>%s"%(var_name, hist_name), "weight*(%s)"%smp.additional_cut)
				dummy.Close()
				hist.SetLineWidth(2)
				if smp.isStacked:
					hist.SetFillColor(smp.color)
					hist.SetLineColor(ROOT.kBlack)
					self.mc_stack.Add(hist)
					legend.AddEntry(hist, smp.title, "f")
				elif not smp.isData:
					hist.SetLineColor(smp.color)
					self.signal_hists.append(hist)
					legend.AddEntry(hist, smp.title, "l")

			canvas = ROOT.TCanvas(var_name, var_name, 800, 800)
			canvas.cd()
			canvas.SetLogy()

			self.mc_stack.Draw("hist")
			self.mc_stack.SetTitle(self.title)
			self.mc_stack.GetXaxis().SetTitle(var_name)
			self.mc_stack.SetMinimum(0.01)
			self.mc_stack.SetMaximum(100000)
			for hist in self.signal_hists:
				hist.Draw("histsame")
			if self.data_hist:
				self.data_hist.Draw("pe1same")
			legend.Draw()
			new_out_path = "%s/%s/"%(self.framework.out_path, self.name)

			try:
				os.makedirs(new_out_path)
			except OSError as e:
				if e.errno != errno.EEXIST:
					raise

			canvas.SaveAs("%s/%s.root"%(new_out_path,var_name))
			canvas.SaveAs("%s/%s.png"%(new_out_path,var_name))


		def plot_roc(self, score, nBins, xmin, xmax, working_points=[]):
			print "Plotting MVA score..."
			self.plot(score, nBins, xmin, xmax, "forROC")
			roc = ROOT.TGraph()
			idx = []
			if working_points:
				idx = self.score_cuts_from_wp(score, working_points, nBins, xmin, xmax)
				wp_graph = ROOT.TGraph()
				wp_graph.SetMarkerStyle(20)
				wp_graph.SetMarkerSize(1)
				wp_graph.SetMarkerColor(ROOT.kRed)
				count = 0

			for i in range(1, nBins+1):
				# print "i = ", i
				sig_above = 0
				sig_total = 0
				bkg_below = 0
				bkg_total = 0
				sig_eff = 0
				bkg_rej = 0

				for h in self.mc_stack.GetHists():
					bkg_below = bkg_below + h.Integral(1, i)
					bkg_total = bkg_total + h.Integral()
				for h in self.signal_hists:
					sig_above = sig_above + h.Integral(i,nBins+1)
					sig_total = sig_total + h.Integral()

				if sig_total:
					sig_eff = sig_above/sig_total

				if bkg_total:
					bkg_rej = bkg_below/bkg_total

				if working_points:
					if i in idx:
						wp_graph.SetPoint(count, sig_eff, bkg_rej)
						count = count+1

				roc.SetPoint(i-1, sig_eff, bkg_rej)

			canvas = ROOT.TCanvas("%s_roc"%self.name, "%s_roc"%self.name, 800, 800)
			canvas.cd()
			roc.SetLineWidth(2)
			roc.GetXaxis().SetTitle("sig. eff.")
			roc.GetYaxis().SetTitle("bkg. rej.")
			roc.Draw("al")
			if working_points:
				wp_graph.Draw("psame")
			new_out_path = "%s/%s/"%(self.framework.out_path, self.name)
			canvas.SaveAs("%s/roc.root"%(new_out_path))
			canvas.SaveAs("%s/roc.png"%(new_out_path))

			return roc

		def score_cuts_from_wp(self, score, signal_wp, nBins, xmin, xmax):
			self.plot(score, nBins, xmin, xmax, "forWP")
			closest_cuts = []
			best_appr = []
			idx = []
			binWidth = (xmax-xmin)/float(nBins)
			for wp in signal_wp:
				closest_cut = 0
				best_approx = 0
				best_idx = 0
				min_diff = 100
				for i in range(1, nBins+1):
					bin_upper_value = xmin+i*binWidth
					sig_below = 0
					sig_total = 0
					for h in self.signal_hists:
						sig_below = sig_below + h.Integral(1,i)
						sig_total = sig_total + h.Integral()

					if sig_total:
						sig_percentile = sig_below/sig_total
					else:
						sig_percentile = 0

					if abs(wp - sig_percentile)<min_diff:
						min_diff = abs(wp - sig_percentile)
						closest_cut = bin_upper_value
						best_approx = sig_percentile
						best_idx = i
				closest_cuts.append(closest_cut)
				best_appr.append(best_approx)
				idx.append(best_idx)
			print closest_cuts
			return idx

	def set_out_path(self, path):
		self.out_path = path
		try:
			os.makedirs(path)
		except OSError as e:
			if e.errno != errno.EEXIST:
				raise

	def add_mva_source(self, name, title, path):
		new_bdt = self.MVASource(self, name, title, path)
		self.MVASources.append(new_bdt)
		return new_bdt

	class RocCurve(object):
		def __init__(self, graph, name, title, color):
			self.graph=graph
			self.name=name
			self.title=title
			self.color=color
			

	def compare_roc_curves(self, roc_list):
		legend = ROOT.TLegend(0.7, 0.8, 0.895, 0.895)
		canvas = ROOT.TCanvas("roc_curves","roc_curves", 800, 800)
		canvas.cd()
		first = True
		for roc in roc_list:
			roc.graph.SetLineColor(roc.color)
			roc.graph.SetLineWidth(2)
			if first:
				roc.graph.Draw("al")
				first = False
			else:
				roc.graph.Draw("lsame")
			legend.AddEntry(roc.graph, roc.title, "l")
		legend.Draw()
		canvas.SaveAs("%s/roc_curves.root"%(self.out_path))
		canvas.SaveAs("%s/roc_curves.png"%(self.out_path))



roc_to_compare = []

a = Analyzer()
a.set_out_path("plots/mva_output_analyzis")
bdt_old = a.add_mva_source("BDT_UF_V1_old", "BDT_UF_V1_old", "/Users/dmitrykondratyev/ML_output/BDTG_UF_v1/")
bdt_old.add_sample("tt", "ttbar", "tt_ll_POW_BDTG_UF_v1.root", "tree", False, True, ROOT.kYellow)
bdt_old.add_sample("dy", "Drell-Yan", "ZJets_AMC_BDTG_UF_v1.root", "tree", False, True, ROOT.kOrange-3)
bdt_old.add_sample("ggh", "ggH", "H2Mu_gg_BDTG_UF_v1.root", "tree", False, False, ROOT.kRed)
bdt_old.add_sample("vbf", "VBF", "H2Mu_VBF_BDTG_UF_v1.root", "tree", False, False, ROOT.kViolet-1)
bdt_old.add_sample("data", "Data 2017B", "SingleMu2017B_BDTG_UF_v1.root", "tree", True, False, ROOT.kBlack)
bdt_old.set_lumi(4823)
bdt_old_roc_graph = bdt_old.plot_roc("MVA", 500, -1, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
bdt_old_roc = a.RocCurve(bdt_old_roc_graph, "bdt_old", "BDT V1 old", ROOT.kGreen)
roc_to_compare.append(bdt_old_roc)

bdt_mit = a.add_mva_source("BDT_MIT", "BDT_MIT", "/Users/dmitrykondratyev/ML_output/BDTG_MIT/")
bdt_mit.add_sample("tt", "ttbar", "tt_ll_POW_BDTG_MIT.root", "tree", False, True, ROOT.kYellow, True, "0.1")
bdt_mit.add_sample("dy", "Drell-Yan", "ZJets_AMC_BDTG_MIT.root", "tree", False, True, ROOT.kOrange-3)
bdt_mit.add_sample("ggh", "ggH", "H2Mu_gg_BDTG_MIT.root", "tree", False, False, ROOT.kRed)
bdt_mit.add_sample("vbf", "VBF", "H2Mu_VBF_BDTG_MIT.root", "tree", False, False, ROOT.kViolet-1)
bdt_mit.add_sample("data", "Data 2017B", "SingleMu2017B_BDTG_MIT.root", "tree", True, False, ROOT.kBlack)
bdt_mit.set_lumi(4823)
bdt_mit_roc_graph = bdt_mit.plot_roc("MVA", 500, -1, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
bdt_mit_roc = a.RocCurve(bdt_mit_roc_graph, "bdt_mit", "BDT MIT V1", ROOT.kBlack)
roc_to_compare.append(bdt_mit_roc)

dnn_v1 = a.add_mva_source("DNN_v1", "DNN_v1", "/Users/dmitrykondratyev/ML_output/Run_2019-03-29_10-49-20/Keras_multi/model_50_D2_25_D2_25_D2/root/")
dnn_v1.add_sample("tt", "ttbar", "output_t*root", "tree_tt_ll_POW", False, True, ROOT.kYellow, False)
dnn_v1.add_sample("dy", "Drell-Yan", "output_t*root", "tree_ZJets_aMC", False, True, ROOT.kOrange-3, False)
dnn_v1.add_sample("ggh", "ggH", "output_t*root", "tree_H2Mu_gg", False, False, ROOT.kRed, False)
dnn_v1.add_sample("vbf", "VBF", "output_t*root", "tree_H2Mu_VBF", False, False, ROOT.kViolet-1, False)
dnn_v1.add_sample("data", "Data 2017B", "output_Data.root", "tree_Data", True, False, ROOT.kBlack)
dnn_v1.set_lumi(4823)
dnn_v1_roc_graph = dnn_v1.plot_roc("ggH_prediction+VBF_prediction", 500, 0, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
dnn_v1_roc = a.RocCurve(dnn_v1_roc_graph, "dnn_v1", "DNN V1", ROOT.kBlue)
roc_to_compare.append(dnn_v1_roc)

dnn_v2 = a.add_mva_source("DNN_v2", "DNN_v2", "/Users/dmitrykondratyev/ML_output/Run_2019-04-02_20-32-47/Keras_multi/model_50_D2_25_D2_25_D2/root/")
dnn_v2.add_sample("tt", "ttbar", "output_t*root", "tree_tt_ll_POW", False, True, ROOT.kYellow, False)
dnn_v2.add_sample("dy", "Drell-Yan", "output_t*root", "tree_ZJets_aMC", False, True, ROOT.kOrange-3, False)
dnn_v2.add_sample("ggh", "ggH", "output_t*root", "tree_H2Mu_gg", False, False, ROOT.kRed, False)
dnn_v2.add_sample("vbf", "VBF", "output_t*root", "tree_H2Mu_VBF", False, False, ROOT.kViolet-1, False)
dnn_v2.add_sample("data", "Data 2017B", "output_Data.root", "tree_Data", True, False, ROOT.kBlack)
dnn_v2.set_lumi(4823)
dnn_v2_roc_graph = dnn_v2.plot_roc("ggH_prediction+VBF_prediction", 500, 0, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
dnn_v2_roc = a.RocCurve(dnn_v2_roc_graph, "dnn_v2", "DNN V2", ROOT.kRed)
roc_to_compare.append(dnn_v2_roc)

a.compare_roc_curves(roc_to_compare)


# Binary test

dnn_test = a.add_mva_source("DNN_test", "DNN_test", "/tmp/dkondrat/ML_output/Run_2019-04-07_18-53-10/Keras_multi/model_50_D2_25_D2_25_D2/root/")

dnn_test.add_sample("bkg", "Background", "output_t*root", "tree_bkg", False, True, ROOT.kYellow, False)
dnn_test.add_sample("sig", "Signal", "output_t*root", "tree_sig", False, False, ROOT.kRed, False)
dnn_test.set_lumi(4823)
dnn_test_roc_graph = dnn_test.plot_roc("sig_prediction", 500, 0, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
dnn_test_roc = a.RocCurve(dnn_test_roc_graph, "dnn_test", "DNN test", ROOT.kRed)
