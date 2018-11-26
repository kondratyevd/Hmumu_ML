import ROOT
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(1)
from math import *		

def BWZ(x,p):
	if abs(x[0]) > 120 and abs(x[0]) < 130:
		ROOT.TF1.RejectPoint()
	result = p[0]*(exp(p[1]*x[0])*2.4952)/((x[0]-91.1876)**2+(2.4952/2)**2)
	return result

def SimpleExponent(x,p):
	if abs(x[0]) > 120 and abs(x[0]) < 130:
		ROOT.TF1.RejectPoint()
	result = p[0]*exp(p[1]*x[0])
	return result

def Linear(x,p):
	if abs(x[0]) > 120 and abs(x[0]) < 130:
		ROOT.TF1.RejectPoint()
	result = p[0]+p[1]*x[0]
	return result

class Fitter(object):
	def __init__(self):
		self.src_list = []
		self.nBins = 100

	def set_num_bins(self, nBins):
		self.nBins = nBins

	class DNN_Output(object):
		def __init__(self, name, RunID, method, color, cut):
			self.name = name
			self.RunID = RunID
			self.method = method
			self.plot_path = "plots/"
			self.data_path = "output/%s/Keras_multi/%s/root/output_Data.root"%(self.RunID, self.method)
			self.mc_path = "output/%s/Keras_multi/%s/root/output_test.root"%(self.RunID, self.method)
			self.color = color
			self.cut = cut

	class DNN_Score(object):
		def __init__(self, name, expression, xmin, xmax, color):
			self.name = name
			self.expression = expression
			self.xmin = xmin
			self.xmax = xmax
			self.color = color

	def add_data_src(self, name, RunID, method, color, cut):
		obj = self.DNN_Output(name, RunID, method, color, cut)
		self.src_list.append(obj)
		print "%s: Method %s added for %s"%(name, method, RunID)
		return obj
			
	def get_mass_hist(self, name, data_src, path, tree_name, nBins, xmin, xmax, normalize = True):
		data = ROOT.TChain(tree_name)
		data.Add(path)	
		hist_name = name
		data_hist = ROOT.TH1D(hist_name, hist_name, 	nBins, xmin, xmax)
		data_hist.SetLineColor(ROOT.kBlack)
		data_hist.SetMarkerStyle(20)
		data_hist.SetMarkerSize(0.8)
		dummy = ROOT.TCanvas("dummy", "dummy", 800, 800)
		dummy.cd()
		data.Draw("mass>>%s"%(hist_name), data_src.cut)
		dummy.Close()
		if normalize:
			data_hist.Scale(1/data_hist.Integral())
		return data_hist, data

	def fit_mass(self, src):
		canv = ROOT.TCanvas("canv", "canv", 800, 800)
		canv.cd()

		mass_hist, tree = self.get_mass_hist("data_fit", src, src.data_path, "tree_Data", 40, 110, 150, normalize=False)
		mass_hist.Draw("pe")
		fit = ROOT.TF1("fit",BWZ,110,160,2)
		fit.SetParameter(0,0.218615)
		fit.SetParameter(1,-0.001417)
		mass_hist.Fit(fit,"","",110,160)
		# tree.UnbinnedFit("fit","mass","(mass>110)&(mass<150)")
		fit.Draw("samel")
		canv.Print("plots/bkg_fit/test.png")
		return fit

	def make_hist_from_fit(self, src, func, nBins, xmin, xmax):
		hist = ROOT.TH1D("background", "", nBins, xmin, xmax)
		bin_width = (xmax - xmin) / float(nBins)
		for i in range(nBins):
			xi = xmin + (0.5+i)*bin_width
			yi = func.Eval(xi)
			hist.SetBinContent(i+1, yi)
			hist.SetBinError(i+1, 0)
		canv = ROOT.TCanvas("canv1", "canv1", 800, 800)
		canv.cd()
		hist.Draw("hist")
		canv.Print("plots/bkg_fit/test_bkg.png")
		return hist


p = Fitter()
v3 = p.add_data_src("V3", "Run_2018-11-08_09-49-45", "model_50_D2_25_D2_25_D2", ROOT.kGreen+2	,"(mass<120)||(mass>130)")
data_obs = p.add_data_src("V3", "Run_2018-11-08_09-49-45", "model_50_D2_25_D2_25_D2", ROOT.kGreen+2	,"(mass>120)&(mass<130)")
sig_weigted = p.add_data_src("V3", "Run_2018-11-08_09-49-45", "model_50_D2_25_D2_25_D2", ROOT.kGreen+2	,"((mass>120)&(mass<130))*weight*5")

fit_function = p.fit_mass(v3)

bkg_from_fit = p.make_hist_from_fit(v3, fit_function, 10, 120, 130)
data_obs_hist, data_obs_tree = p.get_mass_hist("data_obs", data_obs, data_obs.data_path, "tree_Data", 10, 120, 130, normalize=False)
signal_hist, signal_tree = p.get_mass_hist("signal", sig_weigted, sig_weigted.mc_path, "tree_H2Mu_gg", 10, 120, 130, normalize=False)

out_file = ROOT.TFile.Open("combine/test/test_input.root", "recreate")
bkg_from_fit.Write()
data_obs_hist.Write()
signal_hist.Write()
out_file.Close()

canv = ROOT.TCanvas("canv2", "canv2", 800, 800)
canv.cd()
canv.SetLogy()
bkg_from_fit.Draw("hist")
data_obs_hist.Draw("pesame")
signal_hist.Draw("histsame")
signal_hist.SetLineColor(ROOT.kRed)
print "Expected yields:"
print "		signal:      %f events"%signal_hist.Integral()
print "		background:  %f events"%bkg_from_fit.Integral()
bkg_from_fit.GetYaxis().SetRangeUser(0.1, 100000)
canv.Print("plots/bkg_fit/test_bs.png")



