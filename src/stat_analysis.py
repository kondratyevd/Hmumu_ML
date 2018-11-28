import ROOT
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(1)
from math import *		

def BWZ(x,p):
	if abs(x[0]) > 120 and abs(x[0]) < 130:
		ROOT.TF1.RejectPoint()
	result = p[0]*(exp(p[1]*x[0])*2.4952)/((x[0]-91.1876)**2+(2.4952/2)**2)
	return result

class Analyzer(object):
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

	def fit_mass_unbinned(self, data_src, signal_src):
		mass_hist, tree = self.get_mass_hist("data_fit", data_src, data_src.data_path, "tree_Data", 40, 110, 150, normalize=False)
		signal_hist, signal_tree = self.get_mass_hist("signal", signal_src, signal_src.mc_path, "tree_H2Mu_gg", 10, 110, 150, normalize=False)
		var = ROOT.RooRealVar("mass","Dilepton mass",110,150)
		var.setBins(100)
		var.setRange("left",110,120+0.1)
		var.setRange("right",130-0.1,150)
		var.setRange("full",110,150)
		var.setRange("window",120,130)
		ds = ROOT.RooDataSet("data_sidebands","data_sidebands", tree, ROOT.RooArgSet(var), "(mass<120)||(mass>130)")
		data_obs = ROOT.RooDataSet("data_obs","data_obs", tree, ROOT.RooArgSet(var), "(mass>120)&(mass<130)")
		signal_ds = ROOT.RooDataSet("signal_ds","signal_ds", signal_tree, ROOT.RooArgSet(var), "")

		w = ROOT.RooWorkspace("w", False) 
		Import = getattr(ROOT.RooWorkspace, 'import')
		var_window = ROOT.RooRealVar("mass","Dilepton mass",120,130)
		Import(w, var_window)
		Import(w, ds)
		Import(w, data_obs)
		Import(w, signal_ds)

		w.factory("a1 [1.39, 0.7, 2.1]")
		w.factory("a2 [0.46, 0.30, 0.62]")
		w.factory("a3 [-0.26, -0.40, -0.12]")

		w.factory("c1 [0.73, 0.7, 0.75]")
		w.factory("c2 [0.23, 0.2, 0.25]")
		w.factory("c3 [0.04, 0.02, 0.06]")

		w.factory("EXPR::bwz_redux_f('(@1*(@0/100)+@2*(@0/100)^2)',{mass, a2, a3})")
		w.factory("EXPR::background('exp(@2)*(2.5)/(pow(@0-91.2,@1)+pow(2.5/2,@1))',{mass, a1, bwz_redux_f})")
		w.factory("Gaussian::g1(mass,mean1[124.8, 120, 130],width1[1.52,1.5,1.6])")
		w.factory("Gaussian::g2(mass,mean2[122.8, 120, 130],width2[4.24,4.2,4.3])")
		w.factory("Gaussian::g3(mass,mean3[126,   120, 130],width3[2.1, 2,  2.2])")

		w.factory("EXPR::signal('@0*@1+@2*@3+@4*@5',{c1,g1,c2,g2,c3,g3})")
				
		
		# bkg fit
		
		fit_func = w.pdf('background')
		r = fit_func.fitTo(ds, ROOT.RooFit.Range("left,right"),ROOT.RooFit.Save(), ROOT.RooFit.Verbose(False))
		r.Print()
		canv = ROOT.TCanvas("canv1", "canv1", 800, 800)
		canv.cd()
		frame = var.frame()
		ds.plotOn(frame, ROOT.RooFit.Range("left, right"))
		fit_func.plotOn(frame,ROOT.RooFit.Range("full"))
		frame.Draw()
		canv.Print("plots/bkg_fit/unbinned_fit_bwzredux.png")


		# signal fit

		fit_func_signal = w.pdf('signal')
		r1 = fit_func_signal.fitTo(signal_ds, ROOT.RooFit.Range("window"),ROOT.RooFit.Save(), ROOT.RooFit.Verbose(False))
		r1.Print()
		canv = ROOT.TCanvas("canv2", "canv2", 800, 800)
		canv.cd()
		frame = var.frame()
		signal_ds.plotOn(frame)
		fit_func_signal.plotOn(frame, ROOT.RooFit.Range("window"))
		frame.Draw()
		canv.Print("plots/bkg_fit/unbinned_fit_signal.png")

		w.Print()

		out_file = ROOT.TFile.Open("plots/bkg_fit/workspace.root", "recreate")
		out_file.cd()
		w.Write()
		out_file.Close()



a = Analyzer()
v3 = a.add_data_src("V3", "Run_2018-11-08_09-49-45", "model_50_D2_25_D2_25_D2", ROOT.kGreen+2	,"(mass<120)||(mass>130)")
data_obs = a.add_data_src("V3", "Run_2018-11-08_09-49-45", "model_50_D2_25_D2_25_D2", ROOT.kGreen+2	,"(mass>120)&(mass<130)")
sig_weigted = a.add_data_src("V3", "Run_2018-11-08_09-49-45", "model_50_D2_25_D2_25_D2", ROOT.kGreen+2	,"((mass>120)&(mass<130))*weight*5")

fit_function = a.fit_mass(v3)

a.fit_mass_unbinned(v3, sig_weigted)

bkg_from_fit = a.make_hist_from_fit(v3, fit_function, 10, 120, 130)
data_obs_hist, data_obs_tree = a.get_mass_hist("data_obs", data_obs, data_obs.data_path, "tree_Data", 10, 120, 130, normalize=False)
signal_hist, signal_tree = a.get_mass_hist("signal", sig_weigted, sig_weigted.mc_path, "tree_H2Mu_gg", 10, 120, 130, normalize=False)

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



