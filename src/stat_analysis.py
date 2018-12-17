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
        data_hist = ROOT.TH1D(hist_name, hist_name,     nBins, xmin, xmax)
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

    def binned_mass_fit(self, src):
        canv = ROOT.TCanvas("canv", "canv", 800, 800)
        canv.cd()
        src.cut = "(mass<120)||(mass>130)"
        mass_hist, tree = self.get_mass_hist("data_fit", src, src.data_path, "tree_Data", 40, 110, 150, normalize=False)
        mass_hist.Draw("pe")
        fit = ROOT.TF1("fit",BWZ,110,150,2)
        fit.SetParameter(0,0.218615)
        fit.SetParameter(1,-0.001417)
        mass_hist.Fit(fit,"","",110,150)
        fit.Draw("samel")
        canv.Print("combine/test/test.png")
        src.cut = ""
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
        canv.Print("combine/test/test_bkg.png")
        return hist

    def plot_unbinned_fit_bkg(self, data_src):
        mass_hist, tree = self.get_mass_hist("data_fit", data_src, data_src.data_path, "tree_Data", 40, 110, 150, normalize=False)

        var = ROOT.RooRealVar("mass","Dilepton mass",110,150)       
        var.setBins(100)
        var.setRange("left",110,120+0.1)
        var.setRange("right",130-0.1,150)
        var.setRange("full",110,150)
        var.setRange("window",120,130)

        data = ROOT.RooDataSet("data_sidebandata","data", tree, ROOT.RooArgSet(var), "")

        w_sidebands = ROOT.RooWorkspace("w_sb", False) 

        Import = getattr(ROOT.RooWorkspace, 'import')
        
        Import(w_sidebands, var)
        Import(w_sidebands, data)

        w_sidebands.factory("a1 [1.66, 0.7, 2.1]")
        w_sidebands.factory("a2 [0.39, 0.30, 0.62]")
        w_sidebands.factory("a3 [-0.26, -0.40, -0.12]")
        w_sidebands.factory("expr::bwz_redux_f('(@1*(@0/100)+@2*(@0/100)^2)',{mass, a2, a3})")
        w_sidebands.factory("EXPR::background('exp(@2)*(2.5)/(pow(@0-91.2,@1)+pow(2.5/2,@1))',{mass, a1, bwz_redux_f})")
        
        fit_func = w_sidebands.pdf('background')
        r = fit_func.fitTo(data, ROOT.RooFit.Range("left,right"),ROOT.RooFit.Save(), ROOT.RooFit.Verbose(False))
        r.Print()

        frame = var.frame()
        data.plotOn(frame, ROOT.RooFit.Range("left, right"))
        fit_func.plotOn(frame,ROOT.RooFit.Range("full"))

        canv = ROOT.TCanvas("canv1", "canv1", 800, 800)
        canv.cd()
        frame.Draw()
        canv.Print("combine/test/unbinned_fit_bwzredux.png")

        integral_sb = fit_func.createIntegral(ROOT.RooArgSet(var), ROOT.RooFit.Range("left,right"))
        integral_full = fit_func.createIntegral(ROOT.RooArgSet(var), ROOT.RooFit.Range("full"))
        func_int_sb = integral_sb.getVal()
        func_int_full = integral_full.getVal()
        data_int_sb = data.sumEntries("1","left,right")
        data_int_full = data.sumEntries("1","full")
        bkg_int_full = data_int_sb * (func_int_full/func_int_sb)
        print "func_int_sb:  ", func_int_sb
        print "func_int_full:  ", func_int_full
        print "data_int_sb: ", data_int_sb
        print "data_int_full: ", data_int_full
        return bkg_int_full

    def plot_unbinned_fit_sig(self, signal_src):
        signal_hist, signal_tree = self.get_mass_hist("signal", signal_src, signal_src.mc_path, "tree_H2Mu_gg", 10, 110, 150, normalize=False)
        var_window = ROOT.RooRealVar("mass","Dilepton mass",110,150)
        var_window.setRange("full",110,150)
        signal_ds = ROOT.RooDataSet("signal_ds","signal_ds", signal_tree, ROOT.RooArgSet(var_window), "")

        Import = getattr(ROOT.RooWorkspace, 'import')
        w_0 = ROOT.RooWorkspace("w0", False)
        Import(w_0, var_window)

        mixGG = ROOT.RooRealVar("mixGG",  "mixGG", 0.5,0.,1.)
        mixGG1 = ROOT.RooRealVar("mixGG1",  "mixGG1", 0.5,0.,1.)

        mu_res_beta = ROOT.RooRealVar('mu_res_beta','mu_res_beta',0,0,0)
        mu_scale_beta = ROOT.RooRealVar('mu_scale_beta','mu_scale_beta',0,0,0)

        Import(w_0, mu_res_beta)
        Import(w_0, mu_scale_beta)

        res_uncert = 0.1
        mu_res_uncert = ROOT.RooRealVar('mu_res_uncert','mu_res_uncert',res_uncert)

        scale_uncert = 0.0005
        mu_scale_uncert = ROOT.RooRealVar('mu_scale_uncert','mu_scale_uncert',scale_uncert)

        mu_res_uncert.setConstant()
        mu_scale_uncert.setConstant()

        Import(w_0, mu_res_uncert)
        Import(w_0, mu_scale_uncert)
        
        w_0.factory("EXPR::mean1_times_nuis('mean1*(1 + mu_scale_uncert*mu_scale_beta)',{mean1[125.0, 120., 130.],mu_scale_uncert,mu_scale_beta})")
        w_0.factory("EXPR::mean2_times_nuis('mean2*(1 + mu_scale_uncert*mu_scale_beta)',{mean2[125.0, 120., 130.],mu_scale_uncert,mu_scale_beta})")
        w_0.factory("EXPR::mean3_times_nuis('mean3*(1 + mu_scale_uncert*mu_scale_beta)',{mean3[125.0, 120., 130.],mu_scale_uncert,mu_scale_beta})")

        w_0.factory("expr::deltaM21('mean2-mean1',{mean2, mean1})")
        w_0.factory("expr::deltaM31('mean3-mean1',{mean3, mean1})")

        w_0.factory("EXPR::mean2_final('mean2_times_nuis + mu_res_uncert*mu_res_beta*deltaM21',{mean2_times_nuis, mu_res_uncert, mu_res_beta, deltaM21})")
        w_0.factory("EXPR::mean3_final('mean3_times_nuis + mu_res_uncert*mu_res_beta*deltaM31',{mean3_times_nuis, mu_res_uncert, mu_res_beta, deltaM31})")

        w_0.factory("EXPR::width1_times_nuis('width1*(1 + mu_res_uncert*mu_res_beta)',{width1[1.0, 0.5, 5.0],mu_res_uncert,mu_res_beta})")
        w_0.factory("EXPR::width2_times_nuis('width2*(1 + mu_res_uncert*mu_res_beta)',{width2[5.0, 2, 10],mu_res_uncert,mu_res_beta})")
        w_0.factory("EXPR::width3_times_nuis('width3*(1 + mu_res_uncert*mu_res_beta)',{width3[5.0, 1, 10],mu_res_uncert,mu_res_beta})")

        w_0.factory("Gaussian::gaus1(mass, mean1_times_nuis, width1_times_nuis)")
        w_0.factory("Gaussian::gaus2(mass, mean2_final, width2_times_nuis)")
        w_0.factory("Gaussian::gaus3(mass, mean3_final, width3_times_nuis)")

        gaus1 = w_0.pdf('gaus1')
        gaus2 = w_0.pdf('gaus2')
        gaus3 = w_0.pdf('gaus3')
        # smodel = ROOT.RooAddPdf('signal', 'signal', gaus1, gaus2, mixGG)
        gaus12 = ROOT.RooAddPdf('gaus12', 'gaus12', gaus1, gaus2, mixGG)
        smodel = ROOT.RooAddPdf('signal', 'signal', gaus3, gaus12, mixGG1)
        Import(w_0,smodel)

        # w.Print()
        
        signal_ds = ROOT.RooDataSet("signal_ds","signal_ds", signal_tree, ROOT.RooArgSet(var_window), "")
        res = smodel.fitTo(signal_ds, ROOT.RooFit.Range("full"),ROOT.RooFit.Save(), ROOT.RooFit.Verbose(False))
        res.Print()


        frame = var_window.frame()
        signal_ds.plotOn(frame)
        smodel.plotOn(frame, ROOT.RooFit.Range("window"))

        canv = ROOT.TCanvas("canv2", "canv2", 800, 800)
        canv.cd()
        frame.Draw()
        canv.Print("combine/test/unbinned_fit_signal.png")

        return signal_hist.Integral()


    def plot_data_obs(self, data_src):
        var = ROOT.RooRealVar("mass","Dilepton mass",110,150)
        mass_hist, tree = self.get_mass_hist("data_fit", data_src, data_src.data_path, "tree_Data", 40, 110, 150, normalize=False)
        data_obs = ROOT.RooDataSet("data_obs","data_obs", tree, ROOT.RooArgSet(var), "(mass>110)&(mass<150)")
        frame = var.frame()
        data_obs.plotOn(frame)

        canv = ROOT.TCanvas("canv3", "canv3", 800, 800)
        canv.cd()
        frame.Draw()
        canv.Print("combine/test/data_obs.png")

        return mass_hist.Integral()


    def make_workspace(self, data_src, signal_src):
        mass_hist, tree = self.get_mass_hist("data_fit", data_src, data_src.data_path, "tree_Data", 40, 110, 150, normalize=False)
        signal_hist, signal_tree = self.get_mass_hist("signal", signal_src, signal_src.mc_path, "tree_H2Mu_gg", 10, 110, 150, normalize=False)
        var = ROOT.RooRealVar("mass","Dilepton mass",110,150)       
        var.setBins(100)
        var.setRange("window",120,130)
        var.setRange("full",110,150)
        data_obs = ROOT.RooDataSet("data_obs","data_obs", tree, ROOT.RooArgSet(var), "")
        Import = getattr(ROOT.RooWorkspace, 'import')
        w = ROOT.RooWorkspace("w", False)
        Import(w, var)

        w.factory("a1 [1.66, 0.7, 3]")
        w.factory("a2 [0.39, 0.30, 10]")
        w.factory("a3 [-0.26, -10, -0.02]")
        w.factory("expr::bwz_redux_f('(@1*(@0/100)+@2*(@0/100)^2)',{mass, a2, a3})")
        w.factory("EXPR::background('exp(@2)*(2.5)/(pow(@0-91.2,@1)+pow(2.5/2,@1))',{mass, a1, bwz_redux_f})")

        mixGG = ROOT.RooRealVar("mixGG",  "mixGG", 0.5,0.,1.)
        mixGG1 = ROOT.RooRealVar("mixGG1",  "mixGG1", 0.5,0.,1.)

        mu_res_beta = ROOT.RooRealVar('mu_res_beta','mu_res_beta',0,0,0)
        mu_scale_beta = ROOT.RooRealVar('mu_scale_beta','mu_scale_beta',0,0,0)

        Import(w, mu_res_beta)
        Import(w, mu_scale_beta)

        res_uncert = 0.1
        mu_res_uncert = ROOT.RooRealVar('mu_res_uncert','mu_res_uncert',res_uncert)

        scale_uncert = 0.0005
        mu_scale_uncert = ROOT.RooRealVar('mu_scale_uncert','mu_scale_uncert',scale_uncert)

        mu_res_uncert.setConstant()
        mu_scale_uncert.setConstant()

        Import(w, mu_res_uncert)
        Import(w, mu_scale_uncert)
        
        
        w.factory("EXPR::mean1_times_nuis('mean1*(1 + mu_scale_uncert*mu_scale_beta)',{mean1[125.0, 120., 130.],mu_scale_uncert,mu_scale_beta})")
        w.factory("EXPR::mean2_times_nuis('mean2*(1 + mu_scale_uncert*mu_scale_beta)',{mean2[125.0, 120., 130.],mu_scale_uncert,mu_scale_beta})")
        w.factory("EXPR::mean3_times_nuis('mean3*(1 + mu_scale_uncert*mu_scale_beta)',{mean3[125.0, 120., 130.],mu_scale_uncert,mu_scale_beta})")

        w.factory("expr::deltaM21('mean2-mean1',{mean2, mean1})")
        w.factory("expr::deltaM31('mean3-mean1',{mean3, mean1})")

        w.factory("EXPR::mean2_final('mean2_times_nuis + mu_res_uncert*mu_res_beta*deltaM21',{mean2_times_nuis, mu_res_uncert, mu_res_beta, deltaM21})")
        w.factory("EXPR::mean3_final('mean3_times_nuis + mu_res_uncert*mu_res_beta*deltaM31',{mean3_times_nuis, mu_res_uncert, mu_res_beta, deltaM31})")

        w.factory("EXPR::width1_times_nuis('width1*(1 + mu_res_uncert*mu_res_beta)',{width1[1.0, 0.5, 5.0],mu_res_uncert,mu_res_beta})")
        w.factory("EXPR::width2_times_nuis('width2*(1 + mu_res_uncert*mu_res_beta)',{width2[5.0, 2, 10],mu_res_uncert,mu_res_beta})")
        w.factory("EXPR::width3_times_nuis('width3*(1 + mu_res_uncert*mu_res_beta)',{width3[5.0, 1, 10],mu_res_uncert,mu_res_beta})")

        w.factory("Gaussian::gaus1(mass, mean1_times_nuis, width1_times_nuis)")
        w.factory("Gaussian::gaus2(mass, mean2_final, width2_times_nuis)")
        w.factory("Gaussian::gaus3(mass, mean3_final, width3_times_nuis)")

        gaus1 = w.pdf('gaus1')
        gaus2 = w.pdf('gaus2')
        gaus3 = w.pdf('gaus3')
        gaus12 = ROOT.RooAddPdf('gaus12', 'gaus12', gaus1, gaus2, mixGG)
        smodel = ROOT.RooAddPdf('signal', 'signal', gaus3, gaus12, mixGG1)
        Import(w,smodel)

        w.Print()
        
        signal_ds = ROOT.RooDataSet("signal_ds","signal_ds", signal_tree, ROOT.RooArgSet(var), "")
        res = smodel.fitTo(signal_ds, ROOT.RooFit.Range("full"),ROOT.RooFit.Save(), ROOT.RooFit.Verbose(False))
        res.Print()

        sigParamList = ["mean1", "mean2", "width1", "width2", "mixGG"]
        for par in sigParamList:
            par_var = w.var(par)
            par_var.setConstant(True)
        mu_res_beta.setRange(-5,5)
        mu_scale_beta.setRange(-5,5)

        Import(w, smodel)
        Import(w, data_obs)
        out_file = ROOT.TFile.Open("combine/test/workspace.root", "recreate")
        out_file.cd()
        w.Write()
        w.Print()
        out_file.Close()

        frame = var.frame()

        
        bkg = w.pdf("background")
        smodel.plotOn(frame)
        bkg.plotOn(frame)
        canv = ROOT.TCanvas("canv4", "canv4", 800, 800)
        canv.cd()
        frame.Draw()
        canv.Print("combine/test/pdfs.png")

        frame_new = var.frame()
        signal_ds.plotOn(frame_new)
        smodel.plotOn(frame_new)

        canv = ROOT.TCanvas("canv5", "canv5", 800, 800)
        canv.cd()
        frame_new.Draw()
        canv.Print("combine/test/signal_fit.png")


a = Analyzer()
v3 = a.add_data_src("V3", "Run_2018-11-08_09-49-45", "model_50_D2_25_D2_25_D2", ROOT.kGreen+2   ,"")
data_obs = a.add_data_src("V3", "Run_2018-11-08_09-49-45", "model_50_D2_25_D2_25_D2", ROOT.kGreen+2 ,"")
sig_weigted = a.add_data_src("V3", "Run_2018-11-08_09-49-45", "model_50_D2_25_D2_25_D2", ROOT.kGreen+2  ,"(1)*weight*5") # *5 because there are only 20% of MC in test dataset

bkg_integral = a.plot_unbinned_fit_bkg(v3)
sig_integral = a.plot_unbinned_fit_sig(sig_weigted)
data_integral = a.plot_data_obs(v3)
a.make_workspace(v3, sig_weigted)
print "Integrals (unbinned fit):"
print "     signal:      %f events"%sig_integral
print "     background:  %f events"%bkg_integral
print "     data:        %f events"%data_integral

fit_function = a.binned_mass_fit(v3)
bkg_from_fit = a.make_hist_from_fit(v3, fit_function, 40, 110, 150)
data_obs_hist, data_obs_tree = a.get_mass_hist("data_obs", data_obs, data_obs.data_path, "tree_Data", 40, 110, 150, normalize=False)
signal_hist, signal_tree = a.get_mass_hist("signal", sig_weigted, sig_weigted.mc_path, "tree_H2Mu_gg", 40, 110, 150, normalize=False)

out_file = ROOT.TFile.Open("combine/test/test_input.root", "recreate")
bkg_from_fit.Write()
data_obs_hist.Write()
signal_hist.Write()
out_file.Close()

print "Integrals (binned fit):"
print "     signal:      %f events"%signal_hist.Integral()
print "     background:  %f events"%bkg_from_fit.Integral()
print "     data:        %f events"%data_obs_hist.Integral()

canv = ROOT.TCanvas("canv2", "canv2", 800, 800)
canv.cd()
canv.SetLogy()
bkg_from_fit.Draw("hist")
data_obs_hist.Draw("pesame")
signal_hist.Draw("histsame")
signal_hist.SetLineColor(ROOT.kRed)

bkg_from_fit.GetYaxis().SetRangeUser(0.1, 100000)
canv.Print("combine/test/test_bs.png")



