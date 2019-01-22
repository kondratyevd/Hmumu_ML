import ROOT
import os, sys, errno
# ROOT.gStyle.SetOptStat(0)
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
        self.out_dir = "combine/test/"

    def set_num_bins(self, nBins):
        self.nBins = nBins

    def set_out_dir(self, dir):
        self.out_dir = dir
        try:
            os.makedirs(dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    class DNN_Output(object):
        def __init__(self, name, RunID, method, tree_name, color, cut, weight):
            self.name = name
            self.RunID = RunID
            self.method = method
            self.tree_name = tree_name
            self.plot_path = "plots/"
            self.data_path = "output/%s/Keras_multi/%s/root/output_Data.root"%(self.RunID, self.method)
            self.mc_path = "output/%s/Keras_multi/%s/root/output_train.root"%(self.RunID, self.method)
            self.color = color
            self.cut = cut
            self.weight = weight

    class DNN_Score(object):
        def __init__(self, name, expression, xmin, xmax, color):
            self.name = name
            self.expression = expression
            self.xmin = xmin
            self.xmax = xmax
            self.color = color



    def add_data_src(self, name, RunID, method, tree_name, color, cut, weight):
        obj = self.DNN_Output(name, RunID, method, tree_name, color, cut, weight)
        self.src_list.append(obj)
        print "%s: Method %s added for %s"%(name, method, RunID)
        return obj
            
    def get_mass_hist(self, name, src, path, additional_cuts, nBins, xmin, xmax, normalize = True):
        data = ROOT.TChain(src.tree_name)
        data.Add(path)  
        hist_name = name
        data_hist = ROOT.TH1D(hist_name, hist_name, nBins, xmin, xmax)
        data_hist.SetLineColor(ROOT.kBlack)
        data_hist.SetMarkerStyle(20)
        data_hist.SetMarkerSize(0.8)
        dummy = ROOT.TCanvas("dummy", "dummy", 800, 800)
        dummy.cd()
        data.Draw("mass>>%s"%(hist_name), "("+src.cut+additional_cuts+")*"+src.weight)
        dummy.Close()
        if normalize:
            data_hist.Scale(1/data_hist.Integral())
        return data_hist, data

    def binned_mass_fit(self, src):

        mass_hist, tree = self.get_mass_hist("data_fit", src, src.data_path, "&((mass<120)||(mass>130))", 40, 110, 150, normalize=False)
        fit = ROOT.TF1("fit",BWZ,110,150,2)
        fit.SetParameter(0,0.218615)
        fit.SetParameter(1,-0.001417)
        mass_hist.Fit(fit,"","",110,150)

        canv = ROOT.TCanvas("canv", "canv", 800, 800)
        canv.cd()
        mass_hist.Draw("pe")
        canv.Print(self.out_dir+"test.png")

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
        canv.Print(self.out_dir+"test_bkg.png")
        return hist

    def plot_unbinned_fit_bkg(self, data_src):
        mass_hist, tree = self.get_mass_hist("data_fit", data_src, data_src.data_path, "", 40, 110, 150, normalize=False)

        var = ROOT.RooRealVar("mass","Dilepton mass",110,150) 
        max_abs_eta_var = ROOT.RooRealVar("max_abs_eta_mu","Max abs(eta) of muons", 0, 2.4) 
        ggH_prediction_var = ROOT.RooRealVar("ggH_prediction", "ggH_prediction", 0, 1)
        VBF_prediction_var = ROOT.RooRealVar("VBF_prediction", "VBF_prediction", 0, 1)
        DY_prediction_var = ROOT.RooRealVar("DY_prediction", "DY_prediction", 0, 1)
        ttbar_prediction_var = ROOT.RooRealVar("ttbar_prediction", "ttbar_prediction", 0, 1)      
        var.setBins(100)
        var.setRange("left",110,120+0.1)
        var.setRange("right",130-0.1,150)
        var.setRange("full",110,150)
        var.setRange("window",120,130)

        data = ROOT.RooDataSet("data_sidebandata","data", tree, ROOT.RooArgSet(var, max_abs_eta_var, ggH_prediction_var, VBF_prediction_var, DY_prediction_var, ttbar_prediction_var), data_src.cut)

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
        canv.Print(self.out_dir+"unbinned_fit_bwzredux.png")

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

    def plot_data_obs(self, data_src):
        var = ROOT.RooRealVar("mass","Dilepton mass",110,150)
        max_abs_eta_var = ROOT.RooRealVar("max_abs_eta_mu","Max abs(eta) of muons", 0, 2.4)
        ggH_prediction_var = ROOT.RooRealVar("ggH_prediction", "ggH_prediction", 0, 1)
        VBF_prediction_var = ROOT.RooRealVar("VBF_prediction", "VBF_prediction", 0, 1)
        DY_prediction_var = ROOT.RooRealVar("DY_prediction", "DY_prediction", 0, 1)
        ttbar_prediction_var = ROOT.RooRealVar("ttbar_prediction", "ttbar_prediction", 0, 1) 

        mass_hist, tree = self.get_mass_hist("data_fit", data_src, data_src.data_path, "", 40, 110, 150, normalize=False)
        data_obs = ROOT.RooDataSet("data_obs","data_obs", tree, ROOT.RooArgSet(var, max_abs_eta_var, ggH_prediction_var, VBF_prediction_var, DY_prediction_var, ttbar_prediction_var), data_src.cut)
        frame = var.frame()
        data_obs.plotOn(frame)

        canv = ROOT.TCanvas("canv3", "canv3", 800, 800)
        canv.cd()
        frame.Draw()
        canv.Print(self.out_dir+"data_obs.png")

        return mass_hist.Integral()


    def make_workspace(self, data_src, signal_src):
        mass_hist, tree = self.get_mass_hist("data_fit", data_src, data_src.data_path, "", 40, 110, 150, normalize=False)
        signal_hist, signal_tree = self.get_mass_hist("signal", signal_src, signal_src.mc_path, "", 10, 110, 150, normalize=False)
        var = ROOT.RooRealVar("mass","Dilepton mass",110,150)     
        max_abs_eta_var = ROOT.RooRealVar("max_abs_eta_mu","Max abs(eta) of muons", 0, 2.4) 
        ggH_prediction_var = ROOT.RooRealVar("ggH_prediction", "ggH_prediction", 0, 1)
        VBF_prediction_var = ROOT.RooRealVar("VBF_prediction", "VBF_prediction", 0, 1)
        DY_prediction_var = ROOT.RooRealVar("DY_prediction", "DY_prediction", 0, 1)
        ttbar_prediction_var = ROOT.RooRealVar("ttbar_prediction", "ttbar_prediction", 0, 1) 

        var.setBins(100)
        var.setRange("window",120,130)
        var.setRange("full",110,150)
        data_obs = ROOT.RooDataSet("data_obs","data_obs", tree, ROOT.RooArgSet(var, max_abs_eta_var, ggH_prediction_var, VBF_prediction_var, DY_prediction_var, ttbar_prediction_var), data_src.cut)
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

        # mu_res_beta = ROOT.RooRealVar('mu_res_beta','mu_res_beta',0,0,0)
        # mu_scale_beta = ROOT.RooRealVar('mu_scale_beta','mu_scale_beta',0,0,0)

        # Import(w, mu_res_beta)
        # Import(w, mu_scale_beta)

        # res_uncert = 0.1
        # mu_res_uncert = ROOT.RooRealVar('mu_res_uncert','mu_res_uncert',res_uncert)

        # scale_uncert = 0.0005
        # mu_scale_uncert = ROOT.RooRealVar('mu_scale_uncert','mu_scale_uncert',scale_uncert)

        # mu_res_uncert.setConstant()
        # mu_scale_uncert.setConstant()

        # Import(w, mu_res_uncert)
        # Import(w, mu_scale_uncert)
              
        # w.factory("EXPR::mean1_times_nuis('mean1*(1 + mu_scale_uncert*mu_scale_beta)',{mean1[125.0, 120., 130.],mu_scale_uncert,mu_scale_beta})")
        # w.factory("EXPR::mean2_times_nuis('mean2*(1 + mu_scale_uncert*mu_scale_beta)',{mean2[125.0, 120., 130.],mu_scale_uncert,mu_scale_beta})")
        # w.factory("EXPR::mean3_times_nuis('mean3*(1 + mu_scale_uncert*mu_scale_beta)',{mean3[125.0, 120., 130.],mu_scale_uncert,mu_scale_beta})")

        # w.factory("expr::deltaM21('mean2-mean1',{mean2, mean1})")
        # w.factory("expr::deltaM31('mean3-mean1',{mean3, mean1})")

        # w.factory("EXPR::mean2_final('mean2_times_nuis + mu_res_uncert*mu_res_beta*deltaM21',{mean2_times_nuis, mu_res_uncert, mu_res_beta, deltaM21})")
        # w.factory("EXPR::mean3_final('mean3_times_nuis + mu_res_uncert*mu_res_beta*deltaM31',{mean3_times_nuis, mu_res_uncert, mu_res_beta, deltaM31})")

        # w.factory("EXPR::width1_times_nuis('width1*(1 + mu_res_uncert*mu_res_beta)',{width1[1.0, 0.5, 5.0],mu_res_uncert,mu_res_beta})")
        # w.factory("EXPR::width2_times_nuis('width2*(1 + mu_res_uncert*mu_res_beta)',{width2[5.0, 2, 10],mu_res_uncert,mu_res_beta})")
        # w.factory("EXPR::width3_times_nuis('width3*(1 + mu_res_uncert*mu_res_beta)',{width3[5.0, 1, 10],mu_res_uncert,mu_res_beta})")

        # w.factory("Gaussian::gaus1(mass, mean1_times_nuis, width1_times_nuis)")
        # w.factory("Gaussian::gaus2(mass, mean2_final, width2_times_nuis)")
        # w.factory("Gaussian::gaus3(mass, mean3_final, width3_times_nuis)")

        # gaus1 = w.pdf('gaus1')
        # gaus2 = w.pdf('gaus2')
        # gaus3 = w.pdf('gaus3')
        # gaus12 = ROOT.RooAddPdf('gaus12', 'gaus12', gaus1, gaus2, mixGG)
        # smodel = ROOT.RooAddPdf('signal', 'signal', gaus3, gaus12, mixGG1)
        # Import(w,smodel)

        w.factory("Gaussian::gaus1(mass, mean1[125., 120., 130.], width1[1.0, 0.5, 5.0])")
        w.factory("Gaussian::gaus2(mass, mean2[125., 120., 130.], width2[5.0, 2.0, 10.])")
        w.factory("Gaussian::gaus3(mass, mean3[125., 120., 130.], width3[5.0, 1.0, 10.])")
        gaus1 = w.pdf('gaus1')
        gaus2 = w.pdf('gaus2')
        gaus3 = w.pdf('gaus3')
        gaus12 = ROOT.RooAddPdf('gaus12', 'gaus12', gaus1, gaus2, mixGG)

        smodel = ROOT.RooAddPdf('signal', 'signal', gaus3, gaus12, mixGG1)
        Import(w,smodel)
        w.Print()
        
        signal_ds = ROOT.RooDataSet("signal_ds","signal_ds", signal_tree, ROOT.RooArgSet(var, max_abs_eta_var, ggH_prediction_var, VBF_prediction_var, DY_prediction_var, ttbar_prediction_var), signal_src.cut)
        res = smodel.fitTo(signal_ds, ROOT.RooFit.Range("full"),ROOT.RooFit.Save(), ROOT.RooFit.Verbose(False))
        res.Print()

        sigParamList = ["mean1", "mean2", "mean3", "width1", "width2", "width3", "mixGG", "mixGG1"]
        for par in sigParamList:
            par_var = w.var(par)
            par_var.setConstant(True)
        # mu_res_beta.setRange(-5,5)
        # mu_scale_beta.setRange(-5,5)

        Import(w, smodel)
        Import(w, data_obs)
        out_file = ROOT.TFile.Open(self.out_dir+"workspace.root", "recreate")
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
        canv.Print(self.out_dir+"pdfs.png")

        frame_new = var.frame()
        signal_ds.plotOn(frame_new, ROOT.RooFit.Name("signal_ds"))
        smodel.plotOn(frame_new, ROOT.RooFit.Name("signal_3gaus"))

        canv = ROOT.TCanvas("canv5", "canv5", 800, 800)
        canv.cd()
        frame_new.Draw()
        canv.Print(self.out_dir+"signal_fit.png")

        print "3Gaus chi2/d.o.f: ", frame_new.chiSquare("signal_3gaus", "signal_ds", 8)

    def make_workspace_DCB(self, data_src, signal_src):
        mass_hist, tree = self.get_mass_hist("data_fit", data_src, data_src.data_path, "", 40, 110, 150, normalize=False)
        signal_hist, signal_tree = self.get_mass_hist("signal", signal_src, signal_src.mc_path, "", 10, 110, 150, normalize=False)
        var = ROOT.RooRealVar("mass","Dilepton mass",110,150)     
        max_abs_eta_var = ROOT.RooRealVar("max_abs_eta_mu","Max abs(eta) of muons", 0, 2.4) 
        ggH_prediction_var = ROOT.RooRealVar("ggH_prediction", "ggH_prediction", 0, 1)
        VBF_prediction_var = ROOT.RooRealVar("VBF_prediction", "VBF_prediction", 0, 1)
        DY_prediction_var = ROOT.RooRealVar("DY_prediction", "DY_prediction", 0, 1)
        ttbar_prediction_var = ROOT.RooRealVar("ttbar_prediction", "ttbar_prediction", 0, 1) 

        var.setBins(100)
        var.setRange("window",120,130)
        var.setRange("full",110,150)
        data_obs = ROOT.RooDataSet("data_obs","data_obs", tree, ROOT.RooArgSet(var, max_abs_eta_var, ggH_prediction_var, VBF_prediction_var, DY_prediction_var, ttbar_prediction_var), data_src.cut)
        Import = getattr(ROOT.RooWorkspace, 'import')
        w = ROOT.RooWorkspace("w", False)
        Import(w, var)
        
        w.factory("a1 [1.66, 0.7, 3]")
        w.factory("a2 [0.39, 0.30, 10]")
        w.factory("a3 [-0.26, -10, -0.02]")
        w.factory("expr::bwz_redux_f('(@1*(@0/100)+@2*(@0/100)^2)',{mass, a2, a3})")
        w.factory("EXPR::background('exp(@2)*(2.5)/(pow(@0-91.2,@1)+pow(2.5/2,@1))',{mass, a1, bwz_redux_f})")

        # mu_res_beta = ROOT.RooRealVar('mu_res_beta','mu_res_beta',0,0,0)
        # mu_scale_beta = ROOT.RooRealVar('mu_scale_beta','mu_scale_beta',0,0,0)
        # Import(w, mu_res_beta)
        # Import(w, mu_scale_beta)
        # res_uncert = 0.1
        # mu_res_uncert = ROOT.RooRealVar('mu_res_uncert','mu_res_uncert',res_uncert)
        # scale_uncert = 0.0005
        # mu_scale_uncert = ROOT.RooRealVar('mu_scale_uncert','mu_scale_uncert',scale_uncert)
        # mu_res_uncert.setConstant()
        # mu_scale_uncert.setConstant()
        # Import(w, mu_res_uncert)
        # Import(w, mu_scale_uncert)

        # w.factory("expr::mean('mean*(mu_scale_uncert*mu_scale_beta)',{mean[125,120,130], mu_scale_uncert, mu_scale_beta})")
        # w.factory("expr::sigma('sigma*(mu_res_uncert*mu_res_beta)',{sigma[2,0,20], mu_res_uncert, mu_res_beta})")

        # ROOT.gROOT.ProcessLine(".L src/RooDCBShape.cxx")
        ROOT.gSystem.Load("src/RooDCBShape_cxx.so")
        mean = w.factory("mean[125,120,130]")
        sigma = w.factory("sigma[2,0,20]")
        alphaL = w.factory("alphaL[2,0,25]")
        alphaR = w.factory("alphaR[2,0,25]")
        nL = w.factory("nL[1.5,0,25]")
        nR = w.factory("nR[1.5,0,25]")

        w.factory("RooDCBShape::signal(mass, mean[125,120,130], sigma[2,0,20], alphaL[2,0,25] , alphaR[2,0,25], nL[1.5,0,25], nR[1.5,0,25])")
        smodel = w.pdf("signal")
        smodel.Print()        
        signal_ds = ROOT.RooDataSet("signal_ds","signal_ds", signal_tree, ROOT.RooArgSet(var, max_abs_eta_var, ggH_prediction_var, VBF_prediction_var, DY_prediction_var, ttbar_prediction_var), signal_src.cut)
        res = smodel.fitTo(signal_ds, ROOT.RooFit.Range("full"),ROOT.RooFit.Save(), ROOT.RooFit.Verbose(False))
        res.Print()

        sigParamList = ["mean", "sigma", "alphaL", "alphaR", "nL", "nR"]
        for par in sigParamList:
            par_var = w.var(par)
            par_var.setConstant(True)
        # w.var("mu_scale_beta").setRange(-5,5)
        # w.var("mu_res_beta").setRange(-5,5)

        Import(w, smodel)
        Import(w, data_obs)
        out_file = ROOT.TFile.Open(self.out_dir+"workspace_DCB.root", "recreate")
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
        canv.Print(self.out_dir+"pdfs_DCB.png")

        frame_new = var.frame()
        signal_ds.plotOn(frame_new, ROOT.RooFit.Name("signal_ds"))
        smodel.plotOn(frame_new, ROOT.RooFit.Name("signal_DCB"))

        canv = ROOT.TCanvas("canv5", "canv5", 800, 800)
        canv.cd()
        frame_new.Draw()
        canv.Print(self.out_dir+"signal_fit_DCB.png")

        print "DCB chi2/d.o.f: ", frame_new.chiSquare("signal_DCB", "signal_ds", 8)


    def full_plotting_sequence(self, data_src, signal_src):
       
        bkg_integral = self.plot_unbinned_fit_bkg(data_src)
        signal_hist, signal_tree = self.get_mass_hist("signal", signal_src, signal_src.mc_path, "", 10, 110, 150, normalize=False)
        data_hist, data_obs_tree = self.get_mass_hist("data_obs", data_src, data_src.data_path, "", 40, 110, 150, normalize=False)

        sig_integral = signal_hist.Integral()
        data_integral = data_hist.Integral()

        self.plot_data_obs(data_src)
        self.make_workspace(data_src, signal_src)
        self.make_workspace_DCB(data_src, signal_src)
        print "Integrals (unbinned fit):"
        print "     signal:      %f events"%sig_integral
        print "     background:  %f events"%bkg_integral
        print "     data:        %f events"%data_integral
        
        fit_function = self.binned_mass_fit(data_src)
        bkg_from_fit = self.make_hist_from_fit(data_src, fit_function, 40, 110, 150)
        
        out_file = ROOT.TFile.Open(self.out_dir+"test_input.root", "recreate")
        bkg_from_fit.Write()
        data_hist.Write()
        signal_hist.Write()
        out_file.Close()
        
        print "Integrals (binned fit):"
        print "     signal:      %f events"%sig_integral
        print "     background:  %f events"%bkg_from_fit.Integral()
        print "     data:        %f events"%data_integral
        
        canv = ROOT.TCanvas("canv2", "canv2", 800, 800)
        canv.cd()
        canv.SetLogy()
        bkg_from_fit.Draw("hist")
        data_hist.Draw("pesame")
        signal_hist.Draw("histsame")
        signal_hist.SetLineColor(ROOT.kRed)
        
        bkg_from_fit.GetYaxis().SetRangeUser(0.1, 100000)
        canv.Print(self.out_dir+"test_bs.png")

    def signal_fit_DCB(self, signal_src, name, label, xmin, xmax):
        signal_hist, signal_tree = self.get_mass_hist("signal", signal_src, signal_src.mc_path, "", 10, xmin, xmax, normalize=False)
        var = ROOT.RooRealVar("mass","Dilepton mass",xmin,xmax)
        max_abs_eta_var = ROOT.RooRealVar("max_abs_eta_mu","Max abs(eta) of muons", 0, 2.4)
        ggH_prediction_var = ROOT.RooRealVar("ggH_prediction", "ggH_prediction", 0, 1)
        VBF_prediction_var = ROOT.RooRealVar("VBF_prediction", "VBF_prediction", 0, 1)
        DY_prediction_var = ROOT.RooRealVar("DY_prediction", "DY_prediction", 0, 1)
        ttbar_prediction_var = ROOT.RooRealVar("ttbar_prediction", "ttbar_prediction", 0, 1) 

        var.setRange("full",xmin,xmax)

        Import = getattr(ROOT.RooWorkspace, 'import')
        w = ROOT.RooWorkspace("w0", False)
        Import(w, var)
        
        ROOT.gROOT.ProcessLine(".L src/RooDCBShape.cxx")
        w.factory("RooDCBShape::cb(mass, mean[125,120,130], sigma[2,0,20], alphaL[2,0,25] , alphaR[2,0,25], nL[1.5,0,25], nR[1.5,0,25])")
        smodel = w.pdf("cb")
        
        signal_ds = ROOT.RooDataSet("signal_ds","signal_ds", signal_tree, ROOT.RooArgSet(var, max_abs_eta_var, ggH_prediction_var, VBF_prediction_var, DY_prediction_var, ttbar_prediction_var), signal_src.cut)
        res = smodel.fitTo(signal_ds, ROOT.RooFit.Range("full"),ROOT.RooFit.Save(), ROOT.RooFit.Verbose(False))
        res.Print("v")


        frame = var.frame()
        signal_ds.plotOn(frame, ROOT.RooFit.Name("signal_ds"))
        smodel.plotOn(frame, ROOT.RooFit.Range("window"), ROOT.RooFit.Name("signal_cb"), ROOT.RooFit.MarkerColor(ROOT.kRed),ROOT.RooFit.LineColor(ROOT.kRed))


        chi2 = frame.chiSquare("signal_cb", "signal_ds", 6)
        print "DCB chi2/d.o.f: ", chi2
        canv = ROOT.TCanvas("canv2", "canv2", 800, 800)
        canv.cd()
        statbox = smodel.paramOn(frame, ROOT.RooFit.Layout(0.1, 0.4, 0.9))
        frame.getAttText().SetTextSize(0.02)
        t1 = ROOT.TPaveLabel(0.7,0.83,0.9,0.9, "#chi^{2}/dof = %.4f"%chi2,"brNDC")
        t1.SetFillColor(0)
        t1.SetTextSize(0.4)
        frame.addObject(t1)
        frame.SetTitle(label+" DCB")
        frame.Draw()
        statbox.Draw("same")

        # canv = ROOT.TCanvas("canv2", "canv2", 800, 800)
        # canv.cd()
        # frame.Draw()
        # chi2 = frame.chiSquare("signal_cb", "signal_ds", 6)
        # print "DCB chi2/d.o.f: ", chi2
        # latex = ROOT.TLatex()
        # latex1 = ROOT.TLatex()
        # latex.SetNDC()
        # latex1.SetNDC()
        # latex.SetTextSize(0.03)
        # latex1.SetTextSize(0.03)
        # latex.DrawLatex(0.5, 0.84, label+" DCB")
        # latex1.DrawLatex(0.5, 0.8, "#chi^{2}/d.o.f = %f"%chi2)
        canv.Print(self.out_dir+"DCB_"+name+".png")

        return frame

    def signal_fit_3Gaus(self, signal_src, name, label, xmin, xmax):
        signal_hist, signal_tree = self.get_mass_hist("signal", signal_src, signal_src.mc_path, "", 10, xmin, xmax, normalize=False)
        var = ROOT.RooRealVar("mass","Dilepton mass",xmin,xmax)
        max_abs_eta_var = ROOT.RooRealVar("max_abs_eta_mu","Max abs(eta) of muons", 0, 2.4)
        ggH_prediction_var = ROOT.RooRealVar("ggH_prediction", "ggH_prediction", 0, 1)
        VBF_prediction_var = ROOT.RooRealVar("VBF_prediction", "VBF_prediction", 0, 1)
        DY_prediction_var = ROOT.RooRealVar("DY_prediction", "DY_prediction", 0, 1)
        ttbar_prediction_var = ROOT.RooRealVar("ttbar_prediction", "ttbar_prediction", 0, 1) 

        var.setRange("full",xmin,xmax)

        Import = getattr(ROOT.RooWorkspace, 'import')
        w = ROOT.RooWorkspace("w", False)
        Import(w, var)

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
        signal_ds = ROOT.RooDataSet("signal_ds","signal_ds", signal_tree, ROOT.RooArgSet(var, max_abs_eta_var, ggH_prediction_var, VBF_prediction_var, DY_prediction_var, ttbar_prediction_var), signal_src.cut)
        res = smodel.fitTo(signal_ds, ROOT.RooFit.Range("full"),ROOT.RooFit.Save(), ROOT.RooFit.Verbose(False))
        res.Print("v")
        frame = var.frame()
        signal_ds.plotOn(frame, ROOT.RooFit.Name("signal_ds"))
        smodel.plotOn(frame, ROOT.RooFit.Range("window"), ROOT.RooFit.Name("signal_cb"))

        chi2 = frame.chiSquare("signal_cb", "signal_ds", 6)
        print "3Gaus chi2/d.o.f: ", chi2
        canv = ROOT.TCanvas("canv2", "canv2", 800, 800)
        canv.cd()
        statbox = smodel.paramOn(frame, ROOT.RooFit.Layout(0.1, 0.4, 0.9))
        frame.getAttText().SetTextSize(0.02)
        t1 = ROOT.TPaveLabel(0.7,0.83,0.9,0.9, "#chi^{2}/dof = %.4f"%chi2,"brNDC")
        t1.SetFillColor(0)
        t1.SetTextSize(0.4)
        frame.addObject(t1)
        frame.SetTitle(label+" 3Gaus")
        frame.Draw()
        statbox.Draw("same")

        # latex = ROOT.TLatex()
        # latex1 = ROOT.TLatex()
        # latex.SetNDC()
        # latex1.SetNDC()
        # latex.SetTextSize(0.03)
        # latex1.SetTextSize(0.03)
        # latex.DrawLatex(0.4, 0.84, label+" 3Gaus")
        # latex1.DrawLatex(0.5, 0.8, "#chi^{2}/d.o.f = %f"%chi2)

        canv.Print(self.out_dir+"3Gaus_"+name+".png")

        return frame

    def test_signal_fits(self, signal_src, name, label, xmin, xmax):
        frame_DCB = self.signal_fit_DCB(signal_src, name, label, xmin, xmax)
        frame_3Gaus = self.signal_fit_3Gaus(signal_src, name, label, xmin, xmax)

        canv = ROOT.TCanvas("canv", "canv", 800, 800)
        canv.cd()
        frame_DCB.Draw()
        frame_3Gaus.Draw("same")
        canv.Print(self.out_dir+"/combined/"+name+".png")


a = Analyzer()

dnn_cut_1 = "((ggH_prediction+VBF_prediction+1-DY_prediction+1-ttbar_prediction)<1.5)"
dnn_cut_2 = "((ggH_prediction+VBF_prediction+1-DY_prediction+1-ttbar_prediction)>1.5)&(((ggH_prediction+VBF_prediction+1-DY_prediction+1-ttbar_prediction)<2))"
dnn_cut_3 = "((ggH_prediction+VBF_prediction+1-DY_prediction+1-ttbar_prediction)>2)&(((ggH_prediction+VBF_prediction+1-DY_prediction+1-ttbar_prediction)<2.5))"
dnn_cut_4 = "((ggH_prediction+VBF_prediction+1-DY_prediction+1-ttbar_prediction)>2.5)&(((ggH_prediction+VBF_prediction+1-DY_prediction+1-ttbar_prediction)<3))"

data_obs = a.add_data_src("V3", "Run_2018-12-19_14-25-02", "model_50_D2_25_D2_25_D2", "tree_Data", ROOT.kGreen+2 ,"1", "1")
ggH_weigted = a.add_data_src("V3", "Run_2018-12-19_14-25-02", "model_50_D2_25_D2_25_D2", "tree_H2Mu_gg", ROOT.kGreen+2  ,"1", "weight*5/4") 
vbf_weigted = a.add_data_src("V3", "Run_2018-12-19_14-25-02", "model_50_D2_25_D2_25_D2", "tree_H2Mu_VBF", ROOT.kGreen+2  ,"1", "weight*5/4") 

data_obs_in = a.add_data_src("V3", "Run_2018-12-19_14-25-02", "model_50_D2_25_D2_25_D2", "tree_Data", ROOT.kGreen+2 ,"(max_abs_eta_mu<0.9)", "1")
ggH_weigted_in = a.add_data_src("V3", "Run_2018-12-19_14-25-02", "model_50_D2_25_D2_25_D2", "tree_H2Mu_gg", ROOT.kGreen+2  ,"(max_abs_eta_mu<0.9)", "weight*5/4") 
vbf_weigted_in = a.add_data_src("V3", "Run_2018-12-19_14-25-02", "model_50_D2_25_D2_25_D2", "tree_H2Mu_VBF", ROOT.kGreen+2  ,"(max_abs_eta_mu<0.9)", "weight*5/4") 

data_obs_mid = a.add_data_src("V3", "Run_2018-12-19_14-25-02", "model_50_D2_25_D2_25_D2", "tree_Data", ROOT.kGreen+2 ,"(max_abs_eta_mu>0.9)&(max_abs_eta_mu<1.9)", "1")
ggH_weigted_mid = a.add_data_src("V3", "Run_2018-12-19_14-25-02", "model_50_D2_25_D2_25_D2", "tree_H2Mu_gg", ROOT.kGreen+2  ,"(max_abs_eta_mu>0.9)&(max_abs_eta_mu<1.9)", "weight*5/4") 
vbf_weigted_mid = a.add_data_src("V3", "Run_2018-12-19_14-25-02", "model_50_D2_25_D2_25_D2", "tree_H2Mu_VBF", ROOT.kGreen+2  ,"(max_abs_eta_mu>0.9)&(max_abs_eta_mu<1.9)", "weight*5/4") 

data_obs_out = a.add_data_src("V3", "Run_2018-12-19_14-25-02", "model_50_D2_25_D2_25_D2", "tree_Data", ROOT.kGreen+2 ,"(max_abs_eta_mu>1.9)", "1")
ggH_weigted_out = a.add_data_src("V3", "Run_2018-12-19_14-25-02", "model_50_D2_25_D2_25_D2", "tree_H2Mu_gg", ROOT.kGreen+2  ,"(max_abs_eta_mu>1.9)", "weight*5/4") 
vbf_weigted_out = a.add_data_src("V3", "Run_2018-12-19_14-25-02", "model_50_D2_25_D2_25_D2", "tree_H2Mu_VBF", ROOT.kGreen+2  ,"(max_abs_eta_mu>1.9)", "weight*5/4") 

ggH_dnn_cut_1 = a.add_data_src("V3", "Run_2018-12-19_14-25-02", "model_50_D2_25_D2_25_D2", "tree_H2Mu_gg", ROOT.kGreen+2  , dnn_cut_1, "weight*5/4")
ggH_dnn_cut_2 = a.add_data_src("V3", "Run_2018-12-19_14-25-02", "model_50_D2_25_D2_25_D2", "tree_H2Mu_gg", ROOT.kGreen+2  , dnn_cut_2, "weight*5/4")
ggH_dnn_cut_3 = a.add_data_src("V3", "Run_2018-12-19_14-25-02", "model_50_D2_25_D2_25_D2", "tree_H2Mu_gg", ROOT.kGreen+2  , dnn_cut_3, "weight*5/4")
ggH_dnn_cut_4 = a.add_data_src("V3", "Run_2018-12-19_14-25-02", "model_50_D2_25_D2_25_D2", "tree_H2Mu_gg", ROOT.kGreen+2  , dnn_cut_4, "weight*5/4")

a.full_plotting_sequence(data_obs,ggH_weigted)
# a.set_out_dir("combine/test_eta_0to0p9/")
# a.full_plotting_sequence(data_obs_in, ggH_weigted_in)
# a.set_out_dir("combine/test_eta_0p9to1p9/")
# a.full_plotting_sequence(data_obs_mid, ggH_weigted_mid)
# a.set_out_dir("combine/test_eta_1p9to2p4/")
# a.full_plotting_sequence(data_obs_out, ggH_weigted_out)



# class dnn_cut(object):
#     def __init__(self, name, cut, caption):
#         self.name = name
#         self.cut = cut
#         self.caption = caption

# d1 = dnn_cut("dnn_cut_1", dnn_cut_1, "DNN score: 0 - 25%")
# d2 = dnn_cut("dnn_cut_2", dnn_cut_2, "DNN score: 25 - 50%")
# d3 = dnn_cut("dnn_cut_3", dnn_cut_3, "DNN score: 50 - 75%")
# d4 = dnn_cut("dnn_cut_4", dnn_cut_4, "DNN score: 75 - 100%")
# dnn_cuts = [d1, d2, d3, d4]

# class eta_cut(object):
#     def __init__(self, name, cut, caption):
#         self.name = name
#         self.cut = cut
#         self.caption = caption
 
# e1 = eta_cut("inclusive", "1", "inclusive by #eta")
# e2 = eta_cut("in", "(max_abs_eta_mu<0.9)", "max.|#eta| < 0.9")
# e3 = eta_cut("mid", "(max_abs_eta_mu>0.9)&(max_abs_eta_mu<1.9)", "0.9 < max.|#eta| < 1.9")
# e4 = eta_cut("out", "(max_abs_eta_mu>1.9)", "max.|#eta| > 1.9") 

# eta_cuts = [e1, e2, e3, e4]

# a.set_out_dir("combine/sig_fit_test_eta_dnn_110-135/")
# for d in dnn_cuts:
#     for e in eta_cuts:
#         signal_src = a.add_data_src("V3", "Run_2018-12-19_14-25-02", "model_50_D2_25_D2_25_D2", "tree_H2Mu_gg", ROOT.kGreen+2  ,"%s*%s"%(d.cut, e.cut), "weight*5/4")
#         a.test_signal_fits(signal_src, "ggh_%s_%s"%(d.name, e.name), "%s, %s"%(e.caption, d.caption), 110, 135)





# a.test_signal_fits(ggH_weigted, "ggh_inclusive","inclusive")
# a.test_signal_fits(ggH_weigted_in, "ggh_in", "max.|#eta| < 0.9")
# a.test_signal_fits(ggH_weigted_mid, "ggh_mid", "0.9 < max.|#eta| < 1.9")
# a.test_signal_fits(ggH_weigted_out, "ggh_out", "max.|#eta| > 1.9")
# a.test_signal_fits(vbf_weigted, "vbf_inclusive","inclusive")
# a.test_signal_fits(vbf_weigted_in, "vbf_in", "max.|#eta| < 0.9")
# a.test_signal_fits(vbf_weigted_mid, "vbf_mid", "0.9 < max.|#eta| < 1.9")
# a.test_signal_fits(vbf_weigted_out, "vbf_out", "max.|#eta| > 1.9")




