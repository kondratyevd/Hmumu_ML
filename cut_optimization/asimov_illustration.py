import ROOT
from math import sqrt
import os, sys, errno
Import = getattr(ROOT.RooWorkspace, 'import')
ROOT.gStyle.SetOptStat(0)

signal_input = "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/miniaod_skim/ggh_2017_psweights.root"
sig_tree_name = "tree"
data_input = "/mnt/hadoop/store/user/dkondrat/skim/2016/SingleMu_2016/*root"
data_tree_name = "dimuons/tree"

def create_workspace():
    var = ROOT.RooRealVar("mass","",110,135)     
    var.setBins(100)
    var.setRange("window",120,130)
    var.setRange("full",110,135)
    w = ROOT.RooWorkspace("w", False)
    Import(w, var)
    return w

def add_sig_model_3gaus(w, cat_number, input_path, sig_tree, cut):
    var = w.var("mass")
    # var.setBins(5000)
    max_abs_eta_var = ROOT.RooRealVar("max_abs_eta_mu","Max abs(eta) of muons", 0, 2.4) 
    mu1_eta = ROOT.RooRealVar("mu1_eta","mu1_eta", -2.4, 2.4) 
    mu2_eta = ROOT.RooRealVar("mu2_eta","mu2_eta", -2.4, 2.4) 
    signal_tree = ROOT.TChain(sig_tree)
    signal_tree.Add(input_path)  
    print "Loaded tree from "+input_path+" with %i entries."%signal_tree.GetEntries()    
    signal_hist_name = "signal_%i"%cat_number
    signal_hist = ROOT.TH1D(signal_hist_name, signal_hist_name, 100, 110, 135)
    dummy = ROOT.TCanvas("dummy", "dummy", 800, 800)
    dummy.cd()
    signal_tree.Draw("mass>>%s"%(signal_hist_name), cut)
    dummy.Close()
    signal_hist.Scale(1/signal_hist.Integral())
    signal_rate = 1
    # signal_rate = signal_hist.Integral()
    # print signal_rate

    w.factory("cat%i_mixGG [0.5, 0.0, 1.0]"%cat_number)
    w.factory("cat%i_mixGG1 [0.5, 0.0, 1.0]"%cat_number)
    mixGG = w.var("cat%i_mixGG"%cat_number)
    mixGG1 = w.var("cat%i_mixGG1"%cat_number)
    w.factory("Gaussian::cat%i_gaus1(mass, cat%i_mean1[125., 120., 130.], cat%i_width1[1.0, 0.5, 5.0])"%(cat_number, cat_number, cat_number))
    w.factory("Gaussian::cat%i_gaus2(mass, cat%i_mean2[125., 120., 130.], cat%i_width2[5.0, 2.0, 10.])"%(cat_number, cat_number, cat_number))
    w.factory("Gaussian::cat%i_gaus3(mass, cat%i_mean3[125., 120., 130.], cat%i_width3[5.0, 1.0, 10.])"%(cat_number, cat_number, cat_number))
    gaus1 = w.pdf('cat%i_gaus1'%(cat_number))
    gaus2 = w.pdf('cat%i_gaus2'%(cat_number))
    gaus3 = w.pdf('cat%i_gaus3'%(cat_number))
    gaus12 = ROOT.RooAddPdf('cat%i_gaus12'%(cat_number), 'cat%i_gaus12'%(cat_number), gaus1, gaus2, mixGG)
    smodel = ROOT.RooAddPdf('cat%i_ggh'%cat_number, 'cat%i_ggh'%cat_number, gaus3, gaus12, mixGG1)
    # Import(w,smodel)
    w.Print()
    signal_ds = ROOT.RooDataSet("signal_ds","signal_ds", signal_tree, ROOT.RooArgSet(var, max_abs_eta_var, mu1_eta, mu2_eta), cut)
    res = smodel.fitTo(signal_ds, ROOT.RooFit.Range("full"),ROOT.RooFit.Save(), ROOT.RooFit.Verbose(False))
    res.Print()
    sigParamList = ["mean1", "mean2", "mean3", "width1", "width2", "width3", "mixGG", "mixGG1"]
    for par in sigParamList:
        par_var = w.var("cat%s_%s"%(cat_number,par))
        par_var.setConstant(True)
    Import(w, smodel)
    return signal_rate

def add_sig_model_3gaus_nuis(w, cat_number, input_path, sig_tree, cut, beta_scale, beta_res, rec_frac):
    var = w.var("mass")
    # var.setBins(5000)
    max_abs_eta_var = ROOT.RooRealVar("max_abs_eta_mu","Max abs(eta) of muons", 0, 2.4) 
    mu1_eta = ROOT.RooRealVar("mu1_eta","mu1_eta", -2.4, 2.4) 
    mu2_eta = ROOT.RooRealVar("mu2_eta","mu2_eta", -2.4, 2.4) 
    signal_tree = ROOT.TChain(sig_tree)
    signal_tree.Add(input_path)  
    print "Loaded tree from "+input_path+" with %i entries."%signal_tree.GetEntries()    
    signal_hist_name = "signal_%i"%cat_number
    signal_hist = ROOT.TH1D(signal_hist_name, signal_hist_name, 100, 110, 135)
    dummy = ROOT.TCanvas("dummy", "dummy", 800, 800)
    dummy.cd()
    signal_tree.Draw("mass>>%s"%(signal_hist_name), cut)
    dummy.Close()
    signal_hist.Scale(1/signal_hist.Integral())
    signal_rate = 1
    # signal_rate = signal_hist.Integral()
    # print signal_rate

    w.factory("cat%i_mixGG [0.1, 0.0, 1.0]"%cat_number)
    w.factory("cat%i_mixGG1 [0, 0.0, 1.0]"%cat_number)

    w.factory("mu_res_beta [0, 0, 0]")
    w.factory("mu_scale_beta [0, 0, 0]")

    w.factory("mu_res_unc [0.1, 0.1, 0.1]")
    w.factory("mu_scale_unc [0.0005, 0.0005, 0.0005]")

    w.var("mu_res_unc").setConstant(True)
    w.var("mu_scale_unc").setConstant(True)

    mixGG = w.var("cat%i_mixGG"%cat_number)
    mixGG1 = w.var("cat%i_mixGG1"%cat_number)

    w.factory("EXPR::cat%i_mean1_times_nuis('cat%i_mean1*(1 + mu_scale_unc*mu_scale_beta)',{cat%i_mean1[125.0, 120., 130.],mu_scale_unc,mu_scale_beta})"%(cat_number,cat_number,cat_number))
    w.factory("EXPR::cat%i_mean2_times_nuis('cat%i_mean2*(1 + mu_scale_unc*mu_scale_beta)',{cat%i_mean2[125.0, 120., 130.],mu_scale_unc,mu_scale_beta})"%(cat_number,cat_number,cat_number))
    w.factory("EXPR::cat%i_mean3_times_nuis('cat%i_mean3*(1 + mu_scale_unc*mu_scale_beta)',{cat%i_mean3[125.0, 120., 130.],mu_scale_unc,mu_scale_beta})"%(cat_number,cat_number,cat_number))

    w.factory("expr::cat%i_deltaM21('cat%i_mean2-cat%i_mean1',{cat%i_mean2, cat%i_mean1})"%(cat_number,cat_number,cat_number,cat_number,cat_number))
    w.factory("expr::cat%i_deltaM31('cat%i_mean3-cat%i_mean1',{cat%i_mean3, cat%i_mean1})"%(cat_number,cat_number,cat_number,cat_number,cat_number))

    w.factory("EXPR::cat%i_mean2_final('cat%i_mean2_times_nuis + mu_res_unc*mu_res_beta*cat%i_deltaM21',{cat%i_mean2_times_nuis, mu_res_unc, mu_res_beta, cat%i_deltaM21})"%(cat_number,cat_number,cat_number,cat_number,cat_number))
    w.factory("EXPR::cat%i_mean3_final('cat%i_mean3_times_nuis + mu_res_unc*mu_res_beta*cat%i_deltaM31',{cat%i_mean3_times_nuis, mu_res_unc, mu_res_beta, cat%i_deltaM31})"%(cat_number,cat_number,cat_number,cat_number,cat_number))

    w.factory("EXPR::cat%i_width1_times_nuis('cat%i_width1*(1 + mu_res_unc*mu_res_beta)',{cat%i_width1[1.0, 0.5, 5.0],mu_res_unc, mu_res_beta})"%(cat_number,cat_number,cat_number))
    w.factory("EXPR::cat%i_width2_times_nuis('cat%i_width2*(1 + mu_res_unc*mu_res_beta)',{cat%i_width2[5.0, 2.0, 10.],mu_res_unc, mu_res_beta})"%(cat_number,cat_number,cat_number))
    w.factory("EXPR::cat%i_width3_times_nuis('cat%i_width3*(1 + mu_res_unc*mu_res_beta)',{cat%i_width3[5.0, 1.0, 10.],mu_res_unc, mu_res_beta})"%(cat_number,cat_number,cat_number))

    w.factory("Gaussian::cat%i_gaus1(mass, cat%i_mean1_times_nuis, cat%i_width1_times_nuis)"%(cat_number, cat_number, cat_number))
    w.factory("Gaussian::cat%i_gaus2(mass, cat%i_mean2_final, cat%i_width2_times_nuis)"%(cat_number, cat_number, cat_number))
    w.factory("Gaussian::cat%i_gaus3(mass, cat%i_mean3_final, cat%i_width3_times_nuis)"%(cat_number, cat_number, cat_number))


    gaus1 = w.pdf('cat%i_gaus1'%(cat_number))
    gaus2 = w.pdf('cat%i_gaus2'%(cat_number))
    gaus3 = w.pdf('cat%i_gaus3'%(cat_number))

    smodel = ROOT.RooAddPdf('cat%i_ggh'%cat_number, 'cat%i_ggh'%cat_number, ROOT.RooArgList(gaus1, gaus2, gaus3) , ROOT.RooArgList(mixGG, mixGG1), rec_frac)

    # Import(w,smodel)
    w.Print()
    signal_ds = ROOT.RooDataSet("signal_ds","signal_ds", signal_tree, ROOT.RooArgSet(var, max_abs_eta_var, mu1_eta, mu2_eta), cut)
    res = smodel.fitTo(signal_ds, ROOT.RooFit.Range("full"),ROOT.RooFit.Save(), ROOT.RooFit.Verbose(False))
    res.Print()
    # sigParamList = ["mean1", "mean2", "mean3", "width1", "width2", "width3", "mixGG", "mixGG1"]
    # for par in sigParamList:
    #     par_var = w.var("cat%s_%s"%(cat_number,par))
    #     par_var.setConstant(True)
    w.var("mu_res_beta").setRange(-5, 5)
    w.var("mu_res_beta").setVal(beta_res)
    Import(w, smodel)
    return signal_rate

def add_sig_model_dcb(w, cat_number, input_path, sig_tree, cut):
    var = w.var("mass")
    # var.setBins(5000)
    max_abs_eta_var = ROOT.RooRealVar("max_abs_eta_mu","Max abs(eta) of muons", 0, 2.4) 
    mu1_eta = ROOT.RooRealVar("mu1_eta","mu1_eta", -2.4, 2.4) 
    mu2_eta = ROOT.RooRealVar("mu2_eta","mu2_eta", -2.4, 2.4) 
    signal_tree = ROOT.TChain(sig_tree)
    signal_tree.Add(input_path)  
    print "Loaded tree from "+input_path+" with %i entries."%signal_tree.GetEntries()    
    signal_hist_name = "signal_%i"%cat_number
    signal_hist = ROOT.TH1D(signal_hist_name, signal_hist_name, 40, 110, 150)
    dummy = ROOT.TCanvas("dummy", "dummy", 800, 800)
    dummy.cd()
    signal_tree.Draw("mass>>%s"%(signal_hist_name), cut)
    dummy.Close()
    signal_rate = signal_hist.Integral()

    ROOT.gSystem.Load("/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/RooDCBShape_cxx.so")
    w.factory("RooDCBShape::cat%i_ggh(mass, cat%i_mean[125,120,130], cat%i_sigma[2,0,5], cat%i_alphaL[2,0,25] , cat%i_alphaR[2,0,25], cat%i_nL[1.5,0,25], cat%i_nR[1.5,0,25])"%(cat_number,cat_number,cat_number,cat_number,cat_number,cat_number,cat_number))
    smodel = w.pdf("cat%i_ggh"%cat_number)
    w.Print()
    signal_ds = ROOT.RooDataSet("signal_ds","signal_ds", signal_tree, ROOT.RooArgSet(var, max_abs_eta_var, mu1_eta, mu2_eta), cut)
    res = smodel.fitTo(signal_ds, ROOT.RooFit.Range("full"),ROOT.RooFit.Save(), ROOT.RooFit.Verbose(False))
    res.Print()
    sigParamList = ["mean", "sigma", "alphaL", "alphaR", "nL", "nR"]
    for par in sigParamList:
        par_var = w.var("cat%s_%s"%(cat_number,par))
        par_var.setConstant(True)
    Import(w, smodel)
    return signal_rate

def add_data(w, cat_number, input_path, data_tree, cut):
    var = w.var("mass")
    # var.setBins(5000)
    max_abs_eta_var = ROOT.RooRealVar("max_abs_eta_mu","Max abs(eta) of muons", 0, 2.4) 
    mu1_eta = ROOT.RooRealVar("mu1_eta","mu1_eta", -2.4, 2.4) 
    mu2_eta = ROOT.RooRealVar("mu2_eta","mu2_eta", -2.4, 2.4) 
    data_tree = ROOT.TChain(data_tree)
    data_tree.Add(input_path)  
    print "Loaded tree from "+input_path+" with %i entries."%data_tree.GetEntries()
    data_hist_name = "data_%i"%cat_number
    data_hist = ROOT.TH1D(data_hist_name, data_hist_name, 40, 110, 135)
    dummy = ROOT.TCanvas("dummy", "dummy", 800, 800)
    dummy.cd()
    data_tree.Draw("mass>>%s"%(data_hist_name), cut)
    dummy.Close()
    data_hist.Scale(1/data_hist.Integral())
    data = ROOT.RooDataSet("cat%i_data"%cat_number,"cat%i_data"%cat_number, data_tree, ROOT.RooArgSet(var, max_abs_eta_var, mu1_eta, mu2_eta), cut)
    Import(w, data)
    return w.data("cat%i_data"%cat_number), data_hist

def add_bkg_model(w, cat_number, input_path, data_tree, cut):
    data, data_hist = add_data(w, cat_number, input_path, data_tree, cut)
    var = w.var("mass")
    # var.setBins(5000)
    var.setRange("left",110,120+0.1)
    var.setRange("right",130-0.1,135)
    # data = w.data("cat%i_data"%cat_number)
    w.factory("cat%i_a1 [1.66, 0.7, 2.1]"%cat_number)
    w.factory("cat%i_a2 [0.39, 0.30, 0.62]"%cat_number)
    w.factory("cat%i_a3 [-0.26, -0.40, -0.12]"%cat_number)
    w.factory("expr::cat%i_bwz_redux_f('(@1*(@0/100)+@2*(@0/100)^2)',{mass, cat%i_a2, cat%i_a3})"%(cat_number,cat_number,cat_number))
    w.factory("EXPR::cat%i_bkg('exp(@2)*(2.5)/(pow(@0-91.2,@1)+pow(2.5/2,@1))',{mass, cat%i_a1, cat%i_bwz_redux_f})"%(cat_number,cat_number,cat_number))
    fit_func = w.pdf('cat%i_bkg'%cat_number)

    data_binned = ROOT.RooDataHist("data_hist","data_hist", ROOT.RooArgList(var), data_hist)
    Import(w, data_binned)
    data_binned.Print()
    bkg_rate = data_hist.Integral()
    cmdlist = ROOT.RooLinkedList()
    cmd = ROOT.RooFit.Save()
    # cmdlist.Add(ROOT.RooFit.Range("left,right"))
    cmdlist.Add(cmd)
    # cmdlist.Add(ROOT.RooFit.Verbose(False))
    # cmdlist.Add(ROOT.RooFit.PrintLevel(-1000))
    r = fit_func.chi2FitTo(data_binned, cmdlist)

    # r = fit_func.fitTo(data, ROOT.RooFit.Range("full"),ROOT.RooFit.Save(), ROOT.RooFit.Verbose(False))
    r.Print()
    # integral_sb = fit_func.createIntegral(ROOT.RooArgSet(var), ROOT.RooFit.Range("left,right"))
    # integral_full = fit_func.createIntegral(ROOT.RooArgSet(var), ROOT.RooFit.Range("full"))
    # func_int_sb = integral_sb.getVal()
    # func_int_full = integral_full.getVal()
    # data_int_sb = data.sumEntries("1","left,right")
    # data_int_full = data.sumEntries("1","full")
    # bkg_rate = data_int_sb * (func_int_full/func_int_sb)
    # print cut, data_int_full
    # print "="*100
    return bkg_rate    


def plot_initial_shapes(eta_min, eta_max):

    cuts = "(max_abs_eta_mu>%f)&(max_abs_eta_mu<%f)"%(eta_min, eta_max)

    sig_tree = ROOT.TChain(sig_tree_name)
    sig_tree.Add(signal_input)
    sig_hist = ROOT.TH1D("signal_initial", "", 100, 110, 135)
    sig_tree.Draw("mass>>signal_initial", cuts)
    sig_hist.Scale(1/sig_hist.Integral())
    sig_hist.SetLineWidth(3)

    data_tree = ROOT.TChain(data_tree_name)
    data_tree.Add(data_input)
    data_hist = ROOT.TH1D("data_initial", "", 100, 110, 135)    
    data_hist_full = ROOT.TH1D("data_initial_full", "", 100, 110, 135)  
    data_tree.Draw("mass>>data_initial_full", cuts)  
    data_tree.Draw("mass>>data_initial", "(mass<120||mass>130)&%s"%cuts)
    data_hist.Sumw2()
    data_hist.Scale(1/data_hist_full.Integral())
    # data_hist.Scale(1/data_hist.Integral())
    data_hist.SetLineWidth(3)
    data_hist.SetMarkerStyle(20)
    data_hist.SetMarkerSize(1)

    # canvas = ROOT.TCanvas("c", "c", 800, 800)
    # canvas.cd()
    # sig_hist.Draw('hist')
    # data_hist.Draw('histsame')
    # canvas.SaveAs('plots/asimov/initial_shapes.png')
    return sig_hist, data_hist

def plot_fits(eta_min, eta_max):
    cut = "(max_abs_eta_mu>%f)&(max_abs_eta_mu<%f)"%(eta_min, eta_max)

    w = create_workspace()
    sig_rate = add_sig_model_3gaus(w, 0, signal_input, sig_tree_name, cut) 
    bkg_rate = add_bkg_model(w, 0, data_input, data_tree_name, cut)
    var = w.var('mass')
    frame = var.frame(ROOT.RooFit.Bins(100))
    smodel = w.pdf('cat0_ggh')
    bmodel = w.pdf('cat0_bkg')
    smodel.plotOn(frame, ROOT.RooFit.Range("full"), ROOT.RooFit.Name("signal"),ROOT.RooFit.LineColor(ROOT.kRed))
    bmodel.plotOn(frame, ROOT.RooFit.Range("full"), ROOT.RooFit.Name("bkg"),ROOT.RooFit.LineColor(ROOT.kRed))
    # canvas = ROOT.TCanvas("c", "c", 800, 800)
    # canvas.cd()
    # frame.Draw()
    # canvas.SaveAs('plots/asimov/fit.png')
    return w, frame

def plot_fits_nuis(eta_min, eta_max):
    cut = "(max_abs_eta_mu>%f)&(max_abs_eta_mu<%f)"%(eta_min, eta_max)

    w = create_workspace()
    sig_rate = add_sig_model_3gaus_nuis(w, 0, signal_input, sig_tree_name, cut, 0, -1, ROOT.kTRUE) 
    var = w.var('mass')
    frame = var.frame(ROOT.RooFit.Bins(100))
    smodel = w.pdf('cat0_ggh')
    smodel.plotOn(frame, ROOT.RooFit.Range("full"), ROOT.RooFit.Name("signal"),ROOT.RooFit.LineColor(ROOT.kRed))

    w_nuis_false = create_workspace()
    sig_rate = add_sig_model_3gaus_nuis(w_nuis_false, 0, signal_input, sig_tree_name, cut, 0, -1, ROOT.kFALSE) 
    smodel_false = w_nuis_false.pdf('cat0_ggh')
    smodel_false.plotOn(frame, ROOT.RooFit.Range("full"), ROOT.RooFit.Name("signal_false"),ROOT.RooFit.LineColor(ROOT.kBlue))

    # w_nuis_up = create_workspace()
    # sig_rate = add_sig_model_3gaus_nuis(w_nuis_up, 0, signal_input, sig_tree_name, cut, 0, 1) 
    # # var_up = w_nuis_up.var('mass')
    # # frame_nuis_up = var_up.frame(ROOT.RooFit.Bins(100))
    # smodel_up = w_nuis_up.pdf('cat0_ggh')
    # smodel_up.plotOn(frame, ROOT.RooFit.Range("full"), ROOT.RooFit.Name("signal_up"),ROOT.RooFit.LineColor(ROOT.kGreen))

    # w_nuis_down = create_workspace()
    # sig_rate = add_sig_model_3gaus_nuis(w_nuis_down, 0, signal_input, sig_tree_name, cut, 0, -1) 
    # # var_down = w_nuis_down.var('mass')
    # # frame_nuis_down = var_down.frame(ROOT.RooFit.Bins(100))
    # smodel_down = w_nuis_down.pdf('cat0_ggh')
    # smodel_down.plotOn(frame, ROOT.RooFit.Range("full"), ROOT.RooFit.Name("signal_down"),ROOT.RooFit.LineColor(ROOT.kBlue))

    return frame#, frame_nuis_up, frame_nuis_down

def make_asimov_dataset(w):
    nbins = 200
    xmin = 110.
    xmax = 135.
    binwidth = (xmax-xmin)/float(nbins)
    sig_hist_new = ROOT.TH1D("signal_new", "", nbins, xmin, xmax)
    bkg_hist_new = ROOT.TH1D("bkg_new", "", nbins, xmin, xmax)
    sig_hist_new.SetLineWidth(3)
    sig_hist_new.SetLineColor(ROOT.kBlack)
    bkg_hist_new.SetLineWidth(3)
    bkg_hist_new.SetLineColor(ROOT.kBlack)

    var = w.var('mass')
    for i in range(nbins):
        x = xmin+(i+0.5)*binwidth
        var.setVal(x)
        sig = w.pdf('cat0_ggh')
        bkg = w.pdf('cat0_bkg')
        sig_hist_new.SetBinContent(i+1, sig.getVal())
        bkg_hist_new.SetBinContent(i+1, bkg.getVal())
    sig_hist_new.Scale(1/sig_hist_new.Integral())
    bkg_hist_new.Scale(1/bkg_hist_new.Integral())
    return sig_hist_new, bkg_hist_new


sig_hist, data_hist = plot_initial_shapes(0, 0.1)
# frame, frame_nuis_up, frame_nuis_down = plot_fits_nuis(0, 0.1)
# frame = plot_fits_nuis(0, 0.1)

w, frame = plot_fits(0, 0.1)

canvas = ROOT.TCanvas("c", "c", 800, 800)
canvas.cd()
frame.Draw("")
frame.GetYaxis().SetTitle("a. u.")
frame.GetYaxis().SetLabelSize(0)
frame.SetTitle("")
sig_hist.Draw('histsame')
# data_hist.Draw('plesame')

canvas.SaveAs('plots/asimov/new_test.png')

# sig_hist, data_hist = plot_initial_shapes(0, 0.1)
# w, frame = plot_fits(0, 0.1)
# sig_hist_new, bkg_hist_new = make_asimov_dataset(w)

# canvas = ROOT.TCanvas("c", "c", 800, 800)
# canvas.cd()
# sig_hist.Draw('hist')
# sig_hist.GetYaxis().SetTitle("a. u.")
# sig_hist.GetYaxis().SetLabelSize(0)
# sig_hist.SetTitle("")
# data_hist.Draw('plesame')
# canvas.SaveAs('plots/asimov/initial_shapes.png')

# canvas = ROOT.TCanvas("c", "c", 800, 800)
# canvas.cd()
# frame.Draw("")
# frame.GetYaxis().SetTitle("a. u.")
# frame.GetYaxis().SetLabelSize(0)
# frame.SetTitle("")
# sig_hist.Draw('histsame')
# data_hist.Draw('plesame')
# canvas.SaveAs('plots/asimov/fit.png')

# canvas = ROOT.TCanvas("c", "c", 800, 800)
# canvas.cd()
# # frame.Draw("same")
# sig_hist_new.Draw('hist')
# sig_hist_new.GetYaxis().SetTitle("a. u.")
# sig_hist_new.GetYaxis().SetLabelSize(0)
# sig_hist_new.SetTitle("")
# bkg_hist_new.Draw('histsame')
# canvas.SaveAs('plots/asimov/asimov_ds.png')


