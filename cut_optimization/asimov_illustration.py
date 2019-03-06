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
    var = ROOT.RooRealVar("mass","Dilepton mass",110,135)     
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
    signal_hist = ROOT.TH1D(signal_hist_name, signal_hist_name, 40, 110, 135)
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

def add_data(w, cat_number, input_path, data_tree, cut):
    var = w.var("mass")
    var.setBins(5000)
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
    return w.data("cat%i_data"%cat_number)

def add_bkg_model(w, cat_number, input_path, data_tree, cut):
    data = add_data(w, cat_number, input_path, data_tree, cut)
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
    r = fit_func.fitTo(data, ROOT.RooFit.Range("full"),ROOT.RooFit.Save(), ROOT.RooFit.Verbose(False))
    r.Print()
    integral_sb = fit_func.createIntegral(ROOT.RooArgSet(var), ROOT.RooFit.Range("left,right"))
    integral_full = fit_func.createIntegral(ROOT.RooArgSet(var), ROOT.RooFit.Range("full"))
    func_int_sb = integral_sb.getVal()
    func_int_full = integral_full.getVal()
    data_int_sb = data.sumEntries("1","left,right")
    data_int_full = data.sumEntries("1","full")
    bkg_rate = data_int_sb * (func_int_full/func_int_sb)
    print cut, data_int_full
    print "="*100
    return bkg_rate    


def plot_initial_shapes(eta_min, eta_max):

    cuts = "(max_abs_eta_mu>%f)&(max_abs_eta_mu<%f)"%(eta_min, eta_max)

    sig_tree = ROOT.TChain(sig_tree_name)
    sig_tree.Add(signal_input)
    sig_hist = ROOT.TH1D("signal_initial", "", 20, 110, 135)
    sig_tree.Draw("mass>>signal_initial", cuts)
    sig_hist.Scale(1/sig_hist.Integral())
    sig_hist.SetLineWidth(3)

    data_tree = ROOT.TChain(data_tree_name)
    data_tree.Add(data_input)
    data_hist = ROOT.TH1D("data_initial", "", 20, 110, 135)    
    data_tree.Draw("mass>>data_initial", "(mass<120||mass>130)&%s"%cuts)
    data_hist.Scale(1/data_hist.Integral())
    data_hist.SetLineWidth(3)

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
    frame = var.frame(ROOT.RooFit.Bins(20))
    smodel = w.pdf('cat0_ggh')
    bmodel = w.pdf('cat0_bkg')
    smodel.plotOn(frame, ROOT.RooFit.Range("full"), ROOT.RooFit.Name("signal"),ROOT.RooFit.LineColor(ROOT.kRed))
    bmodel.plotOn(frame, ROOT.RooFit.Range("full"), ROOT.RooFit.Name("bkg"),ROOT.RooFit.LineColor(ROOT.kRed))
    # canvas = ROOT.TCanvas("c", "c", 800, 800)
    # canvas.cd()
    # frame.Draw()
    # canvas.SaveAs('plots/asimov/fit.png')
    return frame


sig_hist, data_hist = plot_initial_shapes(0, 0.1)
frame = plot_fits(0, 0.1)

canvas = ROOT.TCanvas("c", "c", 800, 800)
canvas.cd()
frame.Draw("same")
sig_hist.Draw('histsame')
data_hist.Draw('histple')

canvas.SaveAs('plots/asimov/fit.png')

