import ROOT
import os, sys, errno
# ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(1)
from math import *   

var_name = "mass"

def get_mass_hist(name, src, nBins, xmin, xmax):
    data = ROOT.TChain(src.tree_name)
    data.Add(src.path)  
    hist_name = name
    data_hist = ROOT.TH1D(hist_name, hist_name, nBins, xmin, xmax)
    data_hist.SetLineColor(ROOT.kBlack)
    data_hist.SetMarkerStyle(20)
    data_hist.SetMarkerSize(0.8)
    dummy = ROOT.TCanvas("dummy", "dummy", 800, 800)
    dummy.cd()
    data.Draw("%s>>%s"%(var_name,hist_name), src.cut)
    dummy.Close()

    # data_hist.Scale(1/data_hist.Integral())
    return data_hist, data

def signal_fit_DCB(signal_src, name, label, xmin, xmax):
    signal_hist, signal_tree = get_mass_hist("signal", signal_src, 10, xmin, xmax)
    var = ROOT.RooRealVar(var_name,"Dilepton mass",xmin,xmax)
    # max_abs_eta_var = ROOT.RooRealVar("max_abs_eta_mu","Max abs(eta) of muons", 0, 2.4)
    # ggH_prediction_var = ROOT.RooRealVar("ggH_prediction", "ggH_prediction", 0, 1)
    # VBF_prediction_var = ROOT.RooRealVar("VBF_prediction", "VBF_prediction", 0, 1)
    # DY_prediction_var = ROOT.RooRealVar("DY_prediction", "DY_prediction", 0, 1)
    # ttbar_prediction_var = ROOT.RooRealVar("ttbar_prediction", "ttbar_prediction", 0, 1) 

    var.setRange("full",xmin,xmax)

    Import = getattr(ROOT.RooWorkspace, 'import')
    w = ROOT.RooWorkspace("w0", False)
    Import(w, var)
    
    ROOT.gROOT.ProcessLine(".L src/RooDCBShape.cxx")
    w.factory("RooDCBShape::cb(%s, mean[125,120,130], sigma[2,0,20], alphaL[2,0,25] , alphaR[2,0,25], nL[1.5,0,25], nR[1.5,0,25])"%var_name)
    smodel = w.pdf("cb")
    
    signal_ds = ROOT.RooDataSet("signal_ds","signal_ds", signal_tree, ROOT.RooArgSet(var), signal_src.cut)
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
    canv.Print(out_dir+"DCB_"+name+".png")

    return frame
    # return 0

def signal_fit_3Gaus(signal_src, name, label, xmin, xmax):
    signal_hist, signal_tree = get_mass_hist("signal", signal_src, 10, xmin, xmax)
    # print signal_hist.GetEntries()

    var = ROOT.RooRealVar(var_name,"Dilepton mass",xmin,xmax)
    # max_abs_eta_var = ROOT.RooRealVar("max_abs_eta_mu","Max abs(eta) of muons", 0, 2.4)
    # ggH_prediction_var = ROOT.RooRealVar("ggH_prediction", "ggH_prediction", 0, 1)
    # VBF_prediction_var = ROOT.RooRealVar("VBF_prediction", "VBF_prediction", 0, 1)
    # DY_prediction_var = ROOT.RooRealVar("DY_prediction", "DY_prediction", 0, 1)
    # ttbar_prediction_var = ROOT.RooRealVar("ttbar_prediction", "ttbar_prediction", 0, 1) 

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
    w.factory("Gaussian::gaus1(%s, mean1_times_nuis, width1_times_nuis)"%var_name)
    w.factory("Gaussian::gaus2(%s, mean2_final, width2_times_nuis)"%var_name)
    w.factory("Gaussian::gaus3(%s, mean3_final, width3_times_nuis)"%var_name)
    gaus1 = w.pdf('gaus1')
    gaus2 = w.pdf('gaus2')
    gaus3 = w.pdf('gaus3')
    gaus12 = ROOT.RooAddPdf('gaus12', 'gaus12', gaus1, gaus2, mixGG)
    smodel = ROOT.RooAddPdf('signal', 'signal', gaus3, gaus12, mixGG1)
    Import(w,smodel)        
    signal_ds = ROOT.RooDataSet("signal_ds","signal_ds", signal_tree, ROOT.RooArgSet(var), signal_src.cut)
    res = smodel.fitTo(signal_ds, ROOT.RooFit.Range("full"),ROOT.RooFit.Save(), ROOT.RooFit.Verbose(False))
    res.Print("v")
    frame = var.frame()
    signal_ds.plotOn(frame, ROOT.RooFit.Name("signal_ds"))
    smodel.plotOn(frame, ROOT.RooFit.Range("window"), ROOT.RooFit.Name("signal_cb"))

    chi2 = frame.chiSquare("signal_cb", "signal_ds", 6)
    print "3Gaus chi2/d.o.f: ", chi2
    canv = ROOT.TCanvas("canv2", "canv2", 800, 800)
    canv.cd()
    statbox = smodel.paramOn(frame, ROOT.RooFit.Layout(0.1, 0.4, 0.9)) #0.9,0.65,0.8
    frame.getAttText().SetTextSize(0.02)
    t1 = ROOT.TPaveLabel(0.7,0.83,0.9,0.9, "#chi^{2}/dof = %.4f"%chi2,"brNDC")
    t1.SetFillColor(0)
    t1.SetTextSize(0.4)
    frame.addObject(t1)
    frame.SetTitle(label+" 3Gaus")
    frame.Draw()
    statbox.Draw("same")

    canv.Print(out_dir+"3Gaus_"+name+".png")

    return frame
    # return 0

def test_signal_fits(signal_src, name, label, xmin, xmax):
    frame_DCB = signal_fit_DCB(signal_src, name, label, xmin, xmax)
    frame_3Gaus = signal_fit_3Gaus(signal_src, name, label, xmin, xmax)

    canv = ROOT.TCanvas("canv", "canv", 800, 800)
    canv.cd()
    frame_DCB.Draw()
    frame_3Gaus.Draw("same")
    canv.Print(out_dir+"/combined/"+name+".png")

class DataSrc(object):
    def __init__(self, name, path, tree_name, cut):
        self.name = name
        self.path = path
        self.tree_name = tree_name
        self.cut = cut

out_dir = "combine/fsr_test/"
signal_src = DataSrc("ggH", "combine/fsr_test.root", "tree","1")

test_signal_fits(signal_src, "ggh_postFSR_115-132", "ggh_postFSR_115-132", 115, 132)    