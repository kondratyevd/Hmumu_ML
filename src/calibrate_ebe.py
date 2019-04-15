import ROOT
import os, sys, errno
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(1)
from math import *  
from array import array 


def fit_zpeak(cat_name, input_path, out_path, cut, isData=False):
    Import = getattr(ROOT.RooWorkspace, 'import')
    var = ROOT.RooRealVar("mass","mass",50,130)     
    var.setBins(1000)
    var.setRange("full",50,130)
    w = ROOT.RooWorkspace("w", False)
    Import(w, var)
    var = w.var("mass")
    # var.setBins(5000)
    mu1_pt = ROOT.RooRealVar("muons.pt[0]","muons.pt[0]", 0, 10000) 
    mu1_eta = ROOT.RooRealVar("muons.eta[0]","muons.eta[0]", -2.4, 2.4) 
    mu2_eta = ROOT.RooRealVar("muons.eta[1]","muons.eta[1]", -2.4, 2.4)
    tree = ROOT.TChain("dimuons/tree")
    tree.Add(input_path)
    print "Loaded sig tree from "+input_path+" with %i entries."%tree.GetEntries()    

     
    hist_name = "%s"%cat_name
    hist = ROOT.TH1D(hist_name, hist_name, 40, 110, 150)
    dummy = ROOT.TCanvas("dummy", "dummy", 800, 800)
    dummy.cd()
    tree.Draw("muPairs.mass_Roch>>%s"%(hist_name), "(%s)"%(cut))
    dummy.Close()

    print "Entries in hist: ",hist.GetEntries()

    w.factory("mZ[91.1876, 91.1876, 91.1876]")
    w.factory("wZ[2.4952, 2.4952, 2.4952]")
    w.factory("RooBreitWigner::bw_%s(mass, mZ, wZ)"%cat_name)


    ROOT.gSystem.Load("src/RooDCBShape_cxx.so")    
    w.factory("RooDCBShape::dcb_%s(mass, %s_mean[125,120,130], %s_sigma[2,0,5], %s_alphaL[2,0,25] , %s_alphaR[2,0,25], %s_nL[1.5,0,25], %s_nR[1.5,0,25])"%(cat_name,cat_name,cat_name,cat_name,cat_name,cat_name,cat_name))

    w.factory("RooFFTConvPdf::zfit_%s(mass, bw_%s, dcb_%s)"%(cat_name, cat_name, cat_name))
    model = w.pdf("zfit_%s"%cat_name)

    w.Print()
    # ds = ROOT.RooDataSet("ds","ds", tree, ROOT.RooArgSet(var, mu1_pt, mu1_eta, mu2_eta), cut)
    ds = ROOT.RooDataHist("ds", "ds", ROOT.RooArgList(var), hist)
    res = model.fitTo(ds, ROOT.RooFit.Range("full"),ROOT.RooFit.Save(), ROOT.RooFit.Verbose(False))
    res.Print()

    frame = var.frame(ROOT.RooFit.Bins(100))
    ds.plotOn(frame, ROOT.RooFit.Name("ds"))
    model.plotOn(frame, ROOT.RooFit.Range("full"), ROOT.RooFit.Name("zfit_%s"%cat_name),ROOT.RooFit.LineColor(ROOT.kRed))

    canv = ROOT.TCanvas("canv", "canv", 800, 800)
    canv.cd()
    frame.Draw()
    canv.Print("%s/zfit_%s.png"%(out_path, cat_name))


    print cat_name, " width = ", w.var("%s_sigma"%cat_name).getVal()
    return w.var("%s_sigma"%cat_name).getVal()






pt_bins = ["(muons.pt[0]>30)&(muons.pt[0]<50)", "(muons.pt[0]>50)"]

B1 = "(abs(muons.eta[0])<0.9)"
B2 = "(abs(muons.eta[1])<0.9)"
O1 = "(abs(muons.eta[0])>0.9)&(abs(muons.eta[0])<1.9)"
O2 = "(abs(muons.eta[1])>0.9)&(abs(muons.eta[1])<1.9)"
E1 = "(abs(muons.eta[0])>1.9)"
E2 = "(abs(muons.eta[1])>1.9)"

BB = "%s&%s"%(B1, B2)
BO = "(%s&%s)||(%s&%s)"%(B1, O2, B2, O1)
BE = "(%s&%s)||(%s&%s)"%(B1, E2, B2, E1)
OO = "%s&%s"%(O1, O2)
OE = "(%s&%s)||(%s&%s)"%(O1, E2, O2, E1)
EE = "%s&%s"%(E1, E2)

eta_bins = [BB, BO, BE, OO, OE, EE]

categories = {
    "pt_bin1_BB" : "(%s)&(%s)"%(pt_bins[0], BB),
    "pt_bin1_BO" : "(%s)&(%s)"%(pt_bins[0], BO),
    "pt_bin1_BE" : "(%s)&(%s)"%(pt_bins[0], BE),
    "pt_bin1_OO" : "(%s)&(%s)"%(pt_bins[0], OO),
    "pt_bin1_OE" : "(%s)&(%s)"%(pt_bins[0], OE),
    "pt_bin1_EE" : "(%s)&(%s)"%(pt_bins[0], EE),

    "pt_bin2_BB" : "(%s)&(%s)"%(pt_bins[1], BB),
    "pt_bin2_BO" : "(%s)&(%s)"%(pt_bins[1], BO),
    "pt_bin2_BE" : "(%s)&(%s)"%(pt_bins[1], BE),
    "pt_bin2_OO" : "(%s)&(%s)"%(pt_bins[1], OO),
    "pt_bin2_OE" : "(%s)&(%s)"%(pt_bins[1], OE),
    "pt_bin2_EE" : "(%s)&(%s)"%(pt_bins[1], EE)
}

# input_path = "/Users/dmitrykondratyev/Documents/HiggsToMuMu/test_files/dy/*root"
input_path ="/mnt/hadoop/store/user/dkondrat/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/ZJets_AMC/190406_001043/0000/tuple_1*.root"
out_path = "plots/EBECalibration/"

try:
    os.makedirs(out_path)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

for key, value in categories.iteritems():
    fit_zpeak(key, input_path, out_path, value, False)

