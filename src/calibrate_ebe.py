import ROOT
import os, sys, errno
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(1)
from math import *  
from array import array 


def fit_zpeak(cat_name, tree, out_path, cut):
    Import = getattr(ROOT.RooWorkspace, 'import')
    var = ROOT.RooRealVar("mass","mass",80,100)     
    var.setBins(500)
    var.setRange("full",80,100)
    var.setRange("6gev", 85, 97)
    w = ROOT.RooWorkspace("w_"+cat_name, False)
    Import(w, var)
    var = w.var("mass")
     
    hist_name = "%s"%cat_name
    hist = ROOT.TH1D(hist_name, hist_name, 40, 80,100)
    dummy = ROOT.TCanvas("dummy", "dummy", 800, 800)
    dummy.cd()
    tree.Draw("muPairs.mass_Roch>>%s"%(hist_name), "(%s)&muons.isTightID[0]&muons.isTightID[1]"%(cut))
    dummy.Close()

    print "Entries in hist: ",hist.GetEntries()

    w.factory("mZ[91.1876, 91.1876, 91.1876]")
    w.factory("wZ[2.4952, 2.4952, 2.4952]")
    w.factory("RooBreitWigner::bw_%s(mass, mZ, wZ)"%cat_name)


    ROOT.gSystem.Load("src/RooDCBShape_cxx.so")    
    w.factory("RooDCBShape::dcb_%s(mass, %s_mean[0, -10, 10], %s_sigma[1.5,0,6], %s_alphaL[2,0,10] , %s_alphaR[2,0,10], %s_nL[2,0,1000], %s_nR[2,0,1000])"%(cat_name,cat_name,cat_name,cat_name,cat_name,cat_name,cat_name))

    w.factory("RooFFTConvPdf::zfit_%s(mass, bw_%s, dcb_%s)"%(cat_name, cat_name, cat_name))
    model = w.pdf("zfit_%s"%cat_name)


    w.Print()
    ds = ROOT.RooDataHist("ds", "ds", ROOT.RooArgList(var), hist)
    # res = model.fitTo(ds, ROOT.RooFit.Range("full"),ROOT.RooFit.Save(), ROOT.RooFit.Verbose(False))
    res = model.fitTo(ds, ROOT.RooFit.Range("6gev"),ROOT.RooFit.Save(), ROOT.RooFit.Verbose(False))
    res.Print()

    frame = var.frame(ROOT.RooFit.Bins(500))
    ds.plotOn(frame, ROOT.RooFit.Name("ds"))
    # model.plotOn(frame, ROOT.RooFit.Range("full"), ROOT.RooFit.Name("zfit_%s"%cat_name),ROOT.RooFit.LineColor(ROOT.kRed))
    model.plotOn(frame, ROOT.RooFit.Range("6gev"), ROOT.RooFit.Name("zfit_%s"%cat_name),ROOT.RooFit.LineColor(ROOT.kRed))

    canv = ROOT.TCanvas("canv", "canv", 800, 800)
    canv.cd()
    frame.Draw()
    canv.Print("%s/zfit_%s.png"%(out_path, cat_name))


    print cat_name, " width = ", w.var("%s_sigma"%cat_name).getVal()
    return w.var("%s_sigma"%cat_name).getVal(), w.var("%s_sigma"%cat_name).getError()






pt_bins = {
           "pt_bin1": "(muons.pt[0]>30)&(muons.pt[0]<50)",
           "pt_bin2": "(muons.pt[0]>50)"
           }

B1 = "(abs(muons.eta[0])<0.9)"
B2 = "(abs(muons.eta[1])<0.9)"
O1 = "(abs(muons.eta[0])>0.9)&(abs(muons.eta[0])<1.9)"
O2 = "(abs(muons.eta[1])>0.9)&(abs(muons.eta[1])<1.9)"
E1 = "(abs(muons.eta[0])>1.9)"
E2 = "(abs(muons.eta[1])>1.9)"

BB = "%s&%s"%(B1, B2)
BO = "%s&%s"%(B1, O2)
BE = "%s&%s"%(B1, E2)

OB = "%s&%s"%(O1, B2)
OO = "%s&%s"%(O1, O2)
OE = "%s&%s"%(O1, E2)

EB = "%s&%s"%(E1, B2)
EO = "%s&%s"%(E1, O2)
EE = "%s&%s"%(E1, E2)

eta_bins = { "BB": BB,
             "BO": BO,
             "BE": BE,
             "OB": OB,
             "OO": OO,
             "OE": OE,
             "EB": EB,
             "EO": EO,
             "EE": EE
             }
eta_bin_numbers = { 
            "BB": 1,
            "BO": 2,
            "BE": 3,
            "OB": 4,
            "OO": 5,
            "OE": 6,
            "EB": 7,
            "EO": 8,
            "EE": 9
             }


# input_path = "/Users/dmitrykondratyev/Documents/HiggsToMuMu/test_files/dy/*root"
input_path_MC ="/mnt/hadoop/store/user/dkondrat/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/ZJets_AMC/190406_001043/0000/*.root"
input_path_Data ="/mnt/hadoop/store/user/dkondrat/SingleMuon/SingleMu_2017B/190415_222712/0000/*root"
input_path_DataB ="/mnt/hadoop/store/user/dkondrat/SingleMuon/SingleMu_2017B/190415_222712/0000/*root"
input_path_DataC ="/mnt/hadoop/store/user/dkondrat/SingleMuon/SingleMu_2017C/190415_222730/0000/*root"
input_path_DataD ="/mnt/hadoop/store/user/dkondrat/SingleMuon/SingleMu_2017D/190415_222746/0000/*root"
input_path_DataE ="/mnt/hadoop/store/user/dkondrat/SingleMuon/SingleMu_2017E/190415_222801/0000/*root"
input_path_DataF ="/mnt/hadoop/store/user/dkondrat/SingleMuon/SingleMu_2017F/190415_222818/0000/*root"
out_path = "plots/EBECalibration/"

try:
    os.makedirs(out_path)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise


hist_res_MC_pt_bin1 = ROOT.TH1D("res_MC_pt_bin1", "MC resolution pT bin1", 9, 0, 9)
hist_res_MC_pt_bin1.SetLineColor(ROOT.kRed)
hist_res_MC_pt_bin1.SetMarkerColor(ROOT.kRed)
hist_res_MC_pt_bin2 = ROOT.TH1D("res_MC_pt_bin2", "MC resolution pT bin2", 9, 0, 9)
hist_res_MC_pt_bin2.SetLineColor(ROOT.kGreen)
hist_res_MC_pt_bin2.SetMarkerColor(ROOT.kGreen)

hist_res_Data_pt_bin1 = ROOT.TH1D("res_Data_pt_bin1", "Data resolution pT bin1", 9, 0, 9)
hist_res_Data_pt_bin1.SetLineColor(ROOT.kBlack)
hist_res_Data_pt_bin1.SetMarkerColor(ROOT.kBlack)
hist_res_Data_pt_bin2 = ROOT.TH1D("res_Data_pt_bin2", "Data resolution pT bin2", 9, 0, 9)
hist_res_Data_pt_bin2.SetLineColor(ROOT.kBlue)
hist_res_Data_pt_bin2.SetMarkerColor(ROOT.kBlue)

hist_ratio_pt_bin1 = ROOT.TH1D("res_ratio_pt_bin1", "Data/MC pT bin1", 9, 0, 9)
hist_ratio_pt_bin1.SetLineColor(ROOT.kBlack)
hist_ratio_pt_bin1.SetMarkerColor(ROOT.kBlack)
hist_ratio_pt_bin2 = ROOT.TH1D("res_ratio_pt_bin2", "Data/MC pT bin2", 9, 0, 9)
hist_ratio_pt_bin2.SetLineColor(ROOT.kBlue)
hist_ratio_pt_bin2.SetMarkerColor(ROOT.kBlue)

hist_res_MC = {
    "pt_bin1": hist_res_MC_pt_bin1,
    "pt_bin2": hist_res_MC_pt_bin2
}

hist_res_Data = {
    "pt_bin1": hist_res_Data_pt_bin1,
    "pt_bin2": hist_res_Data_pt_bin2
}

hist_ratio = {
    "pt_bin1": hist_ratio_pt_bin1,
    "pt_bin2": hist_ratio_pt_bin2
}

# tree_MC = ROOT.TChain("dimuons/tree")
# tree_MC.Add(input_path_MC)
# print "Loaded MC tree from "+input_path_MC+" with %i entries."%tree_MC.GetEntries() 

tree_Data = ROOT.TChain("dimuons/tree")
tree_Data.Add(input_path_DataB)
tree_Data.Add(input_path_DataC)
tree_Data.Add(input_path_DataD)
tree_Data.Add(input_path_DataE)
# tree_Data.Add(input_path_DataF)

print "Loaded data tree with %i entries."%tree_Data.GetEntries() 


for eta_bin_key, eta_bin_cut in eta_bins.iteritems():
    for pt_bin_key, pt_bin_cut in pt_bins.iteritems():
        name = "%s_%s"%(pt_bin_key, eta_bin_key)
        cut = "(%s)&(%s)"%(pt_bin_cut, eta_bin_cut)

        # width_MC, widthErr_MC = fit_zpeak(name+"_MC", tree_MC, out_path, cut)
        # hist_res_MC[pt_bin_key].GetXaxis().SetBinLabel(eta_bin_numbers[eta_bin_key], eta_bin_key)
        # hist_res_MC[pt_bin_key].GetYaxis().SetTitle("#sigma_{res}(GeV)")
        # hist_res_MC[pt_bin_key].SetBinContent(eta_bin_numbers[eta_bin_key], width_MC)
        # hist_res_MC[pt_bin_key].SetBinError(eta_bin_numbers[eta_bin_key], widthErr_MC)
        # hist_res_MC[pt_bin_key].SetLineWidth(2)
        # hist_res_MC[pt_bin_key].SetMarkerStyle(20)

        width_Data, widthErr_Data = fit_zpeak(name+"_Data", tree_Data, out_path, cut)
        hist_res_Data[pt_bin_key].GetXaxis().SetBinLabel(eta_bin_numbers[eta_bin_key], eta_bin_key)
        hist_res_Data[pt_bin_key].GetYaxis().SetTitle("#sigma_{res}(GeV)")
        hist_res_Data[pt_bin_key].SetBinContent(eta_bin_numbers[eta_bin_key], width_Data)
        hist_res_Data[pt_bin_key].SetBinError(eta_bin_numbers[eta_bin_key], widthErr_Data)
        hist_res_Data[pt_bin_key].SetLineWidth(2)
        hist_res_Data[pt_bin_key].SetMarkerStyle(20)

        # hist_ratio[pt_bin_key].GetXaxis().SetBinLabel(eta_bin_numbers[eta_bin_key], eta_bin_key)
        # hist_ratio[pt_bin_key].GetYaxis().SetTitle("#sigma_{Data} / #sigma_{MC}")
        # hist_ratio[pt_bin_key].SetBinContent(eta_bin_numbers[eta_bin_key], width_Data/width_MC)
        # hist_ratio[pt_bin_key].SetLineWidth(2)
        # hist_ratio[pt_bin_key].SetMarkerStyle(20)
      
canv = ROOT.TCanvas("c", "c", 800, 800)
canv.cd() 
legend = ROOT.TLegend(0.11, 0.7, 0.35, 0.89)

for hist_key, hist in hist_res_MC.iteritems():
    hist_res_MC[hist_key].Draw("histe1same")
    legend.AddEntry(hist_res_MC[hist_key], hist_res_MC[hist_key].GetTitle(), "ple1")

for hist_key, hist in hist_res_Data.iteritems():
    hist_res_Data[hist_key].Draw("histe1same")
    legend.AddEntry(hist_res_Data[hist_key], hist_res_Data[hist_key].GetTitle(), "ple1")

legend.Draw()

canv.SaveAs("%s/resolution.png"%out_path) 
canv.SaveAs("%s/resolution.root"%out_path) 


canv = ROOT.TCanvas("c1", "c1", 800, 800)
canv.cd() 
legend = ROOT.TLegend(0.11, 0.7, 0.35, 0.89)

for hist_key, hist in hist_ratio.iteritems():
    hist_ratio[hist_key].Draw("histe1same")
    legend.AddEntry(hist_ratio[hist_key], hist_ratio[hist_key].GetTitle(), "ple1")

legend.Draw()

canv.SaveAs("%s/DataMCratio.png"%out_path) 
canv.SaveAs("%s/DataMCratio.root"%out_path) 