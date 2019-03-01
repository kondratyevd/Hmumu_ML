import ROOT
import os, sys, errno
from array import array
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(1)
from math import * 

class SignalSrc(object):
    def __init__(self, name, title, path, tree_name, cut, color):
        self.name = name
        self.title = title
        self.path = path
        self.tree_name = tree_name
        self.cut = cut
        self.color = color
        self.hist = ROOT.TH1D("hist_"+name, title, 24, 0, 2.4)
        self.hist.SetName(name)
        self.hist.SetMarkerColor(color)
        self.hist.SetLineColor(color)
        r = g = b = array("f", [0])
        color_i = ROOT.gROOT.GetColor(color)
        color_i.GetRGB(r,g,b)
        color_tr = ROOT.TColor(2000, r[0], g[0], b[0], " ", 0.5)
        self.hist.SetFillColor(color_tr)
        self.hist.SetLineWidth(2)
        self.hist.SetMarkerStyle(20)
        self.hist.SetMarkerSize(0.8)

        self.hist_copy = ROOT.TH1D("hist_copy_"+name, title, 24, 0, 2.4)
        self.hist_copy.SetName(name)
        self.hist_copy.SetMarkerColor(color)
        self.hist_copy.SetLineColor(color)
        # self.hist_copy.SetFillColor(color)
        self.hist_copy.SetLineWidth(2)
        self.hist_copy.SetMarkerStyle(20)
        self.hist_copy.SetMarkerSize(0.8)

        self.graph_chi2 = ROOT.TGraphErrors()
        self.graph_chi2.SetName(name)
        self.graph_chi2.SetMarkerColor(color)
        self.graph_chi2.SetLineColor(color)
        self.graph_chi2.SetLineWidth(2)
        self.graph_chi2.SetMarkerStyle(20)
        self.graph_chi2.SetMarkerSize(0.8)

class FitOutput(object):
    def __init__(self, mean, mean_err, width, width_err, chi2, frame):
        self.mean = mean
        self.mean_err = mean_err
        self.width  = width 
        self.width_err  = width_err 
        self.chi2  = chi2 
        self.frame  = frame 

try:
    os.makedirs("plots")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

def get_mass_hist(src, suff, var_name, nBins, xmin, xmax, cut="", normalize=False):
    tree = ROOT.TChain(src.tree_name)
    tree.Add(src.path)  
    hist = ROOT.TH1D(src.name+suff, src.name, nBins, xmin, xmax)
    dummy = ROOT.TCanvas("dummy", "dummy", 800, 800)
    dummy.cd()
    tree.Draw("%s>>%s"%(var_name,src.name+suff), "(%s)*(%s)"%(cut,src.cut))
    dummy.Close()
    if normalize:
        if hist.Integral():
            hist.Scale(1/hist.Integral())
    return hist, tree        


def fit_in_eta_bin(src, eta_lo, eta_hi, fit_func, var_name, nMassBins, massMin, massMax):

    try:
        os.makedirs("plots/"+src.name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    eta_cut = "((max_abs_eta_mu>%.1f)&(max_abs_eta_mu<%.1f))"%(eta_lo, eta_hi)

    signal_hist, signal_tree = get_mass_hist(src, "_%.1f"%eta_lo, var_name, nMassBins, massMin, massMax, eta_cut, True)

    var = ROOT.RooRealVar(var_name, "Dilepton mass", massMin, massMax)

    max_abs_eta_var = ROOT.RooRealVar("max_abs_eta_mu","Max abs(eta) of muons", 0, 2.4)
    mu1_eta = ROOT.RooRealVar("mu1_eta","mu1_eta", -2.4, 2.4)
    mu2_eta = ROOT.RooRealVar("mu2_eta","mu2_eta", -2.4, 2.4)
    weight_over_lumi = ROOT.RooRealVar("weight_over_lumi","weight_over_lumi", -1000, 1000)
    
    var.setRange("full",massMin,massMax)

    Import = getattr(ROOT.RooWorkspace, 'import')
    w = ROOT.RooWorkspace("w", False)
    Import(w, var)
    
    ROOT.gROOT.ProcessLine(".L RooDCBShape.cxx")
    w.factory("RooDCBShape::dcb(%s, mean[125,120,130], sigma[2,0,5], alphaL[2,0,25] , alphaR[2,0,25], nL[1.5,0,25], nR[1.5,0,25])"%var_name)
    smodel = w.pdf("dcb")
    
    signal_ds = ROOT.RooDataSet("signal_ds_"+src.name,"signal_ds_"+src.name, signal_tree, ROOT.RooArgSet(var, max_abs_eta_var, mu1_eta, mu2_eta, weight_over_lumi), "(%s)*(%s)"%(eta_cut,src.cut))
    res = smodel.fitTo(signal_ds, ROOT.RooFit.Range("full"),ROOT.RooFit.Save(), ROOT.RooFit.Verbose(False))
    res.Print("v")
    minNll = res.minNll()

    frame = var.frame(ROOT.RooFit.Bins(nMassBins))
    signal_ds.plotOn(frame, ROOT.RooFit.Name("signal_ds"))
    smodel.plotOn(frame, ROOT.RooFit.Range("window"), ROOT.RooFit.Name("signal_dcb"),ROOT.RooFit.LineColor(ROOT.kRed))


    chi2 = frame.chiSquare("signal_dcb", "signal_ds", 6)
    print "DCB chi2/d.o.f: ", chi2
    canv = ROOT.TCanvas("canv2", "canv2", 800, 800)
    canv.cd()
    statbox = smodel.paramOn(frame, ROOT.RooFit.Layout(0.1, 0.4, 0.9))
    frame.getAttText().SetTextSize(0.02)
    t1 = ROOT.TPaveLabel(0.7,0.83,0.9,0.9, "#chi^{2}/dof = %.4f"%chi2,"brNDC")
    t1.SetFillColor(0)
    t1.SetTextSize(0.4)
    frame.addObject(t1)
    frame.SetTitle(src.title+" DCB, %.1f<|#eta|<%.1f"%(eta_lo, eta_hi))
    frame.Draw()
    statbox.Draw("same")
    canv.Print("plots/%s/DCB_%.1f-%.1f.png"%(src.name, eta_lo, eta_hi))


    mean = w.var("mean").getVal()
    mean_err = w.var("mean").getError()
    width = w.var("sigma").getVal()
    width_err = w.var("sigma").getError()
    output = FitOutput(mean, mean_err, width, width_err, chi2, frame)
    return output


def make_resolution_plot(sources, label):
    legend = ROOT.TLegend(0.13, 0.75, 0.35, 0.89)
    for src in sources:
        for i in range(1,25):
            eta_lo = (i-1)/10.0
            eta_hi = i/10.0
            fit_output = fit_in_eta_bin(src, eta_lo, eta_hi, "DCB", "mass", 100, 110, 135)
            src.hist.SetBinContent(i, fit_output.width)
            src.hist.SetBinError(i, fit_output.width_err)
            src.hist_copy.SetBinContent(i, fit_output.width)
            src.hist_copy.SetBinError(i, 0)            
            src.graph_chi2.SetPoint(i-1, i/10.0-0.05, fit_output.chi2)
            src.graph_chi2.SetPointError(i-1, 0.05, 0)            
        legend.AddEntry(src.hist, src.title, "pfl")
    canvas = ROOT.TCanvas("c","c",800,800)
    canvas.cd()
    for src in sources:
        src.hist.Draw("pe2same")
        # src.hist_copy.Draw("pe0same")
        src.hist.SetTitle("")
        src.hist.GetXaxis().SetTitle("max. |#eta| of two muons")
        src.hist.GetYaxis().SetTitle("width, GeV")
        src.hist.GetYaxis().SetTitleOffset(1.22)
        src.hist.SetMinimum(0)
        src.hist.SetMaximum(4)
    legend.Draw()
    canvas.SaveAs("plots/%s.png"%label)
    canvas.SaveAs("plots/%s.root"%label)
    canvas.Close()

    canvas_chi2 = ROOT.TCanvas("c1","c1",800,800)
    canvas_chi2.cd()
    for isrc, src in enumerate(sources):
        if not isrc:
            src.graph_chi2.Draw("ape1l")
        else:
            src.graph_chi2.Draw("pe1lsame")            
        src.graph_chi2.SetTitle("")
        src.graph_chi2.GetXaxis().SetTitle("max. |#eta| of two muons")
        src.graph_chi2.GetYaxis().SetTitle("#chi^{2}/d.o.f")
        src.graph_chi2.GetYaxis().SetTitleOffset(1.22)
        src.graph_chi2.GetXaxis().SetRangeUser(0, 2.4)
        src.graph_chi2.SetMinimum(0)
        src.graph_chi2.SetMaximum(3)
    legend.Draw()
    canvas_chi2.SaveAs("plots/%s_chi2.png"%label)
    canvas_chi2.SaveAs("plots/%s_chi2.root"%label)
    canvas_chi2.Close()

# source = SignalSrc("zh_2017", "ZH 2017", "~/Hmumu_analysis/Hmumu_ML/cut_optimization/miniaod_skim/zh_2017.root", "tree", "1", ROOT.kOrange-3)
# fit_output = fit_in_eta_bin(source, 2.3, 2.4, "DCB", "mass", 100, 110, 135)
# print "Chi2/d.o.f = ", fit_output.chi2

# sources_2016 = []
# sources_2016.append(SignalSrc("ggH_2016", "ggH 2016", "/mnt/hadoop/store/user/dkondrat/skim/2016/H2Mu_gg/*root", "dimuons/tree", "1", ROOT.kBlack))
# sources_2016.append(SignalSrc("vbf_2016", "VBF 2016", "/mnt/hadoop/store/user/dkondrat/skim/2016/H2Mu_VBF/*root", "dimuons/tree", "1", ROOT.kBlue))
# make_resolution_plot(sources_2016, "resolution_2016")

# sources_2017 = []
# sources_2017.append(SignalSrc("ggH_2017", "ggH 2017", "~/Hmumu_analysis/Hmumu_ML/cut_optimization/miniaod_skim/ggh_2017.root", "tree", "1", ROOT.kBlack))
# sources_2017.append(SignalSrc("vbf_2017", "VBF 2017", "~/Hmumu_analysis/Hmumu_ML/cut_optimization/miniaod_skim/vbf_2017.root", "tree", "1", ROOT.kBlue))
# sources_2017.append(SignalSrc("wplush_2017", "W+H 2017", "~/Hmumu_analysis/Hmumu_ML/cut_optimization/miniaod_skim/wplush_2017.root", "tree", "1", ROOT.kViolet))
# sources_2017.append(SignalSrc("wminush_2017", "W-H 2017", "~/Hmumu_analysis/Hmumu_ML/cut_optimization/miniaod_skim/wminush_2017.root", "tree", "1", ROOT.kGreen))
# sources_2017.append(SignalSrc("zh_2017", "ZH 2017", "~/Hmumu_analysis/Hmumu_ML/cut_optimization/miniaod_skim/zh_2017.root", "tree", "1", ROOT.kOrange-3))
# sources_2017.append(SignalSrc("tth_2017", "ttH 2017", "~/Hmumu_analysis/Hmumu_ML/cut_optimization/miniaod_skim/tth_2017.root", "tree", "1", ROOT.kRed))
# make_resolution_plot(sources_2017, "resolution_2017")

# sources_2018 = []
# sources_2018.append(SignalSrc("ggH_2018", "ggH 2018", "~/Hmumu_analysis/Hmumu_ML/cut_optimization/miniaod_skim/ggh_2018.root", "tree", "1", ROOT.kBlack))
# sources_2018.append(SignalSrc("vbf_2018", "VBF 2018", "~/Hmumu_analysis/Hmumu_ML/cut_optimization/miniaod_skim/vbf_2018.root", "tree", "1", ROOT.kBlue))
# sources_2018.append(SignalSrc("wplush_2018", "W+H 2018", "~/Hmumu_analysis/Hmumu_ML/cut_optimization/miniaod_skim/wplush_2018.root", "tree", "1", ROOT.kViolet))
# sources_2018.append(SignalSrc("wminush_2018", "W-H 2018", "~/Hmumu_analysis/Hmumu_ML/cut_optimization/miniaod_skim/wminush_2018.root", "tree", "1", ROOT.kGreen))
# sources_2018.append(SignalSrc("zh_2018", "ZH 2018", "~/Hmumu_analysis/Hmumu_ML/cut_optimization/miniaod_skim/zh_2018.root", "tree", "1", ROOT.kOrange-3))
# sources_2018.append(SignalSrc("tth_2018", "ttH 2018", "~/Hmumu_analysis/Hmumu_ML/cut_optimization/miniaod_skim/tth_2018.root", "tree", "1", ROOT.kRed))
# make_resolution_plot(sources_2018, "resolution_2018")

sources_ggh = []
sources_ggh.append(SignalSrc("ggH_2016", "ggH 2016", "/mnt/hadoop/store/user/dkondrat/skim/2016/H2Mu_gg/*root", "dimuons/tree", "1", ROOT.kBlack))
sources_ggh.append(SignalSrc("ggH_2017", "ggH 2017", "~/Hmumu_analysis/Hmumu_ML/cut_optimization/miniaod_skim/ggh_2017_psweights.root", "tree", "1", ROOT.kRed))
sources_ggh.append(SignalSrc("ggH_2018", "ggH 2018", "~/Hmumu_analysis/Hmumu_ML/cut_optimization/miniaod_skim/ggh_2018.root", "tree", "1", ROOT.kBlue))
make_resolution_plot(sources_ggh, "resolution_ggh")

# sources_vbf = []
# sources_vbf.append(SignalSrc("vbf_2016", "VBF 2016", "/mnt/hadoop/store/user/dkondrat/skim/2016/H2Mu_VBF/*root", "dimuons/tree", "1", ROOT.kBlack))
# sources_vbf.append(SignalSrc("vbf_2017", "VBF 2017", "~/Hmumu_analysis/Hmumu_ML/cut_optimization/miniaod_skim/vbf_2017.root", "tree", "1", ROOT.kRed))
# sources_vbf.append(SignalSrc("vbf_2018", "VBF 2018", "~/Hmumu_analysis/Hmumu_ML/cut_optimization/miniaod_skim/vbf_2018.root", "tree", "1", ROOT.kBlue))
# make_resolution_plot(sources_vbf, "resolution_vbf")

# sources_wplush = []
# sources_wplush.append(SignalSrc("wplush_2017", "W+H 2017", "~/Hmumu_analysis/Hmumu_ML/cut_optimization/miniaod_skim/wplush_2017.root", "tree", "1", ROOT.kRed))
# sources_wplush.append(SignalSrc("wplush_2018", "W+H 2018", "~/Hmumu_analysis/Hmumu_ML/cut_optimization/miniaod_skim/wplush_2018.root", "tree", "1", ROOT.kBlue))
# make_resolution_plot(sources_wplush, "resolution_wplush")

# sources_wminush = []
# sources_wminush.append(SignalSrc("wminush_2017", "W-H 2017", "~/Hmumu_analysis/Hmumu_ML/cut_optimization/miniaod_skim/wminush_2017.root", "tree", "1", ROOT.kRed))
# sources_wminush.append(SignalSrc("wminush_2018", "W-H 2018", "~/Hmumu_analysis/Hmumu_ML/cut_optimization/miniaod_skim/wminush_2018.root", "tree", "1", ROOT.kBlue))
# make_resolution_plot(sources_wminush, "resolution_wminush")

# sources_zh = []
# sources_zh.append(SignalSrc("zh_2017", "ZH 2017", "~/Hmumu_analysis/Hmumu_ML/cut_optimization/miniaod_skim/zh_2017.root", "tree", "1", ROOT.kRed))
# sources_zh.append(SignalSrc("zh_2018", "ZH 2018", "~/Hmumu_analysis/Hmumu_ML/cut_optimization/miniaod_skim/zh_2018.root", "tree", "1", ROOT.kBlue))
# make_resolution_plot(sources_zh, "resolution_zh")

# sources_tth = []
# sources_tth.append(SignalSrc("tth_2017", "ttH 2017", "~/Hmumu_analysis/Hmumu_ML/cut_optimization/miniaod_skim/tth_2017.root", "tree", "1", ROOT.kRed))
# sources_tth.append(SignalSrc("tth_2018", "ttH 2018", "~/Hmumu_analysis/Hmumu_ML/cut_optimization/miniaod_skim/tth_2018.root", "tree", "1", ROOT.kBlue))
# make_resolution_plot(sources_tth, "resolution_tth")

