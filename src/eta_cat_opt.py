import ROOT
import os, sys, errno
from array import array

ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptStat(0)
from math import *   

var_name = "mass_postFSR"

def set_out_path(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

class DataSrc(object):
    def __init__(self, name, path, tree_name, cut):
        self.name = name
        self.path = path
        self.tree_name = tree_name
        self.cut = cut

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

    return data_hist, data

signal_src = DataSrc("ggH", "combine/fsr_test_2017.root", "tree","1")

def events_in_fwhm(hist):
    bin1 = hist.FindFirstBinAbove(hist.GetMaximum()/2.0)
    bin2 = hist.FindLastBinAbove(hist.GetMaximum()/2.0)
    # print "Maximum: ", hist.GetMaximum()
    # print "bin1:    ", hist.GetBinContent(bin1)
    # print "bin2:    ", hist.GetBinContent(bin2)    
    return hist.Integral(bin1, bin2)


out_dir = "combine/eta_categories/"
set_out_path(out_dir)

steps = []
for i in range(23):
    steps.append((i+1)/10.0)
# steps = [1, 2]

hist2d = ROOT.TH2D("eta_categories", "", 23, 0.05, 2.35, 23, 0.05, 2.35)

for d1 in steps:
    for d2 in steps:
        if d2<=d1:
            continue
        else:
            print "Processing ",d1,d2

            src_1 = DataSrc("ggH_1_%.1f_%.1f"%(d1,d2), "combine/fsr_test_2017.root", "tree", "(max_abs_eta_mu>0)&(max_abs_eta_mu<%f)"%d1)
            src_2 = DataSrc("ggH_2_%.1f_%.1f"%(d1,d2), "combine/fsr_test_2017.root", "tree", "(max_abs_eta_mu>%f)&(max_abs_eta_mu<%f)"%(d1,d2))
            src_3 = DataSrc("ggH_3_%.1f_%.1f"%(d1,d2), "combine/fsr_test_2017.root", "tree", "(max_abs_eta_mu>%f)&(max_abs_eta_mu<2.4)"%d2)

            hist_1, data_1 = get_mass_hist("ggH_1_%.1f_%.1f"%(d1,d2), src_1, 100, 110, 135)
            hist_2, data_2 = get_mass_hist("ggH_2_%.1f_%.1f"%(d1,d2), src_2, 100, 110, 135)
            hist_3, data_3 = get_mass_hist("ggH_3_%.1f_%.1f"%(d1,d2), src_3, 100, 110, 135)   

            nEvts = events_in_fwhm(hist_1)+events_in_fwhm(hist_2)+events_in_fwhm(hist_3)

            canv = ROOT.TCanvas("canv_%.1f_%.1f"%(d1,d2), "canv_%.1f_%.1f"%(d1,d2), 800, 800)
            canv.cd()

            hist_1.SetLineColor(ROOT.kBlue)
            hist_2.SetLineColor(ROOT.kRed)
            hist_3.SetLineColor(ROOT.kGreen)
            hist_1.Draw('hist')
            hist_2.Draw('histsame')
            hist_3.Draw('histsame')

            maximum = 0
            for h in [hist_1, hist_2, hist_3]:
                if h.GetMaximum()>maximum:
                    maximum = h.GetMaximum()

            hist_1.SetMaximum(maximum*1.1)

            canv.Print(out_dir+"test_%.1f_%.1f.png"%(d1,d2))

            hist2d.Fill(d1, d2, nEvts)

canv = ROOT.TCanvas("canv", "canv", 800, 800)
canv.cd()
hist2d.Draw("colz")
hist2d.SetMinimum(760000)
hist2d.GetXaxis().SetTitle("First cut on max. |#eta|")
hist2d.GetYaxis().SetTitle("Second cut on max. |#eta|")
canv.SetRightMargin(0.145)
canv.SaveAs(out_dir+"test.root")
canv.Print(out_dir+"test.png")

