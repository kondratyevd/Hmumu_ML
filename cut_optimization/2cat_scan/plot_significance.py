import ROOT
ROOT.gStyle.SetOptStat(0)
import argparse

class SignPlot(object):
    def __init__(self, name, title, path, color):
        self.name = name
        self.title = title
        self.path = path
        self.color = color
        self.graph = ROOT.TGraph()
        self.graph.SetName(name)
        self.graph.SetLineColor(color)
        self.graph.SetMarkerColor(color)
        self.graph.SetLineWidth(2)
        self.graph.SetMarkerStyle(20)
        self.graph.SetMarkerSize(1.5)


parser = argparse.ArgumentParser(description='')
parser.add_argument('-p', action='store', dest='process', default="ggh", help='Process')
args = parser.parse_args()


plot_2016 = SignPlot("plot_2016", "%s 3Gaus 2016"%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2016/"%args.process, ROOT.kBlue)
plot_2017 = SignPlot("plot_2017", "%s 2017"%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2017/"%args.process, ROOT.kRed)
plot_2017_psweights = SignPlot("plot_2017_psweights", "%s 3Gaus 2017"%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2017_psweights/"%args.process, ROOT.kBlack)
plot_2017_dcb = SignPlot("plot_2017_dcb", "%s DCB"%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2017_dcb/"%args.process, ROOT.kRed)
plot_2018 = SignPlot("plot_2018", "%s 3Gaus 2018"%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2018/"%args.process, ROOT.kGreen)
# plot_2018_test = SignPlot("plot_2018_test", "%s 2018 (250k evts)"%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2018_test/"%args.process, ROOT.kBlack)
# plots = [plot_2016, plot_2017_psweights, plot_2018]
# plots = [plot_2017_psweights, plot_2017_dcb, plot_2016, plot_2018]
plots = [plot_2017_psweights, plot_2017_dcb]
# plots = [plot_2017_psweights]
# plot_2016_nuis = SignPlot("plot_2016", "%s 2016 w/ nuis."%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2016_nuis/"%args.process, ROOT.kBlue)
# plot_2017_nuis = SignPlot("plot_2017", "%s 2017 w/ nuis."%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2017_nuis/"%args.process, ROOT.kRed)
# plot_2018_nuis = SignPlot("plot_2018", "%s 2018 w/ nuis."%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2018_nuis/"%args.process, ROOT.kGreen)
# plot_2018_test_nuis = SignPlot("plot_2018_test_nuis", "%s 2018 w/ nuis. (250k evts)"%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2018_test_nuis/"%args.process, ROOT.kBlack)

# plot_2017_nuis_0p01 = SignPlot("plot_2017_psweights_0p01", "%s 2017 1%% nuis."%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2017_psweights_nuis_0p01/"%args.process, ROOT.kBlack)
# plot_2017_nuis_0p02 = SignPlot("plot_2017_psweights_0p02", "%s 2017 2%% nuis."%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2017_psweights_nuis_0p02/"%args.process, ROOT.kBlack)
# plot_2017_nuis_0p05 = SignPlot("plot_2017_psweights_0p05", "%s 2017 5%% nuis."%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2017_psweights_nuis_0p05/"%args.process, ROOT.kBlack)
plot_2017_nuis_0p1 = SignPlot("plot_2017_psweights_0p1", "%s 3Gaus scale & res. unc."%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2017_psweights_nuis_0p1/"%args.process, ROOT.kBlack)
plot_2017_nuis_0p1_new = SignPlot("plot_2017_psweights_0p1_new", "%s 3Gaus new"%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2017_psweights_nuis_0p1_new/"%args.process, ROOT.kGreen)
plot_2017_nuis_0p1_old = SignPlot("plot_2017_psweights_0p1_old", "%s 3Gaus old"%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2017_psweights_nuis_0p1_old/"%args.process, ROOT.kOrange)
plot_2017_nuis_0 = SignPlot("plot_2017_psweights_0", "%s 3Gaus only scale unc."%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2017_psweights_nuis_0/"%args.process, ROOT.kBlack)
plot_2017_onlyres = SignPlot("plot_2017_psweights_onlyres", "%s 3Gaus only res. unc."%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2017_psweights_onlyres/"%args.process, ROOT.kBlack)
plot_2017_nuis_onlyinnermost = SignPlot("plot_2017_psweights_onlyinnermost", "%s 3Gaus res. unc. only for innermost Gaus."%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2017_psweights_onlyinnermost/"%args.process, ROOT.kBlack)
plot_2017_dcb_nuis = SignPlot("plot_2017_dcb_nuis", "%s DCB scale & res. unc."%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2017_dcb_nuis/"%args.process, ROOT.kRed)
plot_2017_dcb_nuis_0 = SignPlot("plot_2017_dcb_nuis_0", "%s DCB only scale unc."%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2017_psweights_nuis_0_dcb/"%args.process, ROOT.kRed)
plot_2017_onlyres_dcb = SignPlot("plot_2017_psweights_onlyres_dcb", "%s DCB only res. unc."%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2017_psweights_onlyres_dcb/"%args.process, ROOT.kRed)
# plot_2017_nuis_0p01.graph.SetMarkerStyle(21)
# plot_2017_nuis_0p02.graph.SetMarkerStyle(22)
# plot_2017_nuis_0p05.graph.SetMarkerStyle(23)
plot_2017_nuis_0p1.graph.SetMarkerStyle(24)
plot_2017_nuis_0p1_new.graph.SetMarkerStyle(29)
plot_2017_dcb_nuis.graph.SetMarkerStyle(24)
plot_2017_nuis_0.graph.SetMarkerStyle(23)
plot_2017_dcb_nuis_0.graph.SetMarkerStyle(23)
plot_2017_onlyres.graph.SetMarkerStyle(22)
plot_2017_nuis_onlyinnermost.graph.SetMarkerStyle(22)
plot_2017_onlyres_dcb.graph.SetMarkerStyle(22)
# plots_nuis = [plot_2017_nuis_0p01, plot_2017_nuis_0p02, plot_2017_nuis_0p05, plot_2017_nuis_0p1]
# plots_nuis = [plot_2017_nuis_0p1, plot_2017_nuis_onlyinnermost, plot_2017_dcb_nuis, plot_2017_dcb_nuis_0, plot_2017_nuis_0]
# plots_nuis = [plot_2017_nuis_0p1, plot_2017_dcb_nuis, plot_2017_dcb_nuis_0, plot_2017_nuis_0]
# plots_nuis = [plot_2017_nuis_0, plot_2017_onlyres, plot_2017_nuis_0p1, plot_2017_dcb_nuis_0, plot_2017_onlyres_dcb, plot_2017_dcb_nuis, plot_2017_nuis_onlyinnermost]
plots_nuis = [plot_2017_nuis_0p1, plot_2017_nuis_0p1_new, plot_2017_nuis_0p1_old, plot_2017_dcb_nuis]
# plots_nuis = []
canvas = ROOT.TCanvas("c", "c", 800, 800)
canvas.cd()

legend = ROOT.TLegend(0.65, 0.7, 0.89, 0.89)

for ip, p in enumerate(plots):
    for i in range(1, 24):
        tree = ROOT.TChain("limit")
        tree.Add(p.path+"higgsCombine_2cat_%i.Significance.mH120.root"%i)
        for iev,  event in enumerate(tree):
            p.graph.SetPoint(i-1, i/10.0, event.limit)
    if not ip:
        p.graph.Draw("apl")
        p.graph.GetXaxis().SetRangeUser(0, 2.4)
        p.graph.GetXaxis().SetTitle("Rapidity cut")
        p.graph.GetYaxis().SetTitle("Significance")
        p.graph.GetYaxis().SetTitleOffset(1.35)
    else:
        p.graph.Draw("plsame")
    legend.AddEntry(p.graph, p.title, "pl")

for ip, p in enumerate(plots_nuis):
    for i in range(1, 24):
        tree = ROOT.TChain("limit")
        tree.Add(p.path+"higgsCombine_2cat_%i.Significance.mH120.root"%i)
        for iev,  event in enumerate(tree):
            p.graph.SetPoint(i-1, i/10.0, event.limit)
        p.graph.SetLineStyle(2)
        # p.graph.SetMarkerStyle(21)
        p.graph.Draw("plsame")
    legend.AddEntry(p.graph, p.title, "pl")

legend.Draw()
canvas.Print("/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/significance_%s_nuis_test_4.png"%args.process)
canvas.SaveAs("/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/significance_%s_nuis_test_4.root"%args.process)
