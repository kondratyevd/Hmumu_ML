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


plot_2016 = SignPlot("plot_2016", "%s 2016"%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2016/"%args.process, ROOT.kBlue)
plot_2017 = SignPlot("plot_2017", "%s 2017"%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2017/"%args.process, ROOT.kRed)
plot_2017_psweights = SignPlot("plot_2017_psweights", "%s 2017 PSweights"%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2017_psweights/"%args.process, ROOT.kOrange)
plot_2018 = SignPlot("plot_2018", "%s 2018"%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2018/"%args.process, ROOT.kGreen)
# plot_2018_test = SignPlot("plot_2018_test", "%s 2018 (250k evts)"%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2018_test/"%args.process, ROOT.kBlack)
plots = [plot_2016, plot_2017, plot_2017_psweights, plot_2018]
# plots = [plot_2017]

plot_2016_nuis = SignPlot("plot_2016", "%s 2016 w/ nuis."%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2016_nuis/"%args.process, ROOT.kBlue)
plot_2017_nuis = SignPlot("plot_2017", "%s 2017 w/ nuis."%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2017_nuis/"%args.process, ROOT.kRed)
# plot_2018_nuis = SignPlot("plot_2018", "%s 2018 w/ nuis."%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2018_nuis/"%args.process, ROOT.kGreen)
# plot_2018_test_nuis = SignPlot("plot_2018_test_nuis", "%s 2018 w/ nuis. (250k evts)"%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2018_test_nuis/"%args.process, ROOT.kBlack)

# plot_2017_nuis_0p01 = SignPlot("plot_2017_psweights_0p01", "%s 2017 1%% nuis."%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2017_psweights_nuis_0p01/"%args.process, ROOT.kOrange)
# plot_2017_nuis_0p01 = SignPlot("plot_2017", "%s 2017 w/ 0.1%% nuis."%args.process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2017_nuis_0p001/"%args.process, ROOT.kOrange)

# plots_nuis = [plot_2016_nuis, plot_2017_nuis, plot_2017_nuis_0p01]

# plots_nuis = [plot_2017_nuis_0p1, plot_2017_nuis_0p01]

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
        p.graph.SetMarkerStyle(21)
        p.graph.Draw("plsame")
    legend.AddEntry(p.graph, p.title, "pl")

legend.Draw()
canvas.Print("/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/significance_%s_psweights.png"%args.process)
canvas.SaveAs("/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/significance_%s_psweights.root"%args.process)
