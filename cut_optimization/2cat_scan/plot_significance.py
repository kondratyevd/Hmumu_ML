import ROOT
ROOT.gStyle.SetOptStat(0)

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
        self.graph.SetMarkerSize(2)


parser = argparse.ArgumentParser(description='')
parser.add_argument('-p', action='store', dest='process', default="ggh" help='Process')
args = parser.parse_args()


plot_2016 = SignPlot("plot_2016", "%s 2016"%process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2016/"%process, ROOT.kBlue)
plot_2017 = SignPlot("plot_2017", "%s 2017"%process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2017/"%process, ROOT.kRed)
plot_2018 = SignPlot("plot_2018", "%s 2018"%process, "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/output/%s_2018/"%process, ROOT.kGreen)
plots = [plot_2016, plot_2017, plot_2018]

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
legend.Draw()
canvas.Print("/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/significance_%s.png"%process)
canvas.SaveAs("/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/2cat_scan/significance_%s.root"%process)
