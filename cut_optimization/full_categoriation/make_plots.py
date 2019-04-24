import ROOT
ROOT.gStyle.SetOptStat(0)
import argparse

class SignPlot(object):
    def __init__(self, name, title, path, color, prefix, postfix, linestyle=1):
        self.name = name
        self.title = title
        self.path = path
        self.color = color
        self.prefix = prefix
        self.postfix = postfix
        self.graph = ROOT.TGraph()
        self.graph.SetName(name)
        self.graph.SetLineColor(color)
        self.graph.SetMarkerColor(color)
        self.graph.SetLineWidth(2)
        self.graph.SetLineStyle(linestyle)
        self.graph.SetMarkerStyle(20)
        self.graph.SetMarkerSize(1.5)


parser = argparse.ArgumentParser(description='')

parser.add_argument('--out_path', action='store', dest='out_path', help='Output path')
parser.add_argument('--label', action='store', dest='label', help='label')
args = parser.parse_args()


plot_011_full = SignPlot("BDT_1_1_full", "BDT 1 cut, full", "output/scan_0.1.1/", ROOT.kBlack, "higgsCombine_dnn_option0.1.1_full_", ".Significance.mH120.root")
plot_111_full = SignPlot("DNN_1_1_full", "DNN 1 cut, full", "output/scan_1.1.1/", ROOT.kRed, "higgsCombine_dnn_option1.1.1_full_", ".Significance.mH120.root")


plot_011_mva = SignPlot("BDT_1_1_mva", "BDT 1 cut, mva", "output/scan_0.1.1/", ROOT.kBlack, "higgsCombine_dnn_option0.1.1_mva_", ".Significance.mH120.root", 2)
plot_111_mva = SignPlot("DNN_1_1_mva", "DNN 1 cut, mva", "output/scan_1.1.1/", ROOT.kRed, "higgsCombine_dnn_option1.1.1_mva_", ".Significance.mH120.root", 2)

plots = [ plot_111_full,  plot_111_mva ]
# plots = [plot_011_full, plot_111_full, plot_011_mva , plot_111_mva ]

canvas = ROOT.TCanvas("c", "c", 800, 800)
canvas.cd()

legend = ROOT.TLegend(0.65, 0.7, 0.89, 0.89)

for ip, p in enumerate(plots):
    for i in range(10):
        tree = ROOT.TChain("limit")
        tree.Add(p.path+p.prefix+"%i"%i+p.postfix)
        for iev,  event in enumerate(tree):
            p.graph.SetPoint(i, i/10.0, event.limit)
    if not ip:
        p.graph.Draw("apl")
        p.graph.GetXaxis().SetRangeUser(0, 1)
        p.graph.GetXaxis().SetTitle("MVA cut")
        p.graph.GetYaxis().SetTitle("Significance")
        p.graph.GetYaxis().SetTitleOffset(1.35)
    else:
        p.graph.Draw("plsame")
    legend.AddEntry(p.graph, p.title, "pl")


legend.Draw()
canvas.Print("%s/%s_mva.png"%(args.out_path, args.label))
canvas.SaveAs("%s/%s_mva.root"%(args.out_path, args.label))