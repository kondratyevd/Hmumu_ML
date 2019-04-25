import ROOT
ROOT.gStyle.SetOptStat(0)
import argparse

def plot_scan(plots_list, name):
    canvas = ROOT.TCanvas(name, name, 800, 800)
    canvas.cd()

    legend = ROOT.TLegend(0.65, 0.7, 0.89, 0.89)
    legend.SetHeader("2017 Data (L = 41.4 /fb)")
    min_x = 10
    max_x = -10
    for ip, p in enumerate(plots_list):
        if p.min<min_x:
            min_x=p.min
        if p.max>max_x:
            max_x=p.max 
        for i in range(1, p.nSteps):
            step = (p.max-p.min)/float(p.nSteps)
            x = p.min+i*step
            tree = ROOT.TChain("limit")
            tree.Add(p.path+p.prefix+"%i"%i+p.postfix)
            for iev,  event in enumerate(tree):
                p.graph.SetPoint(i-1, x, event.limit)
        if not ip:
            p.graph.Draw("apl")
            p.graph.GetXaxis().SetRangeUser(min_x, max_x)
            p.graph.GetXaxis().SetTitle("MVA cut")
            p.graph.GetYaxis().SetTitle("Significance")
            p.graph.GetYaxis().SetTitleOffset(1.35)
            p.graph.SetMinimum(0.6)
            p.graph.SetMaximum(1)
        else:
            p.graph.Draw("plsame")
        if p.legEntry:
            legend.AddEntry(p.graph, p.title, "pl")

    # These values are for 2017 Data/MC, lumi = 41394.221, eta categories [0, 0.9, 1.9, 2.4]
    inclusive = ROOT.TLine(min_x,0.696594,max_x,0.696594)
    inclusive.SetLineColor(ROOT.kBlue)
    inclusive.SetLineStyle(2)
    inclusive.SetLineWidth(2)
    inclusive.Draw("same")
    inclusive_eta = ROOT.TLine(min_x,0.738164,max_x,0.738164)
    inclusive_eta.SetLineColor(ROOT.kBlue)
    inclusive_eta.SetLineWidth(2)
    inclusive_eta.Draw("same")
    legend.AddEntry(inclusive_eta, "3 #eta categories", "l")
    legend.AddEntry(inclusive, "Inclusive", "l")

    legend.SetBorderSize(0)
    legend.Draw()
    canvas.Print("%s/%s_%s.png"%(args.out_path, args.label, name))
    canvas.SaveAs("%s/%s_%s.root"%(args.out_path, args.label, name))


class SignPlot(object):
    def __init__(self, min, max, nSteps, name, title, path, prefix, postfix, color, linestyle=1, legEntry=True):
        self.min = min
        self.max = max
        self.nSteps = nSteps
        self.name = name
        self.title = title
        self.path = path
        self.color = color
        self.prefix = prefix
        self.postfix = postfix
        self.legEntry = legEntry        
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

plot_011_mva = SignPlot(-1, 1, 10, "BDT_1_1_mva", "BDT 1 cut, only mva", "output/scan_0.1.1/",  "higgsCombine_dnn_option0.1.1_mva_", ".Significance.mH120.root",ROOT.kBlack, linestyle=2, legEntry=True)
plot_011_full = SignPlot(-1, 1, 10, "BDT_1_1_full", "BDT 1 cut, mva & #eta", "output/scan_0.1.1/", "higgsCombine_dnn_option0.1.1_full_", ".Significance.mH120.root", ROOT.kBlack, legEntry=True)

plot_012_mva = SignPlot(-0.4, 0, 20, "BDT_1_2_mva", "BDT 1 cut tune1, only mva", "output/scan_0.1.2/",  "higgsCombine_dnn_option0.1.2_mva_", ".Significance.mH120.root",ROOT.kBlack, linestyle=2, legEntry=False)
plot_012_full = SignPlot(-0.4, 0, 20, "BDT_1_2_full", "BDT 1 cut tune1, mva & #eta", "output/scan_0.1.2/", "higgsCombine_dnn_option0.1.2_full_", ".Significance.mH120.root", ROOT.kBlack, legEntry=False)

plot_013_mva = SignPlot(0.2, 0.8, 30, "BDT_1_3_mva", "BDT 1 cut tune2, only mva", "output/scan_0.1.3/",  "higgsCombine_dnn_option0.1.3_mva_", ".Significance.mH120.root",ROOT.kBlack, linestyle=2, legEntry=False)
plot_013_full = SignPlot(0.2, 0.8, 30, "BDT_1_3_full", "BDT 1 cut tune2, mva & #eta", "output/scan_0.1.3/", "higgsCombine_dnn_option0.1.3_full_", ".Significance.mH120.root", ROOT.kBlack, legEntry=False)

plots_bdt1 = [plot_011_full, plot_011_mva, plot_012_full, plot_012_mva, plot_013_full, plot_013_mva]

plot_111_mva = SignPlot(1, 3, 10, "DNN_1_1_mva", "DNN 1 cut, only mva", "output/scan_1.1.1/",  "higgsCombine_dnn_option1.1.1_mva_", ".Significance.mH120.root",ROOT.kRed, 2, legEntry=True)
plot_111_full = SignPlot(1, 3, 10, "DNN_1_1_full", "DNN 1 cut, mva & #eta", "output/scan_1.1.1/",  "higgsCombine_dnn_option1.1.1_full_", ".Significance.mH120.root", ROOT.kRed, legEntry=True)

plot_112_mva = SignPlot(1.2, 1.8, 30, "DNN_1_2_mva", "DNN 1 cut tune, only mva", "output/scan_1.1.2/",  "higgsCombine_dnn_option1.1.2_mva_", ".Significance.mH120.root",ROOT.kRed, 2, legEntry=False)
plot_112_full = SignPlot(1.2, 1.8, 30, "DNN_1_2_full", "DNN 1 cut tune, mva & #eta", "output/scan_1.1.2/",  "higgsCombine_dnn_option1.1.2_full_", ".Significance.mH120.root", ROOT.kRed, legEntry=False)

plots_dnn1 = [ plot_111_full,  plot_111_mva, plot_112_full,  plot_112_mva ]

plot_scan(plots_bdt1, "bdt1") # 1st cut, rough estimate
plot_scan(plots_dnn1, "dnn1") # 1st cut, rough estimate
