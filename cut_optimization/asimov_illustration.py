import ROOT

signal_input = "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/miniaod_skim/ggh_2017_psweights.root"
sig_tree_name = "tree"
data_input = "/mnt/hadoop/store/user/dkondrat/skim/2016/SingleMu_2016/*root"
data_tree_name = "dimuons/tree"

def plot_initial_shapes(eta_min, eta_max):

    cuts = "(max_abs_eta>%f)&(max_abs_eta<%f)"

    sig_tree = ROOT.TChain(sig_tree_name)
    sig_tree.Add(signal_input)
    sig_hist = ROOT.TH1D("signal", "", 100, 110, 135)
    sig_tree.Draw("mass>>signal", cuts)
    sig_hist.Scale(1/sig_hist.Integral())

    data_tree = ROOT.TChain(data_tree_name)
    data_tree.Add(data_input)
    data_hist = ROOT.TH1D("data", "", 100, 110, 135)    
    data_tree.Draw("mass>>data", cuts)
    data_hist.Scale(1/data_hist.Integral())

    canvas = ROOT.TCanvas("c", "c", 800, 800)
    canvas.cd()

    data_hist.Draw('hist')
    sig_hist.Draw('histsame')

    canvas.SaveAs('plots/asimov/initial_shapes.png')


plot_initial_shapes(0, 0.1)