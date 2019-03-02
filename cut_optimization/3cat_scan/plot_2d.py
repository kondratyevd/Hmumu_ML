import ROOT
import argparse
ROOT.gStyle.SetOptStat(0)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--label', action='store', dest='label', help='', default='ggh_2017_psweights')
args = parser.parse_args()
# label = ggh_2017_psweights
label = args.label
filedir = 'output/%s/'%label

hist2d = ROOT.TH2D("eta_categories", "", 23, 0.05, 2.35, 23, 0.05, 2.35)

for i in range(23):
    for j in range(i):
#        print j+1, i+1
        combine_filename = "higgsCombine_3cat_%i_%i.Significance.mH120.root"%(j+1, i+1)
        tree = ROOT.TChain("limit")
        tree.Add(combine_filename)
        for event in tree:
            significance = event.limit
        hist2d.Fill((j+1)/10.0, (i+1)/10.0, significance)

canv = ROOT.TCanvas("c", "c", 800, 800)
canv.cd()
hist2d.Draw("colz")
canv.Print("significance_%s.png"%label)
canv.SaveAs("significance_%s.root"%label)        
