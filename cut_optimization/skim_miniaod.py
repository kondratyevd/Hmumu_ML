# import ROOT in batch mode
import sys, os, math, errno
from array import array
oldargv = sys.argv[:]
sys.argv = [ '-b-' ]
import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
sys.argv = oldargv

# load FWLite C++ libraries
ROOT.gSystem.Load("libFWCoreFWLite.so")
ROOT.gSystem.Load("libDataFormatsFWLite.so")
ROOT.FWLiteEnabler.enable()

# load FWlite python libraries
from DataFormats.FWLite import Handle, Events


def deltaR(eta1,phi1,eta2,phi2):
    dEta = eta1 - eta2
    dPhi = phi1 - phi2
    while dPhi>math.pi:
        dPhi = dPhi - 2*math.pi
    while dPhi<-math.pi:
        dPhi = dPhi + 2*math.pi
    return math.sqrt(dEta*dEta+dPhi*dPhi) 

def mu_rel_iso(mu):
    iso  = mu.pfIsolationR04().sumChargedHadronPt
    iso += max( 0., mu.pfIsolationR04().sumNeutralHadronEt + mu.pfIsolationR04().sumPhotonEt - 0.5*mu.pfIsolationR04().sumPUPt )
    return iso/mu.pt()

def mu1_selection(mu):
    passed = (mu.pt()>26) and (abs(mu.eta())<2.4)# and (mu_rel_iso(mu)<0.25)
    return passed

def mu2_selection(mu):
    passed = (mu.pt()>20) and (abs(mu.eta())<2.4)# and (mu_rel_iso(mu)<0.25)
    return passed

def loop_over_events(path, out_path):
    
    muons, muonLabel = Handle("std::vector<pat::Muon>"), "slimmedMuons"
    jets, jetLabel = Handle("std::vector<pat::Jet>"), "slimmedJets"
    pfCands, pfCandsLabel = Handle("std::vector<pat::PackedCandidate>"), "packedPFCandidates"

    genEvtInfo, genEvtInfoLabel = Handle("GenEventInfoProduct"), "generator"


    tree = ROOT.TTree("tree", "tree")
    tree.SetDirectory(0)

    metadata = ROOT.TTree("metadata", "metadata")

    mass = array("f", [0])
    max_abs_eta_mu = array("f", [0])
    dimu_eta = array("f", [0])
    mu1_eta = array("f", [0])
    mu2_eta = array("f", [0])
    GEN_wgt = array("f", [0])
    sumGenWeights = array("f", [0])

    tree.Branch('mass', mass, 'mass/F')
    tree.Branch('max_abs_eta_mu', max_abs_eta_mu, 'max_abs_eta_mu/F')
    tree.Branch('dimu_eta', dimu_eta, 'dimu_eta/F')
    tree.Branch('mu1_eta', mu1_eta, 'mu1_eta/F')    
    tree.Branch('mu2_eta', mu2_eta, 'mu2_eta/F')
    tree.Branch('GEN_wgt', GEN_wgt, 'GEN_wgt/F')

    metadata.Branch('sumGenWeights', sumGenWeights, 'sumGenWeights/F')

    print  "Processing files from ", path
    for filename in os.listdir(path):
        # if filename.endswith("A68BB.root"):
        if filename.endswith(".root"):         
            events = Events(path+filename)
            print "Processing file: ", filename   
            # pos_wgts = 0
            # neg_wgts = 0         
            for iev,event in enumerate(events):
                # if iev>1000:
                #     break
                mass[0] = -999
                max_abs_eta_mu[0] = -999
                event.getByLabel(jetLabel, jets)
                event.getByLabel(muonLabel, muons)
                event.getByLabel(pfCandsLabel, pfCands)
                event.getByLabel(genEvtInfoLabel,  genEvtInfo)

                gen_wgt = genEvtInfo.product().weight()

                if gen_wgt > 0:
                    sumGenWeights[0] = sumGenWeights[0] + 1
                    GEN_wgt[0] = 1
                    # pos_wgts = pos_wgts+1
                else:
                    sumGenWeights[0] = sumGenWeights[0] - 1
                    GEN_wgt[0] = -1
                    # neg_wgts = neg_wgts+1

                if (iev % 1000) is 0: 
                    print "Event # %i"%iev
                    # print "Positive weights: %i,    negative weights: %i"%(pos_wgts, neg_wgts)

            
                mu1 = None
                mu2 = None

                mu1_p4 = None
                mu2_p4 = None
                
                for i_mu,mu in enumerate(muons.product()):
                    iso = mu_rel_iso(mu)
                    if mu1_selection(mu) and not mu1:
                        mu1 = mu
                        mu1_eta[0] = mu.eta()
                        if abs(mu.eta())>max_abs_eta_mu[0]:
                            max_abs_eta_mu[0] = abs(mu.eta())

                    elif mu2_selection(mu) and mu1 and not mu2 and (mu1.charge()*mu.charge()<0):
                        mu2 = mu
                        mu2_eta[0] = mu.eta()
                        if abs(mu.eta())>max_abs_eta_mu[0]:
                            max_abs_eta_mu[0] = abs(mu.eta())

                if (not mu1) or (not mu2):
                    continue

                mass[0] = (mu1.p4() + mu2.p4()).M()
                dimu_eta[0] = (mu1.p4() + mu2.p4()).eta()

                tree.Fill()
    metadata.Fill()                    

    out_file = ROOT.TFile(out_path, "RECREATE")
    out_file.cd()
    metadata.Write()
    tree.Write()
    out_file.Close()





def write_weights_to_tree(dir, label, xSec): 
    totalOrigNumEvts = 0
    for filename in os.listdir(dir):
        if filename.startswith(label):
            print filename
            metadata = ROOT.TChain("metadata")
            metadata.Add(dir+filename)
    
            for i in range(metadata.GetEntries()):
                metadata.GetEntry(i)
                totalOrigNumEvts = totalOrigNumEvts + metadata.GetLeaf("sumGenWeights").GetValue()
    print totalOrigNumEvts
    for filename in os.listdir(dir):
        if filename.startswith(label):
            tree = ROOT.TChain("tree")
            tree.Add(dir+filename)
            new_tree = tree.CloneTree(0)
        
            new_file = ROOT.TFile.Open(dir+filename, "RECREATE")
            new_file.cd()
        
            weight_over_lumi = array('f', [0])
            newBranch = new_tree.Branch("weight_over_lumi", weight_over_lumi, "weight_over_lumi/F")
        
            for i in range(tree.GetEntries()):
                tree.GetEntry(i)
                weight_over_lumi[0] = tree.GetLeaf("GEN_wgt").GetValue()*xSec/totalOrigNumEvts
                new_tree.Fill()
        
            new_tree.Write()
            new_file.Close()



def set_out_path(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def plot_hists(hist_list, name, path, legend):
    canvas = ROOT.TCanvas(name, name, 800, 800)
    canvas.cd()
    for hist in hist_list:
        hist.Draw("histsame")
    legend.Draw()
    canvas.SaveAs(path+name+".png")



output_path = "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/miniaod_skim/"


set_out_path(output_path)


# ggH 2017 ##

# ggh_2017 = "/mnt/hadoop/store/mc/RunIIFall17MiniAODv2/GluGluHToMuMu_M125_13TeV_amcatnloFXFX_pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/20000/"
# loop_over_events(ggh_2017, output_path+"ggh_2017.root")
# write_weights_to_tree(output_path, "ggh_2017", xSec=0.009618) #ggH

# # ggH 2018 #

ggh_2018_1 = "/mnt/hadoop/store/mc/RunIIAutumn18MiniAOD/GluGluHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnloFXFX_pythia8/MINIAODSIM/102X_upgrade2018_realistic_v15-v2/100000/"
ggh_2018_2 = "/mnt/hadoop/store/mc/RunIIAutumn18MiniAOD/GluGluHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnloFXFX_pythia8/MINIAODSIM/102X_upgrade2018_realistic_v15-v2/110000/"
# ggh_2018_3 = "/mnt/hadoop/store/mc/RunIIAutumn18MiniAOD/GluGluHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnloFXFX_pythia8/MINIAODSIM/102X_upgrade2018_realistic_v15-v2/120000/"
# ggh_2018_4 = "/mnt/hadoop/store/mc/RunIIAutumn18MiniAOD/GluGluHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnloFXFX_pythia8/MINIAODSIM/102X_upgrade2018_realistic_v15-v2/80000/"
loop_over_events(ggh_2018_1, output_path+"ggh_2018_1.root")
loop_over_events(ggh_2018_2, output_path+"ggh_2018_2.root")
# loop_over_events(ggh_2018_3, output_path+"ggh_2018_3.root")
# loop_over_events(ggh_2018_4, output_path+"ggh_2018_4.root")
# write_weights_to_tree(output_path, "ggh_2018_", xSec=0.009618) 
# write_weights_to_tree(output_path, "ggh_2018_", xSec=0.009618) 
# write_weights_to_tree(output_path, "ggh_2018_", xSec=0.009618) 
# write_weights_to_tree(output_path, "ggh_2018_", xSec=0.009618) 

# # VBF 2017 ##

# vbf_2017_1 = "/mnt/hadoop/store/mc/RunIIFall17MiniAODv2/VBFHToMuMu_M125_13TeV_amcatnlo_pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/00000/"
# vbf_2017_2 = "/mnt/hadoop/store/mc/RunIIFall17MiniAODv2/VBFHToMuMu_M125_13TeV_amcatnlo_pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/10000/"
# loop_over_events(vbf_2017_1, output_path+"vbf_2017_1.root")
# loop_over_events(vbf_2017_2, output_path+"vbf_2017_2.root")
# write_weights_to_tree(output_path, "vbf_2017_", xSec=0.0008208)

# # VBF 2018 ##

# vbf_2018 = "/mnt/hadoop/store/mc/RunIIAutumn18MiniAOD/VBFHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnlo_pythia8/MINIAODSIM/102X_upgrade2018_realistic_v15-v2/270000/"
# loop_over_events(vbf_2018, output_path+"vbf_2018.root")
# write_weights_to_tree(output_path, "vbf_2018.root", xSec=0.0008208)

# # WplusH 2017 ##

# wplus_2017_1 = "/mnt/hadoop/store/mc/RunIIFall17MiniAODv2/WplusH_HToMuMu_WToAll_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v3/00000/"
# wplus_2017_2 = "/mnt/hadoop/store/mc/RunIIFall17MiniAODv2/WplusH_HToMuMu_WToAll_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v3/10000/"
# wplus_2017_3 = "/mnt/hadoop/store/mc/RunIIFall17MiniAODv2/WplusH_HToMuMu_WToAll_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v3/30000/"
# loop_over_events(wplus_2017_1, output_path+"wplush_2017_1.root")
# loop_over_events(wplus_2017_2, output_path+"wplush_2017_2.root")
# loop_over_events(wplus_2017_3, output_path+"wplush_2017_3.root")
# write_weights_to_tree(output_path, "wplush_2017_", xSec=0.0001858) 
# write_weights_to_tree(output_path, "wplush_2017_", xSec=0.0001858)
# write_weights_to_tree(output_path, "wplush_2017_", xSec=0.0001858)

# # WplusH 2018 ##

# wplush_2018_1 = "/mnt/hadoop/store/mc/RunIIAutumn18MiniAOD/WplusH_HToMuMu_WToAll_M125_TuneCP5_PSweights_13TeV_powheg_pythia8/MINIAODSIM/102X_upgrade2018_realistic_v15-v2/00000/"
# wplush_2018_2 = "/mnt/hadoop/store/mc/RunIIAutumn18MiniAOD/WplusH_HToMuMu_WToAll_M125_TuneCP5_PSweights_13TeV_powheg_pythia8/MINIAODSIM/102X_upgrade2018_realistic_v15-v2/110000/"
# loop_over_events(wplush_2018_1, output_path+"wplush_2018_1.root")
# loop_over_events(wplush_2018_2, output_path+"wplush_2018_2.root")
# write_weights_to_tree(output_path, "wplush_2018_", xSec=0.0001858) 
# write_weights_to_tree(output_path, "wplush_2018_", xSec=0.0001858)

# # WminusH 2017 ##

# wminus_2017_1 = "/mnt/hadoop/store/mc/RunIIFall17MiniAODv2/WminusH_HToMuMu_WToAll_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/40000/"
# wminus_2017_2 = "/mnt/hadoop/store/mc/RunIIFall17MiniAODv2/WminusH_HToMuMu_WToAll_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/70000/"
# loop_over_events(wminus_2017_1, output_path+"wminush_2017_1.root")
# loop_over_events(wminus_2017_2, output_path+"wminush_2017_2.root")
# write_weights_to_tree(output_path, "wminush_2017_", xSec=0.0001164) 
# write_weights_to_tree(output_path, "wminush_2017_", xSec=0.0001164)

# # WminusH 2018 ##

# wminush_2018_1 = "/mnt/hadoop/store/mc/RunIIAutumn18MiniAOD/WminusH_HToMuMu_WToAll_M125_TuneCP5_PSweights_13TeV_powheg_pythia8/MINIAODSIM/102X_upgrade2018_realistic_v15-v2/00000/"
# wminush_2018_2 = "/mnt/hadoop/store/mc/RunIIAutumn18MiniAOD/WminusH_HToMuMu_WToAll_M125_TuneCP5_PSweights_13TeV_powheg_pythia8/MINIAODSIM/102X_upgrade2018_realistic_v15-v2/60000/"
# loop_over_events(wminush_2018_1, output_path+"wminush_2018_1.root")
# loop_over_events(wminush_2018_2, output_path+"wminush_2018_2.root")
# write_weights_to_tree(output_path, "wminush_2018_", xSec=0.0001164) 
# write_weights_to_tree(output_path, "wminush_2018_", xSec=0.0001164)

# # ZH 2017 ##

# zh_2017_1 = "/mnt/hadoop/store/mc/RunIIFall17MiniAODv2/ZH_HToMuMu_ZToAll_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/10000/"
# zh_2017_2 = "/mnt/hadoop/store/mc/RunIIFall17MiniAODv2/ZH_HToMuMu_ZToAll_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/00000/"
# zh_2017_3 = "/mnt/hadoop/store/mc/RunIIFall17MiniAODv2/ZH_HToMuMu_ZToAll_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/70000/"
# zh_2017_4 = "/mnt/hadoop/store/mc/RunIIFall17MiniAODv2/ZH_HToMuMu_ZToAll_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/90000/"
# loop_over_events(zh_2017_1, output_path+"zh_2017_1.root")
# loop_over_events(zh_2017_2, output_path+"zh_2017_2.root")
# loop_over_events(zh_2017_3, output_path+"zh_2017_3.root")
# loop_over_events(zh_2017_4, output_path+"zh_2017_4.root")
# write_weights_to_tree(output_path, "zh_2017_", xSec=0.00003865) #ZH
# write_weights_to_tree(output_path, "zh_2017_", xSec=0.00003865) #ZH
# write_weights_to_tree(output_path, "zh_2017_", xSec=0.00003865) #ZH
# write_weights_to_tree(output_path, "zh_2017_", xSec=0.00003865) #ZH

# # ZH 2018 ##

# zh_2018 = "/mnt/hadoop/store/mc/RunIIAutumn18MiniAOD/ZH_HToMuMu_ZToAll_M125_TuneCP5_PSweights_13TeV_powheg_pythia8/MINIAODSIM/102X_upgrade2018_realistic_v15-v2/60000/"
# loop_over_events(zh_2018, output_path+"zh_2018.root")
# write_weights_to_tree(output_path, "zh_2018.root", xSec=0.00003865)

# # ttH 2017 ##

# ttH_2017 = "/mnt/hadoop/store/mc/RunIIFall17MiniAODv2/ttHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v3/00000/"
# loop_over_events(ttH_2017, output_path+"ttH_2017.root")
# write_weights_to_tree(output_path, "ttH_2017.root", xSec=0.00011034496) 

# # ttH 2018 ##

# tth_2018_1 = "/mnt/hadoop/store/mc/RunIIAutumn18MiniAOD/ttHToMuMu_M125_TuneCP5_PSweights_13TeV-powheg-pythia8/MINIAODSIM/102X_upgrade2018_realistic_v15-v2/40000/"
# tth_2018_2 = "/mnt/hadoop/store/mc/RunIIAutumn18MiniAOD/ttHToMuMu_M125_TuneCP5_PSweights_13TeV-powheg-pythia8/MINIAODSIM/102X_upgrade2018_realistic_v15-v2/80000/"
# tth_2018_3 = "/mnt/hadoop/store/mc/RunIIAutumn18MiniAOD/ttHToMuMu_M125_TuneCP5_PSweights_13TeV-powheg-pythia8/MINIAODSIM/102X_upgrade2018_realistic_v15-v2/270000/"
# loop_over_events(tth_2018_1, output_path+"tth_2018_1.root")
# loop_over_events(tth_2018_2, output_path+"tth_2018_2.root")
# loop_over_events(tth_2018_3, output_path+"tth_2018_3.root")
# write_weights_to_tree(output_path, "tth_2018_", xSec=0.00011034496) 
# write_weights_to_tree(output_path, "tth_2018_", xSec=0.00011034496)
# write_weights_to_tree(output_path, "tth_2018_", xSec=0.00011034496)


