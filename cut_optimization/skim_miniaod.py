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
    sumGenWeights = array("f", [0])

    tree.Branch('mass', mass, 'mass/F')
    tree.Branch('max_abs_eta_mu', max_abs_eta_mu, 'max_abs_eta_mu/F')
    tree.Branch('dimu_eta', dimu_eta, 'dimu_eta/F')
    tree.Branch('mu1_eta', mu1_eta, 'mu1_eta/F')    
    tree.Branch('mu2_eta', mu2_eta, 'mu2_eta/F')

    metadata.Branch('sumGenWeights', sumGenWeights, 'sumGenWeights/F')


    for filename in os.listdir(path):
        # if filename.endswith("A68BB.root"):
        if filename.endswith(".root"):         
            events = Events(path+filename)
            print "Processing file: ", filename            
            for iev,event in enumerate(events):
                mass[0] = -999
                max_abs_eta_mu[0] = -999
                event.getByLabel(jetLabel, jets)
                event.getByLabel(muonLabel, muons)
                event.getByLabel(pfCandsLabel, pfCands)
                event.getByLabel(genEvtInfoLabel,  genEvtInfo)

                gen_wgt = genEvtInfo.product().weight()

                if gen_wgt > 0:
                    sumGenWeights[0] = sumGenWeights[0] + 1
                else:
                    sumGenWeights[0] = sumGenWeights[0] - 1

                if (iev % 1000) is 0: 
                    print "Event # %i"%iev

                # if iev>1000:
                #     break
            
                mu1 = None
                mu2 = None

                mu1_p4 = None
                mu2_p4 = None

                photon = None
                photon1 = None
                photon2 = None
                photon1_found = False
                photon2_found = False
                preselected_photons = []
                
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





def write_weights_to_tree(file_path): 
    tree = ROOT.TChain("tree")
    tree.Add(file_path)
    new_tree = tree.CloneTree(0)

    metadata = ROOT.TChain("metadata")
    metadata.Add(file_path)

    for i in range(metadata.GetEntries()):
        metadata.GetEntry(i)
        nOriginalWeighted = metadata.GetLeaf("sumGenWeights").GetValue()


    new_file = ROOT.TFile.Open(file_path, "RECREATE")
    new_file.cd()

    weight_over_lumi = array('f', [0])
    newBranch = new_tree.Branch("weight_over_lumi", weight_over_lumi, "weight_over_lumi/F")

    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        weight_over_lumi[0] = 0.009618/nOriginalWeighted
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

# dy_path = "/mnt/hadoop/store/mc/RunIISummer16MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/MINIAODSIM/PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6_ext2-v1/110000/4CDE9146-50F1-E611-AE57-02163E014769.root"

# 2017 high stat sample
# ggh_path = "/mnt/hadoop/store/mc/RunIIFall17MiniAODv2/GluGluHToMuMu_M125_13TeV_amcatnloFXFX_pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/20000/"

# Autumn 2018 
# ggh_path = "/mnt/hadoop/store/mc/RunIIAutumn18MiniAOD/GluGluHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/MINIAODSIM/102X_upgrade2018_realistic_v15-v2/270000/"


zh_2017_1 = "/mnt/hadoop/store/mc/RunIIFall17MiniAODv2/ZH_HToMuMu_ZToAll_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/10000/"
zh_2017_2 = "/mnt/hadoop/store/mc/RunIIFall17MiniAODv2/ZH_HToMuMu_ZToAll_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/00000/"
zh_2017_3 = "/mnt/hadoop/store/mc/RunIIFall17MiniAODv2/ZH_HToMuMu_ZToAll_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/70000/"
zh_2017_4 = "/mnt/hadoop/store/mc/RunIIFall17MiniAODv2/ZH_HToMuMu_ZToAll_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/90000/"

output_path = "/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/miniaod_skim/"
output_file = "zh_2017_4.root"

set_out_path(output_path)

# loop_over_events(ggh_path, output_path+output_file)
loop_over_events(zh_2017_4, output_path+output_file)
write_weights_to_tree(output_path+output_file)




