# import ROOT in batch mode
import sys, os, math
oldargv = sys.argv[:]
sys.argv = [ '-b-' ]
import ROOT
ROOT.gROOT.SetBatch(True)
sys.argv = oldargv

# load FWLite C++ libraries
ROOT.gSystem.Load("libFWCoreFWLite.so");
ROOT.gSystem.Load("libDataFormatsFWLite.so");
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

def mu_rel_iso(muon):
    iso  = muon.pfIsolationR04().sumChargedHadronPt
    iso += max( 0., muon.pfIsolationR04().sumNeutralHadronEt + muon.pfIsolationR04().sumPhotonEt - 0.5*muon.pfIsolationR04().sumPUPt )
    return iso/muon.pt()


def loop_over_events(path, color):
    
    muons, muonLabel = Handle("std::vector<pat::Muon>"), "slimmedMuons"
    jets, jetLabel = Handle("std::vector<pat::Jet>"), "slimmedJets"
    pfCands, pfCandsLabel = Handle("std::vector<pat::PackedCandidate>"), "packedPFCandidates"

    events = Events(path)
    jets_eta_hist = ROOT.TH1D("jet_eta", "jet_eta", 100, -5, 5)
    mu1_eta_hist = ROOT.TH1D("mu1_eta", "mu1_eta", 100, -3, 3)
    jets_pt_hist = ROOT.TH1D("jet_pt", "jet_pt", 100, 0, 300)
    njets_hist = ROOT.TH1D("nJets", "nJets", 5,0,5)

    for iev,event in enumerate(events):
 
        event.getByLabel(jetLabel, jets)
        event.getByLabel(muonLabel, muons)
        event.getByLabel(pfCandsLabel, pfCands)

        if (iev % 10000) is 0: 
            print "Event # %i"%iev
    
        mu1_pt = -999
        mu1_eta = -999
        mu1_phi = -999
        mu1_found = False
        mu2_pt = -999
        mu2_eta = -999
        mu2_phi = -999
        mu2_found = False

        pfc_found = False
        pfc_eta = -999
        pfc_phi = -999

        for i_pfc, pfc in enumerate(pfCands.product()):
            if pfc.isElectron() or pfc.isPhoton():
                print "isElectron: ", pfc.isElectron()
                print "isPhoton: ", pfc.isPhoton()
                print " "
            if (pfc.pt()>1) and (abs(pfc.eta()<2.4)):
                pfc_found = True
                pfc_eta = pfc.eta()
                pfc.phi = pfc.phi()


        for i_mu,mu in enumerate(muons.product()):
            iso = mu_rel_iso(mu)

            if (mu.pt()>26) and (abs(mu.eta())<2.4) and (iso<0.25) and not mu1_found:
                mu1_found = True
                mu1_pt = mu.pt()
                mu1_eta = mu.eta()
                mu1_phi = mu.phi()

            elif (mu.pt()>20) and (abs(mu.eta())<2.4) and (iso<0.25) and mu1_found and not mu2_found:
                mu2_found = True
                mu2_pt = mu.pt()
                mu2_eta = mu.eta()
                mu2_phi = mu.phi()
    
            # print i_mu, mu.pt()

        # nJets = 0
        # for i_jet,jet in enumerate(jets.product()):    
        #     if mu1_found and mu2_found:
        #         dR_1 = deltaR(jet.eta(), jet.phi(), mu1_eta, mu1_phi)
        #         dR_2 = deltaR(jet.eta(), jet.phi(), mu2_eta, mu2_phi)   
        #         if jet.pt()>30 and dR_1>0.4 and dR_2>0.4:
        #             nJets = nJets+1
        #             jets_pt_hist.Fill(jet.pt())
        #             jets_eta_hist.Fill(jet.eta())
        #             mu1_eta_hist.Fill(mu1_eta)
        # njets_hist.Fill(nJets)

    # jets_eta_hist.SetLineColor(color)
    # jets_pt_hist.SetLineColor(color)
    mu1_eta_hist.SetLineColor(color)
    # njets_hist.SetLineColor(color)

    # jets_eta_hist.Scale(1/jets_eta_hist.Integral())
    # jets_pt_hist.Scale(1/jets_pt_hist.Integral())
    mu1_eta_hist.Scale(1/mu1_eta_hist.Integral())
    # njets_hist.Scale(1/njets_hist.Integral())
    # return jets_eta_hist, jets_pt_hist, mu1_eta_hist, njets_hist


def plot_hists(hist_list, name, path, legend):
    canvas = ROOT.TCanvas("name", "name", 800, 800)
    canvas.cd()
    for hist in hist_list:
        hist.Draw("histsame")
    legend.Draw()
    canvas.SaveAs(path+name+".png")

dy_path = "/mnt/hadoop/store/mc/RunIISummer16MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/MINIAODSIM/PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6_ext2-v1/110000/4CDE9146-50F1-E611-AE57-02163E014769.root"
ggh_path = "/mnt/hadoop/store/mc/RunIISummer16MiniAODv2/GluGlu_HToMuMu_M125_13TeV_powheg_pythia8/MINIAODSIM/PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/70000/36967CD0-3CC1-E611-A615-D8D385FF1996.root"


out_path = "plots/miniAOD/"

loop_over_events(dy_path, ROOT.kBlue)
loop_over_events(ggh_path,  ROOT.kRed)




