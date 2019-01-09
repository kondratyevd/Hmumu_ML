# import ROOT in batch mode
import sys, os, math, errno
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


def loop_over_events(path):
    
    muons, muonLabel = Handle("std::vector<pat::Muon>"), "slimmedMuons"
    jets, jetLabel = Handle("std::vector<pat::Jet>"), "slimmedJets"
    pfCands, pfCandsLabel = Handle("std::vector<pat::PackedCandidate>"), "packedPFCandidates"

    events = Events(path)

    mass_hist = ROOT.TH1D("mass", "M_{#mu#mu}", 40,110,150)
    mass_fsr_hist = ROOT.TH1D("mass_fsr", "M_{#mu#mu} post-FSR", 40,110,150)

    for iev,event in enumerate(events):
 
        event.getByLabel(jetLabel, jets)
        event.getByLabel(muonLabel, muons)
        event.getByLabel(pfCandsLabel, pfCands)

        if (iev % 1000) is 0: 
            print "Event # %i"%iev

        if iev>5000:
            break
    
        mu1_pt = -999
        mu1_eta = -999
        mu1_phi = -999
        mu1_v = ROOT.TLorentzVector()
        mu1_found = False

        mu2_pt = -999
        mu2_eta = -999
        mu2_phi = -999
        mu2_v = ROOT.TLorentzVector()        
        mu2_found = False

        pfc_found = False
        pfc_eta = -999
        pfc_phi = -999
        pfc_et = -999
        pfc_v = ROOT.TLorentzVector()

        ph_mu1_dR = 999
        ph_mu2_dR = 999

        dimu_mass = -999
        dimu_fsr_mass = -999


        for i_mu,mu in enumerate(muons.product()):
            iso = mu_rel_iso(mu)
            if (mu.pt()>26) and (abs(mu.eta())<2.4) and (iso<0.25) and not mu1_found:
                mu1_found = True
                mu1_pt = mu.pt()
                mu1_eta = mu.eta()
                mu1_phi = mu.phi()
                mu1_v = mu.p4()

            elif (mu.pt()>20) and (abs(mu.eta())<2.4) and (iso<0.25) and mu1_found and not mu2_found:
                mu2_found = True
                mu2_pt = mu.pt()
                mu2_eta = mu.eta()
                mu2_phi = mu.phi()
                mu2_v = mu.p4()

        for i_pfc, pfc in enumerate(pfCands.product()):
            if (pfc.pt()>1) and (abs(pfc.eta()<2.4)) and not pfc_found:
                pfc_found = True
                pfc_eta = pfc.eta()
                pfc_phi = pfc.phi()
                pfc_et = pfc.et()
                pfc_v = pfc.p4()

                if mu1_found and mu2_found:
                    ph_mu1_dR = deltaR(pfc_eta, pfc_phi, mu1_eta, mu1_phi)
                    ph_mu2_dR = deltaR(pfc_eta, pfc_phi, mu2_eta, mu2_phi)

        if mu1_found and mu2_found:
            dimu_mass = (mu1_v + mu2_v).M()
            # mass_hist.Fill(dimu_mass)
            if pfc_found and ((ph_mu1_dR<0.5 and ph_mu1_dR/(pfc_et*pfc_et)<0.012) or (ph_mu2_dR<0.5 and ph_mu2_dR/(pfc_et*pfc_et)<0.012)):
                dimu_fsr_mass = (mu1_v+mu2_v+pfc_v).M()
                mass_fsr_hist.Fill(dimu_fsr_mass)
                mass_hist.Fill(dimu_mass)
            # else:
            #     mass_fsr_hist.Fill(dimu_mass)

    mass_hist.SetLineColor(ROOT.kBlue)
    mass_fsr_hist.SetLineColor(ROOT.kRed)

    return mass_hist, mass_fsr_hist

def set_out_path(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def plot_hists(hist_list, name, path):
    canvas = ROOT.TCanvas("name", "name", 800, 800)
    canvas.cd()
    for hist in hist_list:
        hist.Draw("histsame")
    canvas.SaveAs(path+name+".png")

dy_path = "/mnt/hadoop/store/mc/RunIISummer16MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/MINIAODSIM/PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6_ext2-v1/110000/4CDE9146-50F1-E611-AE57-02163E014769.root"
ggh_path = "/mnt/hadoop/store/mc/RunIISummer16MiniAODv2/GluGlu_HToMuMu_M125_13TeV_powheg_pythia8/MINIAODSIM/PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/70000/36967CD0-3CC1-E611-A615-D8D385FF1996.root"


out_path = "plots/fsr_recovery/"
set_out_path(out_path)

# loop_over_events(dy_path)
mass_hist, mass_fsr_hist =  loop_over_events(ggh_path)

plot_hists([mass_hist, mass_fsr_hist], "test", out_path)




