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

def mu_rel_iso_corrected(mu, photon1, photon2):
    ph1_et = 0
    ph2_et = 0

    if photon1:
        if deltaR(photon1.eta(), photon1.phi(), mu.eta(), mu.phi())<0.4:
            ph1_et = photon1.et()
    if photon2:
        if deltaR(photon2.eta(), photon2.phi(), mu.eta(), mu.phi())<0.4:
            ph2_et = photon2.et()

    iso  = mu.pfIsolationR04().sumChargedHadronPt
    iso += max( 0., mu.pfIsolationR04().sumNeutralHadronEt + mu.pfIsolationR04().sumPhotonEt - ph1_et - ph2_et - 0.5*mu.pfIsolationR04().sumPUPt )
    return iso/mu.pt()

def isolated(mu1, mu2, photon1, photon2):
    if photon1 or photon2:
        mu1_rel_iso_corrected = mu_rel_iso_corrected(mu1, photon1, photon2)
        mu2_rel_iso_corrected = mu_rel_iso_corrected(mu2, photon1, photon2)
        result = (mu1_rel_iso_corrected<0.25) and (mu2_rel_iso_corrected<0.25)
    else:
        result = (mu_rel_iso(mu1)<0.25) and (mu_rel_iso(mu2)<0.25)

    return result


def mu1_selection(mu):
    passed = (mu.pt()>26) and (abs(mu.eta())<2.4)# and (mu_rel_iso(mu)<0.25)
    return passed

def mu2_selection(mu):
    passed = (mu.pt()>20) and (abs(mu.eta())<2.4)# and (mu_rel_iso(mu)<0.25)
    return passed

def photon_preselection(photon, mu1, mu2, pfCands):
    ph_mu1_dR = deltaR(photon.eta(), photon.phi(), mu1.eta(), mu1.phi())
    ph_mu2_dR = deltaR(photon.eta(), photon.phi(), mu2.eta(), mu2.phi())
    passed = (photon.pt()>1) and (abs(photon.eta()<2.4)) and (photon.pdgId()==22) and (ph_mu1_dR<0.5 or ph_mu2_dR<0.5) and (photonPfIso03(photon, pfCands)<1.8)
    return passed

def photonPfIso03(pho, pfCands):
    ptsum = 0.0
    for i_pfc, pfc in enumerate(pfCands.product()):
        dr = deltaR(pho.eta(), pho.phi(), pfc.eta(), pfc.phi())
        if (dr >= 0.3):
            continue
        if (pfc.charge() != 0 and abs(pfc.pdgId()) == 211 and pfc.pt() > 0.2):
            if (dr > 0.0001):
                ptsum += pfc.pt()
        elif (pfc.charge() == 0 and (abs(pfc.pdgId()) == 22 or abs(pfc.pdgId()) == 130) and pfc.pt() > 0.5):
            if (dr > 0.01):
                ptsum += pfc.pt()
    return ptsum/pho.pt()



def loop_over_events(path):
    
    muons, muonLabel = Handle("std::vector<pat::Muon>"), "slimmedMuons"
    jets, jetLabel = Handle("std::vector<pat::Jet>"), "slimmedJets"
    pfCands, pfCandsLabel = Handle("std::vector<pat::PackedCandidate>"), "packedPFCandidates"

    mass_hist = ROOT.TH1D("mass", "", 40,110,150)
    mass_fsr_hist = ROOT.TH1D("mass_fsr", "", 40,110,150)
    mass_hist_tagged = ROOT.TH1D("mass_tagged", "", 40,110,150)
    mass_fsr_hist_tagged = ROOT.TH1D("mass_fsr_tagged", "", 40,110,150)
    # fsr_spectrum = ROOT.TH1D("fsr_spectrum", "", 100,0,50)

    tree = ROOT.TTree("tree", "tree")
    tree.SetDirectory(0)

    mass = array("f", [0])
    fsr_tag = array("i", [0])
    fsr_2tag = array("i", [0])
    mass_postFSR = array("f", [0])
    max_abs_eta_mu = array("f", [0])
    mu1_eta = array("f", [0])
    mu2_eta = array("f", [0])
    fsr_spectrum = array("f", [0])

    tree.Branch('mass', mass, 'mass/F')
    tree.Branch('fsr_tag', fsr_tag, 'fsr_tag/I')
    tree.Branch('fsr_2tag', fsr_2tag, 'fsr_2tag/I')
    tree.Branch('mass_postFSR', mass_postFSR, 'mass_postFSR/F')
    tree.Branch('max_abs_eta_mu', max_abs_eta_mu, 'max_abs_eta_mu/F')
    tree.Branch('mu1_eta', mu1_eta, 'mu1_eta/F')    
    tree.Branch('mu2_eta', mu2_eta, 'mu2_eta/F')
    tree.Branch('fsr_spectrum', fsr_spectrum, 'fsr_spectrum/F')

    for filename in os.listdir(path):
        if filename.endswith("D37BC.root"): 
            events = Events(path+filename)
            print "Processing file: ", filename
            for iev,event in enumerate(events):
                mass[0] = -999
                fsr_tag[0] = -999
                fsr_2tag[0] = -999
                mass_postFSR[0] = -999
                max_abs_eta_mu[0] = -999
                fsr_spectrum[0] = -999
                event.getByLabel(jetLabel, jets)
                event.getByLabel(muonLabel, muons)
                event.getByLabel(pfCandsLabel, pfCands)

                if (iev % 1000) is 0: 
                    print "Event # %i"%iev

                # if iev>10000:
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

                # preselect photons: cuts on pT, eta, id, and close to at least one muon
                for i_pfc, pfc in enumerate(pfCands.product()):
                    if photon_preselection(pfc, mu1, mu2, pfCands):
                        preselected_photons.append(pfc)
    
                # min_dR1_over_et2 = 0.012
                # min_dR2_over_et2 = 0.012
                min_dR1_over_et2 = 999999
                min_dR2_over_et2 = 999999    

                # find closest photon to the first muon
                for pfc in preselected_photons:
                    ph_mu1_dR = deltaR(pfc.eta(), pfc.phi(), mu1.eta(), mu1.phi())
                    if (ph_mu1_dR<0.5) and (ph_mu1_dR/(pfc.et()*pfc.et()) < min_dR1_over_et2):
                        min_dR1_over_et2 = ph_mu1_dR/(pfc.et()*pfc.et())
                        photon1 = pfc  
    
                # find closest photon to the second muon
                for pfc in preselected_photons:
                    ph_mu2_dR = deltaR(pfc.eta(), pfc.phi(), mu1.eta(), mu1.phi())
                    if (ph_mu2_dR<0.5) and (ph_mu2_dR/(pfc.et()*pfc.et()) < min_dR2_over_et2):
                        min_dR2_over_et2 = ph_mu2_dR/(pfc.et()*pfc.et())
                        if photon1:
                            if deltaR(pfc.eta(), pfc.phi(), photon1.eta(), photon1.phi())>0.001:
                                photon2 = pfc 
                        else:
                            photon2 = pfc    
    
                if photon1:
                    mu1_p4 = mu1.p4()+photon1.p4()
                else:
                    mu1_p4 = mu1.p4()
    
                if photon2:
                    mu2_p4 = mu2.p4()+photon2.p4()
                else:
                    mu2_p4 = mu2.p4()                    
                            
                if isolated(mu1, mu2, photon1, photon2):
                    dimu_mass = (mu1.p4() + mu2.p4()).M()
                    dimu_fsr_mass = (mu1_p4+mu2_p4).M()
                    mass[0] = dimu_mass
                    mass_postFSR[0] = dimu_fsr_mass
                    mass_fsr_hist.Fill(dimu_fsr_mass)
                    mass_hist.Fill(dimu_mass)
                    if photon1 or photon2:
                        mass_fsr_hist_tagged.Fill(dimu_fsr_mass)
                        mass_hist_tagged.Fill(dimu_mass)
                        fsr_tag[0] = 1
                    else:
                        fsr_tag[0] = 0

                    if photon1 and photon2:
                        fsr_2tag[0] = 1
                    else:
                        fsr_2tag[0] = 0

                    if photon1:
                        fsr_spectrum[0] = photon1.pt()
                    elif photon2:
                        fsr_spectrum[0] = photon2.pt()
                    tree.Fill()

    mass_hist_tagged.SetLineColor(ROOT.kBlue)
    mass_fsr_hist_tagged.SetLineColor(ROOT.kRed)
    mass_hist.SetLineColor(ROOT.kBlue)
    mass_fsr_hist.SetLineColor(ROOT.kRed)
    out_file = ROOT.TFile("combine/fsr_test_2017_nodRETCut.root", "RECREATE")
    out_file.cd()
    tree.Write()
    out_file.Close()

    return mass_hist, mass_fsr_hist, mass_hist_tagged, mass_fsr_hist_tagged

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

dy_path = "/mnt/hadoop/store/mc/RunIISummer16MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/MINIAODSIM/PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6_ext2-v1/110000/4CDE9146-50F1-E611-AE57-02163E014769.root"
# ggh_path = "/mnt/hadoop/store/mc/RunIISummer16MiniAODv2/GluGlu_HToMuMu_M125_13TeV_powheg_pythia8/MINIAODSIM/PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/70000/36967CD0-3CC1-E611-A615-D8D385FF1996.root"
# ggh_path = "/mnt/hadoop/store/mc/RunIISummer16MiniAODv2/GluGlu_HToMuMu_M125_13TeV_powheg_pythia8/MINIAODSIM/PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/70000/"
ggh_path = "/mnt/hadoop/store/mc/RunIIFall17MiniAODv2/GluGluHToMuMu_M125_13TeV_amcatnloFXFX_pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/20000/"

out_path = "plots/fsr_recovery/"
set_out_path(out_path)

mass_hist, mass_fsr_hist, mass_hist_tagged, mass_fsr_hist_tagged =  loop_over_events(ggh_path)

legend = ROOT.TLegend(0.65,0.7,0.89,0.89)
legend.AddEntry(mass_hist, 'pre-FSR', 'l')
legend.AddEntry(mass_fsr_hist, ' post-FSR', 'l')

plot_hists([mass_fsr_hist, mass_hist], "inclusive_removeFromIso", out_path, legend)
plot_hists([mass_fsr_hist_tagged, mass_hist_tagged], "fsr_tagged_removeFromIso", out_path, legend)



