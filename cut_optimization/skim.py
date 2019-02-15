import ROOT 
import os,errno, math
from array import array
from ROOT import gInterpreter, gSystem
gInterpreter.ProcessLine('#include "interface/JetInfo.h"')
gInterpreter.ProcessLine('#include "interface/EventInfo.h"')
gInterpreter.ProcessLine('#include "interface/VertexInfo.h"')
gInterpreter.ProcessLine('#include "interface/JetPairInfo.h"')
gInterpreter.ProcessLine('#include "interface/MuonInfo.h"')
gInterpreter.ProcessLine('#include "interface/MuPairInfo.h"')
gInterpreter.ProcessLine('#include "interface/EleInfo.h"')
gInterpreter.ProcessLine('#include "interface/MetInfo.h"')
gInterpreter.ProcessLine('#include "interface/MhtInfo.h"')
gInterpreter.ProcessLine('#include "interface/GenParentInfo.h"')
gInterpreter.ProcessLine('#include "interface/GenMuonInfo.h"')
gInterpreter.ProcessLine('#include "interface/GenMuPairInfo.h"')
gInterpreter.ProcessLine('#include "interface/GenJetInfo.h"')

input_dir = "/mnt/hadoop/store/user/dkondrat/"
output_dir = "/tmp/Hmumu_ntuples/updated_data/"
# input_dir = "/Users/dmitrykondratyev/root_files/mc/2017/"
# output_dir = "/Users/dmitrykondratyev/root_files/mc/2017/updated/"

class Source(object):
    def __init__(self, name, in_filename, out_dir_name, isData=False, year="2016", xSec=1):
        self.name = name
        self.in_filename = in_filename
        self.out_dir_name = out_dir_name
        self.isData = isData
        self.year = year
        self.xSec=xSec
        

sources = []
# sources.append(Source('gluglu_1', 'gluglu*.root', 'gluglu', year="2017", xSec=0.009618))

sources.append(Source('H2Mu_gg', '/GluGlu_HToMuMu_M125_13TeV_powheg_pythia8/H2Mu_gg/180827_202700/0000/*root', 'H2Mu_gg', year="2016", xSec=0.009618))

sources.append(Source('SingleMu_2016B_1', '/SingleMuon/SingleMu_2016B/180912_165011/0000/tuple_1*', 'SingleMu_2016B', isData=True))
sources.append(Source('SingleMu_2016B_2', '/SingleMuon/SingleMu_2016B/180912_165011/0000/tuple_2*', 'SingleMu_2016B', isData=True))
sources.append(Source('SingleMu_2016B_3', '/SingleMuon/SingleMu_2016B/180912_165011/0000/tuple_3*', 'SingleMu_2016B', isData=True))
sources.append(Source('SingleMu_2016B_4', '/SingleMuon/SingleMu_2016B/180912_165011/0000/tuple_4*', 'SingleMu_2016B', isData=True))
sources.append(Source('SingleMu_2016B_5', '/SingleMuon/SingleMu_2016B/180912_165011/0000/tuple_5*', 'SingleMu_2016B', isData=True))
sources.append(Source('SingleMu_2016B_6', '/SingleMuon/SingleMu_2016B/180912_165011/0000/tuple_6*', 'SingleMu_2016B', isData=True))
sources.append(Source('SingleMu_2016B_7', '/SingleMuon/SingleMu_2016B/180912_165011/0000/tuple_7*', 'SingleMu_2016B', isData=True))
sources.append(Source('SingleMu_2016B_8', '/SingleMuon/SingleMu_2016B/180912_165011/0000/tuple_8*', 'SingleMu_2016B', isData=True))
sources.append(Source('SingleMu_2016B_9', '/SingleMuon/SingleMu_2016B/180912_165011/0000/tuple_9*', 'SingleMu_2016B', isData=True))  

sources.append(Source('SingleMu_2016C_1', '/SingleMuon/SingleMu_2016C/180912_165036/0000/tuple_1*', 'SingleMu_2016C', isData=True))
sources.append(Source('SingleMu_2016C_2', '/SingleMuon/SingleMu_2016C/180912_165036/0000/tuple_2*', 'SingleMu_2016C', isData=True))
sources.append(Source('SingleMu_2016C_3', '/SingleMuon/SingleMu_2016C/180912_165036/0000/tuple_3*', 'SingleMu_2016C', isData=True))
sources.append(Source('SingleMu_2016C_4', '/SingleMuon/SingleMu_2016C/180912_165036/0000/tuple_4*', 'SingleMu_2016C', isData=True))
sources.append(Source('SingleMu_2016C_5', '/SingleMuon/SingleMu_2016C/180912_165036/0000/tuple_5*', 'SingleMu_2016C', isData=True))
sources.append(Source('SingleMu_2016C_6', '/SingleMuon/SingleMu_2016C/180912_165036/0000/tuple_6*', 'SingleMu_2016C', isData=True))
sources.append(Source('SingleMu_2016C_7', '/SingleMuon/SingleMu_2016C/180912_165036/0000/tuple_7*', 'SingleMu_2016C', isData=True))
sources.append(Source('SingleMu_2016C_8', '/SingleMuon/SingleMu_2016C/180912_165036/0000/tuple_8*', 'SingleMu_2016C', isData=True))
sources.append(Source('SingleMu_2016C_9', '/SingleMuon/SingleMu_2016C/180912_165036/0000/tuple_9*', 'SingleMu_2016C', isData=True))        

sources.append(Source('SingleMu_2016D_1', '/SingleMuon/SingleMu_2016D/180912_165055/0000/tuple_1*', 'SingleMu_2016D', isData=True))
sources.append(Source('SingleMu_2016D_2', '/SingleMuon/SingleMu_2016D/180912_165055/0000/tuple_2*', 'SingleMu_2016D', isData=True))
sources.append(Source('SingleMu_2016D_3', '/SingleMuon/SingleMu_2016D/180912_165055/0000/tuple_3*', 'SingleMu_2016D', isData=True))
sources.append(Source('SingleMu_2016D_4', '/SingleMuon/SingleMu_2016D/180912_165055/0000/tuple_4*', 'SingleMu_2016D', isData=True))
sources.append(Source('SingleMu_2016D_5', '/SingleMuon/SingleMu_2016D/180912_165055/0000/tuple_5*', 'SingleMu_2016D', isData=True))
sources.append(Source('SingleMu_2016D_6', '/SingleMuon/SingleMu_2016D/180912_165055/0000/tuple_6*', 'SingleMu_2016D', isData=True))
sources.append(Source('SingleMu_2016D_7', '/SingleMuon/SingleMu_2016D/180912_165055/0000/tuple_7*', 'SingleMu_2016D', isData=True))
sources.append(Source('SingleMu_2016D_8', '/SingleMuon/SingleMu_2016D/180912_165055/0000/tuple_8*', 'SingleMu_2016D', isData=True))
sources.append(Source('SingleMu_2016D_9', '/SingleMuon/SingleMu_2016D/180912_165055/0000/tuple_9*', 'SingleMu_2016D', isData=True))  

sources.append(Source('SingleMu_2016E_1', '/SingleMuon/SingleMu_2016E/180912_165115/0000/tuple_1*', 'SingleMu_2016E', isData=True))
sources.append(Source('SingleMu_2016E_2', '/SingleMuon/SingleMu_2016E/180912_165115/0000/tuple_2*', 'SingleMu_2016E', isData=True))
sources.append(Source('SingleMu_2016E_3', '/SingleMuon/SingleMu_2016E/180912_165115/0000/tuple_3*', 'SingleMu_2016E', isData=True))
sources.append(Source('SingleMu_2016E_4', '/SingleMuon/SingleMu_2016E/180912_165115/0000/tuple_4*', 'SingleMu_2016E', isData=True))
sources.append(Source('SingleMu_2016E_5', '/SingleMuon/SingleMu_2016E/180912_165115/0000/tuple_5*', 'SingleMu_2016E', isData=True))
sources.append(Source('SingleMu_2016E_6', '/SingleMuon/SingleMu_2016E/180912_165115/0000/tuple_6*', 'SingleMu_2016E', isData=True))
sources.append(Source('SingleMu_2016E_7', '/SingleMuon/SingleMu_2016E/180912_165115/0000/tuple_7*', 'SingleMu_2016E', isData=True))
sources.append(Source('SingleMu_2016E_8', '/SingleMuon/SingleMu_2016E/180912_165115/0000/tuple_8*', 'SingleMu_2016E', isData=True))
sources.append(Source('SingleMu_2016E_9', '/SingleMuon/SingleMu_2016E/180912_165115/0000/tuple_9*', 'SingleMu_2016E', isData=True))   

sources.append(Source('SingleMu_2016F_1_1', '/SingleMuon/SingleMu_2016F_1/180912_165134/0000/tuple_1*', 'SingleMu_2016F_1', isData=True))
sources.append(Source('SingleMu_2016F_1_2', '/SingleMuon/SingleMu_2016F_1/180912_165134/0000/tuple_2*', 'SingleMu_2016F_1', isData=True))
sources.append(Source('SingleMu_2016F_1_3', '/SingleMuon/SingleMu_2016F_1/180912_165134/0000/tuple_3*', 'SingleMu_2016F_1', isData=True))
sources.append(Source('SingleMu_2016F_1_4', '/SingleMuon/SingleMu_2016F_1/180912_165134/0000/tuple_4*', 'SingleMu_2016F_1', isData=True))
sources.append(Source('SingleMu_2016F_1_5', '/SingleMuon/SingleMu_2016F_1/180912_165134/0000/tuple_5*', 'SingleMu_2016F_1', isData=True))
sources.append(Source('SingleMu_2016F_1_6', '/SingleMuon/SingleMu_2016F_1/180912_165134/0000/tuple_6*', 'SingleMu_2016F_1', isData=True))
sources.append(Source('SingleMu_2016F_1_7', '/SingleMuon/SingleMu_2016F_1/180912_165134/0000/tuple_7*', 'SingleMu_2016F_1', isData=True))
sources.append(Source('SingleMu_2016F_1_8', '/SingleMuon/SingleMu_2016F_1/180912_165134/0000/tuple_8*', 'SingleMu_2016F_1', isData=True))
sources.append(Source('SingleMu_2016F_1_9', '/SingleMuon/SingleMu_2016F_1/180912_165134/0000/tuple_9*', 'SingleMu_2016F_1', isData=True))  

sources.append(Source('SingleMu_2016F_2_1', '/SingleMuon/SingleMu_2016F_2/180912_165152/0000/tuple_1*', 'SingleMu_2016F_2', isData=True))
sources.append(Source('SingleMu_2016F_2_2', '/SingleMuon/SingleMu_2016F_2/180912_165152/0000/tuple_2*', 'SingleMu_2016F_2', isData=True))
sources.append(Source('SingleMu_2016F_2_3', '/SingleMuon/SingleMu_2016F_2/180912_165152/0000/tuple_3*', 'SingleMu_2016F_2', isData=True))
sources.append(Source('SingleMu_2016F_2_4', '/SingleMuon/SingleMu_2016F_2/180912_165152/0000/tuple_4*', 'SingleMu_2016F_2', isData=True))
sources.append(Source('SingleMu_2016F_2_5', '/SingleMuon/SingleMu_2016F_2/180912_165152/0000/tuple_5*', 'SingleMu_2016F_2', isData=True))
sources.append(Source('SingleMu_2016F_2_6', '/SingleMuon/SingleMu_2016F_2/180912_165152/0000/tuple_6*', 'SingleMu_2016F_2', isData=True))
sources.append(Source('SingleMu_2016F_2_7', '/SingleMuon/SingleMu_2016F_2/180912_165152/0000/tuple_7*', 'SingleMu_2016F_2', isData=True))
sources.append(Source('SingleMu_2016F_2_8', '/SingleMuon/SingleMu_2016F_2/180912_165152/0000/tuple_8*', 'SingleMu_2016F_2', isData=True))
sources.append(Source('SingleMu_2016F_2_9', '/SingleMuon/SingleMu_2016F_2/180912_165152/0000/tuple_9*', 'SingleMu_2016F_2', isData=True))        

sources.append(Source('SingleMu_2016G_1', '/SingleMuon/SingleMu_2016G/180912_165211/0000/tuple_1*', 'SingleMu_2016G',  isData=True))
sources.append(Source('SingleMu_2016G_2', '/SingleMuon/SingleMu_2016G/180912_165211/0000/tuple_2*', 'SingleMu_2016G',  isData=True))
sources.append(Source('SingleMu_2016G_3', '/SingleMuon/SingleMu_2016G/180912_165211/0000/tuple_3*', 'SingleMu_2016G',  isData=True))
sources.append(Source('SingleMu_2016G_4', '/SingleMuon/SingleMu_2016G/180912_165211/0000/tuple_4*', 'SingleMu_2016G',  isData=True))
sources.append(Source('SingleMu_2016G_5', '/SingleMuon/SingleMu_2016G/180912_165211/0000/tuple_5*', 'SingleMu_2016G',  isData=True))
sources.append(Source('SingleMu_2016G_6', '/SingleMuon/SingleMu_2016G/180912_165211/0000/tuple_6*', 'SingleMu_2016G',  isData=True))
sources.append(Source('SingleMu_2016G_7', '/SingleMuon/SingleMu_2016G/180912_165211/0000/tuple_7*', 'SingleMu_2016G',  isData=True))
sources.append(Source('SingleMu_2016G_8', '/SingleMuon/SingleMu_2016G/180912_165211/0000/tuple_8*', 'SingleMu_2016G',  isData=True))
sources.append(Source('SingleMu_2016G_9', '/SingleMuon/SingleMu_2016G/180912_165211/0000/tuple_9*', 'SingleMu_2016G',  isData=True))  

sources.append(Source('SingleMu_2016H_1_1', '/SingleMuon/SingleMu_2016H_1/180912_165229/0000/tuple_1*','SingleMu_2016H_1',  isData=True))
sources.append(Source('SingleMu_2016H_1_2', '/SingleMuon/SingleMu_2016H_1/180912_165229/0000/tuple_2*','SingleMu_2016H_1',  isData=True))
sources.append(Source('SingleMu_2016H_1_3', '/SingleMuon/SingleMu_2016H_1/180912_165229/0000/tuple_3*','SingleMu_2016H_1',  isData=True))
sources.append(Source('SingleMu_2016H_1_4', '/SingleMuon/SingleMu_2016H_1/180912_165229/0000/tuple_4*','SingleMu_2016H_1',  isData=True))
sources.append(Source('SingleMu_2016H_1_5', '/SingleMuon/SingleMu_2016H_1/180912_165229/0000/tuple_5*','SingleMu_2016H_1',  isData=True))
sources.append(Source('SingleMu_2016H_1_6', '/SingleMuon/SingleMu_2016H_1/180912_165229/0000/tuple_6*','SingleMu_2016H_1',  isData=True))
sources.append(Source('SingleMu_2016H_1_7', '/SingleMuon/SingleMu_2016H_1/180912_165229/0000/tuple_7*','SingleMu_2016H_1',  isData=True))
sources.append(Source('SingleMu_2016H_1_8', '/SingleMuon/SingleMu_2016H_1/180912_165229/0000/tuple_8*','SingleMu_2016H_1',  isData=True))
sources.append(Source('SingleMu_2016H_1_9', '/SingleMuon/SingleMu_2016H_1/180912_165229/0000/tuple_9*','SingleMu_2016H_1',  isData=True)) 

sources.append(Source('SingleMu_2016H_2_1', '/SingleMuon/SingleMu_2016H_2/180912_165249/0000/tuple_1*','SingleMu_2016H_2',  isData=True))
sources.append(Source('SingleMu_2016H_2_2', '/SingleMuon/SingleMu_2016H_2/180912_165249/0000/tuple_2*','SingleMu_2016H_2',  isData=True))
sources.append(Source('SingleMu_2016H_2_3', '/SingleMuon/SingleMu_2016H_2/180912_165249/0000/tuple_3*','SingleMu_2016H_2',  isData=True))
sources.append(Source('SingleMu_2016H_2_4', '/SingleMuon/SingleMu_2016H_2/180912_165249/0000/tuple_4*','SingleMu_2016H_2',  isData=True))
sources.append(Source('SingleMu_2016H_2_5', '/SingleMuon/SingleMu_2016H_2/180912_165249/0000/tuple_5*','SingleMu_2016H_2',  isData=True))
sources.append(Source('SingleMu_2016H_2_6', '/SingleMuon/SingleMu_2016H_2/180912_165249/0000/tuple_6*','SingleMu_2016H_2',  isData=True))
sources.append(Source('SingleMu_2016H_2_7', '/SingleMuon/SingleMu_2016H_2/180912_165249/0000/tuple_7*','SingleMu_2016H_2',  isData=True))
sources.append(Source('SingleMu_2016H_2_8', '/SingleMuon/SingleMu_2016H_2/180912_165249/0000/tuple_8*','SingleMu_2016H_2',  isData=True))
sources.append(Source('SingleMu_2016H_2_9', '/SingleMuon/SingleMu_2016H_2/180912_165249/0000/tuple_9*','SingleMu_2016H_2',  isData=True))
        # ['H2Mu_VBF',    'H2Mu_VBF',     "/VBF_HToMuMu_M125_13TeV_powheg_pythia8/H2Mu_VBF/180827_202716/0000/*root"],

        # ['tt_ll_AMC',   'tt_ll_AMC_1',  "/TTJets_Dilept_TuneCUETP8M2T4_13TeV-amcatnloFXFX-pythia8/tt_ll_AMC/180827_203154/0000/tuple_1*"],
        # ['tt_ll_AMC',   'tt_ll_AMC_2',  "/TTJets_Dilept_TuneCUETP8M2T4_13TeV-amcatnloFXFX-pythia8/tt_ll_AMC/180827_203154/0000/tuple_2*"],
        # ['tt_ll_AMC',   'tt_ll_AMC_3',  "/TTJets_Dilept_TuneCUETP8M2T4_13TeV-amcatnloFXFX-pythia8/tt_ll_AMC/180827_203154/0000/tuple_3*"],
        # ['tt_ll_AMC',   'tt_ll_AMC_4',  "/TTJets_Dilept_TuneCUETP8M2T4_13TeV-amcatnloFXFX-pythia8/tt_ll_AMC/180827_203154/0000/tuple_4*"],
        # ['tt_ll_AMC',   'tt_ll_AMC_5',  "/TTJets_Dilept_TuneCUETP8M2T4_13TeV-amcatnloFXFX-pythia8/tt_ll_AMC/180827_203154/0000/tuple_5*"],
        # ['tt_ll_AMC',   'tt_ll_AMC_6',  "/TTJets_Dilept_TuneCUETP8M2T4_13TeV-amcatnloFXFX-pythia8/tt_ll_AMC/180827_203154/0000/tuple_6*"],
        # ['tt_ll_AMC',   'tt_ll_AMC_7',  "/TTJets_Dilept_TuneCUETP8M2T4_13TeV-amcatnloFXFX-pythia8/tt_ll_AMC/180827_203154/0000/tuple_7*"],   
        # ['tt_ll_AMC',   'tt_ll_AMC_8',  "/TTJets_Dilept_TuneCUETP8M2T4_13TeV-amcatnloFXFX-pythia8/tt_ll_AMC/180827_203154/0000/tuple_8*"],
        # ['tt_ll_AMC',   'tt_ll_AMC_9',  "/TTJets_Dilept_TuneCUETP8M2T4_13TeV-amcatnloFXFX-pythia8/tt_ll_AMC/180827_203154/0000/tuple_9*"],        
        # ['ZJets_MG',    'ZJets_MG_1',   "/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/ZJets_MG/180913_191722/0000/tuple_1*"],
        # ['ZJets_MG',    'ZJets_MG_2',   "/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/ZJets_MG/180913_191722/0000/tuple_2*"],
        # ['ZJets_MG',    'ZJets_MG_3',   "/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/ZJets_MG/180913_191722/0000/tuple_3*"],
        # ['ZJets_MG',    'ZJets_MG_4',   "/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/ZJets_MG/180913_191722/0000/tuple_4*"],
        # ['ZJets_MG',    'ZJets_MG_5',   "/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/ZJets_MG/180913_191722/0000/tuple_5*"],
        # ['ZJets_MG',    'ZJets_MG_6',   "/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/ZJets_MG/180913_191722/0000/tuple_6*"], 
        # ['ZJets_MG',    'ZJets_MG_7',   "/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/ZJets_MG/180913_191722/0000/tuple_7*"],
        # ['ZJets_MG',    'ZJets_MG_8',   "/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/ZJets_MG/180913_191722/0000/tuple_8*"],
        # ['ZJets_MG',    'ZJets_MG_9',   "/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/ZJets_MG/180913_191722/0000/tuple_9*"],       
    

try:
    os.makedirs(output_dir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
 
for src in sources:    
    path = "%s/%s"%(input_dir, src.in_filename)
    tree = ROOT.TChain("dimuons/tree")
    tree.Add(path)
    print src.name, tree.GetEntries()

    new_tree=ROOT.TTree("tree", "tree")
    metadata = ROOT.TChain("dimuons/metadata")
    metadata.Add(path)

    metadata.Draw("sumEventWeights>>eweights_"+src.name)
    sumEventWeightsHist = ROOT.gDirectory.Get("eweights_"+src.name) 
    nOriginalWeighted = sumEventWeightsHist.GetEntries()*sumEventWeightsHist.GetMean()

    new_metadata=metadata.CloneTree()

    mass            = array('f', [0])
    mass_Roch       = array('f', [0])
    mu1_eta         = array('f', [0])
    mu2_eta         = array('f', [0])
    max_abs_eta_mu  = array('f', [0])
    weight_over_lumi= array('f', [0])

    newBranch1 = new_tree.Branch("mass",            mass,           "mass/F")
    newBranch2 = new_tree.Branch("mass_Roch",       mass_Roch     , "mass_Roch/F")
    newBranch3 = new_tree.Branch("mu1_eta",         mu1_eta       , "mu1_eta/F")
    newBranch4 = new_tree.Branch("mu2_eta",         mu2_eta       , "mu2_eta/F")
    newBranch5 = new_tree.Branch("max_abs_eta_mu",  max_abs_eta_mu, "max_abs_eta_mu/F")
    newBranch6 = new_tree.Branch("weight_over_lumi",weight_over_lumi,"weight_over_lumi/F")


    try:
        os.makedirs(output_dir+"/"+src.out_dir_name)
        print "Created directory", output_dir+"/"+src.out_dir_name
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    new_file = ROOT.TFile("%s/%s/%s.root"%(output_dir, src.out_dir_name, src.name),"recreate")
    new_file.cd()
    new_file.mkdir("dimuons")
    new_file.cd("dimuons")


    for i in range(tree.GetEntries()):
        tree.GetEntry(i)

        mass[0]=tree.GetLeaf('muPairs.mass').GetValue()
        mass_Roch[0]=tree.GetLeaf('muPairs.mass_Roch').GetValue()
        mu1_eta[0]=tree.GetLeaf('muons.eta').GetValue(0)
        mu2_eta[0]=tree.GetLeaf('muons.eta').GetValue(1)
        if abs(mu1_eta[0])>abs(mu2_eta[0]):
            max_abs_eta_mu[0]=abs(mu1_eta[0])
        else:
            max_abs_eta_mu[0]=abs(mu2_eta[0])

        if src.isData:
            weight_over_lumi[0] = 1

        else:
            if "2016" in src.year:
                IsoMu_SF_3 = tree.GetLeaf('IsoMu_SF_3').GetValue()
                IsoMu_SF_4 = tree.GetLeaf('IsoMu_SF_4').GetValue()
                MuID_SF_3 = tree.GetLeaf('MuID_SF_3').GetValue()
                MuID_SF_4 = tree.GetLeaf('MuID_SF_4').GetValue()
                MuIso_SF_3 = tree.GetLeaf('MuIso_SF_3').GetValue()
                MuIso_SF_4 = tree.GetLeaf('MuIso_SF_4').GetValue()
                SF = (0.5*(IsoMu_SF_3 + IsoMu_SF_4)*0.5*(MuID_SF_3 + MuID_SF_4)*0.5*(MuIso_SF_3 + MuIso_SF_4))
            elif "2017" in src.year:
                IsoMu_SF_3 = tree.GetLeaf('IsoMu_SF_3').GetValue()
                MuID_SF_3 = tree.GetLeaf('MuID_SF_3').GetValue()
                MuIso_SF_3 = tree.GetLeaf('MuIso_SF_3').GetValue()                
                SF = IsoMu_SF_3 * MuID_SF_3 * MuIso_SF_3
            else:
                SF = 1
            
            PU_wgt = tree.GetLeaf('PU_wgt').GetValue()
            GEN_wgt = tree.GetLeaf('GEN_wgt').GetValue()

            xSec_wgt = src.xSec/nOriginalWeighted

            weight_over_lumi[0] = SF *PU_wgt *xSec_wgt

        new_tree.Fill()
    new_tree.Write()

    new_metadata.Write()

    
    new_file.Close()

