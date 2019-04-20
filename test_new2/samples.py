
class MCSample(object):
	def __init__(self, name, path, xSec):
		self.name = name
		self.path = path
		self.xSec = xSec
		

class DataSample(object):
	def __init__(self, name, path, lumi):
		self.name = name
		self.path = path
		self.lumi = lumi

# Samples with added CS variables

ggH_2017_powheg = MCSample("H2Mu_gg", "/mnt/hadoop/store/user/dkondrat/GluGluHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/H2Mu_gg_powheg/190419_161651/0000/", 0.009618 )
VBF_2017_powheg = MCSample("H2Mu_VBF", "/mnt/hadoop/store/user/dkondrat/VBFHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/H2Mu_VBF_powheg/190419_161716/0000/", 0.0008208 )

ZJets_aMC_2017_hiStat = MCSample("ZJets_aMC",  "/mnt/hadoop/store/user/dkondrat/DYJetsToLL_M-105To160_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/ZJets_AMC_hiStat/190419_161740/0000/", 47.17)

tt_ll_POW_2017 = MCSample("tt_ll_POW", "/mnt/hadoop/store/user/dkondrat/TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8/tt_ll_POW/190419_161808/0000/", 85.656)

SingleMu2017B = DataSample("SingleMu_2017B", "/mnt/hadoop/store/user/dkondrat/SingleMuon/SingleMu_2017B/190419_161448/0000/", 4723.411) 
SingleMu2017C = DataSample("SingleMu_2017C", "/mnt/hadoop/store/user/dkondrat/SingleMuon/SingleMu_2017C/190419_161512/0000/", 9631.612) 
SingleMu2017D = DataSample("SingleMu_2017D", "/mnt/hadoop/store/user/dkondrat/SingleMuon/SingleMu_2017D/190419_161536/0000/", 4247.682) 
SingleMu2017E = DataSample("SingleMu_2017E", "/mnt/hadoop/store/user/dkondrat/SingleMuon/SingleMu_2017E/190419_161602/0000/", 9028.733)
SingleMu2017F = DataSample("SingleMu_2017F", "/mnt/hadoop/store/user/dkondrat/SingleMuon/SingleMu_2017F/190419_161626/0000/", 13443.249) # recalculate lumi!



#local
# input_path = "/Users/dmitrykondratyev/Documents/HiggsToMuMu/test_files/"

# ggH_2017_powheg = MCSample("H2Mu_gg", input_path+"/ggh/", 0.009618)
# VBF_2017_powheg = MCSample("H2Mu_VBF", input_path+"/vbf/", 0.0008208)

# ZJets_aMC_2017 = MCSample("ZJets_aMC", input_path+"/dy/", 5765.4)
# ZJets_aMC_2017_hiStat = MCSample("ZJets_aMC", input_path+"/dy/", 5765.4)

# tt_ll_POW_2017 = MCSample("tt_ll_POW", input_path+"/tt/", 85.656)