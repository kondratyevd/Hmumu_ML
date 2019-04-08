
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

hadoop_path = "/mnt/hadoop/store/user/dkondrat/"
ggH_2017_powheg = MCSample("H2Mu_gg", hadoop_path+"/GluGluHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/H2Mu_gg_powheg/190406_001015/0000/", 0.009618)
VBF_2017_powheg = MCSample("H2Mu_VBF", hadoop_path+"/VBFHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/H2Mu_VBF_powheg/190406_001029/0000/", 0.0008208)

ZJets_aMC_2017 = MCSample("ZJets_aMC", "", 5765.4)
ZJets_aMC_2017_hiStat = MCSample("ZJets_aMC", hadoop_path+"/DYJetsToLL_M-105To160_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/ZJets_AMC_hiStat/190406_001056/0000/", 5765.4)

tt_ll_POW_2017 = MCSample("tt_ll_POW", hadoop_path+"/TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8/tt_ll_POW/190406_001138/0000/", 85.656)

SingleMu2017B = DataSample("SingleMu_2017B", hadoop_path+"/SingleMuon/SingleMu_2017B/190406_000906/0000/", 4793.961)
SingleMu2017C = DataSample("SingleMu_2017C", hadoop_path+"/SingleMuon/SingleMu_2017C/190406_000919/0000/", 9631.612)
SingleMu2017D = DataSample("SingleMu_2017D", hadoop_path+"/SingleMuon/SingleMu_2017D/190406_000933/0000/", 4208.785)
SingleMu2017E = DataSample("SingleMu_2017E", hadoop_path+"/SingleMuon/SingleMu_2017E/190406_000946/0000/", 8955.851)
SingleMu2017F = DataSample("SingleMu_2017F", hadoop_path+"/SingleMuon/SingleMu_2017F/190406_001000/0000/", 12900.503)


#local
# input_path = "/Users/dmitrykondratyev/Documents/HiggsToMuMu/test_files/"

# ggH_2017_powheg = MCSample("H2Mu_gg", input_path+"/ggh/", 0.009618)
# VBF_2017_powheg = MCSample("H2Mu_VBF", input_path+"/vbf/", 0.0008208)

# ZJets_aMC_2017 = MCSample("ZJets_aMC", input_path+"/dy/", 5765.4)
# ZJets_aMC_2017_hiStat = MCSample("ZJets_aMC", input_path+"/dy/", 5765.4)

# tt_ll_POW_2017 = MCSample("tt_ll_POW", input_path+"/tt/", 85.656)