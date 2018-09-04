import ROOT 
import os,errno


# input_dir = "/Users/dmitry/root_files/mc/UF_tuples/"
# output_dir = "/Users/dmitry/root_files/for_tmva/UF_framework_10k/"
# filenames = ["zjets_amc", "gluglu", "vbf"]

input_dir = "/home/dkondra/Hmummu_processed_samples/CMSSW_9_4_6_patch1/src/UfHMuMuCode/UFDiMuonsAnalyzer/crab/output/"
output_dir = "/home/dkondra/Hmummu_processed_samples/CMSSW_9_4_6_patch1/src/UfHMuMuCode/UFDiMuonsAnalyzer/crab/output/10k/"
filenames = [

		'DY_0J',
		'DY_1J',
		'DY_2J_1',
		'DY_2J_2',
		'H2Mu_VBF',
		'H2Mu_gg',
		'H2Mu_WH_neg',
		'H2Mu_WH_pos',
		'H2Mu_ZH',
		'WW',
		'WWW',
		'WWZ',
		'WZZ',
		'WZ_2l',
		'WZ_3l',
		'ZJets_AMC',
		'ZZZ',
		'ZZ_2l_2q',
		'ZZ_2l_2v',
		'ZZ_4l',
		'tW_neg_1',
		'tW_neg_2',
		'tW_pos_1',
		'tW_pos_2',
		'tZq',
		'ttW_1',
		'ttW_2',
		'ttZ',
		'tt_ll_AMC',
		'tt_ll_MG_1',
		'tt_ll_MG_2'

	]

try:
	os.makedirs(output_dir)
except OSError as e:
	if e.errno != errno.EEXIST:
		raise

class file(object):
	def __init__(self, name):
		self.name = name
		self.tree = None
		self.new_tree = None
		
files = []

for fname in filenames:
	files.append(file(fname))

for f in files:
	path = input_dir+f.name+".root"
	f.tree = ROOT.TChain("dimuons/tree")
	f.tree.Add(path)
	# print	"%s:	%i"%(f.name,f.tree.GetEntries())
	f.new_tree=f.tree.CloneTree(10000)

	print f.new_tree.GetEntries()

	new_file = ROOT.TFile(output_dir+f.name+"_10k.root","recreate")
	new_file.cd()
	new_file.mkdir("dimuons")
	new_file.cd("dimuons")
	f.new_tree.Write()
	new_file.Close()


