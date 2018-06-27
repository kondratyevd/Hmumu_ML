import ROOT 
import os,errno


input_dir = "/Users/dmitry/root_files/mc/UF_tuples/"
output_dir = "/Users/dmitry/root_files/for_tmva/UF_framework_10k/"
filenames = ["zjets_amc", "gluglu", "vbf"]

try:
	os.makedirs(output_dir)
except OSError as e:
	if e.errno != errno.EEXIST:
		raise

class file(object):
	def __init__(self, name):
		self.name = name
		
files = []

for fname in filenames:
	files.append(file(fname))

for f in files:
	path = input_dir+f.name+".root"
	setattr(f, 'tree', ROOT.TChain("dimuons/tree"))
	f.tree.Add(path)
	setattr(f, 'new_tree', f.tree.CloneTree(10000))

	print f.new_tree.GetEntries()

	new_file = ROOT.TFile(output_dir+f.name+".root","recreate")
	new_file.cd()
	f.new_tree.Write()
	new_file.Close()


