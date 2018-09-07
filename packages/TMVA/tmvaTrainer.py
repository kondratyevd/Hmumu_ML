import os, sys, errno
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import ROOT

import mva_methods
from mva_methods import compile_method_list

class TMVATrainer(object):
	def __init__(self, framework, package):
		self.framework = framework
		self.package = package
		self.file_list_s = []
		self.file_list_b = []
		ROOT.TMVA.Tools.Instance()
		# ROOT.TMVA.PyMethodBase.PyInitialize()


	def __enter__(self):
		self.outputFile = ROOT.TFile.Open( self.package.mainDir+"TMVA.root", 'RECREATE' )
		print "Opening output file: "+self.package.mainDir+"TMVA.root"
		transformations = ';'.join(self.framework.transf_list)
		self.factory = ROOT.TMVA.Factory( "TMVAClassification", self.outputFile, "!V:!Silent:Color:DrawProgressBar:Transformations=%s:AnalysisType=Classification"%transformations)
		self.dataloader = ROOT.TMVA.DataLoader("dataset")
		# self.load_files()
		# self.load_variables()
		self.load_by_event()
		self.load_methods()
		return self

	def __exit__(self, *args):
		print "Closing output file: "+self.package.mainDir+"TMVA.root"
		self.outputFile.Close()

	def load_files(self):
		for file in self.framework.file_list_s + self.framework.file_list_b:
			tree = ROOT.TChain(self.framework.treePath)
			tree.Add(file.path)
			if file in self.framework.file_list_s:
				self.dataloader.AddSignalTree(tree,file.weight)
			else:
				self.dataloader.AddBackgroundTree(tree,file.weight)

	def load_by_event(self):
		for file in self.framework.file_list_s + self.framework.file_list_b:
			tree = ROOT.TChain(self.framework.treePath)
			tree.Add(file.path)
			# print file.name
			for i in range(1):#(tree.GetEntries()):
				event = []
				tree.GetEntry(i)
				for var in self.framework.variable_list:
					if var.isMultiDim:
						for j in range(var.itemsAdded):
							event.append(tree.GetLeaf("%s[%i]"%(var.name,j)).GetValue(i))
					else:
						event.append(tree.GetLeaf(var.name).GetValue(i))					
				print event

				SF = (0.5*(tree.IsoMu_SF_3 + tree.IsoMu_SF_4)*0.5*(tree.MuID_SF_3 + tree.MuID_SF_4)*0.5*(tree.MuIso_SF_3 + tree.MuIso_SF_4))
				weight = tree.PU_wgt*tree.GEN_wgt*SF*file.xSec/file.nOriginalWeighted*40000 # I take lumi=40000 because it doesn't matter as it is applied to all samples

				# if file in self.framework.file_list_s:
				# 	self.dataloader.AddSignalTrainingEvent(event, weight)
				# else:
				# 	self.dataloader.AddBackgroundTrainingEvent(event, weight)

	def load_variables(self):
		for var in self.framework.variable_list:
			if var.isMultiDim:	
				for i in range(var.itemsAdded):
					self.dataloader.AddVariable("Alt$(%s[%i],%f)"%(var.name,i,var.replacement), var.title+"[%i]"%i, var.units, var.type)
			else:
				self.dataloader.AddVariable("Alt$(%s,%f)"%(var.name,var.replacement), var.title, var.units, var.type)

	def load_methods(self):
		self.dataloader.PrepareTrainingAndTestTree(ROOT.TCut(''), 'nTrain_Signal=0:nTrain_Background=0:SplitMode=Random:NormMode=NumEvents:!V')
		for method in compile_method_list(self.framework, self.package):
			self.factory.BookMethod(self.dataloader, method.type, method.name, method.options)

	def train_methods(self):
		self.factory.TrainAllMethods()
		self.factory.TestAllMethods()
		self.factory.EvaluateAllMethods()
