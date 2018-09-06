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
		self.load_files()
		self.load_variables()
		self.load_methods()
		return self

	def __exit__(self, *args):
		print "Closing output file: "+self.package.mainDir+"TMVA.root"
		self.outputFile.Close()

	def load_files(self):
		for file in self.framework.file_list_s + self.framework.file_list_b:
			tree = ROOT.TChain(self.framework.treePath)
			tree.Add(file.path)
			# new_tree = tree.CloneTree()
			# new_tree.SetName(file.name+"_tree")
			if file in self.framework.file_list_s:
				self.dataloader.AddSignalTree(tree,file.weight)
			else:
				self.dataloader.AddBackgroundTree(tree,file.weight)

	def load_variables(self):
		for var in self.framework.variable_list:
			if var.isMultiDim:	
				for i in range(var.itemsAdded):
					self.dataloader.AddVariable(var.name+"[%i]"%i, var.title+"[%i]"%i, var.units, var.type)
			else:
				self.dataloader.AddVariable(var.name, var.title, var.units, var.type)

	def load_methods(self):
		self.dataloader.PrepareTrainingAndTestTree(ROOT.TCut(''), 'nTrain_Signal=0:nTrain_Background=0:SplitMode=Random:NormMode=NumEvents:!V')
		for method in compile_method_list(self.framework, self.package):
			self.factory.BookMethod(self.dataloader, method.type, method.name, method.options)

	def train_methods(self):
		self.factory.TrainAllMethods()
		self.factory.TestAllMethods()
		self.factory.EvaluateAllMethods()
