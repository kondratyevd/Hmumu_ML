# from datetime import datetime
import os, sys, errno
# import config
from config import variables, pkg_names, Package

sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import ROOT
from ROOT import gInterpreter, gSystem
gInterpreter.ProcessLine('#include "interface/JetInfo.h"')
gInterpreter.ProcessLine('#include "interface/EventInfo.h"')
gInterpreter.ProcessLine('#include "interface/VertexInfo.h"')
gInterpreter.ProcessLine('#include "interface/JetPairInfo.h"')
gInterpreter.ProcessLine('#include "interface/MuonInfo.h"')
gInterpreter.ProcessLine('#include "interface/MuPairInfo.h"')
gInterpreter.ProcessLine('#include "interface/EleInfo.h"')
gInterpreter.ProcessLine('#include "interface/ElePairInfo.h"')
gInterpreter.ProcessLine('#include "interface/MetInfo.h"')
gInterpreter.ProcessLine('#include "interface/MhtInfo.h"')
gInterpreter.ProcessLine('#include "interface/GenParentInfo.h"')
gInterpreter.ProcessLine('#include "interface/GenMuonInfo.h"')
gInterpreter.ProcessLine('#include "interface/GenMuPairInfo.h"')

class Framework(object):
	def __init__(self):
		self.file_list_s = []
		self.file_list_b = []
		self.variable_list = []
		self.nVar = 0
		self.package_list = []
		self.treePath = 'dimuons/tree'
		self.metadataPath = 'dimuons/metadata'
		self.outPath = ''
		self.transf_list = ['I']
		self.RunID = "Run_X/"
		self.prepare_dirs()


	class File(object):
		def __init__(self, source, name, path, xSec, weight):
			self.source = source
			self.name = name
			self.path = path
			self.xSec = xSec
			self.weight = weight
			self.nEvt = 1
			self.nOriginalWeighted = 1
			self.get_original_nEvts()						
		
		def get_original_nEvts(self):
			ROOT.gROOT.SetBatch(1)
			dummy = ROOT.TCanvas("dummmy","dummy",100,100)
			metadata = ROOT.TChain(self.source.metadataPath)
			metadata.Add(self.path)
			print self.source.metadataPath
			print metadata.GetEntries()
			metadata.Draw("originalNumEvents>>nEvt_"+self.name)
			metadata.Draw("sumEventWeights>>eweights_"+self.name)
			nEvtHist = ROOT.gDirectory.Get("nEvt_"+self.name) 
			self.nEvt = nEvtHist.GetEntries()*nEvtHist.GetMean()
  			sumEventWeightsHist = ROOT.gDirectory.Get("eweights_"+self.name) 
			self.nOriginalWeighted = sumEventWeightsHist.GetEntries()*sumEventWeightsHist.GetMean()

	def prepare_dirs(self):
		# now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
		# self.RunID = "Run_"+now+"/"

		with open("output/CURRENT_RUN_ID", "r") as IDfile:
			self.RunID=IDfile.read()

		self.outPath = 'output/'+self.RunID

		self.create_dir(self.outPath)

		info = open("output/CURRENT_RUN_ID","w") 
		info.write(self.RunID)
		info.close() 

	def create_dir(self, path):
		try:
			os.makedirs(path)
		except OSError as e:
			if e.errno != errno.EEXIST:
				raise

	def add_signal(self, name, path, xSec, weight):
		print "Adding %s as signal with xSec %f.."%(name, xSec)
		self.file_list_s.append(self.File(self, name, path, xSec, weight))

	def add_background(self, name, path, xSec, weight):
		print "Adding %s as background with xSec %f.."%(name, xSec)
		self.file_list_b.append(self.File(self, name, path, xSec, weight))

	def set_tree_path(self, treePath):
		self.treePath = treePath

	def add_variable(self, name, nObj):
		if name not in [v.name for v in variables]:
			sys.exit("\n\nERROR: Variable %s not found in the list. Check this file: %s\n\n"%(name,config.__file__))
		else:
			for var in variables:
				if var.name == name:									
					print "Adding input variable %s  [%i] .."%(name, nObj)
					var.itemsAdded = nObj
					self.nVar = self.nVar + nObj
					self.variable_list.append(var)		
			

	def add_package(self, name):

		if name not in pkg_names:
			sys.exit("\n\nERROR: Package %s not found in the list. Check this file: %s\n\n"%(name,config.__file__))
		else:
			print "Will use %s package.."%name	
			self.package_list.append(Package(name, self))		

	def add_transf(self, transf):
		self.transf_list.append(transf)

	def train_methods(self):
		for pkg in self.package_list:
			pkg.train_package()