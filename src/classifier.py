# from datetime import datetime
import os, sys, errno
import config
from config import variables, pkg_names, Package

sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import ROOT
from ROOT import gInterpreter, gSystem

gInterpreter.ProcessLine(' #include "interface/EleInfo.h"')
gInterpreter.ProcessLine(' #include "interface/EventInfo.h"')
gInterpreter.ProcessLine(' #include "interface/GenJetInfo.h"')
gInterpreter.ProcessLine(' #include "interface/GenMuPairInfo.h"')
gInterpreter.ProcessLine(' #include "interface/GenMuonInfo.h"')
gInterpreter.ProcessLine(' #include "interface/GenParentInfo.h"')
gInterpreter.ProcessLine(' #include "interface/GenPartInfo.h"')
gInterpreter.ProcessLine(' #include "interface/JetInfo.h"')
gInterpreter.ProcessLine(' #include "interface/JetPairInfo.h"')
gInterpreter.ProcessLine(' #include "interface/MetInfo.h"')
gInterpreter.ProcessLine(' #include "interface/MhtInfo.h"')
gInterpreter.ProcessLine(' #include "interface/MuPairInfo.h"')
gInterpreter.ProcessLine(' #include "interface/MuonInfo.h"')
gInterpreter.ProcessLine(' #include "interface/TauInfo.h"')
gInterpreter.ProcessLine(' #include "interface/VertexInfo.h"')

class Framework(object):
	def __init__(self):
		self.label = "default"
		self.file_list_s = []
		self.file_list_b = []
		self.dir_list_s = []
		self.dir_list_b = []
		self.files_to_evaluate = []
		self.variable_list = []
		self.spectator_list = []
		self.data_spectator_list = []
		self.signal_categories = []
		self.bkg_categories = []
		self.files = []
		self.data_files = []
		self.more_var_list = []
		self.nVar = 0
		self.package_list = []
		self.method_list = []
		self.year = "2016"
		self.treePath = 'dimuons/tree'
		self.metadataPath = 'dimuons/metadata'
		self.outPath = ''
		self.outDir = '/tmp/dkondrat/ML_output/'
		self.transf_list = ['I']
		self.RunID = "Run_X/"
		self.weighByEvent = False
		self.info_file = None
		self.prepare_dirs()
		self.custom_loss = False
		self.multiclass = True
		self.lumi = 0
		self.hasMass = False

		self.massWindow = [110, 150]

		self.dy_label = "ZJets_aMC"
		self.tt_label = "tt_ll_POW"
		self.ggh_label = "H2Mu_gg"
		self.vbf_label = "H2Mu_VBF"

		self.sig_label = "sig"
		self.bkg_label = "bkg"

		self.ebe_weights = False

	class File(object):
		def __init__(self, source, name, path, xSec, isDir, isData=False, repeat=1):
			self.source = source
			self.name = name
			self.path = path
			self.xSec = xSec
			self.isDir = isDir
			self.nEvt = 1
			self.nOriginalWeighted = 1
			self.weight = 1
			self.weight_over_lumi = 1
			self.isData = isData
			if not isData:
				if "ucsd" not in self.source.year:
					self.get_original_nEvts()
			self.category = ''	
			self.repeat=repeat					
		
		def get_original_nEvts(self):
			ROOT.gROOT.SetBatch(1)
			dummy = ROOT.TCanvas("dummmy","dummy",100,100)
			metadata = ROOT.TChain(self.source.metadataPath)
			if self.isDir:
				metadata.Add(self.path+"/*.root")		
			else:
				metadata.Add(self.path)
			print "metadata: ", metadata.GetEntries(), " entries"
			metadata.Draw("originalNumEvents>>nEvt_"+self.name)
			metadata.Draw("sumEventWeights>>eweights_"+self.name)
			nEvtHist = ROOT.gDirectory.Get("nEvt_"+self.name) 
			self.nEvt = nEvtHist.GetEntries()*nEvtHist.GetMean()
  			sumEventWeightsHist = ROOT.gDirectory.Get("eweights_"+self.name) 
			self.nOriginalWeighted = sumEventWeightsHist.GetEntries()*sumEventWeightsHist.GetMean()
			if self.source.lumi:
				self.weight = self.xSec*self.source.lumi / self.nOriginalWeighted
			else:
				self.weight = self.xSec*40000 / self.nOriginalWeighted
			self.weight_over_lumi = self.xSec / self.nOriginalWeighted

	def set_out_dir(self, _dir):
		self.outDir = _dir

	def prepare_dirs(self):
		with open("output/CURRENT_RUN_ID", "r") as IDfile:
			self.RunID=IDfile.read()

		self.outPath = self.outDir+"/"+self.RunID+"/"
		self.create_dir(self.outPath)

		print "Run ID:	%s"%self.RunID

		info = open("output/CURRENT_RUN_ID","w") 
		info.write(self.RunID)
		info.close() 

		self.info_file = open(self.outPath+"/info.txt","w")

	def create_dir(self, path):
		try:
			os.makedirs(path)
		except OSError as e:
			if e.errno != errno.EEXIST:
				raise

	def weigh_by_event(self, w):
		self.weighByEvent = w
		if w:
			print "Will weigh data event by event"
			self.info_file.write("Samples are weighted event by event \n")
		else:
			print "Samples are weighted proportional to sigma/N" 
			self.info_file.write("Samples are weighted proportional to sigma/N \n")

	def add_comment(self, str):
		self.info_file.write("%s\n"%str)

	def add_signal_file(self, name, path, xSec):
		print "Adding signal file %s with xSec=%f.."%(name, xSec)
		self.file_list_s.append(self.File(self, name, path, xSec, False))
		self.info_file.write("Signal:		%s\n"%path)

	def add_background_file(self, name, path, xSec):
		print "Adding background file %s with xSec=%f.."%(name, xSec)
		self.file_list_b.append(self.File(self, name, path, xSec, False))
		self.info_file.write("Background:	%s\n"%path)

	def add_signal_dir(self, name, path, xSec):
		print "Adding signal directory %s with xSec=%f.."%(path, xSec)
		self.dir_list_s.append(self.File(self, name, path, xSec, True))
		self.info_file.write("Signal dir:		%s\n"%path)

	def add_background_dir(self, name, path, xSec):
		print "Adding background directory %s with xSec=%f.."%(path, xSec)
		self.dir_list_b.append(self.File(self, name, path, xSec, True))
		self.info_file.write("Bkg dir:		%s\n"%path)

	def add_category(self, name, isSignal):
		if isSignal:
			self.signal_categories.append(name)
		else:
			self.bkg_categories.append(name)

	def add_dir_to_category(self, name, path, xSec, category, isDir=True, repeat=1):
		if category in self.signal_categories+self.bkg_categories: 
			print "Adding directory %s with xSec=%f as %s"%(path, xSec, category)
			print "Events will be repeated %i times."%repeat
			file = self.File(self, name, path, xSec, isDir, isData=False, repeat=repeat)
			file.category = category
			self.files.append(file)
			

	def add_data(self, name, path, lumi):
		print "Adding %s with lumi = %f"%(path, lumi)
		file = self.File(self, name, path, 1, True, isData=True)
		file.category = "Data"
		self.data_files.append(file)
		self.lumi += lumi

	def add_mc_to_evaluate(self, name, path, xSec):
		print "Adding %s to evaluate MVA score ..."%(path)
		file = self.File(self, name, path, xSec, False, isData=False)
		self.files_to_evaluate.append(file)

	def add_mc_dir_to_evaluate(self, name, path, xSec):
		print "Adding %s to evaluate MVA score ..."%(path)
		file = self.File(self, name, path, xSec, True, isData=False)
		self.files_to_evaluate.append(file)

	def add_data_to_evaluate(self, name, path):
		print "Adding %s to evaluate MVA score ..."%(path)
		file = self.File(self, name, path, 1, False, isData=True)
		self.files_to_evaluate.append(file)

	def setApplication(self,outPath, xmlPath):
		self.evalOutPath = outPath
		self.evalXmlPath = xmlPath
		self.create_dir(outPath)

	def set_tree_path(self, treePath):
		self.treePath = treePath

	def set_metadata_path(self, treePath):
		self.metadataPath = treePath	

	def set_year(self, year):
		self.year = year

	def add_variable(self, name, nObj):
		if name not in [v.name for v in variables]:
			sys.exit("\n\nERROR: Variable %s not found in the list. Check this file: %s\n\n"%(name,config.__file__))
		else:
			for var in variables:
				if var.name == name:									
					print "Adding input variable %s  [%i] .."%(name, nObj)
					self.info_file.write("Variable:		%s x%i\n"%(name, nObj))
					var.itemsAdded = nObj
					self.nVar = self.nVar + nObj
					self.variable_list.append(var)		
		

	def add_more_var(self, more_var_list):
		self.more_var_list.extend(more_var_list)
		print "Adding more variables: "
		print more_var_list

	def add_spectator(self, name, nObj):
		if name not in [v.name for v in variables]:
			sys.exit("\n\nERROR: Variable %s not found in the list. Check this file: %s\n\n"%(name,config.__file__))
		else:
			for var in variables:
				if var.name == name:									
					print "Adding spectator %s  [%i] .."%(name, nObj)
					self.info_file.write("Spectator:		%s x%i\n"%(name, nObj))
					var.itemsAdded = nObj
					self.nVar = self.nVar + nObj
					self.spectator_list.append(var)

	def add_data_spectator(self, name, nObj):
		if name not in [v.name for v in variables]:
			sys.exit("\n\nERROR: Variable %s not found in the list. Check this file: %s\n\n"%(name,config.__file__))
		else:
			for var in variables:
				if var.name == name:									
					print "Adding spectator %s  [%i] .."%(name, nObj)
					self.info_file.write("Spectator:		%s x%i\n"%(name, nObj))
					var.itemsAdded = nObj
					self.nVar = self.nVar + nObj
					self.data_spectator_list.append(var)

	def add_package(self, name):
		if name not in pkg_names:
			sys.exit("\n\nERROR: Package %s not found in the list. Check this file: %s\n\n"%(name,config.__file__))
		else:
			print "Will use %s package.."%name	
			self.package_list.append(Package(name, self))	
			self.info_file.write("Using package:	%s\n"%name)	

	def add_transf(self, transf):
		self.transf_list.append(transf)

	def add_method(self, name):
		self.method_list.append(name)
		print "Added method %s"%name

	def train_methods(self):
		for pkg in self.package_list:
			pkg.train_package()


	def apply_methods(self):
		for pkg in self.package_list:
			pkg.apply_package()