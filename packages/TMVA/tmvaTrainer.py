import os, sys, errno
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from array import array
import ROOT

import mva_methods
from mva_methods import compile_method_list

class TMVATrainer(object):
	def __init__(self, framework, package):
		self.framework = framework
		self.package = package
		ROOT.TMVA.Tools.Instance()
		# ROOT.TMVA.PyMethodBase.PyInitialize()


	def __enter__(self):
		self.outputFile = ROOT.TFile.Open( self.package.mainDir+"TMVA.root", 'RECREATE' )
		print "Opening output file: "+self.package.mainDir+"TMVA.root"
		transformations = ';'.join(self.framework.transf_list)
		self.factory = ROOT.TMVA.Factory( "TMVAClassification_"+self.framework.label, self.outputFile, "!V:!Silent:Color:DrawProgressBar:Transformations=%s:AnalysisType=Classification"%transformations)
		self.dataloader = ROOT.TMVA.DataLoader("dataset")
		self.load_variables()
		if self.framework.weighByEvent:
			self.load_by_event()
		else:
			self.load_files()			
		self.load_methods()
		return self

	def __exit__(self, *args):
		print "Closing output file: "+self.package.mainDir+"TMVA.root"
		self.outputFile.Close()

	def load_files(self):
		for file in self.framework.file_list_s + self.framework.file_list_b + self.framework.dir_list_s + self.framework.dir_list_b:
			tree = ROOT.TChain(self.framework.treePath)
			if file.isDir:
				tree.Add(file.path+"/*.root")
			else:
				tree.Add(file.path)
			if (file in self.framework.file_list_s) or (file in self.framework.dir_list_s):
				self.dataloader.AddSignalTree(tree,file.weight)
			else:
				self.dataloader.AddBackgroundTree(tree,file.weight)

	def load_by_event(self):
		for file in self.framework.file_list_s + self.framework.file_list_b + self.framework.dir_list_s + self.framework.dir_list_b:
			tree = ROOT.TChain(self.framework.treePath)
			if file.isDir:
				tree.Add(file.path+"/*.root")
			else:
				tree.Add(file.path)
			print tree.GetEntries()

			for i in range(tree.GetEntries()):
				event = ROOT.std.vector(ROOT.double)()
				event.clear()
				tree.GetEntry(i)

				flag, SF = self.eventInfo(tree, self.framework.year)
				GEN_HT = tree.FindLeaf("LHT_HT").GetValue()
				HT_flag = True
				# if ('ZJets_MG' in file.name) and ('HT' not in file.name):
				# 	if GEN_HT>70:
				# 		HT_flag = False


				if (flag and HT_flag):
					for var in self.framework.variable_list:
						if 'muons.pt[0]/muPairs.pt' in var.name:
							event.push_back( tree.FindBranch("muons.pt").FindLeaf("pt").GetValue(0)/tree.FindBranch('muPairs.mass').FindLeaf('mass').GetValue() )
							continue
						if 'muons.pt[1]/muPairs.pt' in var.name:
							event.push_back( tree.FindBranch("muons.pt").FindLeaf("pt").GetValue(1)/tree.FindBranch('muPairs.mass').FindLeaf('mass').GetValue() )
							continue

						if var.abs:
							if var.isMultiDim:
								for j in range(var.itemsAdded):
									if tree.GetLeaf(var.validation).GetValue() > j:
										try:
											event.push_back( abs(ROOT.Double(tree.FindBranch(var.name).FindLeaf(var.leaf).GetValue(j))))
										except:
											event.push_back( var.replacement )
									else:
										event.push_back( var.replacement )	
							else:
								if tree.GetLeaf(var.validation).GetValue() > 0:
									try:
										event.push_back( abs(ROOT.Double(tree.FindBranch(var.name).FindLeaf(var.leaf).GetValue())))	
									except:
										event.push_back( var.replacement )
								else:
									event.push_back( var.replacement )	
						else:
							if var.isMultiDim:
								for j in range(var.itemsAdded):
									if tree.GetLeaf(var.validation).GetValue() > j:
										try:
											event.push_back( ROOT.Double(tree.FindBranch(var.name).FindLeaf(var.leaf).GetValue(j)) )
										except:
											event.push_back( var.replacement )
									else:
										event.push_back( var.replacement )	
							else:
								if tree.GetLeaf(var.validation).GetValue() > 0:
									try:
										event.push_back( ROOT.Double(tree.FindBranch(var.name).FindLeaf(var.leaf).GetValue()))
									except:
										event.push_back( var.replacement )
								else:
									event.push_back( var.replacement )	

				
				weight = tree.PU_wgt*tree.GEN_wgt*SF*file.xSec/file.nOriginalWeighted*40000 # I take lumi=40000 because it doesn't matter as it is applied to all samples

				if self.framework.ebe_weights:
					ebe_weight = tree.FindBranch("muPairs.mass_res").FindLeaf("mass_res").GetValue(0)
					mass = tree.FindBranch("muPairs.mass_Roch").FindLeaf("mass_Roch").GetValue(0)
					if ebe_weight and mass:
						weight = weight*(1/(ebe_weight/mass))

				if i % 2 == 0: # even-numbered events
					if (file in self.framework.file_list_s) or (file in self.framework.dir_list_s):
						self.dataloader.AddSignalTrainingEvent(event, weight)
					else:
						self.dataloader.AddBackgroundTrainingEvent(event, weight)
				else:
					if (file in self.framework.file_list_s) or (file in self.framework.dir_list_s):
						self.dataloader.AddSignalTestEvent(event, weight)
					else:
						self.dataloader.AddBackgroundTestEvent(event, weight)


	def load_variables(self):
		self.dataloader.AddSpectator('muPairs.mass_Roch', 'Dimuon mass (Roch.)', 'GeV', self.framework.massWindow[0], self.framework.massWindow[1])
		for var in self.framework.variable_list:
			if 'muons.pt[0]/muPairs.pt' in var.name:
				self.dataloader.AddVariable('jets.phi', 'mu1_pt/mass', '', 'F') # very random!
				continue
			if 'muons.pt[1]/muPairs.pt' in var.name:
				self.dataloader.AddVariable('jets.eta', 'mu2_pt/mass', '', 'F') # vey random!
				continue
			if var.isMultiDim:	
				for i in range(var.itemsAdded):
					if self.framework.weighByEvent:
						self.dataloader.AddVariable("%s_%i"%(var.name,i), var.title+"[%i]"%i, var.units, var.type)
					else:
						self.dataloader.AddVariable("Alt$(%s[%i],%f)"%(var.name,i,var.replacement), var.title+"[%i]"%i, var.units, var.type)
			else:
				self.dataloader.AddVariable(var.name, var.title, var.units, var.type)


	def load_methods(self):
		self.dataloader.PrepareTrainingAndTestTree(ROOT.TCut(''), 'nTrain_Signal=0:nTrain_Background=0:SplitMode=Random:NormMode=NumEvents:!V')
		for method in compile_method_list(self.framework, self.package):
			if method.name in self.framework.method_list:
				self.factory.BookMethod(self.dataloader, method.type, method.name, method.options)
				self.framework.info_file.write("TMVA method: 	%s %s\n"%(method.name, method.options))

	def train_methods(self):
		# pass
		self.factory.TrainAllMethods()
		self.factory.TestAllMethods()
		self.factory.EvaluateAllMethods()

	def eventInfo(self, tree, year):
		muon1_pt = tree.FindBranch("muons.pt").FindLeaf("pt").GetValue(0)
		muon2_pt = tree.FindBranch("muons.pt").FindLeaf("pt").GetValue(1)
		muon1_hlt2 = tree.FindBranch("muons.isHltMatched").FindLeaf("isHltMatched").GetValue(2)
		muon1_hlt3 = tree.FindBranch("muons.isHltMatched").FindLeaf("isHltMatched").GetValue(3)
		muon2_hlt2 = tree.FindBranch("muons.isHltMatched").FindLeaf("isHltMatched").GetValue(8)
		muon2_hlt3 = tree.FindBranch("muons.isHltMatched").FindLeaf("isHltMatched").GetValue(9)
		muon1_ID = tree.FindBranch("muons.isMediumID").FindLeaf("isMediumID").GetValue(0)
		muon2_ID = tree.FindBranch("muons.isMediumID").FindLeaf("isMediumID").GetValue(1)
		muPair_mass = tree.FindBranch("muPairs.mass_Roch").FindLeaf("mass_Roch").GetValue()
		nMuons = tree.FindBranch("nMuons").FindLeaf("nMuons").GetValue()
		nMuonPairs = tree.FindBranch("nMuPairs").FindLeaf("nMuPairs").GetValue()

		if year is "2016":

			flag = 			((muPair_mass>self.framework.massWindow[0])&
							(muPair_mass<self.framework.massWindow[1])&
							(muon1_ID>0)&
							(muon2_ID>0)&
							(muon1_pt>20)&
							(muon2_pt>20)&
							(
								( muon1_pt > 26 & (muon1_hlt2>0 or muon1_hlt3>0) ) 
							or
								( muon2_pt > 26 & (muon2_hlt2>0 or muon2_hlt3>0) )
							))
			SF = (0.5*(tree.IsoMu_SF_3 + tree.IsoMu_SF_4)*0.5*(tree.MuID_SF_3 + tree.MuID_SF_4)*0.5*(tree.MuIso_SF_3 + tree.MuIso_SF_4))


		elif year is "2017":

			flag = 			((muPair_mass>self.framework.massWindow[0])&
							(muPair_mass<self.framework.massWindow[1])&
							(muon1_ID>0)&
							(muon2_ID>0)&
							(muon1_pt>30)&
							(muon2_pt>20)
							# &
							# (
							# 	( muon1_pt > 30 & (muon1_hlt2>0 or muon1_hlt3>0) ) 
							# or
							# 	( muon2_pt > 30 & (muon2_hlt2>0 or muon2_hlt3>0) )
							# )
							)
			SF = (tree.IsoMu_SF_3 * tree.MuID_SF_3 * tree.MuIso_SF_3 ) 

		return flag, SF






