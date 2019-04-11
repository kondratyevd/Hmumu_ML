import os, sys, errno
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from array import array
import ROOT

class TMVAApplicator(object):
	def __init__(self, framework):
		self.framework = framework
		self.hasMass = framework.hasMass
		ROOT.TMVA.Tools.Instance()

	def __enter__(self):
		self.reader = ROOT.TMVA.Reader( "!Color:!Silent:!V" )
		self.load_variables()
		self.method = self.framework.method_list[0]
		self.reader.BookMVA(self.method, self.framework.evalXmlPath)
		self.loop_over_files()
		return self

	def __exit__(self, *args):
		pass

	def load_variables(self):
		self.branches = {}
		if self.hasMass:
			self.branches["muPairs.mass_Roch"] = array('f', [-999])
			self.reader.AddSpectator('muPairs.mass_Roch', self.branches["muPairs.mass_Roch"])
		for var in self.framework.variable_list:
			if var.isMultiDim:	
				for i in range(var.itemsAdded):
					if self.framework.weighByEvent:
						name = "%s_%i"%(var.name,i)
						self.branches[name] = array('f', [-999])
						self.reader.AddVariable(name, self.branches[name])
					else:
						name = "Alt$(%s[%i],%f)"%(var.name,i,var.replacement)
						self.branches[name] = array('f', [-999])
						self.reader.AddVariable(name, self.branches[name])
			else:
				name = var.name
				self.branches[name] = array('f', [-999])
				self.reader.AddVariable(name, self.branches[name])
		# print self.branches


	def loop_over_files(self):
		for file in self.framework.files_to_evaluate:
			for filename in os.listdir(file.path):
				if filename.endswith(".root"): 
					tree = ROOT.TChain("dimuons/tree")
					tree.Add(file.path+"/"+filename)
					print "Processing ", filename
					try:
						os.makedirs("%s/%s/"%(self.framework.evalOutPath, file.name))
					except OSError as e:
						if e.errno != errno.EEXIST:
							raise

					new_file = ROOT.TFile("%s/%s/%s_%s"%(self.framework.evalOutPath, file.name, self.method, filename),"recreate")
					new_file.cd()

					new_tree=ROOT.TTree()
					new_tree.SetName("tree")
					MVA = array('f', [0])
					mass = array('f', [0])
					max_abs_eta_mu = array('f', [0])
					weight_over_lumi = array('f', [0])
					new_tree.Branch("MVA" , MVA, "MVA/F")
					new_tree.Branch("mass" , mass, "mass/F")
					new_tree.Branch("max_abs_eta_mu" , max_abs_eta_mu, "max_abs_eta_mu/F")
					new_tree.Branch("weight_over_lumi" , weight_over_lumi, "weight_over_lumi/F")

					new_variables = {}

					for var in self.framework.variable_list:

						if var.isMultiDim:	
							for j in range(var.itemsAdded):					
								if self.framework.weighByEvent:
									name = "%s_%i"%(var.name,j)
								else:
									name = "Alt$(%s[%i],%f)"%(var.name,j,var.replacement)
								tree.SetBranchAddress(var.name, self.branches[name]) # ERROR MAY BE HERE
								new_variables[name] = array('f', [0])

								new_tree.Branch(name , new_variables[name], "%s/F"%(name))
						else:
							tree.SetBranchAddress(tree.FindBranch(var.name).GetName(), self.branches[var.name])
							new_variables[var.name] = array('f', [0])

							new_tree.Branch(var.name , new_variables[var.name], "%s/F"%(var.name))
						
					for i in range(tree.GetEntries()):
					# for i in range(100):
						tree.GetEntry(i)
						flag, SF = self.eventInfo(tree, self.framework.year, file.isData)
			
						if flag:
							for var in self.framework.variable_list:

								if var.abs:
									if var.isMultiDim:
										for j in range(var.itemsAdded):
											if tree.GetLeaf(var.validation).GetValue() > j:
												try:
													if "F" in var.type:
														self.branches["%s_%i"%(var.name,j)][0] = abs(ROOT.Double(tree.FindBranch(var.name).FindLeaf(var.leaf).GetValue(j)))
													else:
														self.branches["%s_%i"%(var.name,j)][0] = abs(ROOT.Long(tree.FindBranch(var.name).FindLeaf(var.leaf).GetValue(j)))									
												except:
													self.branches["%s_%i"%(var.name,j)][0] = var.replacement
													print "fail in ", var.name
											else:
												self.branches["%s_%i"%(var.name,j)][0] = var.replacement
											new_variables["%s_%i"%(var.name,j)][0] = self.branches["%s_%i"%(var.name,j)][0]

									else:
										if tree.GetLeaf(var.validation).GetValue() > 0:
											try:
												if "F" in var.type:
													self.branches[var.name][0] = abs(ROOT.Double(tree.FindBranch(var.name).FindLeaf(var.leaf).GetValue()))	
												else:								
													self.branches[var.name][0] = abs(ROOT.Long(tree.FindBranch(var.name).FindLeaf(var.leaf).GetValue()))
											except:
												self.branches[var.name][0] = var.replacement
												print "fail in ", var.name
										else:
											self.branches[var.name][0] = var.replacement
										new_variables[var.name][0] = self.branches[var.name][0]
								else:
									if var.isMultiDim:
										for j in range(var.itemsAdded):									
											if tree.GetLeaf(var.validation).GetValue() > j:
												try:
													if "F" in var.type:
														self.branches["%s_%i"%(var.name,j)][0] = ROOT.Double(tree.FindBranch(var.name).FindLeaf(var.leaf).GetValue(j))
													else:
														self.branches["%s_%i"%(var.name,j)][0] = ROOT.Long(tree.FindBranch(var.name).FindLeaf(var.leaf).GetValue(j))
												except:
													self.branches["%s_%i"%(var.name,j)][0] = var.replacement
													print "fail in ", var.name
											else:
												self.branches["%s_%i"%(var.name,j)][0] = var.replacement
											new_variables["%s_%i"%(var.name,j)][0] = self.branches["%s_%i"%(var.name,j)][0]

									else:
										if tree.GetLeaf(var.validation).GetValue() > 0:
											try:
												if "F" in var.type:
													self.branches[var.name][0] = ROOT.Double(tree.FindBranch(var.name).FindLeaf(var.leaf).GetValue())
												else:
													self.branches[var.name][0] = ROOT.Long(tree.FindBranch(var.name).FindLeaf(var.leaf).GetValue())

											except:
												self.branches[var.name][0] = var.replacement
												print "fail in ", var.name
										else:
											self.branches[var.name][0] = var.replacement

										new_variables[var.name][0] = self.branches[var.name][0]

							# print self.branches
							MVA[0] = self.reader.EvaluateMVA(self.method)
							mass[0] = tree.FindBranch("muPairs.mass_Roch").FindLeaf("mass_Roch").GetValue()
							eta1 = tree.FindBranch("muons.eta").FindLeaf("eta").GetValue(0)
							eta2 = tree.FindBranch("muons.eta").FindLeaf("eta").GetValue(1)

						
							if abs(eta1)>abs(eta2):
								max_abs_eta_mu[0] = abs(eta1)
							else:
								max_abs_eta_mu[0] = abs(eta2)
							if file.isData:
								weight_over_lumi[0] = 1
							else:
								weight_over_lumi[0] = tree.PU_wgt*tree.GEN_wgt*SF*file.xSec/file.nOriginalWeighted
							new_tree.Fill()

					print new_tree.GetEntries()
					new_tree.Write()
					new_file.Close()



	def eventInfo(self, tree, year, isData):
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
			if isData:
				SF = 1
			else:
				SF = (0.5*(tree.IsoMu_SF_3 + tree.IsoMu_SF_4)*0.5*(tree.MuID_SF_3 + tree.MuID_SF_4)*0.5*(tree.MuIso_SF_3 + tree.MuIso_SF_4))


		elif year is "2017":

			flag = 			((muPair_mass>self.framework.massWindow[0])&
							(muPair_mass<self.framework.massWindow[1])&
							(muon1_ID>0)&
							(muon2_ID>0)&
							(muon1_pt>30)&
							(muon2_pt>20)
							)
			if isData:
				SF = 1
			else:
				SF = (tree.IsoMu_SF_3 * tree.MuID_SF_3 * tree.MuIso_SF_3 ) 

		return flag, SF			