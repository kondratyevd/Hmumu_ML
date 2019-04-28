import importlib
import os, sys
# sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

################################ Variables ################################

variables = []
class Variable(object):
	def __init__(self, _name, _leaf, _title, _units, _type, _isMultiDim, _validation, _replacement, _abs):
		self.name = _name
		self.leaf = _leaf
		self.title = _title
		self.units = _units
		self.type = _type
		self.isMultiDim = _isMultiDim
		self.itemsAdded = 0					
		self.replacement = _replacement
		self.validation = _validation
		self.abs = _abs
																	
variables.append(Variable("muPairs.pt"		,	"pt"				,"Dimuon p_{T}", 		"GeV", 		'F', True, 		"nMuPairs"	,	0	, False	))
variables.append(Variable("muPairs.eta"		,	"eta"				,"Dimuon #eta", 		"", 		'F', True, 		"nMuPairs"	,	-5	, False	))
variables.append(Variable("muPairs.phi"		,	"phi"				,"Dimuon #phi", 		"", 		'F', True, 		"nMuPairs"	,	-5	, False	))
variables.append(Variable("muPairs.dEta"	,	"dEta"				,"Dimuon |#delta#eta|", "", 		'F', True, 		"nMuPairs"	,	-1	, True	))
variables.append(Variable("muPairs.dPhi"	,	"dPhi"				,"Dimuon |#delta#phi|", "", 		'F', True, 		"nMuPairs"	,	-1	, True	))
variables.append(Variable("muPairs.mass"	,	"mass"				,"Dimuon mass", 		"GeV", 		'F', True, 		"nMuPairs"	,	0	, False	))
variables.append(Variable("muPairs.mass_Roch",	"mass"				,"Dimuon mass", 		"GeV", 		'F', True,  	"nMuPairs"	,	0	, False	))

variables.append(Variable("muPairs.mass_res",	"mass_res"			,"Ev-by-ev resolution", "GeV", 		'F', True, 		"nMuPairs"	,	0	, False	))
variables.append(Variable("muPairs.cosThetaCS"	,	"cosThetaCS"	,"cosThetaCS",			 "", 		'F', True, 		"nMuPairs"	,	-5	, False	))
variables.append(Variable("muPairs.phiCS"	,	"phiCS"				,"phiCS",				 "", 		'F', True, 		"nMuPairs"	,	-1	, False	))

variables.append(Variable("muons.pt"		,	"pt"				,"Muon p_{T}", 			"GeV",		'F', True, 		"nMuons"	,	0	, False	))

variables.append(Variable("mu1_pt_Roch_over_mass",	"mu1_pt_Roch_over_mass"	,"Muon1 p_{T} / Mmm", 	"",	'F', False, 	"nMuons"	,	0	, False	))
variables.append(Variable("mu2_pt_Roch_over_mass",	"mu2_pt_Roch_over_mass"	,"Muon2 p_{T} / Mmm", 	"",	'F', False, 	"nMuons"	,	0	, False	))

variables.append(Variable("muons.eta"		,	"eta"				,"Muon #eta",  			"",   		'F', True, 		"nMuons"	,	-5	, False	))
variables.append(Variable("muons.phi"		,	"phi"				,"Muon #phi",  			"",   		'F', True, 		"nMuons"	,	-5	, False	))
variables.append(Variable("muons.isMediumID",	"isMediumID"		,"Muon ID", 			"",			'I', True, 		"nMuons"	,	0	, False	))
variables.append(Variable("met.pt"			,	"pt"				,"MET",  				"GeV",   	'F', False, 	"nMuons"	,	0	, False	))
variables.append(Variable("nJets"			,	"nJets"				,"nJets",  				"", 	  	'I', False, 	"nMuons"	,	0	, False	))
variables.append(Variable("nJetsCent"		,	"nJetsCent"			,"nJetsCent",  			"", 		'I', False, 	"nMuons"	,	0	, False	))
variables.append(Variable("nJetsFwd"		,	"nJetsFwd"			,"nJetsFwd",  			"", 		'I', False, 	"nMuons"	,	0	, False	))
variables.append(Variable("nBMed"			,	"nBMed"				,"nBMed",  				"", 	  	'I', False, 	"nMuons"	,	0	, False	))

# variables.append(Variable("jets.pt"			,	"pt"				,"Jet p_{T}",  			"GeV",   	'F', True, 		"nJets"		,	-5	, False	))
# variables.append(Variable("jets.eta"		,	"eta"				,"Jet #eta",  			"",   		'F', True, 		"nJets"		,	-5	, False	))
# variables.append(Variable("jets.phi"		,	"phi"				,"Jet #phi",  			"",   		'F', True, 		"nJets"		,	-5	, False	)) 
# variables.append(Variable("jetPairs.dEta"	,	"dEta"				,"jj |#delta#eta|",  	"",   		'F', True, 		"nJetPairs"	,	-5	, True	)) 
# variables.append(Variable("jetPairs.dPhi"	,	"dPhi"				,"jj |#delta#phi|",  	"",   		'F', True, 		"nJetPairs"	,	-5	, True	)) 
# variables.append(Variable("jetPairs.mass"	,	"mass"				,"jj mass",  			"GeV",   	'F', True, 		"nJetPairs"	,	-5	, False	))

variables.append(Variable("jets.pt"			,	"pt"				,"Jet p_{T}",  			"GeV",   	'F', True, 		"nJets"		,	0	, False	))
variables.append(Variable("jets.eta"		,	"eta"				,"Jet #eta",  			"",   		'F', True, 		"nJets"		,	-5	, False	))
variables.append(Variable("jets.phi"		,	"phi"				,"Jet #phi",  			"",   		'F', True, 		"nJets"		,	-5	, False	)) 
variables.append(Variable("jetPairs.dEta"	,	"dEta"				,"jj |#delta#eta|",  	"",   		'F', True, 		"nJetPairs"	,	0	, True	)) 
variables.append(Variable("jetPairs.dPhi"	,	"dPhi"				,"jj |#delta#phi|",  	"",   		'F', True, 		"nJetPairs"	,	-1	, True	)) 
variables.append(Variable("jetPairs.mass"	,	"mass"				,"jj mass",  			"GeV",   	'F', True, 		"nJetPairs"	,	0	, False	))

# variables.append(Variable("min_dR_mu_jet"	,	"min_dR_mu_jet"		,"min_dR_mu_jet", 		"",			'F', False, 	"nJets"	,	-5	, False	))
# variables.append(Variable("max_dR_mu_jet"	,	"max_dR_mu_jet"		,"max_dR_mu_jet", 		"",			'F', False, 	"nJets"	,	-5	, False	))
# variables.append(Variable("min_dR_mumu_jet"	,	"min_dR_mumu_jet"	,"min_dR_mumu_jet", 	"",			'F', False, 	"nJets"	,	-5	, False	))
# variables.append(Variable("max_dR_mumu_jet"	,	"max_dR_mumu_jet"	,"max_dR_mumu_jet", 	"",			'F', False, 	"nJets"	,	-5	, False	))
# variables.append(Variable("zeppenfeld"		,	"zeppenfeld"		,"zeppenfeld", 			"",			'F', False, 	"nJetPairs"	,-5	, False	))

variables.append(Variable("min_dR_mu_jet"	,	"min_dR_mu_jet"		,"min_dR_mu_jet", 		"",			'F', False, 	"nJets"	,	0	, False	))
variables.append(Variable("max_dR_mu_jet"	,	"max_dR_mu_jet"		,"max_dR_mu_jet", 		"",			'F', False, 	"nJets"	,	0	, False	))
variables.append(Variable("min_dR_mumu_jet"	,	"min_dR_mumu_jet"	,"min_dR_mumu_jet", 	"",			'F', False, 	"nJets"	,	0	, False	))
variables.append(Variable("max_dR_mumu_jet"	,	"max_dR_mumu_jet"	,"max_dR_mumu_jet", 	"",			'F', False, 	"nJets"	,	0	, False	))
variables.append(Variable("zeppenfeld"		,	"zeppenfeld"		,"zeppenfeld", 			"",			'F', False, 	"nJetPairs"	,0	, False	))

variables.append(Variable("PU_wgt"			,	"PU_wgt"				,"PU_wgt",  		"",   		'F', False, 	"nMuons"	,	0	, False	))
variables.append(Variable("GEN_wgt"			,	"GEN_wgt"				,"GEN_wgt",  		"",   		'F', False, 	"nMuons"	,	0	, False	))
variables.append(Variable("IsoMu_SF_3"		,	"IsoMu_SF_3"			,"IsoMu_SF_3",  	"",   		'F', False, 	"nMuons"	,	0	, False	))
variables.append(Variable("MuID_SF_3"		,	"MuID_SF_3"				,"MuID_SF_3",  		"",   		'F', False, 	"nMuons"	,	0	, False	))
variables.append(Variable("MuIso_SF_3"		,	"MuIso_SF_3"			,"MuIso_SF_3",  	"",   		'F', False, 	"nMuons"	,	0	, False	))
variables.append(Variable("IsoMu_SF_4"		,	"IsoMu_SF_4"			,"IsoMu_SF_4",  	"",   		'F', False, 	"nMuons"	,	0	, False	))
variables.append(Variable("MuID_SF_4"		,	"MuID_SF_4"				,"MuID_SF_4",  		"",   		'F', False, 	"nMuons"	,	0	, False	))
variables.append(Variable("MuIso_SF_4"		,	"MuIso_SF_4"			,"MuIso_SF_4",  	"",   		'F', False, 	"nMuons"	,	0	, False	))
variables.append(Variable("muons.isHltMatched",	"isHltMatched"	,		"muons.isHltMatched", "",   	'I', True, 		"nMuons"	,	0	, False	))
################################ Packages ################################




pkg_names = ["TMVA", "Keras", "Keras_multi"]

class Package(object):
	def __init__(self, name, framework):
		self.name = name
		self.framework = framework
		self.mainDir = self.framework.outPath+self.name+"/"
		self.framework.create_dir(self.mainDir)
		self.dirs = {}
		if "Keras" in self.name:
			self.dirs['modelDir'] = self.mainDir+"models/"
			self.dirs['logDir'] = self.framework.outPath+"keras_logs/"
			self.dirs['dataDir'] = self.mainDir+"data/"
		elif self.name is "TMVA":
			self.dirs['modelDir'] = self.mainDir+"models/"
			self.dirs['logDir'] = self.framework.outPath+"keras_logs/"
			
		self.create_dirs()

	def create_dirs(self):
		for dir in self.dirs.values():
			self.framework.create_dir(dir)

	def train_package(self):
		importlib.import_module('packages.%s.train'%self.name).train(self.framework, self)

	def apply_package(self):
		importlib.import_module('packages.%s.apply'%self.name).apply(self.framework, self)


