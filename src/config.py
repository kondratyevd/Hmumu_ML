import importlib
import os, sys
# sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

################################ Variables ################################

variables = []
class Variable(object):
	def __init__(self, _name, _title, _units, _type, _isMultiDim, _itemsExpected):
		self.name = _name
		self.title = _title
		self.units = _units
		self.type = _type
		self.isMultiDim = _isMultiDim
		self.itemsAdded = 0
		self.itemsExpected = _itemsExpected 						#for example, there may be 3 muons but we don't want more than 2. 
																	
variables.append(Variable("muPairs.pt"		,"Dimuon p_{T}", 		"GeV", 		'F', False, 	1))
variables.append(Variable("muPairs.eta"		,"Dimuon #eta", 		"", 		'F', False, 	1))
variables.append(Variable("muPairs.dEta"	,"Dimuon |#delta#eta|", "", 		'F', False, 	1))
variables.append(Variable("muPairs.dPhi"	,"Dimuon |#delta#phi|", "", 		'F', False, 	1))
variables.append(Variable("muPairs.mass"	,"Dimuon mass", 		"GeV", 		'F', False, 	1))

variables.append(Variable("muons.pt"		,"Muon p_{T}", 			"GeV",		'F', True, 		2))
variables.append(Variable("muons.eta"		,"Muon #eta",  			"",   		'F', True, 		2))
variables.append(Variable("muons.phi"		,"Muon #phi",  			"",   		'F', True, 		2))

variables.append(Variable("met.pt"		,"MET",  			"GeV",   		'F', False, 		1))

variables.append(Variable("nJets"			,"nJets",  			"", 	  		'I', False, 	1))
variables.append(Variable("nJetsCent"		,"nJetsCent",  		"", 			 'I', False, 	1))
variables.append(Variable("nJetsFwd"		,"nJetsFwd",  		"", 			 'I', False, 	1))
variables.append(Variable("nBMed"			,"nBMed",  			"", 	  		'I', False, 	1))  

# variables.append(Variable("jets.pt"			,"Jet p_{T}",  		"GeV",   		'F', True, 		2)) 
# variables.append(Variable("jets.eta"		,"Jet #eta",  		"",   			'F', True, 		2)) 
# variables.append(Variable("jets.phi"		,"Jet #phi",  		"",   			'F', True, 		2)) 
# variables.append(Variable("jetPairs.dEta"	,"Jet pair |#delta#eta|",  	"",   	'F', True, 		2)) 
# variables.append(Variable("jetPairs.mass"	,"Jet pair mass",  		"GeV",   	'F', True, 		2))

variables.append(Variable("Alt$(jets.pt[0],-5)"			,"Jet1 p_{T}",  		"GeV",   		'F', False, 	1))
variables.append(Variable("Alt$(jets.pt[1],-5)"			,"Jet2 p_{T}",  		"GeV",   		'F', False, 	1)) 

variables.append(Variable("Alt$(jets.eta[0],-5)"		,"Jet1 #eta",  		"",   			'F', False, 	1))
variables.append(Variable("Alt$(jets.eta[1],-5)"		,"Jet2 #eta",  		"",   			'F', False, 	1)) 

variables.append(Variable("Alt$(jets.phi[0],-5)"		,"Jet1 #phi",  		"",   			'F', False, 	1)) 
variables.append(Variable("Alt$(jets.phi[1],-5)"		,"Jet2 #phi",  		"",   			'F', False, 	1)) 

variables.append(Variable("Alt$(jetPairs.dEta[0],-1)"	,"jj1 |#delta#eta|",  	"",   	'F', False, 	1)) 
variables.append(Variable("Alt$(jetPairs.dEta[1],-1)"	,"jj2 |#delta#eta|",  	"",   	'F', False, 	1)) 

variables.append(Variable("Alt$(jetPairs.mass[0],0)"	,"jj1 mass",  		"GeV",   	'F', False, 	1))
variables.append(Variable("Alt$(jetPairs.mass[1],0)"	,"jj2 mass",  		"GeV",   	'F', False, 	1))

################################ Packages ################################




pkg_names = ["TMVA", "Keras"]

class Package(object):
	def __init__(self, name, framework):
		self.name = name
		self.framework = framework
		self.mainDir = self.framework.outPath+self.name+"/"
		self.framework.create_dir(self.mainDir)
		self.dirs = {}
		if self.name is "Keras":
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


