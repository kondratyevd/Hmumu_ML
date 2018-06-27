
################################## Files ##################################

files = []

class file(object):
	def __init__(self, name, weight, isSignal):
		self.name = name
		self.weight = weight
		self.isSignal = isSignal

files.append(file('zjets_amc', 29.853717, 0))
files.append(file('gluglu', 0.006343, 1))
files.append(file('vbf', 0.000495, 1))

inputDir = "/Users/dmitry/root_files/for_tmva/UF_framework_10k/"
treePath = 'tree'

################################ Variables ################################

variables = []
class variable(object):
	def __init__(self, _name, _title, _units, _type, _isMultiDim, _itemsExpected):
		self.name = _name
		self.title = _title
		self.units = _units
		self.type = _type
		self.isMultiDim = _isMultiDim
		self.itemsExpected = _itemsExpected 						#for example, there may be 3 muons but we don't want more than 2. 
																	
variables.append(variable("muPairs.pt"		,"Dimuon p_{T}", 		"GeV", 		'F', True, 		1))
variables.append(variable("muPairs.eta"		,"Dimuon #eta", 		"", 		'F', True, 		1))
variables.append(variable("muPairs.dEta"	,"Dimuon |#delta#eta|", "", 		'F', True, 		1))
variables.append(variable("muPairs.dPhi"	,"Dimuon |#delta#phi|", "", 		'F', True, 		1))
variables.append(variable("muPairs.mass"	,"Dimuon mass", 		"GeV", 		'F', True, 		1))

variables.append(variable("muons.pt"		,"Muon p_{T}", 			"GeV",		'F', True, 		2))
variables.append(variable("muons.eta"		,"Muon #eta",  			"",   		'F', True, 		2))
variables.append(variable("muons.phi"		,"Muon #phi",  			"",   		'F', True, 		2))

variables.append(variable("nJets"			,"Jet p_{T}",  			"", 	  	'I', False, 	1)) 
variables.append(variable("jets.pt"			,"Jet p_{T}",  			"GeV",   	'F', True, 		5)) 
variables.append(variable("jets.eta"			,"Jet #eta",  		"",   		'F', True, 		5)) 
variables.append(variable("jets.phi"			,"Jet #phi",  		"",   		'F', True, 		5)) 


