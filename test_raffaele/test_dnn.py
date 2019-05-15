import os, sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from src.classifier import Framework

from samples import *

c = Framework()
c.label = "dnn_ucsdfiles"
comment = "4-class DNN, using UCSD files, mass window [120, 130]"	
				# change this line for each run
c.add_comment(comment)
print comment

c.outDir = '/tmp/dkondrat/ML_output/'
treePath = 'dimuons/tree'
metadataPath = 'dimuons/metadata'

c.set_tree_path(treePath)
c.set_metadata_path(metadataPath)

c.set_year("inclusive_ucsd")
c.massWindow = [120,130]
c.multiclass = True
c.dy_label = "ZJets_aMC"
c.tt_label = "tt_ll_POW"
c.ggh_label = "H2Mu_gg"
c.vbf_label = "H2Mu_VBF"


##################### Input samples #######################


c.add_category(c.ggh_label, True)
c.add_dir_to_category(ggH_ucsd_2016.name, ggH_ucsd_2016.path, ggH_ucsd_2016.xSec, c.ggh_label)
c.add_dir_to_category(ggH_ucsd_2017.name, ggH_ucsd_2017.path, ggH_ucsd_2017.xSec, c.ggh_label)
c.add_dir_to_category(ggH_ucsd_2018.name, ggH_ucsd_2018.path, ggH_ucsd_2018.xSec, c.ggh_label)

c.add_category(c.vbf_label, True)
c.add_dir_to_category(VBF_ucsd_2016.name, VBF_ucsd_2016.path, VBF_ucsd_2016.xSec, c.vbf_label)
c.add_dir_to_category(VBF_ucsd_2017.name, VBF_ucsd_2017.path, VBF_ucsd_2017.xSec, c.vbf_label)
c.add_dir_to_category(VBF_ucsd_2018.name, VBF_ucsd_2018.path, VBF_ucsd_2018.xSec, c.vbf_label)

c.add_category(c.dy_label, False)
c.add_dir_to_category(DY_ucsd_2016.name, DY_ucsd_2016.path, DY_ucsd_2016.xSec, c.dy_label)
c.add_dir_to_category(DY_ucsd_2017.name, DY_ucsd_2017.path, DY_ucsd_2017.xSec, c.dy_label)
c.add_dir_to_category(DY_ucsd_2018.name, DY_ucsd_2018.path, DY_ucsd_2018.xSec, c.dy_label)

c.add_category(c.tt_label, False)
c.add_dir_to_category(tt_ucsd_2016.name, tt_ucsd_2016.path, tt_ucsd_2016.xSec, c.tt_label)
c.add_dir_to_category(tt_ucsd_2017.name, tt_ucsd_2017.path, tt_ucsd_2017.xSec, c.tt_label)
c.add_dir_to_category(tt_ucsd_2018.name, tt_ucsd_2018.path, tt_ucsd_2018.xSec, c.tt_label)

##########################################################



###  ------   Raffaele's variables   ------ ###
c.add_variable("hmmpt", 				1) 
c.add_variable("hmmrap", 				1)
c.add_variable("hmmthetacs", 			1) 
c.add_variable("hmmphics",				1)
c.add_variable("met", 					1)

c.add_variable("m1ptOverMass", 	1)
c.add_variable("m2ptOverMass", 	1)
c.add_variable('m1eta',			1)
c.add_variable('m2eta',			1)
c.add_variable("drmj", 		 	1)
c.add_variable("njets", 		1)
c.add_variable("nbjets",		1)
c.add_variable("zepen",			1)

c.add_variable("j1pt",			1)
c.add_variable("j2pt",			1)
c.add_variable("mjj",			1)
c.add_variable("detajj",		1)
c.add_variable("dphijj",		1)
###############################################



c.add_spectator('hmass',		        1)
c.add_spectator('weight',		        1)



c.weigh_by_event(True)

c.add_package("Keras_multi")
c.add_method("model_50_D2_25_D2_25_D2") # Dropout 0.2



c.train_methods()

print "Training is done: "
print comment
print "Output saved to:"
print c.outPath

