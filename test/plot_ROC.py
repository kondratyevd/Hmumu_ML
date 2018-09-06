import os, sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

from src.plot_TMVA import *

p = Plotter_TMVA()
filedir = "/Users/dmitry/Documents/HiggsToMuMu/Hmumu_ML/output/" 
p.add_TMVA_method("BDT1", filedir, "Run_2018-09-06_16-26-23", "BDTG_UF_v1", "#ff0000")
p.add_TMVA_method("BDT2", filedir, "Run_2018-09-06_16-30-51", "BDTG_UF_v1", "#0000ff")

p.get_ROC()
p.set_plot_path("plots/")
p.plot_hist_list("test", p.roc_hist_list)

