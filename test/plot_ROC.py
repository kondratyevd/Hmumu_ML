import os, sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

from src.plot_TMVA import *

p = Plotter_TMVA()
filedir = "/home/dkondra/Hmumu_analysis/Hmumu_ML/output/" 
p.add_TMVA_method("BDT", filedir, "Run_2018-09-10)12-05-15", "BDTG_UF_v1", "#ff0000")
# p.add_TMVA_method("BDT2", filedir, "Run_2018-09-06_16-30-51", "BDTG_UF_v1", "#0000ff")

p.get_ROC()
p.set_plot_path("plots/")
p.plot_hist_list("test_final", p.roc_hist_list)

