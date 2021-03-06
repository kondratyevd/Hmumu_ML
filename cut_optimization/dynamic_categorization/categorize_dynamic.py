import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import ROOT
from math import sqrt
from make_datacards import create_datacard
from make_datacards_ucsd import create_datacard_ucsd
import argparse
import multiprocessing as mp

# Disable most of RooFit output
ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.Eval)
ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.Fitting)
ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.Minimization)
ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.ObjectHandling)
ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.NumIntegration)
ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.DataHandling)
ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.InputArguments)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--sig_in_path', action='store', dest='sig_input_path', help='Input path')
parser.add_argument('--data_in_path', action='store', dest='data_input_path', help='Input path')
parser.add_argument('--data_tree', action='store', dest='data_tree', help='Tree name')
parser.add_argument('--out_path', action='store', dest='output_path', help='Output path')
parser.add_argument('--nuis', action='store_true', dest='nuis', help='Nuisances')
parser.add_argument('--nuis_val', action='store', dest='res_unc_val', help='Resolution uncertainty')
parser.add_argument('--scale_unc_val', action='store', dest='scale_unc_val', help='Scale uncertainty')
parser.add_argument('--smodel', action='store', dest='smodel', help='Signal model')
parser.add_argument('--option', action='store', dest='option', help='option')
parser.add_argument('--method', action='store', dest='method', help='method')
parser.add_argument('--min_var', action='store', dest='min_var', help='min_var', type=float)
parser.add_argument('--max_var', action='store', dest='max_var', help='max_var', type=float)
parser.add_argument('--nSteps', action='store', dest='nSteps', help='nSteps', type=int)
parser.add_argument('--lumi', action='store', dest='lumi', help='lumi', type=float)
parser.add_argument('--penalty', action='store', dest='penalty', help='penalty', type=float)
args = parser.parse_args()

log_mode = 0
parallel = True

def log(s):
    if log_mode is 0:
        pass
    elif log_mode is 1:
        print s
    # elif log_mode is 2:
    #   write into log file

def bins_to_illustration(min, max, bins):
    result = ""
    for iii in range(min, max):
        if (iii in bins):
            result = result+"| "
        result = result+"%i "%iii
    result = result+"| "
    return result

additional_cut = "(1)"

if "binary" in args.method:
    score = "sig_prediction"
elif "DNNmulti" in args.method:
    score = "(ggH_prediction+VBF_prediction+(1-DY_prediction)+(1-ttbar_prediction))"
elif "BDTmva" in args.method:
    score = "MVA"
elif "Rapidity" in args.method:
    score = "max_abs_eta_mu"
elif "UCSD_bdtuf" in args.method:
    score = "bdtuf"
elif "UCSD_bdtucsd_inclusive" in args.method:
    score = "bdtucsd_inclusive"
elif "UCSD_bdtucsd_01jet" in args.method:
    score = "bdtucsd_01jet"
    additional_cut = "(njets<2)"
elif "UCSD_bdtucsd_2jet" in args.method:
    score = "bdtucsd_2jet"
    additional_cut = "(njets>=2)"
elif "UCSD_bdtucsd_bveto" in args.method:
    score = "bdtucsd_2jet_bveto"
    additional_cut = "(njets>=2)"

elif "UCSD_bdtucsd_mjjcut250" in args.method:
    score = "bdtucsd_2jet_bveto"
    additional_cut = "(njets>=2)&(mjj<250)"    
elif "UCSD_bdtucsd_mjjcut300" in args.method:
    score = "bdtucsd_2jet_bveto"
    additional_cut = "(njets>=2)&(mjj<300)"     
elif "UCSD_bdtucsd_mjjcut350" in args.method:
    score = "bdtucsd_2jet_bveto"
    additional_cut = "(njets>=2)&(mjj<350)" 
elif "UCSD_bdtucsd_mjjcut400" in args.method:
    score = "bdtucsd_2jet_bveto"
    additional_cut = "(njets>=2)&(mjj<400)" 
elif "UCSD_bdtucsd_mjjcut450" in args.method:
    score = "bdtucsd_2jet_bveto"
    additional_cut = "(njets>=2)&(mjj<450)" 
elif "UCSD_bdtucsd_mjjcut500" in args.method:
    score = "bdtucsd_2jet_bveto"
    additional_cut = "(njets>=2)&(mjj<500)"     


file_path_old = "/mnt/hadoop/store/user/dkondrat/UCSD_files/"
sig_old = [file_path_old+filename for filename in ["tree_ggH.root","tree_VBF.root","tree_VH.root","tree_ttH.root"]]
bkg_old = [file_path_old+filename for filename in ["tree_DY.root","tree_top.root","tree_VV.root"]]

file_path_2016 = "/mnt/hadoop/store/user/dkondrat/UCSD_files/2016/"
file_path_2017 = "/mnt/hadoop/store/user/dkondrat/UCSD_files/2017/"
file_path_2018 = "/mnt/hadoop/store/user/dkondrat/UCSD_files/2018/"
ggh_name = "tree_ggH.root"
vbf_name = "tree_VBF.root"
vh_name = "tree_VH.root"
tth_name = "tree_ttH.root"
sig_names = [ggh_name, vbf_name, vh_name, tth_name]
sig_2016 = [file_path_2016+file for file in sig_names]
sig_2017 = [file_path_2017+file for file in sig_names]
sig_2018 = [file_path_2018+file for file in sig_names]
dy_name = "tree_DY.root"
tt_name = "tree_top.root"
vv_name = "tree_VV.root"
bkg_names = [dy_name, tt_name, vv_name]
bkg_2016 = [file_path_2016+file for file in bkg_names]
bkg_2017 = [file_path_2017+file for file in bkg_names]
bkg_2018 = [file_path_2018+file for file in bkg_names]

eta_categories = {
    "eta0": "(max_abs_eta_mu>0)&(max_abs_eta_mu<0.9)", 
    "eta1": "(max_abs_eta_mu>0.9)&(max_abs_eta_mu<1.9)", 
    "eta2": "(max_abs_eta_mu>1.9)&(max_abs_eta_mu<2.4)"
}

if args.option is "0": # ucsd categories

    cat_uf = {
        "cat0": "(bdtuf>-1)&(bdtuf<-0.15)",
        "cat1": "(bdtuf>-0.15)&(bdtuf<0.15)",
        "cat2": "(bdtuf>0.15)&(bdtuf<0.4)",
        "cat3": "(bdtuf>0.4)&(bdtuf<0.75)",
        "cat4": "(bdtuf>0.75)&(bdtuf<0.91)",
        "cat5": "(bdtuf>0.91)&(bdtuf<1)"
    }

    eta_cut_0 = "(abs(m1eta)<0.9)&(abs(m2eta)<0.9)"
    eta_cut_1 = "((abs(m1eta)<1.9)&(abs(m2eta)<1.9))&((abs(m1eta)>0.9)||(abs(m2eta)>0.9))"
    eta_cut_2 = "(abs(m1eta)>1.9)||(abs(m2eta)>1.9)"

    cat_uf_eta = {
        "cat00": "(bdtuf>-1)&(bdtuf<-0.15)",
        "cat10": "(bdtuf>-0.15)&(bdtuf<0.15)",
        "cat20": "(bdtuf>0.15)&(bdtuf<0.4)",
        "cat30": "(bdtuf>0.4)&(bdtuf<0.75)",
        "cat40": "(bdtuf>0.75)&(bdtuf<0.91)&(%s)"%(eta_cut_0),
        "cat41": "(bdtuf>0.75)&(bdtuf<0.91)&(%s)"%(eta_cut_1),
        "cat42": "(bdtuf>0.75)&(bdtuf<0.91)&(%s)"%(eta_cut_2),
        "cat5": "(bdtuf>0.91)&(bdtuf<1)"
    }

    my_best_uf = {
        "cat0": "(bdtuf>-1)&(bdtuf<-0.61)",
        "cat1": "(bdtuf>-0.61)&(bdtuf<-0.08)",
        "cat2": "(bdtuf>-0.08)&(bdtuf<0.19)",
        "cat3": "(bdtuf>0.19)&(bdtuf<0.38)",
        "cat4": "(bdtuf>0.38)&(bdtuf<0.67)",
        "cat5": "(bdtuf>0.67)&(bdtuf<0.89)",
        "cat6": "(bdtuf>0.89)&(bdtuf<1)",
    }

    my_best_uf_eta = {
        "cat0": "(bdtuf>-1)&(bdtuf<-0.61)",
        "cat1": "(bdtuf>-0.61)&(bdtuf<-0.08)",
        "cat2": "(bdtuf>-0.08)&(bdtuf<0.19)",
        "cat3": "(bdtuf>0.19)&(bdtuf<0.38)",
        "cat4": "(bdtuf>0.38)&(bdtuf<0.67)",
        "cat50": "(bdtuf>0.67)&(bdtuf<0.89)&(%s)"%(eta_cut_0),
        "cat51": "(bdtuf>0.67)&(bdtuf<0.89)&(%s)"%(eta_cut_1),
        "cat52": "(bdtuf>0.67)&(bdtuf<0.89)&(%s)"%(eta_cut_2),
        "cat6": "(bdtuf>0.89)&(bdtuf<1)",
    }

    cat_ucsd_incl = {
        "cat0": "(bdtucsd_inclusive>-1)&(bdtucsd_inclusive<-0.4)",
        "cat1": "(bdtucsd_inclusive>-0.4)&(bdtucsd_inclusive<0.2)",
        "cat2": "(bdtucsd_inclusive>0.2)&(bdtucsd_inclusive<0.4)",
        "cat3": "(bdtucsd_inclusive>0.4)&(bdtucsd_inclusive<0.6)",
        "cat4": "(bdtucsd_inclusive>0.6)&(bdtucsd_inclusive<0.8)",
        "cat5": "(bdtucsd_inclusive>0.8)&(bdtucsd_inclusive<0.86)",
        "cat6": "(bdtucsd_inclusive>0.86)&(bdtucsd_inclusive<1)" # this category only has 0.84 signal events!
    }

    my_best_incl = {
        "cat0": "(bdtucsd_inclusive>-1)&(bdtucsd_inclusive<-0.71)",
        "cat1": "(bdtucsd_inclusive>-0.71)&(bdtucsd_inclusive<-0.37)",
        "cat2": "(bdtucsd_inclusive>-0.37)&(bdtucsd_inclusive<0.16)",
        "cat3": "(bdtucsd_inclusive>0.16)&(bdtucsd_inclusive<0.58)",
        "cat4": "(bdtucsd_inclusive>0.58)&(bdtucsd_inclusive<0.82)",         
        "cat5": "(bdtucsd_inclusive>0.82)&(bdtucsd_inclusive<1)"
    }


    cat_ucsd_01jet = {
        "cat0": "(bdtucsd_01jet>-1)&(bdtucsd_01jet<-0.2)&(njets<2)",
        "cat1": "(bdtucsd_01jet>-0.2)&(bdtucsd_01jet<0.2)&(njets<2)",
        "cat2": "(bdtucsd_01jet>0.2)&(bdtucsd_01jet<0.55)&(njets<2)",
        "cat3": "(bdtucsd_01jet>0.55)&(bdtucsd_01jet<1)&(njets<2)",
    }

    my_best_01jet = {
        "cat0": "(bdtucsd_01jet>-1)&(bdtucsd_01jet<-0.58)&(njets<2)",
        "cat1": "(bdtucsd_01jet>-0.58)&(bdtucsd_01jet<-0.02)&(njets<2)",
        "cat2": "(bdtucsd_01jet>-0.02)&(bdtucsd_01jet<0.43)&(njets<2)", 
        "cat4": "(bdtucsd_01jet>0.43)&(bdtucsd_01jet<1)&(njets<2)",
    }


    cat_ucsd_2jet = {
        "cat0": "(bdtucsd_2jet>-1)&(bdtucsd_2jet<0)&(njets>=2)",
        "cat1": "(bdtucsd_2jet>0)&(bdtucsd_2jet<0.55)&(njets>=2)",
        "cat2": "(bdtucsd_2jet>0.55)&(bdtucsd_2jet<0.85)&(njets>=2)",
        "cat3": "(bdtucsd_2jet>0.85)&(bdtucsd_2jet<0.91)&(njets>=2)",
        "cat4": "(bdtucsd_2jet>0.91)&(bdtucsd_2jet<1)&(njets>=2)",    
    }

    cat_ucsd_2jet_bveto = {
        "cat0": "(bdtucsd_2jet_bveto>-1)&(bdtucsd_2jet_bveto<0.2)&(njets>=2)",
        "cat1": "(bdtucsd_2jet_bveto>0.2)&(bdtucsd_2jet_bveto<0.55)&(njets>=2)",
        "cat2": "(bdtucsd_2jet_bveto>0.55)&(bdtucsd_2jet_bveto<0.81)&(njets>=2)",
        "cat3": "(bdtucsd_2jet_bveto>0.81)&(bdtucsd_2jet_bveto<0.93)&(njets>=2)",
        "cat4": "(bdtucsd_2jet_bveto>0.93)&(bdtucsd_2jet_bveto<1)&(njets>=2)",    
    }

    my_best_2jet = {
        "cat0": "(bdtucsd_2jet>-1)&(bdtucsd_2jet<-0.6)&(njets>=2)",
        "cat2": "(bdtucsd_2jet>-0.6)&(bdtucsd_2jet<0.22)&(njets>=2)",        
        "cat3": "(bdtucsd_2jet>0.22)&(bdtucsd_2jet<0.73)&(njets>=2)",
        "cat7": "(bdtucsd_2jet>0.73)&(bdtucsd_2jet<0.9)&(njets>=2)",
        "cat8": "(bdtucsd_2jet>0.9)&(bdtucsd_2jet<1)&(njets>=2)",
    }

    my_best_2jet_bveto = {
        "cat0": "(bdtucsd_2jet_bveto>-1)&(bdtucsd_2jet_bveto<-0.04)&(njets>=2)",
        "cat1": "(bdtucsd_2jet_bveto>-0.04)&(bdtucsd_2jet_bveto<0.26)&(njets>=2)",
        "cat2": "(bdtucsd_2jet_bveto>0.26)&(bdtucsd_2jet_bveto<0.77)&(njets>=2)",
        "cat3": "(bdtucsd_2jet_bveto>0.77)&(bdtucsd_2jet_bveto<0.91)&(njets>=2)",
        "cat4": "(bdtucsd_2jet_bveto>0.91)&(bdtucsd_2jet_bveto<1)&(njets>=2)", 
    }

    my_best_2jet_bveto_mjjcut = {
        # "cat0": "(njets>=2)&(mjj<400)",
        "cat0": "(bdtucsd_2jet_bveto>-1)&(bdtucsd_2jet_bveto<0.2)&(njets>=2)&(mjj<400)",
        "cat1": "(bdtucsd_2jet_bveto>0.2)&(bdtucsd_2jet_bveto<1)&(njets>=2)&(mjj<400)",
        # "cat0": "(bdtucsd_2jet_bveto>-1)&(bdtucsd_2jet_bveto<-0.66)&(njets>=2)&(mjj<400)",
        # "cat1": "(bdtucsd_2jet_bveto>-0.66)&(bdtucsd_2jet_bveto<-0.34)&(njets>=2)&(mjj<400)",
        # "cat3": "(bdtucsd_2jet_bveto>-0.34)&(bdtucsd_2jet_bveto<-0.03)&(njets>=2)&(mjj<400)",
        # "cat4": "(bdtucsd_2jet_bveto>-0.03)&(bdtucsd_2jet_bveto<0.5)&(njets>=2)&(mjj<400)",
        # "cat5": "(bdtucsd_2jet_bveto>0.5)&(bdtucsd_2jet_bveto<1)&(njets>=2)&(mjj<400)",   
    }

    create_datacard_ucsd(my_best_2jet_bveto_mjjcut, sig_2016+sig_2017+sig_2018, bkg_2016+bkg_2017+bkg_2018, args.output_path,  "datacard", "workspace")
    os.system('pwd')
    os.system('ls')
    os.system('combine -M Significance --expectSignal=1 -t -1 -d datacard.txt --LoadLibrary /home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/dynamic_categorization/lib/RooDCBShape_cxx.so')
    os.system('rm datacard.txt')
    os.system('rm workspace.root') 
    sys.exit() 

step = (args.max_var - args.min_var)/float(args.nSteps)

s = []              # will store the best significance for the category containing bins i through j
best_splitting = [] # will store the best way to split the category containing bins i through j
memorized = {}
# Initialization
for i in range(args.nSteps):
    row = []
    row_bs = []
    for j in range(args.nSteps):
        row.append(0)
        row_bs.append([])
    s.append(row)
    best_splitting.append(row_bs)


def get_significance(label, bins):
    global memorized
    global additional_cut

    new_bins = []
    for i in range(len(bins)):
        new_bins.append(args.min_var + bins[i]*step)
    log("   Rescaled cut boundaries:")
    log("      %s --> %s"%(', '.join([str(b) for b in bins]), ', '.join([str(b) for b in new_bins])))

    log("   Categories ready:")
    categories = {}
    for i in range(len(new_bins)-1):
        cat_name = "cat%i"%i
        cut = "(%s>%f)&(%s<%f)&(%s)"%(score, new_bins[i], score, new_bins[i+1], additional_cut)
        categories[cat_name] = cut
        log("   %s:  %s"%(cat_name, cut))
    log("   Creating datacards... Please wait...")

    if "UCSD" in args.method:
        try:
            success = create_datacard_ucsd(categories, sig_2016+sig_2017+sig_2018, bkg_2016+bkg_2017+bkg_2018, args.output_path,  "datacard_"+label, "workspace_"+label)
        except:
            "There was an error. Setting significance to 0."
            return 0

    else:
        success = create_datacard(categories, args.sig_input_path, args.data_input_path, args.data_tree, args.output_path,  "datacard_"+label, "workspace_"+label, nuis=args.nuis, res_unc_val=args.res_unc_val, scale_unc_val=args.scale_unc_val, smodel=args.smodel, method=args.method, lumi=args.lumi)

    if not success:
        return 0

    os.system('combine -M Significance --expectSignal=1 -t -1 -n %s -d datacard_%s.txt  --LoadLibrary /home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/dynamic_categorization/lib/RooDCBShape_cxx.so'%(label, label))
    os.system('rm datacard_%s.txt'%label)
    os.system('rm workspace_%s.root'%label)  
      
    significance = 0
    tree = ROOT.TChain("limit")
    tree.Add("higgsCombine%s.Significance.mH120.root"%label)
    for iev,  event in enumerate(tree):
        significance = event.limit
    os.system('rm higgsCombine%s.Significance.mH120.root'%label)

    log("Expected significance %f calculated for bins %s"%(significance, ', '.join([str(b) for b in bins])))

    bins_str = ""
    for ii in range(len(bins)-1):
        bins_str = bins_str+"%f_"%bins[ii]
    bins_str = bins_str+"%f"%bins[len(bins)-1]
    memorized[bins_str]=significance

    log("Memorizing result as "+bins_str )
    log(" ")
    log("############################################")
    log(" ")
    return significance


def solve_subproblem(i,j):
    global s 
    global best_splitting
    global memorized

    log("="*50)
    log("   Solving subproblem P_%i%i"%(i,j))
    log("   The goal is to find best significance in category containing bins #%i through #%i"%(i, j))
    log("="*50)

    s_ij = 0
    best_splitting_ij = []
    for k in range(i, j+1):
        consider_this_option = True
        can_decrease_nCat = False
        if k==i:
            log("   First approach to P_%i%i: merge all bins."%(i,j)   )            
            log("   Merge bins from #%i to #%i into a single category"%(i, j))
            bins = [i,j+1] # here the numbers count not bins, but boundaries between bins, hence j+1
            log("   Splitting is:   "+bins_to_illustration(i, j+1, bins))

            significance = get_significance("%i_%i_%i_%i"%(l, i, j, k), bins)

            sign_merged = significance
            log("   Calculated significance for merged bins!")
            s_ij = significance
            best_splitting_ij = bins

        else:
            log("   Continue solving P_%i%i!"%(i,j))
            log("   Cut between #%i and #%i"%(k-1, k))
            log("   Combine the optimal solutions of P_%i%i and P_%i%i"%(i , k-1, k, j))

            bins = sorted(list(set(best_splitting[i][k-1]) | set(best_splitting[k][j]))) # sorted union of lists will provide the correct category boundaries

            log("   Splitting is "+bins_to_illustration(i, j+1, bins))
            bins_str = ""
            for ii in range(len(bins)-1):
                bins_str = bins_str+"%f_"%bins[ii]
            bins_str = bins_str+"%f"%bins[len(bins)-1]

            if bins_str in memorized.keys():
                log("   We already saw bins "+', '.join([str(b) for b in bins]))
                log("     and the significance for them was %f"%memorized[bins_str])
                significance = memorized[bins_str]
            else:
                significance = sqrt(s[i][k-1]*s[i][k-1]+s[k][j]*s[k][j]) # this would be exactly true for Poisson significance. For Asimov significance it's still a good approximation.
                memorized[bins_str]=significance
                log("   Significance = sqrt(s[%i][%i]^2+s[%i][%i]^2) = %f"%(i, k-1, k, j, significance))
                # significance = get_significance("%i_%i_%i_%i"%(l, i, j, k), bins)  # getting real Asimov significance (takes longer)

            if s_ij:
                gain = ( significance - s_ij ) / s_ij*100.0
            else:
                gain = 999

            log("   Before this option the best s[%i][%i] was %f for splitting %s"%(i, j, s_ij, ', '.join([str(b) for b in best_splitting_ij])))
            log("   This option gives s[%i][%i] = %f for splitting %s"%(i, j, significance, ', '.join([str(b) for b in bins])))
            log("   We gain %f %% if we use the new option."%gain)
            log("   The required gain if %f %% per additional category."%(args.penalty))
            ncat_diff = abs(len(bins) - len(best_splitting_ij))

            if ((len(bins)>len(best_splitting_ij))&(gain<args.penalty*ncat_diff)):
                log("     This option increases number of subcategories by %i from %i to %i, but the improvement is just %f %%, so skip."%(len(bins)-len(best_splitting_ij), len(best_splitting_ij)-1,len(bins)-1,gain))
                consider_this_option = False

            elif ((len(bins)<len(best_splitting_ij))&(gain>-args.penalty*ncat_diff)):
                log("     This option decreases number of subcategories by %i from %i to %i, and the change in significance is just %f %%, so keep it."%(len(best_splitting_ij)-len(bins), len(best_splitting_ij)-1,len(bins)-1, -gain))
                can_decrease_nCat = True

            elif ((len(bins)==len(best_splitting_ij))&(gain>0)):
                log("     This option keeps the same number of categories as the bes option so far, and the significance is increased by %f %%, so keep it."%gain)

            if (((gain>0)&(consider_this_option)) or can_decrease_nCat): 
                s_ij = significance
                best_splitting_ij = bins
                log("   Updating best significance: now s[%i][%i] = %f"%(i,j, s_ij))
            else:
                log("   Don't update best significance.")


    log("   Problem P_%i%i solved! Here's the best solution:"%(i,j))
    log("      Highest significance for P_%i%i is %f and achieved when the splitting is %s"%(i, j, s[i][j], bins_to_illustration(i, j+1, best_splitting[i][j])))
    s[i][j] = s_ij
    best_splitting[i][j] = best_splitting_ij
    return (i, j, s_ij, best_splitting_ij)

def callback(result):
    global s
    global best_splitting
    i, j, s_ij, best_splitting_ij = result
    s[i][j] = s_ij
    best_splitting[i][j] = best_splitting_ij


# Main loop
for l in range(1, args.nSteps+1): # subproblem size: from 1 to N. l=1 initializes the diagonal of s[i,j]. l=N is the big problem.
    print "Scanning categories made of %i bins"%l
    if parallel:
        print "Number of CPUs: ", mp.cpu_count()
        pool = mp.Pool(mp.cpu_count())
        a = [pool.apply_async(solve_subproblem, args = (i,i+l-1), callback=callback) for i in range(0, args.nSteps-l+1)]
        for process in a:
            process.wait()
        pool.close()

    else:   # if not parallel
        for i in range(0, args.nSteps - l + 1): # j = i+l-1
            solve_subproblem(i,i+l-1)

    print "S_ij so far:"
    for ii in range(args.nSteps):
        row = ""
        for jj in range(args.nSteps):
            row = row + "%f "%s[ii][jj]
        print row


# print "Solutions to all subproblems: "
# for i in range(args.nSteps):
#     for j in range(i, args.nSteps):
#         rescaled = []
#         for ii in range(len(best_splitting[i][j])):
#             rescaled.append(args.min_var + best_splitting[i][j][ii]*step)
#         print "P%i%i: "%(i, j), rescaled, "significance = %f"%s[i][j]

# print "----------------------------------------"
# print "Here are the values that were memorized:"
# print memorized
# print "----------------------------------------"

best_bins = best_splitting[0][args.nSteps-1]
print "Best significance overall is %f and achieved when the splitting is "%(s[0][args.nSteps-1]), bins_to_illustration(0, args.nSteps, best_bins)
new_bins = []
for i in range(len(best_bins)):
    new_bins.append(args.min_var + best_bins[i]*step)
print "Best cuts on MVA score are:"
print best_bins, " --> ", new_bins