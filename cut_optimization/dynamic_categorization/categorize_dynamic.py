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

log_mode = 1
parallel = False

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
elif "UCSD_bdtucsd_2jet" in args.method:
    score = "bdtucsd_2jet"




eta_categories = {
    "eta0": "(max_abs_eta_mu>0)&(max_abs_eta_mu<0.9)", 
    "eta1": "(max_abs_eta_mu>0.9)&(max_abs_eta_mu<1.9)", 
    "eta2": "(max_abs_eta_mu>1.9)&(max_abs_eta_mu<2.4)"
}

if args.option is "0": # inclusive
    create_datacard({"cat0": "1"}, args.sig_input_path, args.data_input_path, args.data_tree, args.output_path,  "datacard_inclusive", "workspace_inclusive", nuis=args.nuis, res_unc_val=args.res_unc_val, scale_unc_val=args.scale_unc_val, smodel=args.smodel, method=args.method, lumi=args.lumi)
    create_datacard(eta_categories, args.sig_input_path, args.data_input_path, args.data_tree, args.output_path,  "datacard_inclusive_eta", "workspace_inclusive_eta", nuis=args.nuis, res_unc_val=args.res_unc_val, scale_unc_val=args.scale_unc_val, smodel=args.smodel, method=args.method, lumi=args.lumi)
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

    new_bins = []
    for i in range(len(bins)):
        new_bins.append(args.min_var + bins[i]*step)
    log("   Rescaled cut boundaries:")
    log("      %s --> %s"%(', '.join([str(b) for b in bins]), ', '.join([str(b) for b in new_bins])))

    log("   Categories ready:")
    categories = {}
    for i in range(len(new_bins)-1):
        cat_name = "cat%i"%i
        cut = "(%s>%f)&(%s<%f)"%(score, new_bins[i], score, new_bins[i+1])
        categories[cat_name] = cut
        log("   %s:  %s"%(cat_name, cut))
    log("   Creating datacards... Please wait...")

    if "UCSD" in args.method:
        file_path = "/mnt/hadoop/store/user/dkondrat/UCSD_files/"
        ggh_path = file_path+"tree_ggH.root"
        vbf_path = file_path+"tree_VBF.root"
        dy_path = file_path+"tree_DY.root"
        tt_path = file_path+"tree_top.root"
        vv_path = file_path+"tree_VV.root"
        success = create_datacard_ucsd(categories, ggh_path, vbf_path, dy_path, tt_path, vv_path, args.output_path,  "datacard_"+label, "workspace_"+label)

    else:
        success = create_datacard(categories, args.sig_input_path, args.data_input_path, args.data_tree, args.output_path,  "datacard_"+label, "workspace_"+label, nuis=args.nuis, res_unc_val=args.res_unc_val, scale_unc_val=args.scale_unc_val, smodel=args.smodel, method=args.method, lumi=args.lumi)

    if not success:
        return 0

    os.system('combine -M Significance --expectSignal=1 -t -1 -n %s -d datacard_%s.txt'%(label, label))
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
            s[i][i+l-1], best_splitting[i][i+l-1] = solve_subproblem(i,i+l-1,s,best_splitting,memorized)

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