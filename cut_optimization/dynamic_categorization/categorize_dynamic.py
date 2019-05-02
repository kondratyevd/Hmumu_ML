import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import ROOT
from math import sqrt
from make_datacards import create_datacard
import argparse
import multiprocessing as mp

ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.Eval)
ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.Fitting)
ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.Minimization)
ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.ObjectHandling)
ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.NumIntegration)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--sig_in_path', action='store', dest='sig_input_path', help='Input path')
parser.add_argument('--data_in_path', action='store', dest='data_input_path', help='Input path')
# parser.add_argument('--sig_tree', action='store', dest='sig_tree', help='Tree name')
parser.add_argument('--data_tree', action='store', dest='data_tree', help='Tree name')
parser.add_argument('--out_path', action='store', dest='output_path', help='Output path')
# parser.add_argument('--lumi', action='store', dest='lumi', help='Integrated luminosity')
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

memorized = {}
s = [] # will store the best significance for the category containing bins i through j
best_splitting = [] # will store the best way to split the category containing bins i through j

def bins_to_illustration(min, max, bins):
    result = ""
    for iii in range(min, max):
        if (iii in bins):
            result = result+"| "
        result = result+"%i "%iii
    result = result+"| "
    return result

def get_significance(label, bins, verbose=True):

    if "binary" in args.method:
        score = "sig_prediction"
        min_score = 0
        max_score = 1
    elif "DNNmulti" in args.method:
        score = "(ggH_prediction+VBF_prediction+(1-DY_prediction)+(1-ttbar_prediction))"
        min_score = 1
        max_score = 3
    elif "BDTmva" in args.method:
        score = "MVA"
        min_score = -1
        max_score = 1
    elif "Rapidity" in args.method:
        score = "max_abs_eta_mu"
        min_score = 0
        max_score = 2.4

    step = (args.max_var - args.min_var)/float(args.nSteps)

    new_bins = []
    for i in range(len(bins)):
        new_bins.append(min_score + bins[i]*step)
    if verbose:
        print "   Rescaled cut boundaries:"
        print "      ", bins, " --> ", new_bins

    categories = {}
    for i in range(len(new_bins)-1):
        cat_name = "cat%i"%i
        cut = "(%s>%f)&(%s<%f)"%(score, new_bins[i], score, new_bins[i+1])
        categories[cat_name] = cut

    if verbose:
        print "   Categories ready:"
        print  "   ",categories
        print "   Creating datacards... Please wait..."

    create_datacard(categories, args.sig_input_path, args.data_input_path, args.data_tree, args.output_path,  "datacard_"+label, "workspace_"+label, nuis=args.nuis, res_unc_val=args.res_unc_val, scale_unc_val=args.scale_unc_val, smodel=args.smodel, method=args.method, lumi=args.lumi)

    os.system('combine -M Significance --expectSignal=1 -t -1 -n %s -d datacard_%s.txt'%(label, label))
    os.system('rm datacard_%s.txt'%label)
    os.system('rm workspace_%s.root'%label)  
      
    significance = 0
    tree = ROOT.TChain("limit")
    tree.Add("higgsCombine%s.Significance.mH120.root"%label)
    for iev,  event in enumerate(tree):
        significance = event.limit
    os.system('rm higgsCombine%s.Significance.mH120.root'%label)

    if verbose:
        print "Expected significance %f calculated for bins"%significance, bins

    bins_str = ""
    for ii in range(len(bins)-1):
        bins_str = bins_str+"%f_"%bins[ii]
    bins_str = bins_str+"%f"%bins[len(bins)-1]

    memorized[bins_str]=significance

    if verbose:
        print "Memorizing result as "+bins_str 
        print " "
        print "############################################"
        print " "
    return significance


def solve_subproblem(i,j,s,best_splitting,memorized, verbose=True):
    s_ij = 0
    best_splitting_ij = []
    for k in range(i, j+1):
        consider_this_option = True
        can_decrease_nCat = False
        if k==i:
            if verbose:
                print "   First approach to P_%i%i: merge all bins."%(i,j)               
                print "   Merge bins from #%i to #%i into a single category"%(i, j)

            bins = [i,j+1] # here the numbers count not bins, but boundaries between bins, hence j+1

            if verbose:
                print "   Splitting is:   "+bins_to_illustration(i, j+1, bins)

            significance = get_significance("%i_%i_%i_%i"%(l, i, j, k), bins, verbose)
            sign_merged = significance

            if verbose:
                print "   Calculated significance for merged bins!"

            s_ij = significance
            best_splitting_ij = bins
        else:

            if verbose:
                print "   Continue solving P_%i%i!"%(i,j)
                print "   Cut between #%i and #%i"%(k-1, k)
                print "   Combine the optimal solutions of P_%i%i and P_%i%i:"%(i , k-1, k, j)
                print "   ",best_splitting[i][k-1], "s[%i][%i] = %f"%(i, k-1, s[i][k-1])
                print "   ",best_splitting[k][j], "s[%i][%i] = %f"%(k,j, s[k][j])

            bins = sorted(list(set(best_splitting[i][k-1]) | set(best_splitting[k][j]))) # sorted union of lists will provide the correct category boundaries

            if verbose:
                print "   Splitting is", bins_to_illustration(i, j+1, bins)

            bins_str = ""
            for ii in range(len(bins)-1):
                bins_str = bins_str+"%f_"%bins[ii]
            bins_str = bins_str+"%f"%bins[len(bins)-1]

            if bins_str in memorized.keys():
                if verbose:
                    print "   We already saw bins ", bins
                    print "     and the significance for them was ", memorized[bins_str]
                significance = memorized[bins_str]
            else:
                significance = sqrt(s[i][k-1]*s[i][k-1]+s[k][j]*s[k][j])
                memorized[bins_str]=significance
                if verbose:
                    print "   Significance = sqrt(s[%i][%i]^2+s[%i][%i]^2) = %f"%(i, k-1, k, j, significance)
                # significance = get_significance("%i_%i_%i_%i"%(l, i, j, k), bins)

            if s_ij:
                gain = ( significance - s_ij ) / s_ij*100.0
            else:
                gain = 999

            if verbose:
                print "   Before this option the best s[%i][%i] was %f for splitting "%(i, j, s_ij), best_splitting_ij
                print "   This option gives s[%i][%i] = %f for splitting "%(i, j, significance), bins          
                print "   We gain %f %% if we use the new option."%gain
                print "   The required gain if %f %% per additional category."%(args.penalty)
            ncat_diff = abs(len(bins) - len(best_splitting_ij))

            if ((len(bins)>len(best_splitting_ij))&(gain<args.penalty*ncat_diff)):
                if verbose:
                    print "     This option increases number of subcategories by %i from %i to %i, but the improvement is just %f %%, so skip."%(len(bins)-len(best_splitting_ij), len(best_splitting_ij)-1,len(bins)-1,gain)
                consider_this_option = False # don't split if improvement over merging is not good enough

            if ((len(bins)<len(best_splitting_ij))&(gain>-args.penalty*ncat_diff)):
                if verbose:
                    print "     This option decreases number of subcategories by %i from %i to %i, and the change in significance is just %f %%, so keep it."%(len(best_splitting_ij)-len(bins), len(best_splitting_ij)-1,len(bins)-1, -gain)
                can_decrease_nCat = True

            if ((len(bins)==len(best_splitting_ij))&(gain>0)):
                if verbose:
                    print "     This option keeps the same number of categories as the bes option so far, and the significance is increased by %f %%, so keep it."%gain

            if (((gain>0)&(consider_this_option)) or can_decrease_nCat): 
                s_ij = significance
                best_splitting_ij = bins
                if verbose:
                    print "   Updating best significance: now s[%i][%i] = "%(i,j), s_ij
            else:
                if verbose:
                    print "   Don't update best significance."

    return (i, j, s_ij, best_splitting_ij)


# Initialization
for i in range(args.nSteps):
    row = []
    row_bs = []
    for j in range(args.nSteps):
        row.append(0)
        row_bs.append([])
    s.append(row)
    best_splitting.append(row_bs)

# Main loop
# for l in range(1, args.nSteps+1): # subproblem size: from 1 to N. l=1 initializes the diagonal of s[i,j]
#     print "Scanning categories made of %i bins"%l
#     for i in range(0, args.nSteps - l + 1): # we are considering [bin_i, bin_j]
#         j = i + l - 1
#         print "="*50
#         print "   Solving subproblem P_%i%i"%(i,j)
#         print "   The goal is to find best significance in category containing bins #%i through #%i"%(i, j)
#         print "="*50

#         s[i][j], best_splitting[i][j] = solve_subproblem(i,j,s,best_splitting,memorized, verbose=False)

#         print "S_ij so far:"
#         for ii in range(args.nSteps):
#             row = ""
#             for jj in range(args.nSteps):
#                 row = row + "%f "%s[ii][jj]
#             print row

#         print "   Problem P_%i%i solved! Here's the best solution:"%(i,j)
#         print "      Highest significance for P_%i%i is %f and achieved when the splitting is "%(i, j, s[i][j]), bins_to_illustration(i, j+1, best_splitting[i][j])

def callback(result):
    i, j, s_ij, best_splitting_ij = result
    s[i][j] = s_ij
    best_splitting[i][j] = best_splitting_ij


for l in range(1, args.nSteps+1): # subproblem size: from 1 to N. l=1 initializes the diagonal of s[i,j]
    print "Scanning categories made of %i bins"%l
    print "Number of CPUs: ", mp.cpu_count()
    pool = mp.Pool(mp.cpu_count())

    a = [pool.apply_async(solve_subproblem, args = (i,i+l-1,s,best_splitting,memorized, False), callback=callback) for i in range(0, args.nSteps-l+1)]
    for process in a:
        process.wait()
    # for i in range(0, args.nSteps - l + 1): # we are considering [bin_i, bin_j]
        # j = i + l - 1
        # print "="*50
        # print "   Solving subproblem P_%i%i"%(i,j)
        # print "   The goal is to find best significance in category containing bins #%i through #%i"%(i, j)
        # print "="*50

        # s[i][j], best_splitting[i][j] = solve_subproblem(i,j,s,best_splitting,memorized, verbose=False)
        # s[i][j], best_splitting[i][j] = pool.apply(solve_subproblem, args = (i,j,s,best_splitting,memorized, False))

    pool.close()
    print "S_ij so far:"
    for ii in range(args.nSteps):
        row = ""
        for jj in range(args.nSteps):
            row = row + "%f "%s[ii][jj]
        print row

        # print "   Problem P_%i%i solved! Here's the best solution:"%(i,j)
        # print "      Highest significance for P_%i%i is %f and achieved when the splitting is "%(i, j, s[i][j]), bins_to_illustration(i, j+1, best_splitting[i][j])


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