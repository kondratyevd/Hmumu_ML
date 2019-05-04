import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import ROOT
from math import sqrt
from make_datacards import create_datacard
import argparse
import multiprocessing as mp
from numpy import zeros

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
parser.add_argument('--min_var1', action='store', dest='min_var1', help='min_var1', type=float)
parser.add_argument('--max_var1', action='store', dest='max_var1', help='max_var1', type=float)
parser.add_argument('--min_var2', action='store', dest='min_var2', help='min_var2', type=float)
parser.add_argument('--max_var2', action='store', dest='max_var2', help='max_var2', type=float)
parser.add_argument('--nSteps1', action='store', dest='nSteps1', help='nSteps1', type=int)
parser.add_argument('--nSteps2', action='store', dest='nSteps2', help='nSteps2', type=int)
parser.add_argument('--lumi', action='store', dest='lumi', help='lumi', type=float)
parser.add_argument('--penalty', action='store', dest='penalty', help='penalty', type=float)
args = parser.parse_args()

log_mode = 0
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
    score1 = "sig_prediction"
elif "DNNmulti" in args.method:
    score1 = "(ggH_prediction+VBF_prediction+(1-DY_prediction)+(1-ttbar_prediction))"
elif "BDTmva" in args.method:
    score1 = "MVA"
elif "Rapidity" in args.method:
    score1 = "max_abs_eta_mu"

score2 = "max_abs_eta_mu"

eta_categories = {
    "eta0": "(max_abs_eta_mu>0)&(max_abs_eta_mu<0.9)", 
    "eta1": "(max_abs_eta_mu>0.9)&(max_abs_eta_mu<1.9)", 
    "eta2": "(max_abs_eta_mu>1.9)&(max_abs_eta_mu<2.4)"
}


step1 = (args.max_var1 - args.min_var1)/float(args.nSteps1)
step2 = (args.max_var2 - args.min_var2)/float(args.nSteps2)

s = zeros([args.nSteps1, args.nSteps1, args.nSteps2, args.nSteps2])             # will store the best significance for the category containing bins i through j
best_splitting1 = empty([args.nSteps1, args.nSteps1])  
best_splitting2 = empty([args.nSteps2, args.nSteps2])  
memorized = {}

def get_significance(label, bins1, bins2):
    global memorized

    new_bins1 = []
    for i in range(len(bins1)):
        new_bins1.append(args.min_var1 + bins1[i]*step1)
    new_bins2 = []
    for i in range(len(bins2)):
        new_bins2.append(args.min_var2 + bins2[i]*step2) 

    log("   Rescaled cut boundaries:")
    log("   1st variable:   %s --> %s"%(', '.join([str(b) for b in bins1]), ', '.join([str(b) for b in new_bins1])))
    log("   2nd variable:   %s --> %s"%(', '.join([str(b) for b in bins2]), ', '.join([str(b) for b in new_bins2])))

    log("   Categories ready:")
    categories = {}
    for ii in range(len(new_bins1)-1):
        for jj in range(len(new_bins2)-1):
            cat_name = "cat_%i_%i"%(ii, jj)
            cut = "(%s>%f)&(%s<%f)&(%s>%f)&(%s<%f)"%(score1, new_bins1[ii], score1, new_bins1[ii+1], score2, new_bins2[ii], score2, new_bins2[ii+1])
            categories[cat_name] = cut
            log("   %s:  %s"%(cat_name, cut))
    log("   Creating datacards... Please wait...")

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

    bins_str = ""
    for ii in range(len(bins1)-1):
        bins_str = bins_str+"%f_"%bins1[ii]
    bins_str = bins_str+"+"
    bins_str = bins_str+"%f"%bins1[len(bins1)-1]
    for ii in range(len(bins2)-1):
        bins_str = bins_str+"%f_"%bins2[ii]
    bins_str = bins_str+"%f"%bins2[len(bins2)-1]

    memorized[bins_str]=significance

    log("Memorizing result as "+bins_str )
    log(" ")
    log("############################################")
    log(" ")
    return significance


def solve_subproblem(i1,j1,i2,j2):
    global s 
    global best_splitting
    global memorized

    log("="*50)
    log("   Solving subproblem P_%i_%i_%i_%i"%(i1,j1,i2,j2))
    log("   The goal is to find best significance in category containing bins #%i through #%i by 1st variable and #%i through #%i by 2nd variable"%(i1,j1,i2,j2))
    log("="*50)

    s_ij = 0
    best_splitting_ij_1 = []
    best_splitting_ij_2 = []

    log("   First approach to P_%i_%i_%i_%i: merge all bins"%(i1,j1,i2,j2))          
    log("   Merge bins from #%i to #%i by 1st variable and from #%i to #%i by 2nd variable into a single category"%(i1,j1,i2,j2))
    bins1 = [i1,j1+1] # here the numbers count not bins, but boundaries between bins, hence j+1
    bins2 = [i2,j2+1] # here the numbers count not bins, but boundaries between bins, hence j+1
    log("   Splitting by 1st variable is:   "+bins_to_illustration(i1, j1+1, bins1))
    log("   Splitting by 2nd variable is:   "+bins_to_illustration(i2, j2+1, bins2))

    significance = get_significance("%i_%i_%i_%i_%i"%(i1,j1,i2,j2, 0), bins1, bins2)

    log("   Calculated significance for merged bins!")
    s_ij = significance
    best_splitting_ij_1 = bins1
    best_splitting_ij_2 = bins2

    for k1 in range(i1+1, j1+1):
        consider_this_option = True
        can_decrease_nCat = False

        log("   Continue solving P_%i_%i_%i_%i!"%(i1,j1,i2,j2))
        log("   Cut between #%i and #%i by 1st variable"%(k1-1, k1))
        log("   Combine the optimal solutions of P_%i_%i_%i_%i and P_%i_%i_%i_%i"%(i1 , k1-1, i2, j2, k1, j1, i2, j2))

        bins1 = sorted(list(set(best_splitting1[i1][k1-1]) | set(best_splitting1[k1][j1]))) # sorted union of lists will provide the correct category boundaries
        bins2 = best_splitting2[i2][j2]
        
        log("   Splitting by 1st variable is "+bins_to_illustration(i1, j1+1, bins1))
        log("   Splitting by 2nd variable is "+bins_to_illustration(i2, j2+1, bins2))

        bins_str = ""
        for ii in range(len(bins1)-1):
            bins_str = bins_str+"%f_"%bins1[ii]
        bins_str = bins_str+"+"
        bins_str = bins_str+"%f"%bins1[len(bins1)-1]
        for ii in range(len(bins2)-1):
            bins_str = bins_str+"%f_"%bins2[ii]
        bins_str = bins_str+"%f"%bins2[len(bins2)-1]

        if bins_str in memorized.keys():
            log("   We already saw these bins,")
            log("     and the significance for them was %f"%memorized[bins_str])
            significance = memorized[bins_str]
        else:
            significance = sqrt(s[(i1,k1-1,i2,j2)]*s[(i1,k1-1,i2,j2)]+s[(k1,j1,i2,j2)]*s[(k1,j1,i2,j2)]) # this would be exactly true for Poisson significance. For Asimov significance it's still a good approximation.
            memorized[bins_str]=significance
            log("   Significance = sqrt(s[%i,%i,%i,%i]^2+s[%i,%i,%i,%i]^2) = %f"%(i1, k1-1, i2, j2, k1, j1, i2, j2, significance))

        if s_ij:
            gain = ( significance - s_ij ) / s_ij*100.0
        else:
            gain = 999

        log("   Before this option the best s[%i,%i,%i,%i] was %f."%(i1,j1,i2,j2, s_ij))
        log("   This option gives s[%i,%i,%i,%i] = %f."%(i1,j1,i2,j2, significance ))
        log("   We gain %f %% if we use the new option."%gain)
        log("   The required gain if %f %% per additional category."%(args.penalty))
        
        old_ncat = len(best_splitting_ij_1) + len(best_splitting_ij_2) - 2
        new_ncat = len(bins1) + len(bins2) - 2
        ncat_diff = abs( new_ncat - old_ncat)

        if ((new_ncat>old_ncat)&(gain<args.penalty*ncat_diff)):
            log("     This option increases number of subcategories by %i from %i to %i, but the improvement is just %f %%, so skip."%(ncat_diff, old_ncat,new_ncat,gain))
            consider_this_option = False

        elif ((new_ncat<old_ncat)&(gain>-args.penalty*ncat_diff)):
            log("     This option decreases number of subcategories by %i from %i to %i, and the change in significance is just %f %%, so keep it."%(ncat_diff, old_ncat, new_ncat, -gain))
            can_decrease_nCat = True

        elif ((new_ncat==old_ncat)&(gain>0)):
            log("     This option keeps the same number of categories as the bes option so far, and the significance is increased by %f %%, so keep it."%gain)

        if (((gain>0)&(consider_this_option)) or can_decrease_nCat): 
            s_ij = significance
            best_splitting_ij_1 = bins1
            log("   Updating best significance: now s[%i,%i,%i,%i] = %f"%(i1,j1,i2,j2, s_ij))
        else:
            log("   Don't update best significance.")

    for k2 in range(i2+1, j2+1):
        consider_this_option = True
        can_decrease_nCat = False

        log("   Continue solving P_%i_%i_%i_%i!"%(i1,j1,i2,j2))
        log("   Cut between #%i and #%i by 2nd variable"%(k2-1, k2))
        log("   Combine the optimal solutions of P_%i_%i_%i_%i and P_%i_%i_%i_%i"%(i1 , j1, i2, k2-1, i1, j1, k2, j2))

        bins1 = best_splitting1[i1][j1]
        bins2 = sorted(list(set(best_splitting2[i2][k2-1]) | set(best_splitting2[k2][j2]))) # sorted union of lists will provide the correct category boundaries
        
        log("   Splitting by 1st variable is "+bins_to_illustration(i1, j1+1, bins1))
        log("   Splitting by 2nd variable is "+bins_to_illustration(i2, j2+1, bins2))

        bins_str = ""
        for ii in range(len(bins1)-1):
            bins_str = bins_str+"%f_"%bins1[ii]
        bins_str = bins_str+"+"
        bins_str = bins_str+"%f"%bins1[len(bins1)-1]
        for ii in range(len(bins2)-1):
            bins_str = bins_str+"%f_"%bins2[ii]
        bins_str = bins_str+"%f"%bins2[len(bins2)-1]

        if bins_str in memorized.keys():
            log("   We already saw these bins,")
            log("     and the significance for them was %f"%memorized[bins_str])
            significance = memorized[bins_str]
        else:
            significance = sqrt(s[(i1,j1,i2,k2-1)]*s[(i1,j2,i2,k2-1)]+s[(i1,j1,k2,j2)]*s[(i1,j1,k2,j2)]) # this would be exactly true for Poisson significance. For Asimov significance it's still a good approximation.
            memorized[bins_str]=significance
            log("   Significance = sqrt(s[%i,%i,%i,%i]^2+s[%i,%i,%i,%i]^2) = %f"%(i1, j1, i2,k2-1, i1,j1,k2,j2, significance))

        if s_ij:
            gain = ( significance - s_ij ) / s_ij*100.0
        else:
            gain = 999

        log("   Before this option the best s[%i,%i,%i,%i] was %f."%(i1,j1,i2,j2, s_ij))
        log("   This option gives s[%i,%i,%i,%i] = %f."%(i1,j1,i2,j2, significance ))
        log("   We gain %f %% if we use the new option."%gain)
        log("   The required gain if %f %% per additional category."%(args.penalty))
        
        old_ncat = len(best_splitting_ij_1) + len(best_splitting_ij_2) - 2
        new_ncat = len(bins1) + len(bins2) - 2
        ncat_diff = abs( new_ncat - old_ncat)

        if ((new_ncat>old_ncat)&(gain<args.penalty*ncat_diff)):
            log("     This option increases number of subcategories by %i from %i to %i, but the improvement is just %f %%, so skip."%(ncat_diff, old_ncat,new_ncat,gain))
            consider_this_option = False

        elif ((new_ncat<old_ncat)&(gain>-args.penalty*ncat_diff)):
            log("     This option decreases number of subcategories by %i from %i to %i, and the change in significance is just %f %%, so keep it."%(ncat_diff, old_ncat, new_ncat, -gain))
            can_decrease_nCat = True

        elif ((new_ncat==old_ncat)&(gain>0)):
            log("     This option keeps the same number of categories as the bes option so far, and the significance is increased by %f %%, so keep it."%gain)

        if (((gain>0)&(consider_this_option)) or can_decrease_nCat): 
            s_ij = significance
            best_splitting_ij_1 = bins1
            log("   Updating best significance: now s[%i,%i,%i,%i] = %f"%(i1,j1,i2,j2, s_ij))
        else:
            log("   Don't update best significance.")

    log("   Problem P_%i_%i_%i_%i solved! Here's the best solution:"%(i1,j1,i2,j2))
    log("      Highest significance for P_%i_%i_%i_%i is %f and achieved when the splitting is %s by 1st variable and %s by 2nd variable"%(i1,j1,i2,j2, s[(i1,j1,i2,j2)], bins_to_illustration(i1, j1+1, best_splitting1[i1][j1]), bins_to_illustration(i2, j2+1, best_splitting2[i2][j2])))
    s[(i1,j1,i2,j2)] = s_ij
    best_splitting1[i1][j1] = best_splitting_ij_1
    best_splitting2[i2][j2] = best_splitting_ij_2
    return (i1, j1, i2, j2, s_ij, best_splitting_ij_1, best_splitting_ij_2)

def callback(result):
    global s
    global best_splitting1
    global best_splitting2
    i1, j1, i2, j2, s_ij, best_splitting_ij_1, best_splitting_ij_2 = result
    s[(i1,j1,i2,j2)] = s_ij
    best_splitting1[i1][j1] = best_splitting_ij_1
    best_splitting2[i2][j2] = best_splitting_ij_2


parallel = True

# Main loop
for l1 in range(1, args.nSteps1+1): 
    print "Scanning categories by 1st variable made of %i bins"%l1
    for l2 in range(1, args.nSteps2+1): 
        print "Scanning categories by 2nd variable made of %i bins"%l2

        if parallel:
            pass
            # print "Number of CPUs: ", mp.cpu_count()
            # pool = mp.Pool(mp.cpu_count())
            # a = [pool.apply_async(solve_subproblem, args = (i,i+l-1), callback=callback) for i in range(0, args.nSteps-l+1)]
            # for process in a:
            #     process.wait()
            # pool.close()

        else:   # if not parallel
            for i1 in range(0, args.nSteps - l1 + 1): # j = i+l-1
                for i2 in range(0, args.nSteps - l2 + 1): # j = i+l-1
                    j1 = i1+l1-1
                    j2 = i2+l2-1
                    ii1, jj1, ii2, jj2, s_ij, best_splitting_ij_1, best_splitting_ij_2 = solve_subproblem(i1,j1,i2,j2)
                    s[(i1,j1,i2,j2)] = s_ij
                    best_splitting1[i1][j1] = best_splitting_ij_1
                    best_splitting2[i2][j2] = best_splitting_ij_2

        # print "S_ij so far:"
        # for ii in range(args.nSteps):
        #     row = ""
        #     for jj in range(args.nSteps):
        #         row = row + "%f "%s[ii][jj]
        #     print row


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