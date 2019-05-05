import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import ROOT
from math import sqrt
from make_datacards import create_datacard
import argparse
import multiprocessing as mp
from numpy import zeros, empty

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

log_mode = 1
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

categories = {}

class Category(object):
    """docstring for Category"""
    def __init__(self, var1, i1, j1, var2, i2, j2, inclusive_significance):
        self.var1 = var1
        self.i1 = i1
        self.j1 = j1
        self.var2 = var2
        self.i2 = i2
        self.j2 = j2
        self.inclusive_significance = inclusive_significance
        self.child1 = None
        self.child2 = None
        self.merged = True
        self.label = "%i_%i_%i_%i"%(i1,j1,i2,j2)

    def set_splitting(self, last_cut_var, last_cut):
        global categories

        if last_cut_var is self.var1:
            if (categories["%i_%i_%i_%i"%(self.i1,last_cut-1,self.i2,self.j2)] and categories["%i_%i_%i_%i"%(last_cut,self.j1,self.i2,self.j2)] ):
                self.child1 = categories["%i_%i_%i_%i"%(self.i1,last_cut-1,self.i2,self.j2)]
                self.child2 = categories["%i_%i_%i_%i"%(last_cut,self.j1,self.i2,self.j2)] 
                self.merged = False 
            else: 
                print "Some subproblems of smaller size are not solved yet!"      
                print "Info about category (%i_%i_%i_%i) is missing"%(self.i1,last_cut-1,self.i2,self.j2)
                print "Info about category (%i_%i_%i_%i) is missing"%(last_cut,self.j1,self.i2,self.j2)
                print "Children are not updated"

        elif last_cut_var is self.var2:
            if (categories["%i_%i_%i_%i"%(self.i1,self.j1,self.i2,last_cut-1)] and categories["%i_%i_%i_%i"%(self.i1,self.j1,last_cut,self.j2)] ):
                self.child1 = categories["%i_%i_%i_%i"%(self.i1,self.j1,self.i2,last_cut-1)]
                self.child2 = categories["%i_%i_%i_%i"%(self.i1,self.j1,last_cut,self.j2)]  
                self.merged = False
            else: 
                print "Some subproblems of smaller size are not solved yet!"      
                print "Info about category (%i_%i_%i_%i) is missing"%(self.i1,self.j1,self.i2,last_cut-1)
                print "Info about category (%i_%i_%i_%i) is missing"%(self.i1,self.j1,last_cut,self.j2)
                print "Children are not updated"
        elif not last_cut_var:
            print "Leave the category %s merged."%self.label
            self.merged = True
            self.child1 = None
            self.child2 = None
        else:
            print "Incorrect variable: '%s'"%last_cut_var

    def get_combined_significance(self):
        if self.merged:
            return self.inclusive_significance
        else:
            s1 = self.child1.get_combined_significance()*self.child1.get_combined_significance()
            s2 = self.child2.get_combined_significance()*self.child2.get_combined_significance()
            return sqrt(s1+s2)
            
    def get_ncat(self):
        if self.merged:
            return 1
        else:
            return self.child1.get_ncat()+self.child2.get_ncat()

    def print_structure(self):
        if self.merged:
            if self.i1 == self.j1:
                struct1 = "%s: [%i]"%(self.var1, self.i1)
            else:
                struct1 = "%s: [%i, %i]"%(self.var1, self.i1, self.j1)
            if self.i2 == self.j2:
                struct2 = "%s: [%i]"%(self.var2, self.i2)
            else:
                struct2 = "%s: [%i, %i]"%(self.var2, self.i2, self.j2)
            print "%s, %s"%(struct1, struct2)
        else:
            self.child1.print_structure()
            self.child2.print_structure()

step1 = (args.max_var1 - args.min_var1)/float(args.nSteps1)
step2 = (args.max_var2 - args.min_var2)/float(args.nSteps2)

s = zeros([args.nSteps1, args.nSteps1, args.nSteps2, args.nSteps2])             # will store the best significance for the category containing bins i through j
best_splitting1 = [] 
best_splitting2 = []  
memorized = {}

for i in range(args.nSteps1):
    row_bs = []
    for j in range(args.nSteps1):
        row_bs.append([])
    best_splitting1.append(row_bs)

for i in range(args.nSteps2):
    row_bs = []
    for j in range(args.nSteps2):
        row_bs.append([])
    best_splitting2.append(row_bs)


def get_significance(label, bins1, bins2):

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
    categories_for_combine = {}
    for ii in range(len(new_bins1)-1):
        for jj in range(len(new_bins2)-1):
            cat_name = "cat_%i_%i"%(ii, jj)
            cut = "(%s>%f)&(%s<%f)&(%s>%f)&(%s<%f)"%(score1, new_bins1[ii], score1, new_bins1[ii+1], score2, new_bins2[ii], score2, new_bins2[ii+1])
            categories_for_combine[cat_name] = cut
            log("   %s:  %s"%(cat_name, cut))
    log("   Creating datacards... Please wait...")

    success = create_datacard(categories_for_combine, args.sig_input_path, args.data_input_path, args.data_tree, args.output_path,  "datacard_"+label, "workspace_"+label, nuis=args.nuis, res_unc_val=args.res_unc_val, scale_unc_val=args.scale_unc_val, smodel=args.smodel, method=args.method, lumi=args.lumi)

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

    return significance


def solve_subproblem(i1,j1,i2,j2):
    global s 
    global best_splitting
    global memorized
    global categories
    global score1, score2

    log("="*50)
    log("   Solving subproblem P_%i_%i_%i_%i"%(i1,j1,i2,j2))
    log("   The goal is to find best significance in category containing bins #%i through #%i by 1st variable and #%i through #%i by 2nd variable"%(i1,j1,i2,j2))
    log("="*50)

    s_ij = 0
    best_splitting_var = ""
    best_splitting = 0

    log("   First approach to P_%i_%i_%i_%i: merge all bins"%(i1,j1,i2,j2))          
    log("   Merge bins from #%i to #%i by 1st variable and from #%i to #%i by 2nd variable into a single category"%(i1,j1,i2,j2))
    bins1 = [i1,j1+1] # here the numbers count not bins, but boundaries between bins, hence j+1
    bins2 = [i2,j2+1] # here the numbers count not bins, but boundaries between bins, hence j+1
    log("   Splitting by 1st variable is:   "+bins_to_illustration(i1, j1+1, bins1))
    log("   Splitting by 2nd variable is:   "+bins_to_illustration(i2, j2+1, bins2))

    significance = get_significance("%i_%i_%i_%i_%i"%(i1,j1,i2,j2, 0), bins1, bins2)

    log("   Calculated significance for merged bins!")
    log("   Creating a 'Category' object..")
    cat_ij = Category(score1, i1, j1, score2, i2, j2, significance)
    categories["%i_%i_%i_%i"%(i1,j1,i2,j2)] = cat_ij
    s_ij = significance
    ncat_best = 1

    for k1 in range(i1+1, j1+1):
        consider_this_option = True
        can_decrease_nCat = False

        log("   Continue solving P_%i_%i_%i_%i!"%(i1,j1,i2,j2))
        log("   Cut between #%i and #%i by 1st variable"%(k1-1, k1))
        log("   Combine the optimal solutions of P_%i_%i_%i_%i and P_%i_%i_%i_%i"%(i1 , k1-1, i2, j2, k1, j1, i2, j2))

        cat_ij.set_splitting(score1, k1)
        significance = cat_ij.get_combined_significance()

        if s_ij:
            gain = ( significance - s_ij ) / s_ij*100.0
        else:
            gain = 999

        log("   Before this option the best s[%i,%i,%i,%i] was %f."%(i1,j1,i2,j2, s_ij))
        log("   This option gives s[%i,%i,%i,%i] = %f."%(i1,j1,i2,j2, significance ))
        log("   We gain %f %% if we use the new option."%gain)
        log("   The required gain if %f %% per additional category."%(args.penalty))
        
        old_ncat = ncat_best
        new_ncat = cat_ij.get_ncat()
        ncat_diff = abs(new_ncat - ncat_best)

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
            best_splitting_var = score1 
            best_splitting = k1
            ncat_best = new_ncat
            log("   Updating best significance: now s[%i,%i,%i,%i] = %f"%(i1,j1,i2,j2, s_ij))
        else:
            log("   Don't update best significance.")

    for k2 in range(i2+1, j2+1):
        consider_this_option = True
        can_decrease_nCat = False

        log("   Continue solving P_%i_%i_%i_%i!"%(i1,j1,i2,j2))
        log("   Cut between #%i and #%i by 2nd variable"%(k2-1, k2))
        log("   Combine the optimal solutions of P_%i_%i_%i_%i and P_%i_%i_%i_%i"%(i1 , j1, i2, k2-1, i1, j1, k2, j2))

        cat_ij.set_splitting(score2, k2)
        significance = cat_ij.get_combined_significance()

        if s_ij:
            gain = ( significance - s_ij ) / s_ij*100.0
        else:
            gain = 999

        log("   Before this option the best s[%i,%i,%i,%i] was %f."%(i1,j1,i2,j2, s_ij))
        log("   This option gives s[%i,%i,%i,%i] = %f."%(i1,j1,i2,j2, significance ))
        log("   We gain %f %% if we use the new option."%gain)
        log("   The required gain if %f %% per additional category."%(args.penalty))
        
        old_ncat = ncat_best
        new_ncat = cat_ij.get_ncat()
        ncat_diff = abs(new_ncat - ncat_best)

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
            best_splitting_var = score2
            best_splitting = k2
            log("   Updating best significance: now s[%i,%i,%i,%i] = %f"%(i1,j1,i2,j2, s_ij))
        else:
            log("   Don't update best significance.")
    
    cat_ij.set_splitting(best_splitting_var, best_splitting)
    return cat_ij.label, best_splitting_var, best_splitting

def callback(result):
    label, score, cut = result
    categories[label].set_splitting(score, cut)


parallel = True

# Main loop
for l1 in range(1, args.nSteps1+1): 
    print "Scanning categories by 1st variable made of %i bins"%l1
    for l2 in range(1, args.nSteps2+1): 
        print "Scanning categories by 2nd variable made of %i bins"%l2

        if parallel:
            for i1 in range(0, args.nSteps1 - l1 + 1): 
                j1=i1+l1-1
                print "Number of CPUs: ", mp.cpu_count()
                pool = mp.Pool(mp.cpu_count())
                a = [pool.apply_async(solve_subproblem, args = (i1,j1,i2,i2+l2-1), callback=callback) for i2 in range(0, args.nSteps2-l2+1)]
                for process in a:
                    process.wait()
                    print process.is_alive()
                pool.close()
                pool.join()
                pool.terminate()
                for i2 in range(0, args.nSteps2 - l2 + 1): # j = i+l-1
                    j1 = i1+l1-1
                    j2 = i2+l2-1
                    label = "%i_%i_%i_%i"%(i1,j1,i2,j2)
                    print "Subproblem %s solved; the best significance is %f for the following subcategories:"%(label, categories[label].get_combined_significance())
                    categories[label].print_structure()

        else:   # if not parallel
            for i1 in range(0, args.nSteps1 - l1 + 1): # j = i+l-1
                for i2 in range(0, args.nSteps2 - l2 + 1): # j = i+l-1
                    j1 = i1+l1-1
                    j2 = i2+l2-1
                    label, score, cut = solve_subproblem(i1,j1,i2,j2)
                    categories[label].set_splitting(score, cut)
                    print "Subproblem %s solved; the best significance is %f for the following subcategories:"%(label, categories[label].get_combined_significance())
                    categories[label].print_structure()



final_label = "0_%i_0_%i"%(args.nSteps1-1, args.nSteps2-1)
print "Best significance overall is %f and achieved when the splitting is: "%(categories[final_label].get_combined_significance())
categories[final_label].print_structure()