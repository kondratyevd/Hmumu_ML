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


class Categorizer(object):
    def __init__(self, var1, min_var1, max_var1, nSteps1, var2, min_var2, max_var2, nSteps2, log_mode, parallel):
        self.categories = []
        self.var1 = var1
        self.min_var1 = min_var1
        self.max_var1 = max_var1
        self.nSteps1 = nSteps1
        self.var2 = var2
        self.min_var2 = min_var2
        self.max_var2 = max_var2
        self.nSteps2 = nSteps2
        self.step1 = (self.max_var1 - self.min_var1)/float(self.nSteps1)
        self.step2 = (self.max_var2 - self.min_var2)/float(self.nSteps2)
        self.log_mode = log_mode
        self.parallel = parallel

    def log(self, s):
        if self.log_mode is 0:
            pass
        elif self.log_mode is 1:
            print s
        # elif log_mode is 2:
        #   write into log file

    def bins_to_illustration(self, min, max, bins):
        result = ""
        for iii in range(min, max):
            if (iii in bins):
                result = result+"| "
            result = result+"%i "%iii
        result = result+"| "
        return result


    class Category(object):
        def __init__(self, framework, i1, j1, i2, j2, inclusive_significance):
            self.framework = framework
            self.i1 = i1
            self.j1 = j1
            self.i2 = i2
            self.j2 = j2
            self.inclusive_significance = inclusive_significance
            self.child1 = None
            self.child2 = None
            self.merged = True
            self.label = "%i_%i_%i_%i"%(i1,j1,i2,j2)

        def set_splitting(self, last_cut_var, last_cut):
            if not last_cut_var:
                print "Leave the category %s merged."%self.label
                self.child1 = None
                self.child2 = None

            elif last_cut_var is self.framework.var1:
                label_1 = "%i_%i_%i_%i"%(self.i1,last_cut-1,self.i2,self.j2)
                label_2 = "%i_%i_%i_%i"%(last_cut,self.j1,self.i2,self.j2)
                print "      Searching for %s and %s..."%(label_1, label_2)
                for c in self.framework.categories:
                    if c.label == label_1:
                        self.framework.log("      First child found: %s"%c.label)
                        self.child1 = c
                    elif c.label == label_2 :
                        self.framework.log("      Second child found: %s"%c.label)                        
                        self.child2 = c

            elif last_cut_var is self.framework.var2:
                label_1 = "%i_%i_%i_%i"%(self.i1,self.j1,self.i2,last_cut-1)
                label_2 = "%i_%i_%i_%i"%(self.i1,self.j1,last_cut,self.j2)
                print "      Searching for %s and %s..."%(label_1, label_2)
                for c in self.framework.categories:
                    if c.label == label_1 :
                        self.framework.log("      First child found: %s"%c.label)
                        self.child1 = c
                    elif c.label == label_2 :
                        self.framework.log("      Second child found: %s"%c.label)
                        self.child2 = c
            else:
                print "Incorrect variable: '%s'"%last_cut_var
                self.child1 = None
                self.child2 = None

            if self.child1 and self.child2:
                self.merged = False
            else:
                self.merged = True

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
                    struct1 = "%s: [%i]"%(self.framework.var1, self.i1)
                else:
                    struct1 = "%s: [%i, %i]"%(self.framework.var1, self.i1, self.j1)
                if self.i2 == self.j2:
                    struct2 = "%s: [%i]"%(self.framework.var2, self.i2)
                else:
                    struct2 = "%s: [%i, %i]"%(self.framework.var2, self.i2, self.j2)
                print "%s, %s"%(struct1, struct2)
            else:
                self.child1.print_structure()
                self.child2.print_structure()


    def get_significance(self, label, bins1, bins2):
        new_bins1 = []
        for i in range(len(bins1)):
            new_bins1.append(self.min_var1 + bins1[i]*self.step1)
        new_bins2 = []
        for i in range(len(bins2)):
            new_bins2.append(self.min_var2 + bins2[i]*self.step2) 

        self.log("   Rescaled cut boundaries:")
        self.log("   1st variable:   %s --> %s"%(', '.join([str(b) for b in bins1]), ', '.join([str(b) for b in new_bins1])))
        self.log("   2nd variable:   %s --> %s"%(', '.join([str(b) for b in bins2]), ', '.join([str(b) for b in new_bins2])))
        self.log("   Categories ready:")
        categories_for_combine = {}
        for ii in range(len(new_bins1)-1):
            for jj in range(len(new_bins2)-1):
                cat_name = "cat_%i_%i"%(ii, jj)
                cut = "(%s>%f)&(%s<%f)&(%s>%f)&(%s<%f)"%(self.var1, new_bins1[ii], self.var1, new_bins1[ii+1], self.var2, new_bins2[ii], self.var2, new_bins2[ii+1])
                categories_for_combine[cat_name] = cut
                self.log("   %s:  %s"%(cat_name, cut))
        self.log("   Creating datacards... Please wait...")

        success = create_datacard(categories_for_combine, args.sig_input_path, args.data_input_path, args.data_tree, args.output_path,  "datacard_"+label, "workspace_"+label, nuis=args.nuis, res_unc_val=args.res_unc_val, scale_unc_val=args.scale_unc_val, smodel=args.smodel, method=args.method, lumi=args.lumi)
        if not success:
            print "Datacards were not created (might be not enough events in the category). Set significance to 0."
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


    def solve_subproblem(self, i1,j1,i2,j2):

        self.log("="*50)
        self.log("   Solving subproblem P_%i_%i_%i_%i"%(i1,j1,i2,j2))
        self.log("   The goal is to find best significance in category containing bins #%i through #%i by 1st variable and #%i through #%i by 2nd variable"%(i1,j1,i2,j2))
        self.log("="*50)

        s_ij = 0
        best_splitting_var = ""
        best_splitting = 0

        self.log("   First approach to P_%i_%i_%i_%i: merge all bins"%(i1,j1,i2,j2))          
        self.log("   Merge bins from #%i to #%i by 1st variable and from #%i to #%i by 2nd variable into a single category"%(i1,j1,i2,j2))
        bins1 = [i1,j1+1] # here the numbers count not bins, but boundaries between bins, hence j+1
        bins2 = [i2,j2+1] # here the numbers count not bins, but boundaries between bins, hence j+1
        self.log("   Splitting by 1st variable is:   "+self.bins_to_illustration(i1, j1+1, bins1))
        self.log("   Splitting by 2nd variable is:   "+self.bins_to_illustration(i2, j2+1, bins2))

        significance = self.get_significance("%i_%i_%i_%i_%i"%(i1,j1,i2,j2, 0), bins1, bins2)

        self.log("   Calculated significance for merged bins!")
        self.log("   Creating a 'Category' object..")
        cat_ij = self.Category(self, i1, j1, i2, j2, significance)
        s_ij = significance
        ncat_best = 1

        for k1 in range(i1+1, j1+1):
            consider_this_option = True
            can_decrease_nCat = False

            self.log("   Continue solving P_%i_%i_%i_%i!"%(i1,j1,i2,j2))
            self.log("   Cut between #%i and #%i by 1st variable"%(k1-1, k1))
            self.log("   Combine the optimal solutions of P_%i_%i_%i_%i and P_%i_%i_%i_%i"%(i1 , k1-1, i2, j2, k1, j1, i2, j2))

            cat_ij.set_splitting(self.var1, k1)
            significance = cat_ij.get_combined_significance()

            if s_ij:
                gain = ( significance - s_ij ) / s_ij*100.0
            else:
                gain = 999

            self.log("   Before this option the best s[%i,%i,%i,%i] was %f."%(i1,j1,i2,j2, s_ij))
            self.log("   This option gives s[%i,%i,%i,%i] = %f."%(i1,j1,i2,j2, significance ))
            self.log("   We gain %f %% if we use the new option."%gain)
            self.log("   The required gain if %f %% per additional category."%(args.penalty))
            
            old_ncat = ncat_best
            new_ncat = cat_ij.get_ncat()
            ncat_diff = abs(new_ncat - ncat_best)

            if ((new_ncat>old_ncat)&(gain<args.penalty*ncat_diff)):
                self.log("     This option increases number of subcategories by %i from %i to %i, but the improvement is just %f %%, so skip."%(ncat_diff, old_ncat,new_ncat,gain))
                consider_this_option = False

            elif ((new_ncat<old_ncat)&(gain>-args.penalty*ncat_diff)):
                self.log("     This option decreases number of subcategories by %i from %i to %i, and the change in significance is just %f %%, so keep it."%(ncat_diff, old_ncat, new_ncat, -gain))
                can_decrease_nCat = True

            elif ((new_ncat==old_ncat)&(gain>0)):
                self.log("     This option keeps the same number of categories as the bes option so far, and the significance is increased by %f %%, so keep it."%gain)

            if (((gain>0)&(consider_this_option)) or can_decrease_nCat): 
                s_ij = significance
                best_splitting_var = self.var1
                best_splitting = k1
                ncat_best = new_ncat
                self.log("   Updating best significance: now s[%i,%i,%i,%i] = %f"%(i1,j1,i2,j2, s_ij))
            else:
                self.log("   Don't update best significance.")

        for k2 in range(i2+1, j2+1):
            consider_this_option = True
            can_decrease_nCat = False

            self.log("   Continue solving P_%i_%i_%i_%i!"%(i1,j1,i2,j2))
            self.log("   Cut between #%i and #%i by 2nd variable"%(k2-1, k2))
            self.log("   Combine the optimal solutions of P_%i_%i_%i_%i and P_%i_%i_%i_%i"%(i1 , j1, i2, k2-1, i1, j1, k2, j2))

            cat_ij.set_splitting(self.var2, k2)
            significance = cat_ij.get_combined_significance()

            if s_ij:
                gain = ( significance - s_ij ) / s_ij*100.0
            else:
                gain = 999

            self.log("   Before this option the best s[%i,%i,%i,%i] was %f."%(i1,j1,i2,j2, s_ij))
            self.log("   This option gives s[%i,%i,%i,%i] = %f."%(i1,j1,i2,j2, significance ))
            self.log("   We gain %f %% if we use the new option."%gain)
            self.log("   The required gain if %f %% per additional category."%(args.penalty))
            
            old_ncat = ncat_best
            new_ncat = cat_ij.get_ncat()
            ncat_diff = abs(new_ncat - ncat_best)

            if ((new_ncat>old_ncat)&(gain<args.penalty*ncat_diff)):
                self.log("     This option increases number of subcategories by %i from %i to %i, but the improvement is just %f %%, so skip."%(ncat_diff, old_ncat,new_ncat,gain))
                consider_this_option = False

            elif ((new_ncat<old_ncat)&(gain>-args.penalty*ncat_diff)):
                self.log("     This option decreases number of subcategories by %i from %i to %i, and the change in significance is just %f %%, so keep it."%(ncat_diff, old_ncat, new_ncat, -gain))
                can_decrease_nCat = True

            elif ((new_ncat==old_ncat)&(gain>0)):
                self.log("     This option keeps the same number of categories as the bes option so far, and the significance is increased by %f %%, so keep it."%gain)

            if (((gain>0)&(consider_this_option)) or can_decrease_nCat): 
                s_ij = significance
                best_splitting_var = self.var2
                best_splitting = k2
                self.log("   Updating best significance: now s[%i,%i,%i,%i] = %f"%(i1,j1,i2,j2, s_ij))
            else:
                self.log("   Don't update best significance.")
        
        cat_ij.set_splitting(best_splitting_var, best_splitting)
        return cat_ij

    def callback(self, result):
        print "Retrieving result:", category.label
        category = result
        self.categories.append(category)

    def main_loop(self):
        for l1 in range(1, self.nSteps1+1): 
            print "Scanning categories by 1st variable (%s) made of %i bins"%(self.var1, l1)
            for l2 in range(1, self.nSteps2+1): 
                print "Scanning categories by 2nd variable (%s) made of %i bins"%(self.var2, l2)
                if self.parallel:
                    for i1 in range(0, self.nSteps1 - l1 + 1): 
                        j1=i1+l1-1
                        pool = mp.Pool(mp.cpu_count())
                        a = [pool.apply_async(self.solve_subproblem, args = (i1,j1,i2,i2+l2-1), callback=self.callback) for i2 in range(0, self.nSteps2-l2+1)]
                        for process in a:
                            process.wait()
                        pool.close()
                        # pool.join()

                        for i2 in range(0, self.nSteps2 - l2 + 1): # j = i+l-1
                            j1 = i1+l1-1
                            j2 = i2+l2-1
                            label = "%i_%i_%i_%i"%(i1,j1,i2,j2)
                            for c in self.categories:
                                if label==c.label:
                                    significance = c.get_combined_significance()
                                    print "Subproblem %s solved; the best significance is %f for the following subcategories:"%(label, significance)
                                    c.print_structure()

                else:   # if not parallel
                    for i1 in range(0, self.nSteps1 - l1 + 1): # j = i+l-1
                        for i2 in range(0, self.nSteps2 - l2 + 1): # j = i+l-1
                            j1 = i1+l1-1
                            j2 = i2+l2-1
                            category = self.solve_subproblem(i1,j1,i2,j2)
                            self.categories.append(category)
                            print "Subproblem %s solved; the best significance is %f for the following subcategories:"%(category.label, category.get_combined_significance())
                            category.print_structure()
                        print "Categories so far:"
                        for c in self.categories:
                            print c.label, c.get_combined_significance()


        final_category = None
        final_label = "0_%i_0_%i"%(self.nSteps1-1, self.nSteps2-1)
        for c in self.categories:
            if final_label==c.label:
                final_category = c
        print "Best significance overall is %f and achieved when the splitting is: "%(final_category.get_combined_significance())
        final_category.print_structure()



if "binary" in args.method:
    score1 = "sig_prediction"
elif "DNNmulti" in args.method:
    score1 = "(ggH_prediction+VBF_prediction+(1-DY_prediction)+(1-ttbar_prediction))"
elif "BDTmva" in args.method:
    score1 = "MVA"
elif "Rapidity" in args.method:
    score1 = "max_abs_eta_mu"

score2 = "max_abs_eta_mu"

framework = Categorizer(score1, args.min_var1, args.max_var1, args.nSteps1, score2, args.min_var1, args.max_var2, args.nSteps2, log_mode=1, parallel=False)
# framework.main_loop()


def solve_subproblem(i1,j1,i2,j2):
    global framework
    print "="*50
    print "   Solving subproblem P_%i_%i_%i_%i"%(i1,j1,i2,j2)
    print "   The goal is to find best significance in category containing bins #%i through #%i by 1st variable and #%i through #%i by 2nd variable"%(i1,j1,i2,j2)
    print "="*50

    s_ij = 0
    best_splitting_var = ""
    best_splitting = 0

    print "   First approach to P_%i_%i_%i_%i: merge all bins"%(i1,j1,i2,j2)
    print "   Merge bins from #%i to #%i by 1st variable and from #%i to #%i by 2nd variable into a single category"%(i1,j1,i2,j2)
    bins1 = [i1,j1+1] # here the numbers count not bins, but boundaries between bins, hence j+1
    bins2 = [i2,j2+1] # here the numbers count not bins, but boundaries between bins, hence j+1
    print "   Splitting by 1st variable is:   "+framework.bins_to_illustration(i1, j1+1, bins1)
    print "   Splitting by 2nd variable is:   "+framework.bins_to_illustration(i2, j2+1, bins2)

    significance = framework.get_significance("%i_%i_%i_%i_%i"%(i1,j1,i2,j2, 0), bins1, bins2)

    print "   Calculated significance for merged bins!"
    print "   Creating a 'Category' object.."
    cat_ij = framework.Category(framework, i1, j1, i2, j2, significance)
    s_ij = significance
    ncat_best = 1

    for k1 in range(i1+1, j1+1):
        consider_this_option = True
        can_decrease_nCat = False

        print "   Continue solving P_%i_%i_%i_%i!"%(i1,j1,i2,j2)
        print "   Cut between #%i and #%i by 1st variable"%(k1-1, k1)
        print "   Combine the optimal solutions of P_%i_%i_%i_%i and P_%i_%i_%i_%i"%(i1 , k1-1, i2, j2, k1, j1, i2, j2)

        cat_ij.set_splitting(framework.var1, k1)
        significance = cat_ij.get_combined_significance()

        if s_ij:
            gain = ( significance - s_ij ) / s_ij*100.0
        else:
            gain = 999

        print "   Before this option the best s[%i,%i,%i,%i] was %f."%(i1,j1,i2,j2, s_ij)
        print "   This option gives s[%i,%i,%i,%i] = %f."%(i1,j1,i2,j2, significance )
        print "   We gain %f %% if we use the new option."%gain
        print "   The required gain if %f %% per additional category."%(args.penalty)
        
        old_ncat = ncat_best
        new_ncat = cat_ij.get_ncat()
        ncat_diff = abs(new_ncat - ncat_best)

        if ((new_ncat>old_ncat)&(gain<args.penalty*ncat_diff)):
            print "     This option increases number of subcategories by %i from %i to %i, but the improvement is just %f %%, so skip."%(ncat_diff, old_ncat,new_ncat,gain)
            consider_this_option = False

        elif ((new_ncat<old_ncat)&(gain>-args.penalty*ncat_diff)):
            print "     This option decreases number of subcategories by %i from %i to %i, and the change in significance is just %f %%, so keep it."%(ncat_diff, old_ncat, new_ncat, -gain)
            can_decrease_nCat = True

        elif ((new_ncat==old_ncat)&(gain>0)):
            print "     This option keeps the same number of categories as the bes option so far, and the significance is increased by %f %%, so keep it."%gain

        if (((gain>0)&(consider_this_option)) or can_decrease_nCat): 
            s_ij = significance
            best_splitting_var = framework.var1
            best_splitting = k1
            ncat_best = new_ncat
            print "   Updating best significance: now s[%i,%i,%i,%i] = %f"%(i1,j1,i2,j2, s_ij)
        else:
            print "   Don't update best significance."

    for k2 in range(i2+1, j2+1):
        consider_this_option = True
        can_decrease_nCat = False

        print "   Continue solving P_%i_%i_%i_%i!"%(i1,j1,i2,j2)
        print "   Cut between #%i and #%i by 2nd variable"%(k2-1, k2)
        print "   Combine the optimal solutions of P_%i_%i_%i_%i and P_%i_%i_%i_%i"%(i1 , j1, i2, k2-1, i1, j1, k2, j2)

        cat_ij.set_splitting(framework.var2, k2)
        significance = cat_ij.get_combined_significance()

        if s_ij:
            gain = ( significance - s_ij ) / s_ij*100.0
        else:
            gain = 999

        print "   Before this option the best s[%i,%i,%i,%i] was %f."%(i1,j1,i2,j2, s_ij)
        print "   This option gives s[%i,%i,%i,%i] = %f."%(i1,j1,i2,j2, significance )
        print "   We gain %f %% if we use the new option."%gain
        print "   The required gain if %f %% per additional category."%(args.penalty)
        
        old_ncat = ncat_best
        new_ncat = cat_ij.get_ncat()
        ncat_diff = abs(new_ncat - ncat_best)

        if ((new_ncat>old_ncat)&(gain<args.penalty*ncat_diff)):
            print "     This option increases number of subcategories by %i from %i to %i, but the improvement is just %f %%, so skip."%(ncat_diff, old_ncat,new_ncat,gain)
            consider_this_option = False

        elif ((new_ncat<old_ncat)&(gain>-args.penalty*ncat_diff)):
            print "     This option decreases number of subcategories by %i from %i to %i, and the change in significance is just %f %%, so keep it."%(ncat_diff, old_ncat, new_ncat, -gain)
            can_decrease_nCat = True

        elif ((new_ncat==old_ncat)&(gain>0)):
            print "     This option keeps the same number of categories as the bes option so far, and the significance is increased by %f %%, so keep it."%gain

        if (((gain>0)&(consider_this_option)) or can_decrease_nCat): 
            s_ij = significance
            best_splitting_var = framework.var2
            best_splitting = k2
            print "   Updating best significance: now s[%i,%i,%i,%i] = %f"%(i1,j1,i2,j2, s_ij)
        else:
            print "   Don't update best significance."
    
    cat_ij.set_splitting(best_splitting_var, best_splitting)
    return cat_ij

def callback(result):
    print "Retrieving result:", category.label
    category = result
    framework.categories.append(category)

# Implementing parallel processing outside of the structure

for l1 in range(1, framework.nSteps1+1): 
    print "Scanning categories by 1st variable (%s) made of %i bins"%(framework.var1, l1)
    for l2 in range(1, framework.nSteps2+1): 
        print "Scanning categories by 2nd variable (%s) made of %i bins"%(framework.var2, l2)
        if framework.parallel:
            for i1 in range(0, framework.nSteps1 - l1 + 1): 
                j1=i1+l1-1
                i2max = framework.nSteps2-l2+1
                pool = mp.Pool(mp.cpu_count())
                a = [pool.apply_async(solve_subproblem, args = (i1,j1,i2,i2+l2-1), callback=callback) for i2 in range(0, i2max)]
                for process in a:
                    process.wait()
                pool.close()
                # pool.join()

                for i2 in range(0, framework.nSteps2 - l2 + 1): # j = i+l-1
                    j1 = i1+l1-1
                    j2 = i2+l2-1
                    label = "%i_%i_%i_%i"%(i1,j1,i2,j2)
                    for c in framework.categories:
                        if label==c.label:
                            significance = c.get_combined_significance()
                            print "Subproblem %s solved; the best significance is %f for the following subcategories:"%(label, significance)
                            c.print_structure()



final_category = None
final_label = "0_%i_0_%i"%(framework.nSteps1-1, framework.nSteps2-1)
for c in framework.categories:
    if final_label==c.label:
        final_category = c
print "Best significance overall is %f and achieved when the splitting is: "%(final_category.get_combined_significance())
final_category.print_structure()



