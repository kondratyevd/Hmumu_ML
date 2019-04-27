import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import ROOT
from make_datacards import create_datacard
import argparse

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
parser.add_argument('--nIter', action='store', dest='nIter', help='nIter', type=int)
parser.add_argument('--lumi', action='store', dest='lumi', help='lumi', type=float)
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



def get_significance(label, bins):

    if "binary" in args.method:
        score = "sig_prediction"
        min_score = 0
        max_score = 1
    elif "DNNmulti" in args.method:
        score = "(ggH_prediction+VBF_prediction+(1-DY_prediction)+(1-ttbar_prediction))"
        min_score = 1
        max_score = 3
    elif "BDT" in args.method:
        score = "MVA"
        min_score = -1
        max_score = 1
    elif "Rapidity" in args.method:
        score = "max_abs_eta_mu"
        min_score = 0
        max_score = 2.4

    print "Will use method", args.method
    print "    min score =", min_score
    print "    max score =", max_score

    step = (args.max_var - args.min_var)/float(args.nSteps)

    print "Rescaling cut boundaries:"
    new_bins = []
    for i in range(len(bins)):
        new_bins.append(min_score + bins[i]*step)

    print bins, " --> ", new_bins

    categories = {}
    for i in range(len(new_bins)-1):
        cat_name = "cat%i"%i
        cut = "(%s>%f)&(%s<%f)"%(score, new_bins[i], score, new_bins[i+1])
        categories[cat_name] = cut

    print "Categories ready:"
    print  categories
    print "Creating datacards..."

    create_datacard(categories, args.sig_input_path, args.data_input_path, args.data_tree, args.output_path,  "datacard_"+label, "workspace_"+label, nuis=args.nuis, res_unc_val=args.res_unc_val, scale_unc_val=args.scale_unc_val, smodel=args.smodel, method=args.method, lumi=args.lumi)

    os.system('combine -M Significance --expectSignal=1 -t -1 -n %s -d datacard_%s.txt'%(label, label))

    significance = 0
    tree = ROOT.TChain("limit")
    tree.Add("higgsCombine%s.Significance.mH120.root"%label)
    for iev,  event in enumerate(tree):
        significance = event.limit
    print "Expected significance =", significance
    return significance


best_splitting = [] # will store the best way to split the category containing bins i through j
best_significance = 0


for i in range(1, args.nIter+1): # number of iteration
    print "Iteration %i of %i"%(i, args.nIter)
    for j in range(1, args.nSteps): # possible values of the boundary between categories
        print "   Try to split in j= %i"%(j)
        if j in set(best_splitting):
            "   j=%i is already present in splitting:"%(j), best_splitting
            continue
        bins = sorted(list(set(best_splitting) | set([j])))
        significance = get_significance("%i_%i"%(i, j), bins)

        if (significance>best_significance*1.02):
            best_significance = significance
            best_splitting = bins
    print "Best significance after %i iterations: %f for splitting"%(i, best_significance), best_splitting


print "Rescaling cut boundaries:"
new_bins = []
for i in range(len(best_splitting)):
    new_bins.append(args.min_var + best_splitting[i]*step)
print "Best cuts on MVA score are:"
print best_splitting, " --> ", new_bins