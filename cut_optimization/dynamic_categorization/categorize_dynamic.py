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
parser.add_argument('--min_mva', action='store', dest='min_mva', help='min_mva', type=float)
parser.add_argument('--max_mva', action='store', dest='max_mva', help='max_mva', type=float)
parser.add_argument('--nSteps', action='store', dest='nSteps', help='nSteps', type=int)
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

step = (args.max_mva - args.min_mva)/float(args.nSteps)



def get_significance(label, bins):

    if "binary" in args.method:
        score = "sig_prediction"
        min_score = 0
        max_score = 1
    elif "multi" in args.method:
        score = "(ggH_prediction+VBF_prediction+(1-DY_prediction)+(1-ttbar_prediction))"
        min_score = 1
        max_score = 3
    elif "BDT" in args.method:
        score = "MVA"
        min_score = -1
        max_score = 1

    print "Will use method", args.method
    print "    min score =", min_score
    print "    max score =", max_score

    step = (args.max_mva - args.min_mva)/float(args.nSteps)

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


s = [] # will store the best significance for the category containing bins i through j
best_splitting = [] # will store the best way to split the category containing bins i through j

for i in range(args.nSteps):
    row = []
    row_bs = []
    for j in range(args.nSteps):
        row.append(0)
        row_bs.append([])
    s.append(row)
    best_splitting.append(row_bs)


for l in range(1, args.nSteps+1): # subsequence length: from 1 to N. l=1 is the initialization of s[i,j] diagonal
    print "Scanning categories made of %i bins"%l
    for i in range(0, args.nSteps - l + 1): # we are considering [bin_i, bin_j]
        j = i + l - 1
        print "   Retrieveing best significance in category [%i, %i]"%(i, j)
        for k in range(i, j+1):
            best_splitting_ij = []
            print "      What if we cut at k = %i?"%k
            if k==i:
                print "         No cut"
                print "         Merging bins %i and %i into a single category"%(i, j)
                bins = [i,j+1] # here the numbers count not bins, but boundaries between bins, hence j+1
                print "         Splitting is", bins
                significance = get_significance("%i_%i_%i_%i"%(l, i, j, k), bins)

            else:
                print "         Cut between %i and %i"%(k-1, k)
                print "         Use the splitting that provided best significance in categories %i-%i and %i-%i"%(i , k-1, k, j)
                bins = sorted(list(set(best_splitting[i][k-1]) | set(best_splitting[k][j]))) # sorted union of lists will provide the correct category boundaries
                print "         Splitting is", bins
                significance = get_significance("%i_%i_%i_%i"%(l, i, j, k), bins)
                significance = significance/1.02 # we only want to split if the splitting gives at least 2% gain in significance

            if (significance > s[i][j]): 
                if k!=i:
                    significance = significance*1.02 # revert
                s[i][j] = significance
                best_splitting[i][j] = bins
                print "Best significance for category %i-%i is %f and achieved when the splitting is "%(i, j, significance), bins


for i in range(args.nSteps):
    row = ""
    for j in range(args.nSteps):
        row = row + "%f "%s[i][j]
    print row

best_bins = best_splitting[0][args.nSteps-1]
print "Best significance overall is %f and achieved when the splitting is "%(s[0][args.nSteps-1]), best_bins
print "Rescaling cut boundaries:"
new_bins = []
for i in range(len(best_bins)):
    new_bins.append(min_score + best_bins[i]*step)
print "Best cuts on MVA score are:"
print best_bins, " --> ", new_bins