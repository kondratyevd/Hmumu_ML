import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
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
args = parser.parse_args()

lumi = 41394.221

existing_categories_011 = {}
existing_categories_111 = {}

exisiting_mva = {
	'0.1.1': existing_categories_011,
	'1.1.1': existing_categories_111,
}



if "binary" in args.method:
	score = "sig_prediction"
elif "multi" in args.method:
	score = "(ggH_prediction+VBF_prediction+(1-DY_prediction)+(1-ttbar_prediction))"
elif "BDT" in args.method:
	score = "MVA"


eta_categories = {
	"eta0":	"(max_abs_eta_mu>0)&(max_abs_eta_mu<0.9)", 
	"eta1": "(max_abs_eta_mu>0.9)&(max_abs_eta_mu<1.9)", 
	"eta2": "(max_abs_eta_mu>1.9)&(max_abs_eta_mu<2.4)"
}

step = (args.max_mva - args.min_mva)/float(args.nSteps)


if args.option is "0": # inclusive
	create_datacard({"cat0": "1"}, args.sig_input_path, args.data_input_path, args.data_tree, args.output_path,  "datacard_inclusive", "workspace_inclusive", nuis=args.nuis, res_unc_val=args.res_unc_val, scale_unc_val=args.scale_unc_val, smodel=args.smodel, method=args.method, lumi=lumi)
	create_datacard(eta_categories, args.sig_input_path, args.data_input_path, args.data_tree, args.output_path,  "datacard_inclusive_eta", "workspace_inclusive_eta", nuis=args.nuis, res_unc_val=args.res_unc_val, scale_unc_val=args.scale_unc_val, smodel=args.smodel, method=args.method, lumi=lumi)

for i in range(args.nSteps): # scan from min to max
	sliding_cut = args.min_mva+i*step
	print "--- Move sliding cut to %f -- "%(sliding_cut)

	new_mva_categories = {}
	new_mva_categories["mva0"] = '((%s>%f)&(%s<%f))'%(score, args.min_mva, score, sliding_cut) # [min, cut]
	new_mva_categories["mva1"] = '((%s>%f)&(%s<%f))'%(score, sliding_cut, score, args.max_mva) # [cut, max]
	new_mva_categories.update(exisiting_mva[args.option])
	print "Will use the following MVA categories:"
	print new_mva_categories
	create_datacard(new_mva_categories, args.sig_input_path, args.data_input_path, args.data_tree, args.output_path,  "datacard_dnn_option%s_mva_%i"%(args.option, i), "workspace_dnn_option%s_mva_%i"%(args.option, i), nuis=args.nuis, res_unc_val=args.res_unc_val, scale_unc_val=args.scale_unc_val, smodel=args.smodel, method=args.method, lumi=lumi)

	print "Adding splitting by eta:"
	full_categories = {}
	for key_mva, value_mva in new_mva_categories.iteritems():
		for key_eta, value_eta in eta_categories.iteritems():
			new_key = "%s%s"%(key_mva, key_eta)
			new_value = "(%s)&(%s)"%(value_mva, value_eta)
			full_categories[new_key] = new_value
	print "Will use the following MVA and ETA categories:"
	print full_categories
	create_datacard(full_categories, args.sig_input_path, args.data_input_path, args.data_tree, args.output_path,  "datacard_dnn_option%s_full_%i"%(args.option, i), "workspace_dnn_option%s_full_%i"%(args.option, i), nuis=args.nuis, res_unc_val=args.res_unc_val, scale_unc_val=args.scale_unc_val, smodel=args.smodel, method=args.method, lumi=lumi)

# # categories_inclusive = {"cat0": "1"}
# # create_datacard(categories_inclusive, args.sig_input_path, args.data_input_path, args.data_tree, args.output_path,  "datacard_dnn_option%s_inclusive"%args.option, "workspace_dnn_option%s_inclusive"%args.option, nuis=args.nuis, res_unc_val=args.res_unc_val, scale_unc_val=args.scale_unc_val, smodel=args.smodel, method=args.method, lumi=lumi)

# categories_full = {}
# for i in range(len(eta_cut_full)):
# 	categories_full["cat%i"%i] = "(%s)&(%s)"%(eta_cut_full[i], mva_cut_full[i])
# create_datacard(categories_full, args.sig_input_path, args.data_input_path, args.data_tree, args.output_path,  "datacard_dnn_option%s_full"%args.option, "workspace_dnn_option%s_full"%args.option, nuis=args.nuis, res_unc_val=args.res_unc_val, scale_unc_val=args.scale_unc_val, smodel=args.smodel, method=args.method, lumi=lumi)

# # categories_mva = {}
# # for i in range(len(mva_cut)):
# # 	categories_mva["cat%i"%i] = "(%s)"%(mva_cut[i]) # only mva categorization
# # create_datacard(categories_mva, args.sig_input_path, args.data_input_path, args.data_tree, args.output_path,  "datacard_dnn_option%s_mva"%args.option, "workspace_dnn_option%s_mva"%args.option, nuis=args.nuis, res_unc_val=args.res_unc_val, scale_unc_val=args.scale_unc_val, smodel=args.smodel, method=args.method, lumi=lumi)

# # categories_eta = {}
# # for i in range(len(eta_cut)):
# # 	categories_eta["cat%i"%i] = "(%s)"%(eta_cut[i]) # only eta categorization
# # create_datacard(categories_eta, args.sig_input_path, args.data_input_path, args.data_tree, args.output_path,  "datacard_dnn_option%s_eta"%args.option, "workspace_dnn_option%s_eta"%args.option, nuis=args.nuis, res_unc_val=args.res_unc_val, scale_unc_val=args.scale_unc_val, smodel=args.smodel, method=args.method, lumi=lumi)



