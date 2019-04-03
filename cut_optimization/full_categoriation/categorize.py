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
args = parser.parse_args()


eta_cut = [
	"(max_abs_eta_mu>1.9)&(max_abs_eta_mu<2.4)", #cat1
	"(max_abs_eta_mu>0.9)&(max_abs_eta_mu<1.9)", #cat2
	"(max_abs_eta_mu>0)&(max_abs_eta_mu<0.9)", #cat3
]

eta_cut_full = [
	"1", #cat0
	"(max_abs_eta_mu>1.9)&(max_abs_eta_mu<2.4)", #cat1
	"(max_abs_eta_mu>0.9)&(max_abs_eta_mu<1.9)", #cat2
	"(max_abs_eta_mu>0)&(max_abs_eta_mu<0.9)", #cat3
	"(max_abs_eta_mu>1.9)&(max_abs_eta_mu<2.4)", #cat4
	"(max_abs_eta_mu>0.9)&(max_abs_eta_mu<1.9)", #cat5
	"(max_abs_eta_mu>0)&(max_abs_eta_mu<0.9)", #cat6
	"(max_abs_eta_mu>1.9)&(max_abs_eta_mu<2.4)", #cat7
	"(max_abs_eta_mu>0.9)&(max_abs_eta_mu<1.9)", #cat8
	"(max_abs_eta_mu>0)&(max_abs_eta_mu<0.9)", #cat9
	"(max_abs_eta_mu>1.9)&(max_abs_eta_mu<2.4)", #cat10
	"(max_abs_eta_mu>0.9)&(max_abs_eta_mu<1.9)", #cat11
	"(max_abs_eta_mu>0)&(max_abs_eta_mu<0.9)", #cat12
	"1", #cat13
	"1", #cat14
]

score = "(ggH_prediction+VBF_prediction)"
mva_cuts = [0.61, 0.756, 0.814, 0.842, 0.916, 0.956] # DNN 3layers V1, score = ggH+VBF

mva_cut = [
	"(%s<%f)"%(score, mva_cuts[0]), #cat0
	"(%s>%f)&(%s<%f)"%(score, mva_cuts[0], score, mva_cuts[1]), #cat1
	"(%s>%f)&(%s<%f)"%(score, mva_cuts[1], score, mva_cuts[2]), #cat4
	"(%s>%f)&(%s<%f)"%(score, mva_cuts[2], score, mva_cuts[3]), #cat7
	"(%s>%f)&(%s<%f)"%(score, mva_cuts[3], score, mva_cuts[4]), #cat10
	"(%s>%f)&(%s<%f)"%(score, mva_cuts[4], score, mva_cuts[5]), #cat13
	"(%s>%f)"%(score, mva_cuts[5]), #cat14
]

mva_cut_full = [
	"(%s<%f)"%(score, mva_cuts[0]), #cat0
	"(%s>%f)&(%s<%f)"%(score, mva_cuts[0], score, mva_cuts[1]), #cat1
	"(%s>%f)&(%s<%f)"%(score, mva_cuts[0], score, mva_cuts[1]), #cat2
	"(%s>%f)&(%s<%f)"%(score, mva_cuts[0], score, mva_cuts[1]), #cat3
	"(%s>%f)&(%s<%f)"%(score, mva_cuts[1], score, mva_cuts[2]), #cat4
	"(%s>%f)&(%s<%f)"%(score, mva_cuts[1], score, mva_cuts[2]), #cat5
	"(%s>%f)&(%s<%f)"%(score, mva_cuts[1], score, mva_cuts[2]), #cat6
	"(%s>%f)&(%s<%f)"%(score, mva_cuts[2], score, mva_cuts[3]), #cat7
	"(%s>%f)&(%s<%f)"%(score, mva_cuts[2], score, mva_cuts[3]), #cat8
	"(%s>%f)&(%s<%f)"%(score, mva_cuts[2], score, mva_cuts[3]), #cat9
	"(%s>%f)&(%s<%f)"%(score, mva_cuts[3], score, mva_cuts[4]), #cat10
	"(%s>%f)&(%s<%f)"%(score, mva_cuts[3], score, mva_cuts[4]), #cat11
	"(%s>%f)&(%s<%f)"%(score, mva_cuts[3], score, mva_cuts[4]), #cat12
	"(%s>%f)&(%s<%f)"%(score, mva_cuts[4], score, mva_cuts[5]), #cat13
	"(%s>%f)"%(score, mva_cuts[5]), #cat14
]


# categories_inclusive = {"cat0": "1"}
# create_datacard(categories_inclusive, args.sig_input_path, args.data_input_path, args.data_tree, args.output_path,  "datacard_dnn_v1_inclusive", "workspace_dnn_v1_inclusive", nuis=args.nuis, res_unc_val=args.res_unc_val, scale_unc_val=args.scale_unc_val, smodel=args.smodel)

# categories_full = {}
# for i in range(len(eta_cut_full)):
# 	categories_full["cat%i"%i] = "(%s)&(%s)"%(eta_cut_full[i], mva_cut_full[i])
# create_datacard(categories_full, args.sig_input_path, args.data_input_path, args.data_tree, args.output_path,  "datacard_dnn_v1_full", "workspace_dnn_v1_full", nuis=args.nuis, res_unc_val=args.res_unc_val, scale_unc_val=args.scale_unc_val, smodel=args.smodel)

# categories_mva = {}
# for i in range(len(mva_cut)):
# 	categories_mva["cat%i"%i] = "(%s)"%(mva_cut[i]) # only mva categorization
# create_datacard(categories_mva, args.sig_input_path, args.data_input_path, args.data_tree, args.output_path,  "datacard_dnn_v1_mva", "workspace_dnn_v1_mva", nuis=args.nuis, res_unc_val=args.res_unc_val, scale_unc_val=args.scale_unc_val, smodel=args.smodel)

# categories_eta = {}
# for i in range(len(eta_cut)):
# 	categories_eta["cat%i"%i] = "(%s)"%(eta_cut[i]) # only eta categorization
# create_datacard(categories_eta, args.sig_input_path, args.data_input_path, args.data_tree, args.output_path,  "datacard_dnn_v1_eta", "workspace_dnn_v1_eta", nuis=args.nuis, res_unc_val=args.res_unc_val, scale_unc_val=args.scale_unc_val, smodel=args.smodel)



score = "(ggH_prediction+VBF_prediction)"
mva_cuts_v2 = [0.612, 0.766, 0.808, 0.838, 0.922, 0.956] # DNN 3layers V2, score = ggH+VBF

mva_cut_v2 = [
	"(%s<%f)"%(score, mva_cuts_v2[0]), #cat0
	"(%s>%f)&(%s<%f)"%(score, mva_cuts_v2[0], score, mva_cuts_v2[1]), #cat1
	"(%s>%f)&(%s<%f)"%(score, mva_cuts_v2[1], score, mva_cuts_v2[2]), #cat4
	"(%s>%f)&(%s<%f)"%(score, mva_cuts_v2[2], score, mva_cuts_v2[3]), #cat7
	"(%s>%f)&(%s<%f)"%(score, mva_cuts_v2[3], score, mva_cuts_v2[4]), #cat10
	"(%s>%f)&(%s<%f)"%(score, mva_cuts_v2[4], score, mva_cuts_v2[5]), #cat13
	"(%s>%f)"%(score, mva_cuts_v2[5]), #cat14
]

mva_cut_v2_full = [
	"(%s<%f)"%(score, mva_cuts_v2[0]), #cat0
	"(%s>%f)&(%s<%f)"%(score, mva_cuts_v2[0], score, mva_cuts_v2[1]), #cat1
	"(%s>%f)&(%s<%f)"%(score, mva_cuts_v2[0], score, mva_cuts_v2[1]), #cat2
	"(%s>%f)&(%s<%f)"%(score, mva_cuts_v2[0], score, mva_cuts_v2[1]), #cat3
	"(%s>%f)&(%s<%f)"%(score, mva_cuts_v2[1], score, mva_cuts_v2[2]), #cat4
	"(%s>%f)&(%s<%f)"%(score, mva_cuts_v2[1], score, mva_cuts_v2[2]), #cat5
	"(%s>%f)&(%s<%f)"%(score, mva_cuts_v2[1], score, mva_cuts_v2[2]), #cat6
	"(%s>%f)&(%s<%f)"%(score, mva_cuts_v2[2], score, mva_cuts_v2[3]), #cat7
	"(%s>%f)&(%s<%f)"%(score, mva_cuts_v2[2], score, mva_cuts_v2[3]), #cat8
	"(%s>%f)&(%s<%f)"%(score, mva_cuts_v2[2], score, mva_cuts_v2[3]), #cat9
	"(%s>%f)&(%s<%f)"%(score, mva_cuts_v2[3], score, mva_cuts_v2[4]), #cat10
	"(%s>%f)&(%s<%f)"%(score, mva_cuts_v2[3], score, mva_cuts_v2[4]), #cat11
	"(%s>%f)&(%s<%f)"%(score, mva_cuts_v2[3], score, mva_cuts_v2[4]), #cat12
	"(%s>%f)&(%s<%f)"%(score, mva_cuts_v2[4], score, mva_cuts_v2[5]), #cat13
	"(%s>%f)"%(score, mva_cuts_v2[5]), #cat14
]


categories_v2_full = {}
for i in range(len(eta_cut_v2_full)):
	categories_v2_full["cat%i"%i] = "(%s)&(%s)"%(eta_cut_full[i], mva_cut_v2_full[i])
create_datacard(categories_v2_full, args.sig_input_path, args.data_input_path, args.data_tree, args.output_path,  "datacard_dnn_v2_full", "workspace_dnn_v2_full", nuis=args.nuis, res_unc_val=args.res_unc_val, scale_unc_val=args.scale_unc_val, smodel=args.smodel)

categories_v2_mva = {}
for i in range(len(mva_cut)):
	categories_v2_mva["cat%i"%i] = "(%s)"%(mva_cut_v2[i]) # only mva categorization
create_datacard(categories_v2_mva, args.sig_input_path, args.data_input_path, args.data_tree, args.output_path,  "datacard_dnn_v2_mva", "workspace_dnn_v2_mva", nuis=args.nuis, res_unc_val=args.res_unc_val, scale_unc_val=args.scale_unc_val, smodel=args.smodel)

