import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from make_datacards import create_datacard
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--sig_in_path', action='store', dest='sig_input_path', help='Input path')
parser.add_argument('--data_in_path', action='store', dest='data_input_path', help='Input path')
parser.add_argument('--sig_tree', action='store', dest='sig_tree', help='Tree name')
parser.add_argument('--data_tree', action='store', dest='data_tree', help='Tree name')
parser.add_argument('--out_path', action='store', dest='output_path', help='Output path')
parser.add_argument('--lumi', action='store', dest='lumi', help='Integrated luminosity')
parser.add_argument('--nuis', action='store_true', dest='nuis', help='Nuisances')
parser.add_argument('--nuis_val', action='store', dest='nuis_val', help='Resolution uncertainty')
parser.add_argument('--smodel', action='store', dest='smodel', help='Signal model')
args = parser.parse_args()

args = parser.parse_args()

for i in range(15, 23):
    for j in range(i):
        bins = [0.0, (j+1)/10.0, (i+1)/10.0, 2.4]
        create_datacard(bins, args.sig_input_path, args.sig_tree, args.data_input_path, args.data_tree, args.output_path, "datacard_3cat_%i_%i"%(j+1, i+1), "workspace_3cat_%i_%i"%(j+1, i+1), args.lumi, nuis=args.nuis, nuis_val=args.nuis_val, smodel=args.smodel)