import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from make_datacards import create_datacard
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--in_path', action='store', dest='input_path', help='Input path')
parser.add_argument('--out_path', action='store', dest='output_path', help='Output path')
parser.add_argument('--nuis', action='store_true', dest='nuis', help='Include nuisances')

args = parser.parse_args()

for i in range(23):
    bins = [0, (i+1)/10.0, 2.4]
    create_datacard(bins, args.input_path, args.output_path, "datacard_2cat_%i"%(i+1), "workspace_2cat_%i"%(i+1), nuis=args.nuis)