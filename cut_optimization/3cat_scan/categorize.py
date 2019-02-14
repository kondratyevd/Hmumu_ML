import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from make_datacards import create_datacard
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--in_path', action='store', dest='input_path', help='Input path')
parser.add_argument('--out_path', action='store', dest='output_path', help='Output path')

args = parser.parse_args()

for i in range(24):
    for j in range(i):
        bins = [0.0, (j+1)/10.0, (i+1)/10.0, 2.4]
        create_datacard(bins, args.input_path, args.output_path, "datacard_3cat_%i_%i"%(j+1, i+1), "workspace_3cat_%i_%i"%(j+1, i+1), nuis=False)