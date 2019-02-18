import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from make_datacards import create_datacard
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--sig_in_path', action='store', dest='sig_input_path', help='Input path')
parser.add_argument('--data_in_path', action='store', dest='data_input_path', help='Input path')
parser.add_argument('--out_path', action='store', dest='output_path', help='Output path')

args = parser.parse_args()

create_datacard([0, 2.4], args.sig_input_path, args.data_input_path, args.output_path, "datacard", "workspace")