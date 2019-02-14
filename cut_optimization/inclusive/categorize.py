import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from make_datacards import create_datacard
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--in_path', action='store', dest='input_path', help='Input path')
parser.add_argument('--out_path', action='store', dest='output_path', help='Output path')

args = parser.parse_args()

create_datacard([0, 2.4], args.input_path, args.output_path, "datacard", "workspace")