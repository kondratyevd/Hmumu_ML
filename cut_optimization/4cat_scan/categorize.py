import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from make_datacards import create_datacard
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--in_path', action='store', dest='input_path', help='Input path')
parser.add_argument('--out_path', action='store', dest='output_path', help='Output path')

args = parser.parse_args()

second_cut_options = {
    # "1p8": 1.8,
    "1p9": 1.9,
    "2p0": 2.0,
    }
scan_options = [
    "Bscan","Oscan", "Escan"
    ]
for key, value in second_cut_options.iteritems():
    for scan in scan_options:
        if "O" in scan:
            for i in range(int((value - 1)*10)):
                bins = [0, 0.9, (i+10)/10.0, value, 2.4]
                create_datacard(bins, args.input_path, args.output_path+"_%s_%s/"%(key, scan), "datacard_0p9_%i_%s"%((i+10), key), "workspace_0p9_%i_%s"%((i+10), key))
        if "E" in scan:
            for i in range(23-int((value)*10)):
                bins = [0, 0.9, value, i/10.0+value+0.1, 2.4]
                create_datacard(bins, args.input_path, args.output_path+"_%s_%s/"%(key, scan), "datacard_0p9_%i_%s"%((i+1+value*10), key), "workspace_0p9_%i_%s"%((i+1+value*10), key))
        if "B" in scan:
            for i in range(8):
                bins = [0, (i+1)/10.0 ,0.9, value, 2.4]
                create_datacard(bins, args.input_path, args.output_path+"_%s_%s/"%(key, scan), "datacard_0p9_%i_%s"%((i+1), key), "workspace_0p9_%i_%s"%((i+1), key))