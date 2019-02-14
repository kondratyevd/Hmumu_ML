from run_options import *
import argparse


parser = argparse.ArgumentParser(description='')
parser.add_argument('--in_path', action='store', dest='input_path', help='Input path')
parser.add_argument('--out_path', action='store', dest='output_path', help='Output path')
# parser.add_argument('--input_path', dest='input_path', default='', action='store_const', help='input path')

args = parser.parse_args()


# print args.input_path

# input_path = "/Users/dmitrykondratyev/Documents/HiggsToMuMu/Hmumu_ML/output/Run_2018-12-19_14-25-02/Keras_multi/model_50_D2_25_D2_25_D2/root/"

# output_path = "output/inclusive/"

# run_inclusive(args.input_path, args.output_path)
run_2cat_scan(args.input_path, args.output_path)