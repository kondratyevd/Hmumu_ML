import os, sys
sys.path.append( os.path.dirname(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) ))

from src.setup import files, inputDir, treePath, variables
# from src.setup import variables
import pandas
import uproot



def GetDataFrame():

	df = pandas.DataFrame()

	for f in files:

		open_file = uproot.open(inputDir+f.name+".root")
		uproot_tree = open_file[treePath]

		single_file_df = pandas.DataFrame()

		for var in [v for v in variables if v.use]:

			up_var = uproot_tree[var.name].array()

			if var.isMultiDim:
				
				# splitting the multidimensional input variables so each column corresponds to a one-dimensional variable
				# Only <nItems> objects are kept, the number is specified in setup.py 

				single_var_df = pandas.DataFrame(data = up_var.tolist())
				single_var_df.drop(single_var_df.columns[var.nItems:],axis=1,inplace=True)
				single_var_df.columns = [var.name+"[%i]"%i for i in range(var.nItems)]

				single_file_df = pandas.concat([single_file_df, single_var_df], axis=1)

			else:
				single_file_df[var.name] = up_var
		
		single_file_df['sample_weight'] = f.weight
		single_file_df['category'] = f.isSignal

		df = pandas.concat([df,single_file_df])
	
	df.dropna(axis=0, how='any', inplace=True)
	return df