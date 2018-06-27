import os, sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import importlib
from setup import usePackage

for pkg in usePackage:
	importlib.import_module('packages.%s.train'%pkg).Train()

