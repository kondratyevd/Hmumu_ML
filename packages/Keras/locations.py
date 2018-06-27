import sys, os, errno
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from src.setup import RunID



outputDir = "output/"+RunID+"Keras/"
logsDir = "output/"+RunID+"keras_logs/"
modelsDir = outputDir+"models/"
dataDir = outputDir+"data/"

try:
	os.makedirs(outputDir)
except OSError as e:
	if e.errno != errno.EEXIST:
		raise

try:
	os.makedirs(logsDir)
except OSError as e:
	if e.errno != errno.EEXIST:
		raise

try:
	os.makedirs(modelsDir)
except OSError as e:
	if e.errno != errno.EEXIST:
		raise

try:
	os.makedirs(dataDir)
except OSError as e:
	if e.errno != errno.EEXIST:
		raise