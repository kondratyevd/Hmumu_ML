from datetime import datetime
import sys, os, errno

now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
RunID = "Run_"+now+"/"

try:
	os.makedirs('output/'+RunID)
except OSError as e:
	if e.errno != errno.EEXIST:
		raise


info = open("output/CURRENT_RUN_ID","w") 
info.write(RunID)
info.close() 