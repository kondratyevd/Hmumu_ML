from datetime import datetime
import sys, os, errno

def prepare():
	now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	RunID = "Run_"+now+"/"

	try:
		os.makedirs('output/'+RunID)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise

	with open("output/CURRENT_RUN_ID","w") as info:
		info.write(RunID)


prepare()