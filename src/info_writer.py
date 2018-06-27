

class Writer(object):
	def __init__(self, path):
		self.file = open(path+"info.txt","w") 
	
	def Write(self,str):	
		self.file.write(str)
