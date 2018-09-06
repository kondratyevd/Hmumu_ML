import ROOT
ROOT.gStyle.SetOptStat(0)
		

class Plotter_TMVA(object):
	def __init__(self):
		self.method_list = []
		# self.filedir = "/Users/dmitry/Documents/HiggsToMuMu/Hmumu_ML/output/"

	class Method(object):
		def __init__(self, title, filedir, RunID, name, color):
			self.name = name
			self.title = title
			self.filedir = filedir
			self.RunID = RunID
			self.color = color
			self.plot_path = "plots/"

	def add_TMVA_method(self, title, filedir, RunID, method_name, color):
		method = self.Method(title, filedir, RunID, method_name, color)
		self.method_list.append(method)
		print "Method %s added for %s"%(method_name, RunID)
		pass

	def add_Keras_method(self, RunID, method_name):
		pass

	def get_scatter(self):
		pass	
	def get_transformed_scatter(self):
		pass

	def get_ROC(self):
		self.roc_hist_list = []
		for method in self.method_list:
			file = ROOT.TFile.Open(method.filedir+method.RunID+"/TMVA/TMVA.root")
			hist = file.Get('dataset/Method_%s/%s/MVA_%s_rejBvsS'%(method.name,method.name,method.name))
			hist.SetLineColor(ROOT.TColor.GetColor(method.color))
			hist.SetTitle(method.title)
			hist.SetDirectory(0)
			self.roc_hist_list.append(hist)

	def get_MVA_score(self):
		pass
	def get_acc(self):
		pass
	def get_loss(self):
		pass

	def set_plot_path(self, path):
		self.plot_path = path

	def plot_hist_list(self, label, hist_list):
		canvas = ROOT.TCanvas("canvas_%s"%label, "canvas_%s"%label, 800, 800)
		canvas.cd()
		legend = ROOT.TLegend(0.11,0.11,0.25,0.25)	
		for hist in hist_list:
			hist.Draw('histsame')		
			legend.AddEntry(hist, hist.GetTitle(), "l")
		legend.Draw()
		canvas.Print("%s%s.png"%(self.plot_path,label))
		canvas.Close()
		pass

# plot scatter plots

# plot transformed scatter plots

# plot MVA scores

# plot accuracy

# plot loss