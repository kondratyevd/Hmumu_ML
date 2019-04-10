import ROOT
import os, sys, errno
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(1)
from math import *  
from array import array 


class Analyzer(object):

    def __init__(self):
        self.out_path = ""
        self.MVASources = []
        
    class MVASource(object):
        def __init__(self, framework, name, title, path):
            self.framework = framework
            self.name = name
            self.title = title 
            self.path = path
            self.samples = []
            self.data_hist = None
            self.signal_hists = []
            self.mc_stack = ROOT.THStack()
            self.lumi = 0

        class Sample(object):
            def __init__(self, source, name, title, filename, treename, isData, isStacked, color, isWeightOverLumi, additional_cut):
                self.source = source
                self.name = name
                self.title = title
                self.filename = filename
                self.treename = treename
                self.isData = isData
                self.isStacked = isStacked
                self.color = color          
                self.isWeightOverLumi = isWeightOverLumi
                self.additional_cut = additional_cut

            def fit_with_dcb(self, label):
                width = 0
                Import = getattr(ROOT.RooWorkspace, 'import')
                var = ROOT.RooRealVar("mass","Dilepton mass",110,150)     
                var.setBins(100)
                var.setRange("window",120,130)
                var.setRange("full",110,150)
                w = ROOT.RooWorkspace("w", False)
                Import(w, var)
                max_abs_eta_var = ROOT.RooRealVar("max_abs_eta_mu","Max abs(eta) of muons", 0, 2.4) 
                ggh_pred_var = ROOT.RooRealVar("ggH_prediction", "ggH_prediction", 0, 1)
                vbf_pred_var = ROOT.RooRealVar("VBF_prediction", "VBF_prediction", 0, 1)
                dy_pred_var = ROOT.RooRealVar("DY_prediction", "DY_prediction", 0, 1)
                tt_pred_var = ROOT.RooRealVar("ttbar_prediction", "ttbar_prediction", 0, 1) 
                signal_tree = ROOT.TChain(self.treename)
                signal_tree.Add("%s/%s"%(self.source.path, self.filename))  
                print "Loaded tree from "+self.source.path+" with %i entries."%signal_tree.GetEntries()    
                signal_hist_name = "signal_%s"%label
                signal_hist = ROOT.TH1D(signal_hist_name, signal_hist_name, 40, 110, 150)

                dummy = ROOT.TCanvas("dummy", "dummy", 800, 800)
                dummy.cd()
                signal_tree.Draw("mass>>%s"%(signal_hist_name), "(%s)*weight_over_lumi*%s"%(self.additional_cut, self.source.lumi))
                dummy.Close()
                signal_rate = signal_hist.Integral()
                ROOT.gSystem.Load("/home/dkondra/Hmumu_analysis/Hmumu_ML/lib/RooDCBShape_9g_cxx.so")
                # ROOT.gSystem.Load("/Users/dmitrykondratyev/Documents/HiggsToMuMu/Hmumu_ML/cut_optimization/RooDCBShape_cxx.so")
                w.factory("RooDCBShape_9g::%s_ggh(mass, %s_mean[125,120,130], %s_sigma[2,0,5], %s_alphaL[2,0,25] , %s_alphaR[2,0,25], %s_nL[1.5,0,25], %s_nR[1.5,0,25])"%(label,label,label,label,label,label,label))
                smodel = w.pdf("%s_ggh"%label)
                w.Print()
                signal_ds = ROOT.RooDataSet("signal_ds","signal_ds", signal_tree, ROOT.RooArgSet(var, max_abs_eta_var, ggh_pred_var, vbf_pred_var, dy_pred_var, tt_pred_var), self.additional_cut)
                res = smodel.fitTo(signal_ds, ROOT.RooFit.Range("full"),ROOT.RooFit.Save(), ROOT.RooFit.Verbose(False))
                res.Print()

                Import(w, smodel)

                return w.var("%s_sigma"%label).getVal(), w.var("%s_sigma"%label).getError()



        def set_lumi(self, lumi):
            self.lumi = lumi

        def add_sample(self, name, title, filename, treename, isData, isStacked, color, isWeightOverLumi=True, additional_cut="1"):
            new_sample = self.Sample(self, name, title, filename, treename, isData, isStacked, color, isWeightOverLumi, additional_cut)
            self.samples.append(new_sample)
            return new_sample
            
        def plot(self, var_name, nBins, xmin, xmax, label="", draw=True, shapes=False):
            trees = {}
            self.signal_hists = []
            self.data_hist = None
            self.mc_stack = ROOT.THStack()
            legend = ROOT.TLegend(0.7, 0.8, 0.895, 0.895)

            for smp in self.samples:
                trees[smp.name] = ROOT.TChain(smp.treename)
                trees[smp.name].Add("%s/%s"%(self.path, smp.filename))
                print "Tree for %s added with %i entries"%(smp.name, trees[smp.name].GetEntries())
                hist_name = "hist_%s_%s_%s_%s"%(self.name, smp.name, var_name, label)
                hist = ROOT.TH1D(hist_name, hist_name, nBins, xmin, xmax)

                dummy = ROOT.TCanvas("dummy_"+hist_name, "dummy_"+hist_name, 100, 100)
                dummy.cd()
                if smp.isData:
                    trees[smp.name].Draw("%s>>%s"%(var_name, hist_name))
                    self.data_hist = hist
                    self.data_hist.SetMarkerColor(smp.color)
                    self.data_hist.SetLineColor(smp.color)
                    self.data_hist.SetMarkerStyle(20)
                    self.data_hist.SetMarkerSize(0.8)
                    legend.AddEntry(hist, "Data %i /pb"%self.lumi, "pe")
                else:
                    if smp.isWeightOverLumi:
                        trees[smp.name].Draw("%s>>%s"%(var_name, hist_name), "weight_over_lumi*%f*(%s)"%(self.lumi, smp.additional_cut))
                    else:
                        trees[smp.name].Draw("%s>>%s"%(var_name, hist_name), "weight*(%s)"%smp.additional_cut)
                dummy.Close()
                print "Hist %s integral: %f"%(hist_name, hist.Integral())
                hist.SetLineWidth(2)
                if smp.isStacked:
                    hist.SetFillColor(smp.color)
                    hist.SetLineColor(ROOT.kBlack)
                    self.mc_stack.Add(hist)
                    legend.AddEntry(hist, smp.title, "f")
                elif not smp.isData:
                    hist.SetLineColor(smp.color)
                    self.signal_hists.append(hist)
                    legend.AddEntry(hist, smp.title, "l")
                    if shapes:
                        hist.Scale(1/hist.Integral())

            if draw:
                canvas = ROOT.TCanvas(var_name, var_name, 800, 800)
                canvas.cd()
                canvas.SetLogy()

                if not shapes:
                    self.mc_stack.Draw("hist")
                    self.mc_stack.SetTitle(self.title)
                    self.mc_stack.GetXaxis().SetTitle(var_name)
                    self.mc_stack.SetMinimum(0.01)
                    self.mc_stack.SetMaximum(100000)
                for hist in self.signal_hists:
                    hist.Draw("histsame")
                    hist.GetXaxis().SetTitle(var_name)
                if self.data_hist:
                    self.data_hist.Draw("pe1same")
                legend.Draw()
                new_out_path = "%s/%s/"%(self.framework.out_path, self.name)

                try:
                    os.makedirs(new_out_path)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise

                canvas.SaveAs("%s/%s.root"%(new_out_path,label))
                canvas.SaveAs("%s/%s.png"%(new_out_path,label))


        def plot_roc(self, score, nBins, xmin, xmax, working_points=[], label = "test"):
            print "Plotting MVA score..."
            self.plot(score, nBins, xmin, xmax, label+"_forROC")
            roc = ROOT.TGraph()
            idx = []
            if working_points:
                idx = self.score_cuts_from_wp(score, working_points, nBins, xmin, xmax)
                wp_graph = ROOT.TGraph()
                wp_graph.SetMarkerStyle(20)
                wp_graph.SetMarkerSize(1)
                wp_graph.SetMarkerColor(ROOT.kRed)
                count = 0

            for i in range(1, nBins+1):
                # print "i = ", i
                sig_above = 0
                sig_total = 0
                bkg_below = 0
                bkg_total = 0
                sig_eff = 0
                bkg_rej = 0

                for h in self.mc_stack.GetHists():
                    bkg_below = bkg_below + h.Integral(1, i)
                    bkg_total = bkg_total + h.Integral()
                for h in self.signal_hists:
                    sig_above = sig_above + h.Integral(i,nBins+1)
                    sig_total = sig_total + h.Integral()

                if sig_total:
                    sig_eff = sig_above/sig_total

                if bkg_total:
                    bkg_rej = bkg_below/bkg_total

                if working_points:
                    if i in idx:
                        wp_graph.SetPoint(count, sig_eff, bkg_rej)
                        count = count+1

                roc.SetPoint(i-1, sig_eff, bkg_rej)

            canvas = ROOT.TCanvas("%s_roc"%self.name, "%s_roc"%self.name, 800, 800)
            canvas.cd()
            roc.SetLineWidth(2)
            roc.GetXaxis().SetTitle("sig. eff.")
            roc.GetYaxis().SetTitle("bkg. rej.")
            roc.Draw("al")
            if working_points:
                wp_graph.Draw("psame")
            new_out_path = "%s/%s/"%(self.framework.out_path, self.name)
            canvas.SaveAs("%s/roc.root"%(new_out_path))
            canvas.SaveAs("%s/roc.png"%(new_out_path))

            return roc

        def score_cuts_from_wp(self, score, signal_wp, nBins, xmin, xmax):
            # self.plot(score, nBins, xmin, xmax, "forWP", draw=False)
            closest_cuts = []
            best_appr = []
            idx = []
            binWidth = (xmax-xmin)/float(nBins)
            for wp in signal_wp:
                closest_cut = 0
                best_approx = 0
                best_idx = 0
                min_diff = 100
                for i in range(1, nBins+1):
                    bin_upper_value = xmin+i*binWidth
                    sig_below = 0
                    sig_total = 0
                    for h in self.signal_hists:
                        sig_below = sig_below + h.Integral(1,i)
                        sig_total = sig_total + h.Integral()

                    if sig_total:
                        sig_percentile = sig_below/sig_total
                    else:
                        sig_percentile = 0

                    if abs(wp - sig_percentile)<min_diff:
                        min_diff = abs(wp - sig_percentile)
                        closest_cut = bin_upper_value
                        best_approx = sig_percentile
                        best_idx = i
                closest_cuts.append(closest_cut)
                best_appr.append(best_approx)
                idx.append(best_idx)
            print closest_cuts
            return idx

    def set_out_path(self, path):
        self.out_path = path
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def add_mva_source(self, name, title, path):
        new_bdt = self.MVASource(self, name, title, path)
        self.MVASources.append(new_bdt)
        return new_bdt

    class RocCurve(object):
        def __init__(self, graph, name, title, color, linestyle=1):
            self.graph=graph
            self.name=name
            self.title=title
            self.color=color
            self.linestyle = linestyle
    
    def roc_from_tmva(self, source, title, filename, hist_path, color, linestyle):
        file = ROOT.TFile.Open("%s/%s"%(source.path, filename), "r")
        hist = file.Get(hist_path)
        print hist.GetEntries()
        hist.SetDirectory(0)
        roc = self.RocCurve(hist, source.name+"_roc", title, color, linestyle) 
        file.Close()    
        return roc

    def compare_roc_curves(self, roc_list):
        legend = ROOT.TLegend(0.7, 0.8, 0.895, 0.895)
        canvas = ROOT.TCanvas("roc_curves","roc_curves", 800, 800)
        canvas.cd()
        first = True
        for roc in roc_list:
            roc.graph.SetLineColor(roc.color)
            roc.graph.SetLineWidth(2)
            roc.graph.SetLineStyle(roc.linestyle)
            if first:
                roc.graph.Draw("al")
                first = False
            else:
                roc.graph.Draw("lsame")
            legend.AddEntry(roc.graph, roc.title, "l")
        legend.Draw()
        canvas.SaveAs("%s/roc_curves.root"%(self.out_path))
        canvas.SaveAs("%s/roc_curves.png"%(self.out_path))


    def plot_width_vs_score(self, score, source, name, title, nBins, color, markerStyle, process = "ggH"):
        graph = ROOT.TH1D("wvss"+name, title, nBins, 0, 1)
        for i in range(nBins):
            cut_lo = i/float(nBins)
            cut_hi = (i+1)/float(nBins)
            if "ggH" in process:
                sample_bin = source.add_sample("ggh", "ggH", "output_t*root", "tree_H2Mu_gg", False, False, ROOT.kRed, True, "(%s>%f)&(%s<%f)"%(score, cut_lo, score, cut_hi))
            elif "VBF" in process:
                sample_bin = source.add_sample("vbf", "VBF", "output_t*root", "tree_H2Mu_VBF", False, False, ROOT.kViolet-1, True, "(%s>%f)&(%s<%f)"%(score, cut_lo, score, cut_hi))
            width_bin, error_bin = sample_bin.fit_with_dcb("%s_bin_%i"%(name,i+1))
            graph.SetBinContent(i+1, width_bin)
            graph.SetBinError(i+1, error_bin)
        graph.SetMarkerColor(color)
        graph.SetLineColor(color)
        graph.SetMarkerStyle(markerStyle)
        return graph


roc_to_compare = []

a = Analyzer()
a.set_out_path("plots/mva_output_analyzis")

# Option 6
dnn_multi = a.add_mva_source("DNN_Multi", "DNN_Multi", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-08_11-37-05//Keras_multi/model_50_D2_25_D2_25_D2/root/")
dnn_multi.add_sample("tt", "ttbar", "output_t*root", "tree_tt_ll_POW", False, True, ROOT.kYellow, True)
dnn_multi.add_sample("dy", "Drell-Yan", "output_t*root", "tree_ZJets_aMC", False, True, ROOT.kOrange-3, True)
dnn_multi.add_sample("ggh", "ggH", "output_t*root", "tree_H2Mu_gg", False, False, ROOT.kRed, True)
dnn_multi.add_sample("vbf", "VBF", "output_t*root", "tree_H2Mu_VBF", False, False, ROOT.kViolet-1, True)
dnn_multi.add_sample("data", "Data 2017 (40.5/fb)", "output_Data.root", "tree_Data", True, False, ROOT.kBlack)
dnn_multi.set_lumi(40490.712)
dnn_multi_roc_graph = dnn_multi.plot_roc("ggH_prediction+VBF_prediction", 500, 0, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
dnn_multi_roc = a.RocCurve(dnn_multi_roc_graph, "dnn_multi", "DNN_Multi", ROOT.kViolet)
# roc_to_compare.append(dnn_multi_roc)

# Option 7
dnn_multi_hiStat = a.add_mva_source("DNN_Multi_hiStat", "DNN_Multi_hiStat", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-08_11-51-21//Keras_multi/model_50_D2_25_D2_25_D2/root/")
dnn_multi_hiStat.add_sample("tt", "ttbar", "output_t*root", "tree_tt_ll_POW", False, True, ROOT.kYellow, True)
dnn_multi_hiStat.add_sample("dy", "Drell-Yan", "output_t*root", "tree_ZJets_aMC", False, True, ROOT.kOrange-3, True)
dnn_multi_hiStat.add_sample("ggh", "ggH", "output_t*root", "tree_H2Mu_gg", False, False, ROOT.kRed, True)
dnn_multi_hiStat.add_sample("vbf", "VBF", "output_t*root", "tree_H2Mu_VBF", False, False, ROOT.kViolet-1, True)
dnn_multi_hiStat.add_sample("data", "Data 2017 (40.5/fb)", "output_Data.root", "tree_Data", True, False, ROOT.kBlack)
dnn_multi_hiStat.set_lumi(40490.712)
dnn_multi_hiStat_roc_graph = dnn_multi_hiStat.plot_roc("ggH_prediction+VBF_prediction", 500, 0, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
dnn_multi_hiStat_roc = a.RocCurve(dnn_multi_hiStat_roc_graph, "dnn_multi_hiStat", "DNN_Multi_hiStat", ROOT.kViolet)
roc_to_compare.append(dnn_multi_hiStat_roc)

# Option 8
dnn_multi_hiStat_ebe = a.add_mva_source("DNN_Multi_hiStat_ebe", "DNN_Multi_hiStat_ebe", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-08_11-37-09//Keras_multi/model_50_D2_25_D2_25_D2/root/")
dnn_multi_hiStat_ebe.add_sample("tt", "ttbar", "output_t*root", "tree_tt_ll_POW", False, True, ROOT.kYellow, True)
dnn_multi_hiStat_ebe.add_sample("dy", "Drell-Yan", "output_t*root", "tree_ZJets_aMC", False, True, ROOT.kOrange-3, True)
dnn_multi_hiStat_ebe.add_sample("ggh", "ggH", "output_t*root", "tree_H2Mu_gg", False, False, ROOT.kRed, True)
dnn_multi_hiStat_ebe.add_sample("vbf", "VBF", "output_t*root", "tree_H2Mu_VBF", False, False, ROOT.kViolet-1, True)
dnn_multi_hiStat_ebe.add_sample("data", "Data 2017 (40.5/fb)", "output_Data.root", "tree_Data", True, False, ROOT.kBlack)
dnn_multi_hiStat_ebe.set_lumi(40490.712)
dnn_multi_hiStat_ebe_roc_graph = dnn_multi_hiStat_ebe.plot_roc("ggH_prediction+VBF_prediction", 500, 0, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
dnn_multi_hiStat_ebe_roc = a.RocCurve(dnn_multi_hiStat_ebe_roc_graph, "dnn_multi_hiStat_ebe", "DNN_Multi_hiStat_ebe", ROOT.kGreen)
# roc_to_compare.append(dnn_multi_hiStat_ebe_roc)

# Option 9
dnn_binary_hiStat = a.add_mva_source("DNN_Binary_hiStat", "DNN_Binary_hiStat", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-08_11-37-12//Keras_multi/model_50_D2_25_D2_25_D2/root/")
dnn_binary_hiStat.add_sample("bkg", "bkg", "output_t*root", "tree_bkg", False, True, ROOT.kYellow, True)
dnn_binary_hiStat.add_sample("sig", "sig", "output_t*root", "tree_sig", False, False, ROOT.kRed, True)
dnn_binary_hiStat.add_sample("data", "Data 2017 (40.5/fb)", "output_Data.root", "tree_Data", True, False, ROOT.kBlack)
dnn_binary_hiStat.set_lumi(40490.712)
dnn_binary_hiStat_roc_graph = dnn_binary_hiStat.plot_roc("sig_prediction", 500, 0, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
dnn_binary_hiStat_roc = a.RocCurve(dnn_binary_hiStat_roc_graph, "dnn_binary_hiStat", "DNN_Binary_hiStat", ROOT.kGreen)
roc_to_compare.append(dnn_binary_hiStat_roc)

# Option 9.1 - plot only shapes
dnn_binary_hiStat_1 = a.add_mva_source("DNN_Binary_hiStat_1", "DNN_Binary_hiStat_1", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-08_11-37-12//Keras_multi/model_50_D2_25_D2_25_D2/root/")
dnn_binary_hiStat_1.add_sample("bkg", "bkg", "output_t*root", "tree_bkg", False, False, ROOT.kBlue, True)
dnn_binary_hiStat_1.add_sample("sig", "sig", "output_t*root", "tree_sig", False, False, ROOT.kRed, True)
dnn_binary_hiStat_1.set_lumi(40490.712)
dnn_binary_hiStat_1.plot("sig_prediction", 500, 0, 1, label="shapes", draw=True, shapes=True)

# Option 10
dnn_binary_hiStat_ebe = a.add_mva_source("DNN_Binary_hiStat_ebe", "DNN_Binary_hiStat_ebe", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-08_11-37-16//Keras_multi/model_50_D2_25_D2_25_D2/root/")
dnn_binary_hiStat_ebe.add_sample("bkg", "bkg", "output_t*root", "tree_bkg", False, True, ROOT.kYellow, True)
dnn_binary_hiStat_ebe.add_sample("sig", "sig", "output_t*root", "tree_sig", False, False, ROOT.kRed, True)
dnn_binary_hiStat_ebe.add_sample("data", "Data 2017 (40.5/fb)", "output_Data.root", "tree_Data", True, False, ROOT.kBlack)
dnn_binary_hiStat_ebe.set_lumi(40490.712)
dnn_binary_hiStat_ebe_roc_graph = dnn_binary_hiStat_ebe.plot_roc("sig_prediction", 500, 0, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
dnn_binary_hiStat_ebe_roc = a.RocCurve(dnn_binary_hiStat_ebe_roc_graph, "dnn_binary_hiStat_ebe", "DNN_Binary_hiStat_ebe", ROOT.kYellow)
# roc_to_compare.append(dnn_binary_hiStat_ebe_roc)

# Option 11
dnn_binary = a.add_mva_source("DNN_Binary", "DNN_Binary", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-09_16-22-10//Keras_multi/model_50_D2_25_D2_25_D2/root/")
dnn_binary.add_sample("bkg", "bkg", "output_t*root", "tree_bkg", False, True, ROOT.kYellow, True)
dnn_binary.add_sample("sig", "sig", "output_t*root", "tree_sig", False, False, ROOT.kRed, True)
dnn_binary.add_sample("data", "Data 2017 (40.5/fb)", "output_Data.root", "tree_Data", True, False, ROOT.kBlack)
dnn_binary.set_lumi(40490.712)
dnn_binary_roc_graph = dnn_binary.plot_roc("sig_prediction", 500, 0, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
dnn_binary_roc = a.RocCurve(dnn_binary_roc_graph, "dnn_binary", "DNN_Binary", ROOT.kBlue)
# roc_to_compare.append(dnn_binary_roc)

# Option 0
bdt_uf = a.add_mva_source("BDT_UF", "BDT_UF", "/home/dkondra/tmp/BDTG_UF/")
bdt_uf.add_sample("tt", "ttbar", "tt_ll_POW_BDTG_UF_v1.root", "tree", False, True, ROOT.kYellow)
bdt_uf.add_sample("dy", "Drell-Yan", "ZJets_aMC_BDTG_UF_v1.root", "tree", False, True, ROOT.kOrange-3)
bdt_uf.add_sample("ggh", "ggH", "H2Mu_gg_BDTG_UF_v1.root", "tree", False, False, ROOT.kRed)
bdt_uf.add_sample("vbf", "VBF", "H2Mu_VBF_BDTG_UF_v1.root", "tree", False, False, ROOT.kViolet-1)
bdt_uf.add_sample("data", "Data 2017 (40.5/fb)", "SingleMu_2017*_BDTG_UF_v1.root", "tree", True, False, ROOT.kBlack)
bdt_uf.set_lumi(40490.712)
bdt_uf_roc_graph = bdt_uf.plot_roc("MVA", 500, -1, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
# bdt_uf_roc = a.RocCurve(bdt_uf_roc_graph, "bdt_uf", "BDT UF loStat", ROOT.kBlack)
bdt_uf_roc = a.RocCurve(bdt_uf_roc_graph, "bdt_uf", "BDT UF loStat", ROOT.kBlue)
# roc_to_compare.append(bdt_uf_roc)
# bdt_uf_roc_tmva = a.roc_from_tmva(bdt_uf, "BDT_UF", "TMVA.root", "dataset/Method_BDTG_UF_v1/BDTG_UF_v1/MVA_BDTG_UF_v1_rejBvsS", ROOT.kBlack, 2)
# roc_to_compare.append(bdt_uf_roc_tmva)



# Option 1
bdt_uf_hiStat = a.add_mva_source("BDT_UF_hiStat", "BDT_UF_hiStat", "/home/dkondra/tmp/BDTG_UF_hiStat/")
bdt_uf_hiStat.add_sample("tt", "ttbar", "tt_ll_POW_BDTG_UF_v1.root", "tree", False, True, ROOT.kYellow)
# bdt_uf_hiStat.add_sample("dy", "Drell-Yan", "ZJets_aMC_BDTG_UF_v1.root", "tree", False, True, ROOT.kOrange-3)
bdt_uf_hiStat.add_sample("dy", "Drell-Yan", "ZJets_aMC/*.root", "tree", False, True, ROOT.kOrange-3)
bdt_uf_hiStat.add_sample("ggh", "ggH", "H2Mu_gg_BDTG_UF_v1.root", "tree", False, False, ROOT.kRed)
bdt_uf_hiStat.add_sample("vbf", "VBF", "H2Mu_VBF_BDTG_UF_v1.root", "tree", False, False, ROOT.kViolet-1)
bdt_uf_hiStat.add_sample("data", "Data 2017 (40.5/fb)", "SingleMu_2017*_BDTG_UF_v1.root", "tree", True, False, ROOT.kBlack)
bdt_uf_hiStat.set_lumi(40490.712)
bdt_uf_hiStat_roc_graph = bdt_uf_hiStat.plot_roc("MVA", 500, -1, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
bdt_uf_hiStat_roc = a.RocCurve(bdt_uf_hiStat_roc_graph, "bdt_uf_hiStat", "BDT UF hiStat", ROOT.kBlack)
roc_to_compare.append(bdt_uf_hiStat_roc)
# bdt_uf_hiStat_roc_tmva = a.roc_from_tmva(bdt_uf_hiStat, "BDT_UF_hiStat", "TMVA.root", "dataset/Method_BDTG_UF_v1/BDTG_UF_v1/MVA_BDTG_UF_v1_rejBvsS", ROOT.kBlue, 2)
# roc_to_compare.append(bdt_uf_hiStat_roc_tmva)

# Option 2
# bdt_uf_hiStat_ebe = a.add_mva_source("BDT_UF_hiStat_ebe", "BDT_UF_hiStat_ebe", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-08_11-36-30/TMVA/")
# bdt_uf_hiStat_ebe_roc_tmva = a.roc_from_tmva(bdt_uf_hiStat_ebe, "BDT_UF_hiStat_ebe", "TMVA.root", "dataset/Method_BDTG_UF_v1/BDTG_UF_v1/MVA_BDTG_UF_v1_rejBvsS", ROOT.kRed, 2)
# roc_to_compare.append(bdt_uf_hiStat_ebe_roc_tmva)

# Option 3
bdt_ucsd = a.add_mva_source("BDT_UCSD", "BDT_UCSD", "/home/dkondra/tmp/BDTG_UCSD/")
bdt_ucsd.add_sample("tt", "ttbar", "tt_ll_POW_BDTG_UCSD.root", "tree", False, True, ROOT.kYellow)
bdt_ucsd.add_sample("dy", "Drell-Yan", "ZJets_aMC_BDTG_UCSD.root", "tree", False, True, ROOT.kOrange-3)
bdt_ucsd.add_sample("ggh", "ggH", "H2Mu_gg_BDTG_UCSD.root", "tree", False, False, ROOT.kRed)
bdt_ucsd.add_sample("vbf", "VBF", "H2Mu_VBF_BDTG_UCSD.root", "tree", False, False, ROOT.kViolet-1)
bdt_ucsd.add_sample("data", "Data 2017 (40.5/fb)", "SingleMu_2017*_BDTG_UCSD.root", "tree", True, False, ROOT.kBlack)
bdt_ucsd.set_lumi(40490.712)
bdt_ucsd_roc_graph = bdt_ucsd.plot_roc("MVA", 500, -1, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
bdt_ucsd_roc = a.RocCurve(bdt_ucsd_roc_graph, "bdt_ucsd", "BDT UCSD loStat", ROOT.kBlue)
# roc_to_compare.append(bdt_ucsd_roc)

# bdt_ucsd_roc_tmva = a.roc_from_tmva(bdt_ucsd, "BDT_UCSD", "TMVA.root", "dataset/Method_BDTG_UCSD/BDTG_UCSD/MVA_BDTG_UCSD_rejBvsS", ROOT.kGreen, 2)
# roc_to_compare.append(bdt_ucsd_roc_tmva)

# Option 4
bdt_ucsd_hiStat = a.add_mva_source("BDT_UCSD_hiStat", "BDT_UCSD_hiStat", "/home/dkondra/tmp/BDTG_UCSD_hiStat/")
bdt_ucsd_hiStat.add_sample("tt", "ttbar", "tt_ll_POW_BDTG_UCSD.root", "tree", False, True, ROOT.kYellow)
# bdt_ucsd_hiStat.add_sample("dy", "Drell-Yan", "ZJets_aMC_BDTG_UCSD.root", "tree", False, True, ROOT.kOrange-3)
bdt_ucsd_hiStat.add_sample("dy", "Drell-Yan", "ZJets_aMC/*.root", "tree", False, True, ROOT.kOrange-3)
bdt_ucsd_hiStat.add_sample("ggh", "ggH", "H2Mu_gg_BDTG_UCSD.root", "tree", False, False, ROOT.kRed)
bdt_ucsd_hiStat.add_sample("vbf", "VBF", "H2Mu_VBF_BDTG_UCSD.root", "tree", False, False, ROOT.kViolet-1)
bdt_ucsd_hiStat.add_sample("data", "Data 2017 (40.5/fb)", "SingleMu_2017*_BDTG_UCSD.root", "tree", True, False, ROOT.kBlack)
bdt_ucsd_hiStat.set_lumi(40490.712)
bdt_ucsd_hiStat_roc_graph = bdt_ucsd_hiStat.plot_roc("MVA", 500, -1, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
bdt_ucsd_hiStat_roc = a.RocCurve(bdt_ucsd_hiStat_roc_graph, "bdt_ucsd_hiStat", "BDT UCSD hiStat", ROOT.kOrange-3)
roc_to_compare.append(bdt_ucsd_hiStat_roc)
# bdt_ucsd_hiStat_roc_tmva = a.roc_from_tmva(bdt_ucsd_hiStat, "BDT_UCSD_hiStat", "TMVA.root", "dataset/Method_BDTG_UCSD/BDTG_UCSD/MVA_BDTG_UCSD_rejBvsS", ROOT.kViolet, 2)
# roc_to_compare.append(bdt_ucsd_hiStat_roc_tmva)


# Option 5
# bdt_ucsd_hiStat_ebe = a.add_mva_source("BDT_UCSD_hiStat_ebe", "BDT_UCSD_hiStat_ebe", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-08_11-36-49/TMVA/")
# bdt_ucsd_hiStat_ebe_roc_tmva = a.roc_from_tmva(bdt_ucsd_hiStat_ebe, "BDT_UCSD_hiStat_ebe", "TMVA.root", "dataset/Method_BDTG_UCSD/BDTG_UCSD/MVA_BDTG_UCSD_rejBvsS", ROOT.kYellow, 2)
# roc_to_compare.append(bdt_ucsd_hiStat_ebe_roc_tmva)


a.compare_roc_curves(roc_to_compare)


# Binary test

# dnn_test = a.add_mva_source("DNN_test", "DNN_test", "/tmp/dkondrat/ML_output/Run_2019-04-07_18-53-10/Keras_multi/model_50_D2_25_D2_25_D2/root/")

# dnn_test.add_sample("bkg", "Background", "output_t*root", "tree_bkg", False, True, ROOT.kYellow, False)
# dnn_test.add_sample("sig", "Signal", "output_t*root", "tree_sig", False, False, ROOT.kRed, False)
# dnn_test.set_lumi(4823)
# dnn_test_roc_graph = dnn_test.plot_roc("sig_prediction", 500, 0, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
# dnn_test_roc = a.RocCurve(dnn_test_roc_graph, "dnn_test", "DNN test", ROOT.kRed)


# score = "ggH_prediction+VBF_prediction"
# nBins = 50
# gr1 = a.plot_width_vs_score(score, dnn_multi_hiStat_ebe, "dnn_multi_hiStat_ebe", "DNN hiStat w/ ebe res. ggH", nBins, ROOT.kRed, 20, process = "ggH")
# gr2 = a.plot_width_vs_score(score, dnn_multi_hiStat_ebe, "dnn_multi_hiStat_ebe", "DNN hiStat w/ ebe res. VBF", nBins, ROOT.kBlue, 20, process = "VBF")

# gr3 = a.plot_width_vs_score(score, dnn_multi, "dnn_multi", "DNN loStat w/o ebe res. ggH", nBins, ROOT.kBlack, 20, process = "ggH")
# gr4 = a.plot_width_vs_score(score, dnn_multi, "dnn_multi", "DNN loStat w/o ebe res. VBF", nBins, ROOT.kGreen, 20, process = "VBF")

# canvas = ROOT.TCanvas("c_wvss", "c_wvss", 800, 800)
# canvas.cd()
# legend = ROOT.TLegend(0.7, 0.8, 0.895, 0.895)
# gr1.Draw("ple1")
# gr2.Draw("ple1same")
# gr3.Draw("ple1same")
# gr4.Draw("ple1same")
# legend.AddEntry(gr1, gr1.GetTitle(), "pe1")
# legend.AddEntry(gr2, gr2.GetTitle(), "pe1")
# legend.AddEntry(gr3, gr3.GetTitle(), "pe1")
# legend.AddEntry(gr4, gr4.GetTitle(), "pe1")
# legend.Draw()
# canvas.SaveAs("%s/width_vs_score.png"%(a.out_path))
# canvas.SaveAs("%s/width_vs_score.root"%(a.out_path))