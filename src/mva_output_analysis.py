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
            legend = ROOT.TLegend(0.6, 0.7, 0.895, 0.895)

            for smp in self.samples:
                trees[smp.name] = ROOT.TChain(smp.treename)
                trees[smp.name].Add("%s/%s"%(self.path, smp.filename))
                print "Tree for %s added with %i entries"%(smp.name, trees[smp.name].GetEntries())
                hist_name = "hist_%s_%s_%s"%(self.name, smp.name, label)
                hist = ROOT.TH1D(hist_name, hist_name, nBins, xmin, xmax)

                dummy = ROOT.TCanvas("dummy_"+hist_name, "dummy_"+hist_name, 100, 100)
                dummy.cd()
                if smp.isData:
                    trees[smp.name].Draw("%s>>%s"%(var_name, hist_name), smp.additional_cut)
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
                print "Hist %s: entries = %i, integral=%f"%(hist_name, hist.GetEntries(), hist.Integral())
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
                    # self.mc_stack.SetMaximum(500)
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

    # def plot_mass_in_slices(self, score, nBinx, xmin, xmax):
    #     idx = self.score_cuts_from_wp(score, working_points, nBins, xmin, xmax)
    #     idx = [xmin]+idx+[xmax]
    #     for i in len(idx)-1:
    #         cut_lo = idx[i]
    #         cut_hi = idx[i+1]

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
        legend = ROOT.TLegend(0.6, 0.7, 0.895, 0.895)
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

# # Option 0
# bdt_uf = a.add_mva_source("BDT_UF", "BDT_UF", "/home/dkondra/tmp/BDTG_UF/")
# bdt_uf.add_sample("tt", "ttbar", "tt_ll_POW_BDTG_UF_v1.root", "tree", False, True, ROOT.kYellow)
# bdt_uf.add_sample("dy", "Drell-Yan", "ZJets_aMC_BDTG_UF_v1.root", "tree", False, True, ROOT.kOrange-3)
# bdt_uf.add_sample("ggh", "ggH", "H2Mu_gg_BDTG_UF_v1.root", "tree", False, False, ROOT.kRed)
# bdt_uf.add_sample("vbf", "VBF", "H2Mu_VBF_BDTG_UF_v1.root", "tree", False, False, ROOT.kViolet-1)
# bdt_uf.add_sample("data", "Data 2017 (40.5/fb)", "SingleMu_2017*_BDTG_UF_v1.root", "tree", True, False, ROOT.kBlack)
# bdt_uf.set_lumi(40490.712)
# bdt_uf_roc_graph = bdt_uf.plot_roc("MVA", 500, -1, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
# # bdt_uf_roc = a.RocCurve(bdt_uf_roc_graph, "bdt_uf", "BDT UF loStat", ROOT.kBlack)
# bdt_uf_roc = a.RocCurve(bdt_uf_roc_graph, "bdt_uf", "BDT UF loStat", ROOT.kBlue)
# # roc_to_compare.append(bdt_uf_roc)
# # bdt_uf_roc_tmva = a.roc_from_tmva(bdt_uf, "BDT_UF", "TMVA.root", "dataset/Method_BDTG_UF_v1/BDTG_UF_v1/MVA_BDTG_UF_v1_rejBvsS", ROOT.kBlack, 2)
# # roc_to_compare.append(bdt_uf_roc_tmva)



# # Option 1
# bdt_uf_hiStat = a.add_mva_source("BDT_UF_hiStat", "BDT_UF_hiStat", "/home/dkondra/tmp/BDTG_UF_hiStat/")
# bdt_uf_hiStat.add_sample("tt", "ttbar", "tt_ll_POW_BDTG_UF_v1.root", "tree", False, True, ROOT.kYellow)
# # bdt_uf_hiStat.add_sample("dy", "Drell-Yan", "ZJets_aMC_BDTG_UF_v1.root", "tree", False, True, ROOT.kOrange-3)
# bdt_uf_hiStat.add_sample("dy", "Drell-Yan", "ZJets_aMC/*.root", "tree", False, True, ROOT.kOrange-3)
# bdt_uf_hiStat.add_sample("ggh", "ggH", "H2Mu_gg_BDTG_UF_v1.root", "tree", False, False, ROOT.kRed)
# bdt_uf_hiStat.add_sample("vbf", "VBF", "H2Mu_VBF_BDTG_UF_v1.root", "tree", False, False, ROOT.kViolet-1)
# bdt_uf_hiStat.add_sample("data", "Data 2017 (40.5/fb)", "SingleMu_2017*_BDTG_UF_v1.root", "tree", True, False, ROOT.kBlack)
# bdt_uf_hiStat.set_lumi(40490.712)
# bdt_uf_hiStat_roc_graph = bdt_uf_hiStat.plot_roc("MVA", 500, -1, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
# bdt_uf_hiStat_roc = a.RocCurve(bdt_uf_hiStat_roc_graph, "bdt_uf_hiStat", "BDT UF hiStat", ROOT.kBlack)
# # roc_to_compare.append(bdt_uf_hiStat_roc)
# # bdt_uf_hiStat_roc_tmva = a.roc_from_tmva(bdt_uf_hiStat, "BDT_UF_hiStat", "TMVA.root", "dataset/Method_BDTG_UF_v1/BDTG_UF_v1/MVA_BDTG_UF_v1_rejBvsS", ROOT.kBlue, 2)
# # roc_to_compare.append(bdt_uf_hiStat_roc_tmva)

# # Option 2
# # bdt_uf_hiStat_ebe = a.add_mva_source("BDT_UF_hiStat_ebe", "BDT_UF_hiStat_ebe", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-08_11-36-30/TMVA/")
# # bdt_uf_hiStat_ebe_roc_tmva = a.roc_from_tmva(bdt_uf_hiStat_ebe, "BDT_UF_hiStat_ebe", "TMVA.root", "dataset/Method_BDTG_UF_v1/BDTG_UF_v1/MVA_BDTG_UF_v1_rejBvsS", ROOT.kRed, 2)
# # roc_to_compare.append(bdt_uf_hiStat_ebe_roc_tmva)

# # Option 3
# bdt_ucsd = a.add_mva_source("BDT_UCSD", "BDT_UCSD", "/home/dkondra/tmp/BDTG_UCSD/")
# bdt_ucsd.add_sample("tt", "ttbar", "tt_ll_POW_BDTG_UCSD.root", "tree", False, True, ROOT.kYellow)
# bdt_ucsd.add_sample("dy", "Drell-Yan", "ZJets_aMC_BDTG_UCSD.root", "tree", False, True, ROOT.kOrange-3)
# bdt_ucsd.add_sample("ggh", "ggH", "H2Mu_gg_BDTG_UCSD.root", "tree", False, False, ROOT.kRed)
# bdt_ucsd.add_sample("vbf", "VBF", "H2Mu_VBF_BDTG_UCSD.root", "tree", False, False, ROOT.kViolet-1)
# bdt_ucsd.add_sample("data", "Data 2017 (40.5/fb)", "SingleMu_2017*_BDTG_UCSD.root", "tree", True, False, ROOT.kBlack)
# bdt_ucsd.set_lumi(40490.712)
# bdt_ucsd_roc_graph = bdt_ucsd.plot_roc("MVA", 500, -1, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
# bdt_ucsd_roc = a.RocCurve(bdt_ucsd_roc_graph, "bdt_ucsd", "BDT UCSD loStat", ROOT.kBlue)
# roc_to_compare.append(bdt_ucsd_roc)

# bdt_ucsd_roc_tmva = a.roc_from_tmva(bdt_ucsd, "BDT_UCSD", "TMVA.root", "dataset/Method_BDTG_UCSD/BDTG_UCSD/MVA_BDTG_UCSD_rejBvsS", ROOT.kGreen, 2)
# roc_to_compare.append(bdt_ucsd_roc_tmva)

# # Option 4
# bdt_ucsd_hiStat = a.add_mva_source("BDT_UCSD_hiStat", "BDT_UCSD_hiStat", "/home/dkondra/tmp/BDTG_UCSD_hiStat/")
# bdt_ucsd_hiStat.add_sample("tt", "ttbar", "tt_ll_POW_BDTG_UCSD.root", "tree", False, True, ROOT.kYellow)
# # bdt_ucsd_hiStat.add_sample("dy", "Drell-Yan", "ZJets_aMC_BDTG_UCSD.root", "tree", False, True, ROOT.kOrange-3)
# bdt_ucsd_hiStat.add_sample("dy", "Drell-Yan", "ZJets_aMC/*.root", "tree", False, True, ROOT.kOrange-3)
# bdt_ucsd_hiStat.add_sample("ggh", "ggH", "H2Mu_gg_BDTG_UCSD.root", "tree", False, False, ROOT.kRed)
# bdt_ucsd_hiStat.add_sample("vbf", "VBF", "H2Mu_VBF_BDTG_UCSD.root", "tree", False, False, ROOT.kViolet-1)
# bdt_ucsd_hiStat.add_sample("data", "Data 2017 (40.5/fb)", "SingleMu_2017*_BDTG_UCSD.root", "tree", True, False, ROOT.kBlack)
# bdt_ucsd_hiStat.set_lumi(40490.712)
# bdt_ucsd_hiStat_roc_graph = bdt_ucsd_hiStat.plot_roc("MVA", 500, -1, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
# bdt_ucsd_hiStat_roc = a.RocCurve(bdt_ucsd_hiStat_roc_graph, "bdt_ucsd_hiStat", "BDT UCSD hiStat", ROOT.kOrange-3)
# roc_to_compare.append(bdt_ucsd_hiStat_roc)
# # bdt_ucsd_hiStat_roc_tmva = a.roc_from_tmva(bdt_ucsd_hiStat, "BDT_UCSD_hiStat", "TMVA.root", "dataset/Method_BDTG_UCSD/BDTG_UCSD/MVA_BDTG_UCSD_rejBvsS", ROOT.kViolet, 2)
# # roc_to_compare.append(bdt_ucsd_hiStat_roc_tmva)


# # Option 5
# # bdt_ucsd_hiStat_ebe = a.add_mva_source("BDT_UCSD_hiStat_ebe", "BDT_UCSD_hiStat_ebe", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-08_11-36-49/TMVA/")
# # bdt_ucsd_hiStat_ebe_roc_tmva = a.roc_from_tmva(bdt_ucsd_hiStat_ebe, "BDT_UCSD_hiStat_ebe", "TMVA.root", "dataset/Method_BDTG_UCSD/BDTG_UCSD/MVA_BDTG_UCSD_rejBvsS", ROOT.kYellow, 2)
# # roc_to_compare.append(bdt_ucsd_hiStat_ebe_roc_tmva)


# # Option 6
# dnn_multi = a.add_mva_source("DNN_Multi", "DNN_Multi", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-08_11-37-05//Keras_multi/model_50_D2_25_D2_25_D2/root/")
# dnn_multi.add_sample("tt", "ttbar", "output_t*root", "tree_tt_ll_POW", False, True, ROOT.kYellow, True)
# dnn_multi.add_sample("dy", "Drell-Yan", "output_t*root", "tree_ZJets_aMC", False, True, ROOT.kOrange-3, True)
# dnn_multi.add_sample("ggh", "ggH", "output_t*root", "tree_H2Mu_gg", False, False, ROOT.kRed, True)
# dnn_multi.add_sample("vbf", "VBF", "output_t*root", "tree_H2Mu_VBF", False, False, ROOT.kViolet-1, True)
# dnn_multi.add_sample("data", "Data 2017 (40.5/fb)", "output_Data.root", "tree_Data", True, False, ROOT.kBlack)
# dnn_multi.set_lumi(40490.712)
# dnn_multi_roc_graph = dnn_multi.plot_roc("ggH_prediction+VBF_prediction", 500, 0, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95], label="my_score")
# # dnn_multi_roc_graph = dnn_multi.plot_roc("(ggH_prediction+VBF_prediction+(1-DY_prediction)+(1-ttbar_prediction))", 500, 1, 3, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95], label="adish_score")
# dnn_multi_roc = a.RocCurve(dnn_multi_roc_graph, "dnn_multi", "DNN_Multi", ROOT.kGreen)
# roc_to_compare.append(dnn_multi_roc)


# # dnn_cuts_multi = [0.63, 0.788, 0.832, 0.844, 0.908, 0.940]
# # score_multi = "ggH_prediction+VBF_prediction"
# # dnn_multi_1 = a.add_mva_source("DNN_multi_1", "DNN_multi_1", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-08_11-37-05//Keras_multi/model_50_D2_25_D2_25_D2/root/")
# # dnn_multi_1.add_sample("cat0", "cat0", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kBlack, True, "((%s)>%f)&((%s)<%f)"%(score_multi, 0.0, score_multi, dnn_cuts_multi[0]))
# # dnn_multi_1.add_sample("cat1", "cat1", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kViolet, True, "((%s)>%f)&((%s)<%f)"%(score_multi, dnn_cuts_multi[0], score_multi, dnn_cuts_multi[1]))
# # dnn_multi_1.add_sample("cat2", "cat2", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kBlue, True, "((%s)>%f)&((%s)<%f)"%(score_multi, dnn_cuts_multi[1], score_multi, dnn_cuts_multi[2]))
# # dnn_multi_1.add_sample("cat3", "cat3", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kGreen, True, "((%s)>%f)&((%s)<%f)"%(score_multi, dnn_cuts_multi[2], score_multi, dnn_cuts_multi[3]))
# # dnn_multi_1.add_sample("cat4", "cat4", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kYellow, True, "((%s)>%f)&((%s)<%f)"%(score_multi, dnn_cuts_multi[3], score_multi, dnn_cuts_multi[4]))
# # dnn_multi_1.add_sample("cat5", "cat5", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kOrange, True, "((%s)>%f)&((%s)<%f)"%(score_multi, dnn_cuts_multi[4], score_multi, dnn_cuts_multi[5]))
# # dnn_multi_1.add_sample("cat6", "cat6", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kRed, True, "((%s)>%f)&((%s)<%f)"%(score_multi, dnn_cuts_multi[5], score_multi, 1))
# # # dnn_multi_1.add_sample("sig", "sig", "output_t*root", "tree_sig", False, False, ROOT.kRed, True)
# # dnn_multi_1.set_lumi(40490.712)
# # dnn_multi_1.plot("mass", 40, 110, 150, label="shapes", draw=True, shapes=True)


# # Option 7
# dnn_multi_hiStat = a.add_mva_source("DNN_Multi_hiStat", "DNN_Multi_hiStat", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-08_11-51-21//Keras_multi/model_50_D2_25_D2_25_D2/root/")
# dnn_multi_hiStat.add_sample("tt", "ttbar", "output_t*root", "tree_tt_ll_POW", False, True, ROOT.kYellow, True)
# dnn_multi_hiStat.add_sample("dy", "Drell-Yan", "output_t*root", "tree_ZJets_aMC", False, True, ROOT.kOrange-3, True)
# dnn_multi_hiStat.add_sample("ggh", "ggH", "output_t*root", "tree_H2Mu_gg", False, False, ROOT.kRed, True)
# dnn_multi_hiStat.add_sample("vbf", "VBF", "output_t*root", "tree_H2Mu_VBF", False, False, ROOT.kViolet-1, True)
# dnn_multi_hiStat.add_sample("data", "Data 2017 (40.5/fb)", "output_Data.root", "tree_Data", True, False, ROOT.kBlack)
# dnn_multi_hiStat.set_lumi(40490.712)
# dnn_multi_hiStat_roc_graph = dnn_multi_hiStat.plot_roc("ggH_prediction+VBF_prediction", 500, 0, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
# dnn_multi_hiStat_roc = a.RocCurve(dnn_multi_hiStat_roc_graph, "dnn_multi_hiStat", "DNN_Multi_hiStat", ROOT.kViolet)
# roc_to_compare.append(dnn_multi_hiStat_roc)

# # dnn_cuts_multi_hiStat = [0.046, 0.094, 0.116, 0.134, 0.262, 0.388]
# # score_multi_hiStat = "ggH_prediction+VBF_prediction"
# # dnn_multi_hiStat_1 = a.add_mva_source("DNN_multi_hiStat_1", "DNN_multi_hiStat_1", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-08_11-51-21//Keras_multi/model_50_D2_25_D2_25_D2/root/")
# # dnn_multi_hiStat_1.add_sample("cat0", "cat0", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kBlack, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat, 0.0, score_multi_hiStat, dnn_cuts_multi_hiStat[0]))
# # dnn_multi_hiStat_1.add_sample("cat1", "cat1", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kViolet, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat, dnn_cuts_multi_hiStat[0], score_multi_hiStat, dnn_cuts_multi_hiStat[1]))
# # dnn_multi_hiStat_1.add_sample("cat2", "cat2", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kBlue, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat, dnn_cuts_multi_hiStat[1], score_multi_hiStat, dnn_cuts_multi_hiStat[2]))
# # dnn_multi_hiStat_1.add_sample("cat3", "cat3", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kGreen, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat, dnn_cuts_multi_hiStat[2], score_multi_hiStat, dnn_cuts_multi_hiStat[3]))
# # dnn_multi_hiStat_1.add_sample("cat4", "cat4", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kYellow, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat, dnn_cuts_multi_hiStat[3], score_multi_hiStat, dnn_cuts_multi_hiStat[4]))
# # dnn_multi_hiStat_1.add_sample("cat5", "cat5", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kOrange, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat, dnn_cuts_multi_hiStat[4], score_multi_hiStat, dnn_cuts_multi_hiStat[5]))
# # dnn_multi_hiStat_1.add_sample("cat6", "cat6", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kRed, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat, dnn_cuts_multi_hiStat[5], score_multi_hiStat, 1))
# # # dnn_multi_hiStat_1.add_sample("sig", "sig", "output_t*root", "tree_sig", False, False, ROOT.kRed, True)
# # dnn_multi_hiStat_1.set_lumi(40490.712)
# # dnn_multi_hiStat_1.plot("mass", 40, 110, 150, label="shapes", draw=True, shapes=True)

# # Option 8
# dnn_multi_hiStat_ebe = a.add_mva_source("DNN_Multi_hiStat_ebe", "DNN_Multi_hiStat_ebe", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-08_11-37-09//Keras_multi/model_50_D2_25_D2_25_D2/root/")
# dnn_multi_hiStat_ebe.add_sample("tt", "ttbar", "output_t*root", "tree_tt_ll_POW", False, True, ROOT.kYellow, True)
# dnn_multi_hiStat_ebe.add_sample("dy", "Drell-Yan", "output_t*root", "tree_ZJets_aMC", False, True, ROOT.kOrange-3, True)
# dnn_multi_hiStat_ebe.add_sample("ggh", "ggH", "output_t*root", "tree_H2Mu_gg", False, False, ROOT.kRed, True)
# dnn_multi_hiStat_ebe.add_sample("vbf", "VBF", "output_t*root", "tree_H2Mu_VBF", False, False, ROOT.kViolet-1, True)
# dnn_multi_hiStat_ebe.add_sample("data", "Data 2017 (40.5/fb)", "output_Data.root", "tree_Data", True, False, ROOT.kBlack)
# dnn_multi_hiStat_ebe.set_lumi(40490.712)
# dnn_multi_hiStat_ebe_roc_graph = dnn_multi_hiStat_ebe.plot_roc("ggH_prediction+VBF_prediction", 500, 0, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
# dnn_multi_hiStat_ebe_roc = a.RocCurve(dnn_multi_hiStat_ebe_roc_graph, "dnn_multi_hiStat_ebe", "DNN_Multi_hiStat_ebe", ROOT.kGreen)
# # roc_to_compare.append(dnn_multi_hiStat_ebe_roc)

# # Option 9
# dnn_binary_hiStat = a.add_mva_source("DNN_Binary_hiStat", "DNN_Binary_hiStat", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-08_11-37-12//Keras_multi/model_50_D2_25_D2_25_D2/root/")
# dnn_binary_hiStat.add_sample("bkg", "bkg", "output_t*root", "tree_bkg", False, True, ROOT.kYellow, True)
# dnn_binary_hiStat.add_sample("sig", "sig", "output_t*root", "tree_sig", False, False, ROOT.kRed, True)
# dnn_binary_hiStat.add_sample("data", "Data 2017 (40.5/fb)", "output_Data.root", "tree_Data", True, False, ROOT.kBlack)
# dnn_binary_hiStat.set_lumi(40490.712)
# dnn_binary_hiStat_roc_graph = dnn_binary_hiStat.plot_roc("sig_prediction", 500, 0, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
# dnn_binary_hiStat_roc = a.RocCurve(dnn_binary_hiStat_roc_graph, "dnn_binary_hiStat", "DNN_Binary_hiStat", ROOT.kGreen)
# # roc_to_compare.append(dnn_binary_hiStat_roc)

# # Option 9.1 - plot only shapes
# dnn_cuts = [0.054, 0.112, 0.17, 0.248, 0.436, 0.56]
# dnn_binary_hiStat_1 = a.add_mva_source("DNN_Binary_hiStat_1", "DNN_Binary_hiStat_1", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-08_11-37-12//Keras_multi/model_50_D2_25_D2_25_D2/root/")
# dnn_binary_hiStat_1.add_sample("cat0", "cat0", "output_t*root", "tree_bkg", False, False, ROOT.kBlack, True, "(sig_prediction>%f)&(sig_prediction<%f)"%(0.0, dnn_cuts[0]))
# dnn_binary_hiStat_1.add_sample("cat1", "cat1", "output_t*root", "tree_bkg", False, False, ROOT.kViolet, True, "(sig_prediction>%f)&(sig_prediction<%f)"%(dnn_cuts[0], dnn_cuts[1]))
# dnn_binary_hiStat_1.add_sample("cat2", "cat2", "output_t*root", "tree_bkg", False, False, ROOT.kBlue, True, "(sig_prediction>%f)&(sig_prediction<%f)"%(dnn_cuts[1], dnn_cuts[2]))
# dnn_binary_hiStat_1.add_sample("cat3", "cat3", "output_t*root", "tree_bkg", False, False, ROOT.kGreen, True, "(sig_prediction>%f)&(sig_prediction<%f)"%(dnn_cuts[2], dnn_cuts[3]))
# dnn_binary_hiStat_1.add_sample("cat4", "cat4", "output_t*root", "tree_bkg", False, False, ROOT.kYellow, True, "(sig_prediction>%f)&(sig_prediction<%f)"%(dnn_cuts[3], dnn_cuts[4]))
# dnn_binary_hiStat_1.add_sample("cat5", "cat5", "output_t*root", "tree_bkg", False, False, ROOT.kOrange, True, "(sig_prediction>%f)&(sig_prediction<%f)"%(dnn_cuts[4], dnn_cuts[5]))
# dnn_binary_hiStat_1.add_sample("cat6", "cat6", "output_t*root", "tree_bkg", False, False, ROOT.kRed, True, "(sig_prediction>%f)&(sig_prediction<%f)"%(dnn_cuts[5], 1))
# # dnn_binary_hiStat_1.add_sample("sig", "sig", "output_t*root", "tree_sig", False, False, ROOT.kRed, True)
# dnn_binary_hiStat_1.set_lumi(40490.712)
# dnn_binary_hiStat_1.plot("mass", 200, 110, 150, label="shapes", draw=True, shapes=True)

# # Option 10
# dnn_binary_hiStat_ebe = a.add_mva_source("DNN_Binary_hiStat_ebe", "DNN_Binary_hiStat_ebe", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-08_11-37-16//Keras_multi/model_50_D2_25_D2_25_D2/root/")
# dnn_binary_hiStat_ebe.add_sample("bkg", "bkg", "output_t*root", "tree_bkg", False, True, ROOT.kYellow, True)
# dnn_binary_hiStat_ebe.add_sample("sig", "sig", "output_t*root", "tree_sig", False, False, ROOT.kRed, True)
# dnn_binary_hiStat_ebe.add_sample("data", "Data 2017 (40.5/fb)", "output_Data.root", "tree_Data", True, False, ROOT.kBlack)
# dnn_binary_hiStat_ebe.set_lumi(40490.712)
# dnn_binary_hiStat_ebe_roc_graph = dnn_binary_hiStat_ebe.plot_roc("sig_prediction", 500, 0, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
# dnn_binary_hiStat_ebe_roc = a.RocCurve(dnn_binary_hiStat_ebe_roc_graph, "dnn_binary_hiStat_ebe", "DNN_Binary_hiStat_ebe", ROOT.kYellow)
# # roc_to_compare.append(dnn_binary_hiStat_ebe_roc)

# # Option 11
# dnn_binary = a.add_mva_source("DNN_Binary", "DNN_Binary", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-09_16-22-10//Keras_multi/model_50_D2_25_D2_25_D2/root/")
# dnn_binary.add_sample("bkg", "bkg", "output_t*root", "tree_bkg", False, True, ROOT.kYellow, True)
# dnn_binary.add_sample("sig", "sig", "output_t*root", "tree_sig", False, False, ROOT.kRed, True)
# dnn_binary.add_sample("data", "Data 2017 (40.5/fb)", "output_Data.root", "tree_Data", True, False, ROOT.kBlack)
# dnn_binary.set_lumi(40490.712)
# dnn_binary_roc_graph = dnn_binary.plot_roc("sig_prediction", 500, 0, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
# dnn_binary_roc = a.RocCurve(dnn_binary_roc_graph, "dnn_binary", "DNN_Binary", ROOT.kBlue)


# # roc_to_compare.append(dnn_binary_roc)

# # Option 12
# dnn_multi_hiStat_m120To130 = a.add_mva_source("DNN_Multi_hiStat_m120To130", "DNN_Multi_hiStat_m120To130", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-11_14-31-12//Keras_multi/model_50_D2_25_D2_25_D2/root/")
# dnn_multi_hiStat_m120To130.add_sample("tt", "ttbar", "output_t*root", "tree_tt_ll_POW", False, True, ROOT.kYellow, True)
# dnn_multi_hiStat_m120To130.add_sample("dy", "Drell-Yan", "output_t*root", "tree_ZJets_aMC", False, True, ROOT.kOrange-3, True)
# dnn_multi_hiStat_m120To130.add_sample("ggh", "ggH", "output_t*root", "tree_H2Mu_gg", False, False, ROOT.kRed, True)
# dnn_multi_hiStat_m120To130.add_sample("vbf", "VBF", "output_t*root", "tree_H2Mu_VBF", False, False, ROOT.kViolet-1, True)
# dnn_multi_hiStat_m120To130.add_sample("data", "Data 2017 (40.5/fb)", "output_Data.root", "tree_Data", True, False, ROOT.kBlack)
# dnn_multi_hiStat_m120To130.set_lumi(40490.712)
# dnn_multi_hiStat_m120To130_roc_graph = dnn_multi_hiStat_m120To130.plot_roc("ggH_prediction+VBF_prediction", 500, 0, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
# dnn_multi_hiStat_m120To130_roc = a.RocCurve(dnn_multi_hiStat_m120To130_roc_graph, "dnn_multi_hiStat_m120To130", "DNN_Multi_hiStat_m120To130", ROOT.kViolet, 2)
# roc_to_compare.append(dnn_multi_hiStat_m120To130_roc)


# dnn_cuts_multi_hiStat = [0.152, 0.262, 0.324, 0.354, 0.570, 0.72]
# score_multi_hiStat = "ggH_prediction+VBF_prediction"
# dnn_multi_hiStat_m120To130_1 = a.add_mva_source("DNN_multi_hiStat_m120To130_1", "DNN_multi_hiStat_m120To130_1", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-08_11-51-21//Keras_multi/model_50_D2_25_D2_25_D2/root/")
# dnn_multi_hiStat_m120To130_1.add_sample("cat0", "cat0", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kBlack, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat, 0.0, score_multi_hiStat, dnn_cuts_multi_hiStat[0]))
# dnn_multi_hiStat_m120To130_1.add_sample("cat1", "cat1", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kViolet, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat, dnn_cuts_multi_hiStat[0], score_multi_hiStat, dnn_cuts_multi_hiStat[1]))
# dnn_multi_hiStat_m120To130_1.add_sample("cat2", "cat2", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kBlue, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat, dnn_cuts_multi_hiStat[1], score_multi_hiStat, dnn_cuts_multi_hiStat[2]))
# dnn_multi_hiStat_m120To130_1.add_sample("cat3", "cat3", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kGreen, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat, dnn_cuts_multi_hiStat[2], score_multi_hiStat, dnn_cuts_multi_hiStat[3]))
# dnn_multi_hiStat_m120To130_1.add_sample("cat4", "cat4", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kYellow, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat, dnn_cuts_multi_hiStat[3], score_multi_hiStat, dnn_cuts_multi_hiStat[4]))
# dnn_multi_hiStat_m120To130_1.add_sample("cat5", "cat5", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kOrange, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat, dnn_cuts_multi_hiStat[4], score_multi_hiStat, dnn_cuts_multi_hiStat[5]))
# dnn_multi_hiStat_m120To130_1.add_sample("cat6", "cat6", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kRed, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat, dnn_cuts_multi_hiStat[5], score_multi_hiStat, 1))
# # dnn_multi_hiStat_m120To130_1.add_sample("sig", "sig", "output_t*root", "tree_sig", False, False, ROOT.kRed, True)
# dnn_multi_hiStat_m120To130_1.set_lumi(40490.712)
# dnn_multi_hiStat_m120To130_1.plot("mass", 10, 120, 130, label="shapes", draw=True, shapes=True)



# # Option 13
# dnn_multi_hiStat_m120To130_repeated = a.add_mva_source("DNN_Multi_hiStat_m120To130_repeated", "DNN_Multi_hiStat_m120To130_repeated", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-15_18-13-46//Keras_multi/model_50_D2_25_D2_25_D2/root/")
# dnn_multi_hiStat_m120To130_repeated.add_sample("tt", "ttbar", "output_t*root", "tree_tt_ll_POW", False, True, ROOT.kYellow, True, "1./15.")
# dnn_multi_hiStat_m120To130_repeated.add_sample("dy", "Drell-Yan", "output_t*root", "tree_ZJets_aMC", False, True, ROOT.kOrange-3, True)
# dnn_multi_hiStat_m120To130_repeated.add_sample("ggh", "ggH", "output_t*root", "tree_H2Mu_gg", False, False, ROOT.kRed, True, "1./4.")
# dnn_multi_hiStat_m120To130_repeated.add_sample("vbf", "VBF", "output_t*root", "tree_H2Mu_VBF", False, False, ROOT.kViolet-1, True, "1./6.")
# dnn_multi_hiStat_m120To130_repeated.add_sample("data", "Data 2017 (40.5/fb)", "output_Data.root", "tree_Data", True, False, ROOT.kBlack)
# dnn_multi_hiStat_m120To130_repeated.set_lumi(40490.712)
# dnn_multi_hiStat_m120To130_repeated_roc_graph = dnn_multi_hiStat_m120To130_repeated.plot_roc("ggH_prediction+VBF_prediction", 500, 0, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
# dnn_multi_hiStat_m120To130_repeated_roc = a.RocCurve(dnn_multi_hiStat_m120To130_repeated_roc_graph, "dnn_multi_hiStat_m120To130_repeated", "DNN_Multi_hiStat_m120To130_repeated", ROOT.kCyan, 2)
# roc_to_compare.append(dnn_multi_hiStat_m120To130_repeated_roc)


# dnn_cuts_multi_hiStat = [0.152, 0.262, 0.324, 0.354, 0.570, 0.72]
# score_multi_hiStat = "ggH_prediction+VBF_prediction"
# dnn_multi_hiStat_m120To130_repeated_1 = a.add_mva_source("DNN_multi_hiStat_m120To130_repeated_1", "DNN_multi_hiStat_m120To130_repeated_1", "/scratch/gilbreth/dkondra/ML_output//Run_2019-04-15_18-13-46//Keras_multi/model_50_D2_25_D2_25_D2/root/")
# dnn_multi_hiStat_m120To130_repeated_1.add_sample("cat0", "cat0", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kBlack, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat, 0.0, score_multi_hiStat, dnn_cuts_multi_hiStat[0]))
# dnn_multi_hiStat_m120To130_repeated_1.add_sample("cat1", "cat1", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kViolet, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat, dnn_cuts_multi_hiStat[0], score_multi_hiStat, dnn_cuts_multi_hiStat[1]))
# dnn_multi_hiStat_m120To130_repeated_1.add_sample("cat2", "cat2", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kBlue, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat, dnn_cuts_multi_hiStat[1], score_multi_hiStat, dnn_cuts_multi_hiStat[2]))
# dnn_multi_hiStat_m120To130_repeated_1.add_sample("cat3", "cat3", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kGreen, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat, dnn_cuts_multi_hiStat[2], score_multi_hiStat, dnn_cuts_multi_hiStat[3]))
# dnn_multi_hiStat_m120To130_repeated_1.add_sample("cat4", "cat4", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kYellow, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat, dnn_cuts_multi_hiStat[3], score_multi_hiStat, dnn_cuts_multi_hiStat[4]))
# dnn_multi_hiStat_m120To130_repeated_1.add_sample("cat5", "cat5", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kOrange, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat, dnn_cuts_multi_hiStat[4], score_multi_hiStat, dnn_cuts_multi_hiStat[5]))
# dnn_multi_hiStat_m120To130_repeated_1.add_sample("cat6", "cat6", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kRed, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat, dnn_cuts_multi_hiStat[5], score_multi_hiStat, 1))
# # dnn_multi_hiStat_m120To130_repeated_1.add_sample("sig", "sig", "output_t*root", "tree_sig", False, False, ROOT.kRed, True)
# dnn_multi_hiStat_m120To130_repeated_1.set_lumi(40490.712)
# dnn_multi_hiStat_m120To130_repeated_1.plot("mass", 10, 120, 130, label="shapes", draw=True, shapes=True)



# Option 1.1
# dnn_multi_hiStat_m110to150 = a.add_mva_source("DNN_Multi_hiStat_m110to150", "DNN_Multi_hiStat_m110to150", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-21_11-06-33//Keras_multi/model_50_D2_25_D2_25_D2/root/")
# dnn_multi_hiStat_m110to150.add_sample("tt", "ttbar", "output_t*root", "tree_tt_ll_POW", False, True, ROOT.kYellow, True)
# dnn_multi_hiStat_m110to150.add_sample("dy", "Drell-Yan", "output_t*root", "tree_ZJets_aMC", False, True, ROOT.kOrange-3, True)
# dnn_multi_hiStat_m110to150.add_sample("ggh", "ggH", "output_t*root", "tree_H2Mu_gg", False, False, ROOT.kRed, True)
# dnn_multi_hiStat_m110to150.add_sample("vbf", "VBF", "output_t*root", "tree_H2Mu_VBF", False, False, ROOT.kViolet-1, True)
# dnn_multi_hiStat_m110to150.add_sample("data", "Data 2017 (40.5/fb)", "output_Data.root", "tree_Data", True, False, ROOT.kBlack, False)
# dnn_multi_hiStat_m110to150.set_lumi(41394.221)
# dnn_multi_hiStat_m110to150_roc_graph = dnn_multi_hiStat_m110to150.plot_roc("ggH_prediction+VBF_prediction+(1-DY_prediction)+(1-ttbar_prediction)", 500, 1, 3, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
# dnn_multi_hiStat_m110to150_roc = a.RocCurve(dnn_multi_hiStat_m110to150_roc_graph, "dnn_multi_hiStat_m110to150", "DNN_Multi_hiStat_m110to150", ROOT.kBlack)
# roc_to_compare.append(dnn_multi_hiStat_m110to150_roc)

# dnn_cuts_multi_hiStat_m110To150 = [1.088, 1.184, 1.24, 1.28, 1.548, 1.824]
# score_multi_hiStat_m110To150 = "ggH_prediction+VBF_prediction+(1-DY_prediction)+(1-ttbar_prediction)"
# dnn_multi_hiStat_m110To150_sculpt = a.add_mva_source("DNN_multi_hiStat_m110To150_sculpt", "DNN_multi_hiStat_m110To150_sculpt", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-21_11-06-33//Keras_multi/model_50_D2_25_D2_25_D2/root/")
# dnn_multi_hiStat_m110To150_sculpt.add_sample("cat0", "cat0", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kBlack, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m110To150, 1.0, score_multi_hiStat_m110To150, dnn_cuts_multi_hiStat_m110To150[0]))
# dnn_multi_hiStat_m110To150_sculpt.add_sample("cat1", "cat1", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kViolet, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m110To150, dnn_cuts_multi_hiStat_m110To150[0], score_multi_hiStat_m110To150, dnn_cuts_multi_hiStat_m110To150[1]))
# dnn_multi_hiStat_m110To150_sculpt.add_sample("cat2", "cat2", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kBlue, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m110To150, dnn_cuts_multi_hiStat_m110To150[1], score_multi_hiStat_m110To150, dnn_cuts_multi_hiStat_m110To150[2]))
# dnn_multi_hiStat_m110To150_sculpt.add_sample("cat3", "cat3", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kGreen, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m110To150, dnn_cuts_multi_hiStat_m110To150[2], score_multi_hiStat_m110To150, dnn_cuts_multi_hiStat_m110To150[3]))
# dnn_multi_hiStat_m110To150_sculpt.add_sample("cat4", "cat4", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kYellow, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m110To150, dnn_cuts_multi_hiStat_m110To150[3], score_multi_hiStat_m110To150, dnn_cuts_multi_hiStat_m110To150[4]))
# dnn_multi_hiStat_m110To150_sculpt.add_sample("cat5", "cat5", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kOrange, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m110To150, dnn_cuts_multi_hiStat_m110To150[4], score_multi_hiStat_m110To150, dnn_cuts_multi_hiStat_m110To150[5]))
# dnn_multi_hiStat_m110To150_sculpt.add_sample("cat6", "cat6", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kRed, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m110To150, dnn_cuts_multi_hiStat_m110To150[5], score_multi_hiStat_m110To150, 3.0))
# dnn_multi_hiStat_m110To150_sculpt.set_lumi(40490.712)
# dnn_multi_hiStat_m110To150_sculpt.plot("mass", 40, 110, 150, label="shapes", draw=True, shapes=True)


# Option 1.2
# dnn_multi_hiStat_m110to150_CS = a.add_mva_source("DNN_Multi_hiStat_m110to150_CS", "DNN_Multi_hiStat_m110to150_CS", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-21_11-06-35//Keras_multi/model_50_D2_25_D2_25_D2/root/")
# dnn_multi_hiStat_m110to150_CS.add_sample("tt", "ttbar", "output_t*root", "tree_tt_ll_POW", False, True, ROOT.kYellow, True)
# dnn_multi_hiStat_m110to150_CS.add_sample("dy", "Drell-Yan", "output_t*root", "tree_ZJets_aMC", False, True, ROOT.kOrange-3, True)
# dnn_multi_hiStat_m110to150_CS.add_sample("ggh", "ggH", "output_t*root", "tree_H2Mu_gg", False, False, ROOT.kRed, True)
# dnn_multi_hiStat_m110to150_CS.add_sample("vbf", "VBF", "output_t*root", "tree_H2Mu_VBF", False, False, ROOT.kViolet-1, True)
# dnn_multi_hiStat_m110to150_CS.add_sample("data", "Data 2017 (40.5/fb)", "output_Data.root", "tree_Data", True, False, ROOT.kBlack, False)
# dnn_multi_hiStat_m110to150_CS.set_lumi(41394.221)
# dnn_multi_hiStat_m110to150_CS_roc_graph = dnn_multi_hiStat_m110to150_CS.plot_roc("ggH_prediction+VBF_prediction+(1-DY_prediction)+(1-ttbar_prediction)", 500, 1, 3, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
# dnn_multi_hiStat_m110to150_CS_roc = a.RocCurve(dnn_multi_hiStat_m110to150_CS_roc_graph, "dnn_multi_hiStat_m110to150_CS", "DNN_Multi_hiStat_m110to150_CS", ROOT.kRed)
# roc_to_compare.append(dnn_multi_hiStat_m110to150_CS_roc)

# dnn_cuts_multi_hiStat_m110To150_CS = [1.096, 1.184, 1.244, 1.296, 1.732, 2.048]
# score_multi_hiStat_m110To150_CS = "ggH_prediction+VBF_prediction+(1-DY_prediction)+(1-ttbar_prediction)"
# dnn_multi_hiStat_m110To150_CS_sculpt = a.add_mva_source("DNN_multi_hiStat_m110To150_CS_sculpt", "DNN_multi_hiStat_m110To150_CS_sculpt", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-21_11-06-35//Keras_multi/model_50_D2_25_D2_25_D2/root/")
# dnn_multi_hiStat_m110To150_CS_sculpt.add_sample("cat0", "cat0", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kBlack, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m110To150_CS, 1.0, score_multi_hiStat_m110To150_CS, dnn_cuts_multi_hiStat_m110To150_CS[0]))
# dnn_multi_hiStat_m110To150_CS_sculpt.add_sample("cat1", "cat1", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kViolet, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m110To150_CS, dnn_cuts_multi_hiStat_m110To150_CS[0], score_multi_hiStat_m110To150_CS, dnn_cuts_multi_hiStat_m110To150_CS[1]))
# dnn_multi_hiStat_m110To150_CS_sculpt.add_sample("cat2", "cat2", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kBlue, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m110To150_CS, dnn_cuts_multi_hiStat_m110To150_CS[1], score_multi_hiStat_m110To150_CS, dnn_cuts_multi_hiStat_m110To150_CS[2]))
# dnn_multi_hiStat_m110To150_CS_sculpt.add_sample("cat3", "cat3", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kGreen, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m110To150_CS, dnn_cuts_multi_hiStat_m110To150_CS[2], score_multi_hiStat_m110To150_CS, dnn_cuts_multi_hiStat_m110To150_CS[3]))
# dnn_multi_hiStat_m110To150_CS_sculpt.add_sample("cat4", "cat4", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kYellow, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m110To150_CS, dnn_cuts_multi_hiStat_m110To150_CS[3], score_multi_hiStat_m110To150_CS, dnn_cuts_multi_hiStat_m110To150_CS[4]))
# dnn_multi_hiStat_m110To150_CS_sculpt.add_sample("cat5", "cat5", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kOrange, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m110To150_CS, dnn_cuts_multi_hiStat_m110To150_CS[4], score_multi_hiStat_m110To150_CS, dnn_cuts_multi_hiStat_m110To150_CS[5]))
# dnn_multi_hiStat_m110To150_CS_sculpt.add_sample("cat6", "cat6", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kRed, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m110To150_CS, dnn_cuts_multi_hiStat_m110To150_CS[5], score_multi_hiStat_m110To150_CS, 3.0))
# dnn_multi_hiStat_m110To150_CS_sculpt.set_lumi(40490.712)
# dnn_multi_hiStat_m110To150_CS_sculpt.plot("mass", 40, 110, 150, label="shapes", draw=True, shapes=True)

# Option 1.3
# dnn_multi_hiStat_m120to130 = a.add_mva_source("DNN_Multi_hiStat_m120to130", "DNN_Multi_hiStat_m120to130", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-21_11-06-39//Keras_multi/model_50_D2_25_D2_25_D2/root/")
# dnn_multi_hiStat_m120to130.add_sample("tt", "ttbar", "output_t*root", "tree_tt_ll_POW", False, True, ROOT.kYellow, True)
# dnn_multi_hiStat_m120to130.add_sample("dy", "Drell-Yan", "output_t*root", "tree_ZJets_aMC", False, True, ROOT.kOrange-3, True)
# dnn_multi_hiStat_m120to130.add_sample("ggh", "ggH", "output_t*root", "tree_H2Mu_gg", False, False, ROOT.kRed, True)
# dnn_multi_hiStat_m120to130.add_sample("vbf", "VBF", "output_t*root", "tree_H2Mu_VBF", False, False, ROOT.kViolet-1, True)
# dnn_multi_hiStat_m120to130.add_sample("data", "Data 2017 (40.5/fb)", "output_Data.root", "tree_Data", True, False, ROOT.kBlack)
# dnn_multi_hiStat_m120to130.set_lumi(41394.221)
# dnn_multi_hiStat_m120to130_roc_graph = dnn_multi_hiStat_m120to130.plot_roc("ggH_prediction+VBF_prediction+(1-DY_prediction)+(1-ttbar_prediction)", 500, 1, 3, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
# dnn_multi_hiStat_m120to130_roc = a.RocCurve(dnn_multi_hiStat_m120to130_roc_graph, "dnn_multi_hiStat_m120to130", "DNN_Multi_hiStat_m120to130", ROOT.kBlack, 2)
# roc_to_compare.append(dnn_multi_hiStat_m120to130_roc)

# dnn_cuts_multi_hiStat_m120To130 = [1.3, 1.516, 1.62, 1.6880000000000002, 2.064, 2.3360000000000003]
# score_multi_hiStat_m120To130 = "ggH_prediction+VBF_prediction+(1-DY_prediction)+(1-ttbar_prediction)"
# dnn_multi_hiStat_m120To130_sculpt = a.add_mva_source("DNN_multi_hiStat_m120To130_sculpt", "DNN_multi_hiStat_m120To130_sculpt", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-21_11-06-39//Keras_multi/model_50_D2_25_D2_25_D2/root/")
# dnn_multi_hiStat_m120To130_sculpt.add_sample("cat0", "cat0", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kBlack, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m120To130, 1.0, score_multi_hiStat_m120To130, dnn_cuts_multi_hiStat_m120To130[0]))
# dnn_multi_hiStat_m120To130_sculpt.add_sample("cat1", "cat1", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kViolet, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m120To130, dnn_cuts_multi_hiStat_m120To130[0], score_multi_hiStat_m120To130, dnn_cuts_multi_hiStat_m120To130[1]))
# dnn_multi_hiStat_m120To130_sculpt.add_sample("cat2", "cat2", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kBlue, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m120To130, dnn_cuts_multi_hiStat_m120To130[1], score_multi_hiStat_m120To130, dnn_cuts_multi_hiStat_m120To130[2]))
# dnn_multi_hiStat_m120To130_sculpt.add_sample("cat3", "cat3", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kGreen, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m120To130, dnn_cuts_multi_hiStat_m120To130[2], score_multi_hiStat_m120To130, dnn_cuts_multi_hiStat_m120To130[3]))
# dnn_multi_hiStat_m120To130_sculpt.add_sample("cat4", "cat4", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kYellow, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m120To130, dnn_cuts_multi_hiStat_m120To130[3], score_multi_hiStat_m120To130, dnn_cuts_multi_hiStat_m120To130[4]))
# dnn_multi_hiStat_m120To130_sculpt.add_sample("cat5", "cat5", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kOrange, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m120To130, dnn_cuts_multi_hiStat_m120To130[4], score_multi_hiStat_m120To130, dnn_cuts_multi_hiStat_m120To130[5]))
# dnn_multi_hiStat_m120To130_sculpt.add_sample("cat6", "cat6", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kRed, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m120To130, dnn_cuts_multi_hiStat_m120To130[5], score_multi_hiStat_m120To130, 3.0))
# dnn_multi_hiStat_m120To130_sculpt.set_lumi(41394.221)
# dnn_multi_hiStat_m120To130_sculpt.plot("mass", 40, 110, 150, label="shapes", draw=True, shapes=True)

# # Option 1.4
# dnn_multi_hiStat_m120to130_CS = a.add_mva_source("DNN_Multi_hiStat_m120to130_CS", "DNN_Multi_hiStat_m120to130_CS", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-21_11-06-40//Keras_multi/model_50_D2_25_D2_25_D2/root/")
# dnn_multi_hiStat_m120to130_CS.add_sample("tt", "ttbar", "output_t*root", "tree_tt_ll_POW", False, True, ROOT.kYellow, True)
# dnn_multi_hiStat_m120to130_CS.add_sample("dy", "Drell-Yan", "output_t*root", "tree_ZJets_aMC", False, True, ROOT.kOrange-3, True)
# dnn_multi_hiStat_m120to130_CS.add_sample("ggh", "ggH", "output_t*root", "tree_H2Mu_gg", False, False, ROOT.kRed, True)
# dnn_multi_hiStat_m120to130_CS.add_sample("vbf", "VBF", "output_t*root", "tree_H2Mu_VBF", False, False, ROOT.kViolet-1, True)
# dnn_multi_hiStat_m120to130_CS.add_sample("data", "Data 2017 (40.5/fb)", "output_Data.root", "tree_Data", True, False, ROOT.kBlack)
# dnn_multi_hiStat_m120to130_CS.set_lumi(41394.221)
# dnn_multi_hiStat_m120to130_CS_roc_graph = dnn_multi_hiStat_m120to130_CS.plot_roc("ggH_prediction+VBF_prediction+(1-DY_prediction)+(1-ttbar_prediction)", 500, 1, 3, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
# dnn_multi_hiStat_m120to130_CS_roc = a.RocCurve(dnn_multi_hiStat_m120to130_CS_roc_graph, "dnn_multi_hiStat_m120to130_CS", "DNN_Multi_hiStat_m120to130_CS", ROOT.kRed)
# roc_to_compare.append(dnn_multi_hiStat_m120to130_CS_roc)

# dnn_cuts_multi_hiStat_m120To130_CS = [1.28, 1.504, 1.6280000000000001, 1.7000000000000002, 2.08, 2.356]
# score_multi_hiStat_m120To130_CS = "ggH_prediction+VBF_prediction+(1-DY_prediction)+(1-ttbar_prediction)"
# dnn_multi_hiStat_m120To130_CS_sculpt = a.add_mva_source("DNN_multi_hiStat_m120To130_CS_sculpt", "DNN_multi_hiStat_m120To130_CS_sculpt", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-21_11-06-40//Keras_multi/model_50_D2_25_D2_25_D2/root/")
# dnn_multi_hiStat_m120To130_CS_sculpt.add_sample("cat0", "cat0", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kBlack, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m120To130_CS, 1.0, score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[0]))
# dnn_multi_hiStat_m120To130_CS_sculpt.add_sample("cat1", "cat1", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kViolet, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[0], score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[1]))
# dnn_multi_hiStat_m120To130_CS_sculpt.add_sample("cat2", "cat2", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kBlue, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[1], score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[2]))
# dnn_multi_hiStat_m120To130_CS_sculpt.add_sample("cat3", "cat3", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kGreen, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[2], score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[3]))
# dnn_multi_hiStat_m120To130_CS_sculpt.add_sample("cat4", "cat4", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kYellow, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[3], score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[4]))
# dnn_multi_hiStat_m120To130_CS_sculpt.add_sample("cat5", "cat5", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kOrange, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[4], score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[5]))
# dnn_multi_hiStat_m120To130_CS_sculpt.add_sample("cat6", "cat6", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kRed, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[5], score_multi_hiStat_m120To130_CS, 3.0))
# dnn_multi_hiStat_m120To130_CS_sculpt.set_lumi(41394.221)
# dnn_multi_hiStat_m120To130_CS_sculpt.plot("mass", 40, 110, 150, label="shapes", draw=True, shapes=True)





# # Option 3.1
# bdt_ucsd_hiStat_cs = a.add_mva_source("BDT_UCSD_hiStat_cs", "BDT_UCSD_hiStat", "/home/dkondra/tmp/BDTG_UCSD_hiStat_cs/")
# bdt_ucsd_hiStat_cs.add_sample("tt", "ttbar", "tt_ll_POW/*.root", "tree", False, True, ROOT.kYellow)
# bdt_ucsd_hiStat_cs.add_sample("dy", "Drell-Yan", "ZJets_aMC/*.root", "tree", False, True, ROOT.kOrange-3)
# bdt_ucsd_hiStat_cs.add_sample("ggh", "ggH", "H2Mu_gg/*.root", "tree", False, False, ROOT.kRed)
# bdt_ucsd_hiStat_cs.add_sample("vbf", "VBF", "H2Mu_VBF/*.root", "tree", False, False, ROOT.kViolet-1)
# bdt_ucsd_hiStat_cs.add_sample("datab", "Data 2017B (40.5/fb)", "SingleMu_2017B/*.root", "tree", True, False, ROOT.kBlack)
# # bdt_ucsd_hiStat_cs.add_sample("datac", "Data 2017C (40.5/fb)", "SingleMu_2017C/*.root", "tree", True, False, ROOT.kBlack)
# # bdt_ucsd_hiStat_cs.add_sample("datad", "Data 2017D (40.5/fb)", "SingleMu_2017D/*.root", "tree", True, False, ROOT.kBlack)
# # bdt_ucsd_hiStat_cs.add_sample("datae", "Data 2017E (40.5/fb)", "SingleMu_2017E/*.root", "tree", True, False, ROOT.kBlack)
# # bdt_ucsd_hiStat_cs.add_sample("dataf", "Data 2017F (40.5/fb)", "SingleMu_2017F/*.root", "tree", True, False, ROOT.kBlack)
# bdt_ucsd_hiStat_cs.set_lumi(4723.411)
# bdt_ucsd_hiStat_cs_roc_graph = bdt_ucsd_hiStat_cs.plot_roc("MVA", 500, -1, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
# bdt_ucsd_hiStat_cs_roc = a.RocCurve(bdt_ucsd_hiStat_cs_roc_graph, "bdt_ucsd_hiStat_cs", "BDT UCSD hiStat", ROOT.kBlack)
# roc_to_compare.append(bdt_ucsd_hiStat_cs_roc)


# # Option 3.2
# bdt_ucsd_hiStat_cs_ebe = a.add_mva_source("BDT_UCSD_hiStat_cs_ebe", "BDT_UCSD_hiStat_ebe", "/home/dkondra/tmp/BDTG_UCSD_hiStat_cs_ebe/")
# bdt_ucsd_hiStat_cs_ebe.add_sample("tt", "ttbar", "tt_ll_POW/*.root", "tree", False, True, ROOT.kYellow)
# bdt_ucsd_hiStat_cs_ebe.add_sample("dy", "Drell-Yan", "ZJets_aMC/*.root", "tree", False, True, ROOT.kOrange-3)
# bdt_ucsd_hiStat_cs_ebe.add_sample("ggh", "ggH", "H2Mu_gg/*.root", "tree", False, False, ROOT.kRed)
# bdt_ucsd_hiStat_cs_ebe.add_sample("vbf", "VBF", "H2Mu_VBF/*.root", "tree", False, False, ROOT.kViolet-1)
# bdt_ucsd_hiStat_cs_ebe.add_sample("datab", "Data 2017B (40.5/fb)", "SingleMu_2017B/*.root", "tree", True, False, ROOT.kBlack)
# # bdt_ucsd_hiStat_cs.add_sample("datac", "Data 2017C (40.5/fb)", "SingleMu_2017C/*.root", "tree", True, False, ROOT.kBlack)
# # bdt_ucsd_hiStat_cs.add_sample("datad", "Data 2017D (40.5/fb)", "SingleMu_2017D/*.root", "tree", True, False, ROOT.kBlack)
# # bdt_ucsd_hiStat_cs.add_sample("datae", "Data 2017E (40.5/fb)", "SingleMu_2017E/*.root", "tree", True, False, ROOT.kBlack)
# # bdt_ucsd_hiStat_cs.add_sample("dataf", "Data 2017F (40.5/fb)", "SingleMu_2017F/*.root", "tree", True, False, ROOT.kBlack)
# bdt_ucsd_hiStat_cs_ebe.set_lumi(4723.411)
# bdt_ucsd_hiStat_cs_ebe_roc_graph = bdt_ucsd_hiStat_cs_ebe.plot_roc("MVA", 500, -1, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
# bdt_ucsd_hiStat_cs_ebe_roc = a.RocCurve(bdt_ucsd_hiStat_cs_ebe_roc_graph, "bdt_ucsd_hiStat_cs_ebe", "BDT UCSD hiStat EBE", ROOT.kOrange-3, 2)
# roc_to_compare.append(bdt_ucsd_hiStat_cs_ebe_roc)


# bdt_cuts_ebe = [-0.484, 0.04800000000000004, 0.28400000000000003, 0.43199999999999994, 0.6560000000000001, 0.764]
# bdt_score = "MVA"
# bdt_ucsd_hiStat_cs_ebe_res = a.add_mva_source("BDT_UCSD_hiStat_cs_ebe_res", "BDT_UCSD_hiStat_ebe_res", "/home/dkondra/tmp/BDTG_UCSD_hiStat_cs_ebe/")
# bdt_ucsd_hiStat_cs_ebe_res.add_sample("cat0", "cat0", "H2Mu_gg/*root", "tree", False, False, ROOT.kBlack, True, "((%s)>%f)&((%s)<%f)"%(bdt_score, -1.0, bdt_score, bdt_cuts_ebe[0]))
# bdt_ucsd_hiStat_cs_ebe_res.add_sample("cat1", "cat1", "H2Mu_gg/*root", "tree", False, False, ROOT.kViolet, True, "((%s)>%f)&((%s)<%f)"%(bdt_score, bdt_cuts_ebe[0], bdt_score, bdt_cuts_ebe[1]))
# bdt_ucsd_hiStat_cs_ebe_res.add_sample("cat2", "cat2", "H2Mu_gg/*root", "tree", False, False, ROOT.kBlue, True, "((%s)>%f)&((%s)<%f)"%(bdt_score, bdt_cuts_ebe[1], bdt_score, bdt_cuts_ebe[2]))
# bdt_ucsd_hiStat_cs_ebe_res.add_sample("cat3", "cat3", "H2Mu_gg/*root", "tree", False, False, ROOT.kGreen, True, "((%s)>%f)&((%s)<%f)"%(bdt_score, bdt_cuts_ebe[2], bdt_score, bdt_cuts_ebe[3]))
# bdt_ucsd_hiStat_cs_ebe_res.add_sample("cat4", "cat4", "H2Mu_gg/*root", "tree", False, False, ROOT.kYellow, True, "((%s)>%f)&((%s)<%f)"%(bdt_score, bdt_cuts_ebe[3], bdt_score, bdt_cuts_ebe[4]))
# bdt_ucsd_hiStat_cs_ebe_res.add_sample("cat5", "cat5", "H2Mu_gg/*root", "tree", False, False, ROOT.kOrange, True, "((%s)>%f)&((%s)<%f)"%(bdt_score, bdt_cuts_ebe[4], bdt_score, bdt_cuts_ebe[5]))
# bdt_ucsd_hiStat_cs_ebe_res.add_sample("cat6", "cat6", "H2Mu_gg/*root", "tree", False, False, ROOT.kRed, True, "((%s)>%f)&((%s)<%f)"%(bdt_score, bdt_cuts_ebe[5], bdt_score, 1.0))
# bdt_ucsd_hiStat_cs_ebe_res.set_lumi(4723.411)
# bdt_ucsd_hiStat_cs_ebe_res.plot("mass", 40, 110, 150, label="shapes", draw=True, shapes=True)

# # # Option 3.3
# dnn_multi_hiStat_m120to130_CS1 = a.add_mva_source("DNN_Multi_hiStat_m120to130_CS1", "DNN_Multi_hiStat_m120to130_CS1", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-27_13-20-29//Keras_multi/model_50_D2_25_D2_25_D2/root/")
# dnn_multi_hiStat_m120to130_CS1.add_sample("tt", "ttbar", "output_t*root", "tree_tt_ll_POW", False, True, ROOT.kYellow, True)
# dnn_multi_hiStat_m120to130_CS1.add_sample("dy", "Drell-Yan", "output_t*root", "tree_ZJets_aMC", False, True, ROOT.kOrange-3, True)
# dnn_multi_hiStat_m120to130_CS1.add_sample("ggh", "ggH", "output_t*root", "tree_H2Mu_gg", False, False, ROOT.kRed, True)
# dnn_multi_hiStat_m120to130_CS1.add_sample("vbf", "VBF", "output_t*root", "tree_H2Mu_VBF", False, False, ROOT.kViolet-1, True)
# dnn_multi_hiStat_m120to130_CS1.add_sample("data", "Data 2017 (40.5/fb)", "output_Data.root", "tree_Data", True, False, ROOT.kBlack)
# dnn_multi_hiStat_m120to130_CS1.set_lumi(41394.221)
# dnn_multi_hiStat_m120to130_CS1_roc_graph = dnn_multi_hiStat_m120to130_CS1.plot_roc("ggH_prediction+VBF_prediction+(1-DY_prediction)+(1-ttbar_prediction)", 500, 1, 3, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
# dnn_multi_hiStat_m120to130_CS1_roc = a.RocCurve(dnn_multi_hiStat_m120to130_CS1_roc_graph, "dnn_multi_hiStat_m120to130_CS1", "DNN_Multi_hiStat_m120to130_CS1", ROOT.kRed, 2)
# roc_to_compare.append(dnn_multi_hiStat_m120to130_CS1_roc)

# dnn_cuts_multi_hiStat_m120To130_CS = [1.296, 1.508, 1.624, 1.692, 2.1, 2.368]
# score_multi_hiStat_m120To130_CS = "ggH_prediction+VBF_prediction+(1-DY_prediction)+(1-ttbar_prediction)"
# dnn_multi_hiStat_m120To130_CS1_sculpt = a.add_mva_source("DNN_multi_hiStat_m120To130_CS1_sculpt", "DNN_multi_hiStat_m120To130_CS1_sculpt", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-27_13-20-29//Keras_multi/model_50_D2_25_D2_25_D2/root/")
# dnn_multi_hiStat_m120To130_CS1_sculpt.add_sample("cat0", "cat0", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kBlack, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m120To130_CS, 1.0, score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[0]))
# dnn_multi_hiStat_m120To130_CS1_sculpt.add_sample("cat1", "cat1", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kViolet, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[0], score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[1]))
# dnn_multi_hiStat_m120To130_CS1_sculpt.add_sample("cat2", "cat2", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kBlue, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[1], score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[2]))
# dnn_multi_hiStat_m120To130_CS1_sculpt.add_sample("cat3", "cat3", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kGreen, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[2], score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[3]))
# dnn_multi_hiStat_m120To130_CS1_sculpt.add_sample("cat4", "cat4", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kYellow, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[3], score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[4]))
# dnn_multi_hiStat_m120To130_CS1_sculpt.add_sample("cat5", "cat5", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kOrange, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[4], score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[5]))
# dnn_multi_hiStat_m120To130_CS1_sculpt.add_sample("cat6", "cat6", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kRed, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[5], score_multi_hiStat_m120To130_CS, 3.0))
# dnn_multi_hiStat_m120To130_CS1_sculpt.set_lumi(41394.221)
# dnn_multi_hiStat_m120To130_CS1_sculpt.plot("mass", 40, 110, 150, label="shapes", draw=True, shapes=True)

# Option 3.4
# dnn_multi_hiStat_m120to130_CS_noSingleMu = a.add_mva_source("DNN_Multi_hiStat_m120to130_CS_noSingleMu", "DNN_Multi_hiStat_m120to130_CS_noSingleMu", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-23_09-45-55//Keras_multi/model_50_D2_25_D2_25_D2/root/")
# dnn_multi_hiStat_m120to130_CS_noSingleMu.add_sample("tt", "ttbar", "output_t*root", "tree_tt_ll_POW", False, True, ROOT.kYellow, True)
# dnn_multi_hiStat_m120to130_CS_noSingleMu.add_sample("dy", "Drell-Yan", "output_t*root", "tree_ZJets_aMC", False, True, ROOT.kOrange-3, True)
# dnn_multi_hiStat_m120to130_CS_noSingleMu.add_sample("ggh", "ggH", "output_t*root", "tree_H2Mu_gg", False, False, ROOT.kRed, True)
# dnn_multi_hiStat_m120to130_CS_noSingleMu.add_sample("vbf", "VBF", "output_t*root", "tree_H2Mu_VBF", False, False, ROOT.kViolet-1, True)
# dnn_multi_hiStat_m120to130_CS_noSingleMu.add_sample("data", "Data 2017 (40.5/fb)", "output_Data.root", "tree_Data", True, False, ROOT.kBlack)
# dnn_multi_hiStat_m120to130_CS_noSingleMu.set_lumi(41394.221)
# dnn_multi_hiStat_m120to130_CS_noSingleMu_roc_graph = dnn_multi_hiStat_m120to130_CS_noSingleMu.plot_roc("ggH_prediction+VBF_prediction+(1-DY_prediction)+(1-ttbar_prediction)", 500, 1, 3, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
# dnn_multi_hiStat_m120to130_CS_noSingleMu_roc = a.RocCurve(dnn_multi_hiStat_m120to130_CS_noSingleMu_roc_graph, "dnn_multi_hiStat_m120to130_CS_noSingleMu", "DNN_Multi_hiStat_m120to130_CS_noSingleMu", ROOT.kBlue)
# # roc_to_compare.append(dnn_multi_hiStat_m120to130_CS_noSingleMu_roc)

# dnn_cuts_multi_hiStat_m120To130_CS = [1.308, 1.504, 1.616, 1.6840000000000002, 2.02, 2.308]
# score_multi_hiStat_m120To130_CS = "ggH_prediction+VBF_prediction+(1-DY_prediction)+(1-ttbar_prediction)"
# dnn_multi_hiStat_m120To130_CS_noSingleMu_sculpt = a.add_mva_source("DNN_multi_hiStat_m120To130_CS_noSingleMu_sculpt", "DNN_multi_hiStat_m120To130_CS_noSingleMu_sculpt", "/scratch/gilbreth/dkondra/ML_output/Run_2019-04-23_09-45-55//Keras_multi/model_50_D2_25_D2_25_D2/root/")
# dnn_multi_hiStat_m120To130_CS_noSingleMu_sculpt.add_sample("cat0", "cat0", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kBlack, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m120To130_CS, 1.0, score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[0]))
# dnn_multi_hiStat_m120To130_CS_noSingleMu_sculpt.add_sample("cat1", "cat1", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kViolet, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[0], score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[1]))
# dnn_multi_hiStat_m120To130_CS_noSingleMu_sculpt.add_sample("cat2", "cat2", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kBlue, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[1], score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[2]))
# dnn_multi_hiStat_m120To130_CS_noSingleMu_sculpt.add_sample("cat3", "cat3", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kGreen, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[2], score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[3]))
# dnn_multi_hiStat_m120To130_CS_noSingleMu_sculpt.add_sample("cat4", "cat4", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kYellow, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[3], score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[4]))
# dnn_multi_hiStat_m120To130_CS_noSingleMu_sculpt.add_sample("cat5", "cat5", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kOrange, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[4], score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[5]))
# dnn_multi_hiStat_m120To130_CS_noSingleMu_sculpt.add_sample("cat6", "cat6", "output_t*root", "tree_ZJets_aMC", False, False, ROOT.kRed, True, "((%s)>%f)&((%s)<%f)"%(score_multi_hiStat_m120To130_CS, dnn_cuts_multi_hiStat_m120To130_CS[5], score_multi_hiStat_m120To130_CS, 3.0))
# dnn_multi_hiStat_m120To130_CS_noSingleMu_sculpt.set_lumi(41394.221)
# dnn_multi_hiStat_m120To130_CS_noSingleMu_sculpt.plot("mass", 40, 110, 150, label="shapes", draw=True, shapes=True)


# Option 4 - Raffaele's BDT

# bdt_raffaele = a.add_mva_source("BDT_raffaele", "BDT Raffaele", "/mnt/hadoop/store/user/dkondrat/UCSD_files/")
# bdt_raffaele.add_sample("vv", "VV", "tree_VV.root", "tree", False, True, ROOT.kGreen-1, False)
# bdt_raffaele.add_sample("tt", "ttbar", "tree_top.root", "tree", False, True, ROOT.kYellow, False)
# bdt_raffaele.add_sample("dy", "Drell-Yan", "tree_DY.root", "tree", False, True, ROOT.kOrange-3, False)
# bdt_raffaele.add_sample("ggh", "ggH", "tree_ggH.root", "tree", False, False, ROOT.kRed, False)
# bdt_raffaele.add_sample("vbf", "VBF", "tree_VBF.root", "tree", False, False, ROOT.kViolet-1, False)
# bdt_raffaele.set_lumi(4723.411)
# bdt_raffaele_roc_graph = bdt_raffaele.plot_roc("bdtucsd_inclusive", 200, -1, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
# bdt_raffaele_roc = a.RocCurve(bdt_raffaele_roc_graph, "bdt_raffaele", "BDT Raffaele", ROOT.kBlack, 2)
# roc_to_compare.append(bdt_raffaele_roc)

# a.compare_roc_curves(roc_to_compare)

# bdt_raffaele = a.add_mva_source("BDT_raffaele_01jet", "BDT Raffaele 01jet", "/mnt/hadoop/store/user/dkondrat/UCSD_files/")
# bdt_raffaele.add_sample("vv", "VV", "tree_VV.root", "tree", False, True, ROOT.kGreen-1, False)
# bdt_raffaele.add_sample("tt", "ttbar", "tree_top.root", "tree", False, True, ROOT.kYellow, False)
# bdt_raffaele.add_sample("dy", "Drell-Yan", "tree_DY.root", "tree", False, True, ROOT.kOrange-3, False)
# bdt_raffaele.add_sample("ggh", "ggH", "tree_ggH.root", "tree", False, False, ROOT.kRed, False)
# bdt_raffaele.add_sample("vbf", "VBF", "tree_VBF.root", "tree", False, False, ROOT.kViolet-1, False)
# bdt_raffaele.set_lumi(4723.411)
# bdt_raffaele_roc_graph = bdt_raffaele.plot_roc("bdtucsd_01jet", 200, -1, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
# bdt_raffaele_roc = a.RocCurve(bdt_raffaele_roc_graph, "bdt_raffaele_01jet", "BDT Raffaele 01jet", ROOT.kBlue, 2)
# roc_to_compare.append(bdt_raffaele_roc)

# a.compare_roc_curves(roc_to_compare)

# bdt_raffaele = a.add_mva_source("BDT_raffaele_2jet", "BDT Raffaele 2jet", "/mnt/hadoop/store/user/dkondrat/UCSD_files/")
# bdt_raffaele.add_sample("vv", "VV", "tree_VV.root", "tree", False, True, ROOT.kGreen-1, False)
# bdt_raffaele.add_sample("tt", "ttbar", "tree_top.root", "tree", False, True, ROOT.kYellow, False)
# bdt_raffaele.add_sample("dy", "Drell-Yan", "tree_DY.root", "tree", False, True, ROOT.kOrange-3, False)
# bdt_raffaele.add_sample("ggh", "ggH", "tree_ggH.root", "tree", False, False, ROOT.kRed, False)
# bdt_raffaele.add_sample("vbf", "VBF", "tree_VBF.root", "tree", False, False, ROOT.kViolet-1, False)
# bdt_raffaele.set_lumi(4723.411)
# bdt_raffaele_roc_graph = bdt_raffaele.plot_roc("bdtucsd_2jet", 200, -1, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
# bdt_raffaele_roc = a.RocCurve(bdt_raffaele_roc_graph, "bdt_raffaele_2jet", "BDT Raffaele 2jet", ROOT.kBlue, 2)
# roc_to_compare.append(bdt_raffaele_roc)

# a.compare_roc_curves(roc_to_compare)

# bdt_raffaele = a.add_mva_source("BDT_raffaele_UF", "BDT Raffaele UF", "/mnt/hadoop/store/user/dkondrat/UCSD_files/")
# bdt_raffaele.add_sample("vv", "VV", "tree_VV.root", "tree", False, True, ROOT.kGreen-1, False)
# bdt_raffaele.add_sample("tt", "ttbar", "tree_top.root", "tree", False, True, ROOT.kYellow, False)
# bdt_raffaele.add_sample("dy", "Drell-Yan", "tree_DY.root", "tree", False, True, ROOT.kOrange-3, False)
# bdt_raffaele.add_sample("ggh", "ggH", "tree_ggH.root", "tree", False, False, ROOT.kRed, False)
# bdt_raffaele.add_sample("vbf", "VBF", "tree_VBF.root", "tree", False, False, ROOT.kViolet-1, False)
# bdt_raffaele.set_lumi(4723.411)
# bdt_raffaele_roc_graph = bdt_raffaele.plot_roc("bdtuf", 200, -1, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
# bdt_raffaele_roc = a.RocCurve(bdt_raffaele_roc_graph, "bdt_raffaele_UF", "BDT Raffaele UF", ROOT.kBlue, 2)
# roc_to_compare.append(bdt_raffaele_roc)



# bdt_raffaele_all = a.add_mva_source("BDT_raffaele_all_01jet", "BDT Raffaele 01jet", "/mnt/hadoop/store/user/dkondrat/UCSD_files/")
# bdt_raffaele_all.add_sample("vv2016", "VV", "/2016/tree_VV.root", "tree", False, True, ROOT.kGreen-1, False)
# bdt_raffaele_all.add_sample("vv2017", "VV", "/2017/tree_VV.root", "tree", False, True, ROOT.kGreen-1, False)
# bdt_raffaele_all.add_sample("vv2018", "VV", "/2018/tree_VV.root", "tree", False, True, ROOT.kGreen-1, False)
# bdt_raffaele_all.add_sample("tt2016", "ttbar", "/2016/tree_top.root", "tree", False, True, ROOT.kYellow, False)
# bdt_raffaele_all.add_sample("tt2017", "ttbar", "/2017/tree_top.root", "tree", False, True, ROOT.kYellow, False)
# bdt_raffaele_all.add_sample("tt2018", "ttbar", "/2018/tree_top.root", "tree", False, True, ROOT.kYellow, False)
# bdt_raffaele_all.add_sample("dy2016", "Drell-Yan", "/2016/tree_DY.root", "tree", False, True, ROOT.kOrange-3, False)
# bdt_raffaele_all.add_sample("dy2017", "Drell-Yan", "/2017/tree_DY.root", "tree", False, True, ROOT.kOrange-3, False)
# bdt_raffaele_all.add_sample("dy2018", "Drell-Yan", "/2018/tree_DY.root", "tree", False, True, ROOT.kOrange-3, False)
# bdt_raffaele_all.add_sample("ggh2016", "ggH", "/2016/tree_ggH.root", "tree", False, False, ROOT.kRed, False, "3")
# bdt_raffaele_all.add_sample("vbf2016", "VBF", "/2016/tree_VBF.root", "tree", False, False, ROOT.kViolet-1, False, "3")
# bdt_raffaele_all.set_lumi(4723.411)
# bdt_raffaele_all_roc_graph = bdt_raffaele_all.plot_roc("bdtucsd_01jet", 200, -1, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
# bdt_raffaele_all_roc = a.RocCurve(bdt_raffaele_all_roc_graph, "bdt_raffaele_all_01jet", "BDT Raffaele 01jet", ROOT.kBlue, 2)
# roc_to_compare.append(bdt_raffaele_all_roc)


bdt_raffaele_all = a.add_mva_source("BDT_raffaele_all_2jet_mjjcut", "BDT Raffaele 2jet_mjjcut", "/mnt/hadoop/store/user/dkondrat/UCSD_files/")
bdt_raffaele_all.add_sample("vv2016", "VV", "/2016/tree_VV.root", "tree", False, True, ROOT.kGreen-1, False, "(mjj<500)")
bdt_raffaele_all.add_sample("vv2017", "VV", "/2017/tree_VV.root", "tree", False, True, ROOT.kGreen-1, False, "(mjj<500)")
bdt_raffaele_all.add_sample("vv2018", "VV", "/2018/tree_VV.root", "tree", False, True, ROOT.kGreen-1, False, "(mjj<500)")
bdt_raffaele_all.add_sample("tt2016", "ttbar", "/2016/tree_top.root", "tree", False, True, ROOT.kYellow, False, "(mjj<500)")
bdt_raffaele_all.add_sample("tt2017", "ttbar", "/2017/tree_top.root", "tree", False, True, ROOT.kYellow, False, "(mjj<500)")
bdt_raffaele_all.add_sample("tt2018", "ttbar", "/2018/tree_top.root", "tree", False, True, ROOT.kYellow, False, "(mjj<500)")
bdt_raffaele_all.add_sample("dy2016", "Drell-Yan", "/2016/tree_DY.root", "tree", False, True, ROOT.kOrange-3, False, "(mjj<500)")
bdt_raffaele_all.add_sample("dy2017", "Drell-Yan", "/2017/tree_DY.root", "tree", False, True, ROOT.kOrange-3, False, "(mjj<500)")
bdt_raffaele_all.add_sample("dy2018", "Drell-Yan", "/2018/tree_DY.root", "tree", False, True, ROOT.kOrange-3, False, "(mjj<500)")
bdt_raffaele_all.add_sample("ggh2016", "ggH", "/2016/tree_ggH.root", "tree", False, False, ROOT.kRed, False, "3.8*(mjj<500)")
bdt_raffaele_all.add_sample("vbf2016", "VBF", "/2016/tree_VBF.root", "tree", False, False, ROOT.kViolet-1, False, "3.8*(mjj<500)")

bdt_raffaele_all.set_lumi(4723.411)
bdt_raffaele_all_roc_graph = bdt_raffaele_all.plot_roc("bdtucsd_2jet_bveto", 200, -1, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
bdt_raffaele_all_roc = a.RocCurve(bdt_raffaele_all_roc_graph, "bdt_raffaele_all_2jet_mjjcut", "BDT Raffaele 2jet_mjjcut", ROOT.kBlue, 2)
# roc_to_compare.append(bdt_raffaele_all_roc)


# bdt_raffaele_all = a.add_mva_source("BDT_ucsd_incl", "BDT UCSD inclusive", "/mnt/hadoop/store/user/dkondrat/UCSD_files/")
# bdt_raffaele_all.add_sample("vv2016", "VV", "/2016/tree_VV.root", "tree", False, True, ROOT.kGreen-1, False)
# bdt_raffaele_all.add_sample("vv2017", "VV", "/2017/tree_VV.root", "tree", False, True, ROOT.kGreen-1, False)
# bdt_raffaele_all.add_sample("vv2018", "VV", "/2018/tree_VV.root", "tree", False, True, ROOT.kGreen-1, False)
# bdt_raffaele_all.add_sample("tt2016", "ttbar", "/2016/tree_top.root", "tree", False, True, ROOT.kYellow, False)
# bdt_raffaele_all.add_sample("tt2017", "ttbar", "/2017/tree_top.root", "tree", False, True, ROOT.kYellow, False)
# bdt_raffaele_all.add_sample("tt2018", "ttbar", "/2018/tree_top.root", "tree", False, True, ROOT.kYellow, False)
# bdt_raffaele_all.add_sample("dy2016", "Drell-Yan", "/2016/tree_DY.root", "tree", False, True, ROOT.kOrange-3, False)
# bdt_raffaele_all.add_sample("dy2017", "Drell-Yan", "/2017/tree_DY.root", "tree", False, True, ROOT.kOrange-3, False)
# bdt_raffaele_all.add_sample("dy2018", "Drell-Yan", "/2018/tree_DY.root", "tree", False, True, ROOT.kOrange-3, False)
# bdt_raffaele_all.add_sample("ggh2016", "ggH", "/2016/tree_ggH.root", "tree", False, False, ROOT.kRed, False, "3")
# bdt_raffaele_all.add_sample("vbf2016", "VBF", "/2016/tree_VBF.root", "tree", False, False, ROOT.kViolet-1, False, "3")

# bdt_raffaele_all.set_lumi(4723.411)
# bdt_raffaele_all_roc_graph = bdt_raffaele_all.plot_roc("bdtucsd_inclusive", 200, -1, 1, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
# bdt_raffaele_all_roc = a.RocCurve(bdt_raffaele_all_roc_graph, "BDT_ucsd_incl", "BDT UCSD inclusive", ROOT.kBlue, 2)
# roc_to_compare.append(bdt_raffaele_all_roc)


# # DNN on UCSD files
# dnn_ucsd_files = a.add_mva_source("DNN_ucsd_files", "DNN w/ ucsd files", "~/tmp/Run_2019-05-15_16-55-47//Keras_multi/model_50_D2_25_D2_25_D2/root/")
# dnn_ucsd_files.add_sample("tt", "ttbar", "output_t*root", "tree_tt_ll_POW", False, True, ROOT.kYellow, True)
# dnn_ucsd_files.add_sample("dy", "Drell-Yan", "output_t*root", "tree_ZJets_aMC", False, True, ROOT.kOrange-3, True)
# dnn_ucsd_files.add_sample("ggh", "ggH", "output_t*root", "tree_H2Mu_gg", False, False, ROOT.kRed, True)
# dnn_ucsd_files.add_sample("vbf", "VBF", "output_t*root", "tree_H2Mu_VBF", False, False, ROOT.kViolet-1, True)
# dnn_ucsd_files.add_sample("data", "Data 2017 (40.5/fb)", "output_Data.root", "tree_Data", True, False, ROOT.kBlack)
# dnn_ucsd_files.set_lumi(1)
# dnn_ucsd_files_roc_graph = dnn_ucsd_files.plot_roc("ggH_prediction+VBF_prediction+(1-DY_prediction)+(1-ttbar_prediction)", 500, 1, 3, [0.08, 0.39, 0.61, 0.76, 0.91, 0.95])
# dnn_ucsd_files_roc = a.RocCurve(dnn_ucsd_files_roc_graph, "dnn_ucsd_files", "DNN w/ ucsd files", ROOT.kRed)
# roc_to_compare.append(dnn_ucsd_files_roc)



a.compare_roc_curves(roc_to_compare)

# score = "ggH_prediction+VBF_prediction"
# nBins = 50
# gr1 = a.plot_width_vs_score(score, dnn_multi_hiStat_ebe, "dnn_multi_hiStat_ebe", "DNN hiStat w/ ebe res. ggH", nBins, ROOT.kRed, 20, process = "ggH")
# gr2 = a.plot_width_vs_score(score, dnn_multi_hiStat_ebe, "dnn_multi_hiStat_ebe", "DNN hiStat w/ ebe res. VBF", nBins, ROOT.kBlue, 20, process = "VBF")

# gr3 = a.plot_width_vs_score(score, dnn_multi, "dnn_multi", "DNN loStat w/o ebe res. ggH", nBins, ROOT.kBlack, 20, process = "ggH")
# gr4 = a.plot_width_vs_score(score, dnn_multi, "dnn_multi", "DNN loStat w/o ebe res. VBF", nBins, ROOT.kGreen, 20, process = "VBF")

# canvas = ROOT.TCanvas("c_wvss", "c_wvss", 800, 800)
# canvas.cd()
# legend = ROOT.TLegend(0.6, 0.7, 0.895, 0.895)
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