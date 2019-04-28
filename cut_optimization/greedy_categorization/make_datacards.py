import ROOT
from math import sqrt
import os, sys, errno
Import = getattr(ROOT.RooWorkspace, 'import')

def create_workspace():
    var = ROOT.RooRealVar("mass","Dilepton mass",110,150)     
    var.setBins(100)
    var.setRange("window",120,130)
    var.setRange("full",110,150)
    w = ROOT.RooWorkspace("w", False)
    Import(w, var)

    return w

def add_data(w, cat_name, input_path, data_tree, cut, method):
    var = w.var("mass")
    var.setBins(5000)
    max_abs_eta_var = ROOT.RooRealVar("max_abs_eta_mu","Max abs(eta) of muons", 0, 2.4) 
    mu1_eta = ROOT.RooRealVar("mu1_eta","mu1_eta", -2.4, 2.4) 
    mu2_eta = ROOT.RooRealVar("mu2_eta","mu2_eta", -2.4, 2.4)
    if "binary" in method:
        sig_pred_var = ROOT.RooRealVar("sig_prediction", "sig_prediction", 0, 1)
        bkg_pred_var = ROOT.RooRealVar("bkg_prediction", "bkg_prediction", 0, 1)        
    elif "multi" in method:
        ggh_pred_var = ROOT.RooRealVar("ggH_prediction", "ggH_prediction", 0, 1)
        vbf_pred_var = ROOT.RooRealVar("VBF_prediction", "VBF_prediction", 0, 1)
        dy_pred_var = ROOT.RooRealVar("DY_prediction", "DY_prediction", 0, 1)
        tt_pred_var = ROOT.RooRealVar("ttbar_prediction", "ttbar_prediction", 0, 1) 
    elif "BDT" in method:
        mva_var = ROOT.RooRealVar("MVA", "MVA", -1, 1)
    data_tree = ROOT.TChain(data_tree)
    data_tree.Add(input_path)  
    print "Loaded tree from "+input_path+" with %i entries."%data_tree.GetEntries()
    data_hist_name = "data_%s"%cat_name
    data_hist = ROOT.TH1D(data_hist_name, data_hist_name, 40, 110, 150)
    dummy = ROOT.TCanvas("dummy", "dummy", 800, 800)
    dummy.cd()
    data_tree.Draw("mass>>%s"%(data_hist_name), cut)
    dummy.Close()
    if "binary" in method:
        data = ROOT.RooDataSet("%s_data"%cat_name,"%s_data"%cat_name, data_tree, ROOT.RooArgSet(var, max_abs_eta_var, mu1_eta, mu2_eta, sig_pred_var, bkg_pred_var), cut)
    elif "multi" in method:
        data = ROOT.RooDataSet("%s_data"%cat_name,"%s_data"%cat_name, data_tree, ROOT.RooArgSet(var, max_abs_eta_var, mu1_eta, mu2_eta, ggh_pred_var, vbf_pred_var, dy_pred_var, tt_pred_var), cut)    
    elif "BDT" in method:
        data = ROOT.RooDataSet("%s_data"%cat_name,"%s_data"%cat_name, data_tree, ROOT.RooArgSet(var, max_abs_eta_var, mu1_eta, mu2_eta, mva_var), cut)    

    Import(w, data)
    return w.data("%s_data"%cat_name)

def add_sig_model(w, cat_name, input_path, cut, method, lumi):
    var = w.var("mass")
    var.setBins(5000)
    max_abs_eta_var = ROOT.RooRealVar("max_abs_eta_mu","Max abs(eta) of muons", 0, 2.4) 
    mu1_eta = ROOT.RooRealVar("mu1_eta","mu1_eta", -2.4, 2.4) 
    mu2_eta = ROOT.RooRealVar("mu2_eta","mu2_eta", -2.4, 2.4)
    if "binary" in method:
        sig_pred_var = ROOT.RooRealVar("sig_prediction", "sig_prediction", 0, 1)
        bkg_pred_var = ROOT.RooRealVar("bkg_prediction", "bkg_prediction", 0, 1)
        signal_tree = ROOT.TChain("tree_sig")
        signal_tree.Add(input_path)
        print "Loaded sig tree from "+input_path+" with %i entries."%signal_tree.GetEntries()    
    elif "multi" in method:
        ggh_pred_var = ROOT.RooRealVar("ggH_prediction", "ggH_prediction", 0, 1)
        vbf_pred_var = ROOT.RooRealVar("VBF_prediction", "VBF_prediction", 0, 1)
        dy_pred_var = ROOT.RooRealVar("DY_prediction", "DY_prediction", 0, 1)
        tt_pred_var = ROOT.RooRealVar("ttbar_prediction", "ttbar_prediction", 0, 1) 

        tree_list = ROOT.TList()

        ggh_tree = ROOT.TChain("tree_H2Mu_gg")
        ggh_tree.Add(input_path)
        ggh_tree_clone = ggh_tree.CloneTree()
        ggh_tree_clone.SetDirectory(0)
        tree_list.Add(ggh_tree_clone)


        vbf_tree = ROOT.TChain("tree_H2Mu_VBF")
        vbf_tree.Add(input_path)  
        vbf_tree_clone = vbf_tree.CloneTree()
        vbf_tree_clone.SetDirectory(0)  
        tree_list.Add(vbf_tree_clone)

        signal_tree = ROOT.TTree.MergeTrees(tree_list)
        print "Loaded ggH tree from "+input_path+" with %i entries."%ggh_tree.GetEntries()    
        print "Loaded VBF tree from "+input_path+" with %i entries."%vbf_tree.GetEntries() 
    elif "BDT" in method:
        mva_var = ROOT.RooRealVar("MVA", "MVA", -1, 1)
        signal_tree = ROOT.TChain("tree")
        signal_tree.Add(input_path)
        print "Loaded sig tree from "+input_path+" with %i entries."%signal_tree.GetEntries()    
    
    signal_tree.SetName("signal_tree")
 
    signal_hist_name = "signal_%s"%cat_name
    signal_hist = ROOT.TH1D(signal_hist_name, signal_hist_name, 40, 110, 150)
    dummy = ROOT.TCanvas("dummy", "dummy", 800, 800)
    dummy.cd()
    signal_tree.Draw("mass>>%s"%(signal_hist_name), "(%s)*weight_over_lumi*%s"%(cut, lumi))
    dummy.Close()
    signal_rate = signal_hist.Integral()
    print signal_rate

    w.factory("%s_mix1 [0.5, 0.0, 1.0]"%cat_name)
    w.factory("%s_mix2 [0.5, 0.0, 1.0]"%cat_name)
    mix1 = w.var("%s_mix1"%cat_name)
    mix2 = w.var("%s_mix2"%cat_name)
    w.factory("Gaussian::%s_gaus1(mass, %s_mean1[125., 120., 130.], %s_width1[1.0, 0.5, 5.0])"%(cat_name, cat_name, cat_name))
    w.factory("Gaussian::%s_gaus2(mass, %s_mean2[125., 120., 130.], %s_width2[5.0, 2.0, 10.])"%(cat_name, cat_name, cat_name))
    w.factory("Gaussian::%s_gaus3(mass, %s_mean3[125., 120., 130.], %s_width3[5.0, 1.0, 10.])"%(cat_name, cat_name, cat_name))
    gaus1 = w.pdf('%s_gaus1'%(cat_name))
    gaus2 = w.pdf('%s_gaus2'%(cat_name))
    gaus3 = w.pdf('%s_gaus3'%(cat_name))
    smodel = ROOT.RooAddPdf('%s_sig'%cat_name, '%s_sig'%cat_name, ROOT.RooArgList(gaus1, gaus2, gaus3) , ROOT.RooArgList(mix1, mix2), ROOT.kTRUE)

    # w.Print()
    if "binary" in method:
        signal_ds = ROOT.RooDataSet("signal_ds","signal_ds", signal_tree, ROOT.RooArgSet(var, max_abs_eta_var, mu1_eta, mu2_eta, sig_pred_var, bkg_pred_var), cut)
    elif "multi" in method:
        signal_ds = ROOT.RooDataSet("signal_ds","signal_ds", signal_tree, ROOT.RooArgSet(var, max_abs_eta_var, mu1_eta, mu2_eta, ggh_pred_var, vbf_pred_var, dy_pred_var, tt_pred_var), cut)
    elif "BDT" in method:
        signal_ds = ROOT.RooDataSet("signal_ds","signal_ds", signal_tree, ROOT.RooArgSet(var, max_abs_eta_var, mu1_eta, mu2_eta, mva_var), cut)
    res = smodel.fitTo(signal_ds, ROOT.RooFit.Range("full"),ROOT.RooFit.Save(), ROOT.RooFit.Verbose(False), ROOT.RooFit.PrintLevel(-1000))
    # res.Print()
    sigParamList = ["mean1", "mean2", "mean3", "width1", "width2", "width3", "mix1", "mix2"]
    for par in sigParamList:
        par_var = w.var("%s_%s"%(cat_name,par))
        par_var.setConstant(True)
    Import(w, smodel)
    return signal_rate

def add_sig_model_with_nuisances(w, cat_name, input_path, cut, res_unc_val, scale_unc_val, method, lumi):
    var = w.var("mass")
    var.setBins(5000)
    max_abs_eta_var = ROOT.RooRealVar("max_abs_eta_mu","Max abs(eta) of muons", 0, 2.4) 
    mu1_eta = ROOT.RooRealVar("mu1_eta","mu1_eta", -2.4, 2.4) 
    mu2_eta = ROOT.RooRealVar("mu2_eta","mu2_eta", -2.4, 2.4)
    if "binary" in method:
        sig_pred_var = ROOT.RooRealVar("sig_prediction", "sig_prediction", 0, 1)
        bkg_pred_var = ROOT.RooRealVar("bkg_prediction", "bkg_prediction", 0, 1)
        signal_tree = ROOT.TChain("tree_sig")
        signal_tree.Add(input_path)
        print "Loaded sig tree from "+input_path+" with %i entries."%signal_tree.GetEntries()    
    elif "multi" in method:
        ggh_pred_var = ROOT.RooRealVar("ggH_prediction", "ggH_prediction", 0, 1)
        vbf_pred_var = ROOT.RooRealVar("VBF_prediction", "VBF_prediction", 0, 1)
        dy_pred_var = ROOT.RooRealVar("DY_prediction", "DY_prediction", 0, 1)
        tt_pred_var = ROOT.RooRealVar("ttbar_prediction", "ttbar_prediction", 0, 1) 

        tree_list = ROOT.TList()

        ggh_tree = ROOT.TChain("tree_H2Mu_gg")
        ggh_tree.Add(input_path)
        ggh_tree_clone = ggh_tree.CloneTree()
        ggh_tree_clone.SetDirectory(0)
        tree_list.Add(ggh_tree_clone)


        vbf_tree = ROOT.TChain("tree_H2Mu_VBF")
        vbf_tree.Add(input_path)  
        vbf_tree_clone = vbf_tree.CloneTree()
        vbf_tree_clone.SetDirectory(0)  
        tree_list.Add(vbf_tree_clone)

        signal_tree = ROOT.TTree.MergeTrees(tree_list)
        print "Loaded ggH tree from "+input_path+" with %i entries."%ggh_tree.GetEntries()    
        print "Loaded VBF tree from "+input_path+" with %i entries."%vbf_tree.GetEntries()  
    elif "BDT" in method:
        mva_var = ROOT.RooRealVar("MVA", "MVA", -1, 1)
        signal_tree = ROOT.TChain("tree")
        signal_tree.Add(input_path)

    signal_tree.SetName("signal_tree")  
    signal_hist_name = "signal_%s"%cat_name
    signal_hist = ROOT.TH1D(signal_hist_name, signal_hist_name, 40, 110, 150)
    dummy = ROOT.TCanvas("dummy", "dummy", 800, 800)
    dummy.cd()
    signal_tree.Draw("mass>>%s"%(signal_hist_name), "(%s)*weight_over_lumi*%s"%(cut, lumi))
    dummy.Close()
    signal_rate = signal_hist.Integral()
    print signal_rate

    w.factory("%s_mix1 [0.5, 0.0, 1.0]"%cat_name)
    w.factory("%s_mix2 [0., 0.0, 1.0]"%cat_name)

    w.factory("mu_res_beta [0, 0, 0]")
    w.factory("mu_scale_beta [0, 0, 0]")

    w.factory("mu_res_unc [%s, %s, %s]"%(res_unc_val, res_unc_val, res_unc_val))
    w.factory("mu_scale_unc [%s, %s, %s]"%(scale_unc_val, scale_unc_val, scale_unc_val))

    w.var("mu_res_unc").setConstant(True)
    w.var("mu_scale_unc").setConstant(True)

    mix1 = w.var("%s_mix1"%cat_name)
    mix2 = w.var("%s_mix2"%cat_name)

    w.factory("EXPR::%s_mean1_times_nuis('%s_mean1*(1 + mu_scale_unc*mu_scale_beta)',{%s_mean1[125.0, 120., 130.],mu_scale_unc,mu_scale_beta})"%(cat_name,cat_name,cat_name))
    w.factory("EXPR::%s_mean2_times_nuis('%s_mean2*(1 + mu_scale_unc*mu_scale_beta)',{%s_mean2[125.0, 120., 130.],mu_scale_unc,mu_scale_beta})"%(cat_name,cat_name,cat_name))
    w.factory("EXPR::%s_mean3_times_nuis('%s_mean3*(1 + mu_scale_unc*mu_scale_beta)',{%s_mean3[125.0, 120., 130.],mu_scale_unc,mu_scale_beta})"%(cat_name,cat_name,cat_name))

    w.factory("expr::%s_deltaM21('%s_mean2-%s_mean1',{%s_mean2, %s_mean1})"%(cat_name,cat_name,cat_name,cat_name,cat_name))
    w.factory("expr::%s_deltaM31('%s_mean3-%s_mean1',{%s_mean3, %s_mean1})"%(cat_name,cat_name,cat_name,cat_name,cat_name))

    w.factory("EXPR::%s_mean2_final('%s_mean2_times_nuis + mu_res_unc*mu_res_beta*%s_deltaM21',{%s_mean2_times_nuis, mu_res_unc, mu_res_beta, %s_deltaM21})"%(cat_name,cat_name,cat_name,cat_name,cat_name))
    w.factory("EXPR::%s_mean3_final('%s_mean3_times_nuis + mu_res_unc*mu_res_beta*%s_deltaM31',{%s_mean3_times_nuis, mu_res_unc, mu_res_beta, %s_deltaM31})"%(cat_name,cat_name,cat_name,cat_name,cat_name))

    w.factory("EXPR::%s_width1_times_nuis('%s_width1*(1 + mu_res_unc*mu_res_beta)',{%s_width1[1.0, 0.5, 5.0],mu_res_unc, mu_res_beta})"%(cat_name,cat_name,cat_name))
    w.factory("EXPR::%s_width2_times_nuis('%s_width2*(1 + mu_res_unc*mu_res_beta)',{%s_width2[5.0, 2.0, 10.],mu_res_unc, mu_res_beta})"%(cat_name,cat_name,cat_name))
    w.factory("EXPR::%s_width3_times_nuis('%s_width3*(1 + mu_res_unc*mu_res_beta)',{%s_width3[5.0, 1.0, 10.],mu_res_unc, mu_res_beta})"%(cat_name,cat_name,cat_name))

    w.factory("Gaussian::%s_gaus1(mass, %s_mean1_times_nuis, %s_width1_times_nuis)"%(cat_name, cat_name, cat_name))
    w.factory("Gaussian::%s_gaus2(mass, %s_mean2_final, %s_width2_times_nuis)"%(cat_name, cat_name, cat_name))
    w.factory("Gaussian::%s_gaus3(mass, %s_mean3_final, %s_width3_times_nuis)"%(cat_name, cat_name, cat_name))

 
    # w.factory("Gaussian::%s_gaus2(mass, %s_mean2_times_nuis, %s_width2[5.0, 2.0, 10])"%(cat_name, cat_name, cat_name))
    # w.factory("Gaussian::%s_gaus3(mass, %s_mean3_times_nuis, %s_width3[5.0, 1.0, 10])"%(cat_name, cat_name, cat_name))

    # w.factory("Gaussian::%s_gaus1(mass, %s_mean1[125.,120.,130.], %s_width1_times_nuis)"%(cat_name, cat_name, cat_name))
    # w.factory("Gaussian::%s_gaus2(mass, %s_mean2[125.,120.,130.], %s_width2[5.,2.,10.])"%(cat_name, cat_name, cat_name))
    # w.factory("Gaussian::%s_gaus3(mass, %s_mean3[125.,120.,130.], %s_width3[5.,1.,10.])"%(cat_name, cat_name, cat_name))

    gaus1 = w.pdf('%s_gaus1'%(cat_name))
    gaus2 = w.pdf('%s_gaus2'%(cat_name))
    gaus3 = w.pdf('%s_gaus3'%(cat_name))
    # gaus12 = ROOT.RooAddPdf('%s_gaus12'%(cat_name), '%s_gaus12'%(cat_name), gaus1, gaus2, mix1)
    # smodel = ROOT.RooAddPdf('%s_sig'%cat_name, '%s_sig'%cat_name, gaus3, gaus12, mix2)
    smodel = ROOT.RooAddPdf('%s_sig'%cat_name, '%s_sig'%cat_name, ROOT.RooArgList(gaus1, gaus2, gaus3) , ROOT.RooArgList(mix1, mix2), ROOT.kTRUE)

    # Import(w,smodel)
    w.Print()
    if "binary" in method:
        signal_ds = ROOT.RooDataSet("signal_ds","signal_ds", signal_tree, ROOT.RooArgSet(var, max_abs_eta_var, mu1_eta, mu2_eta, sig_pred_var, bkg_pred_var), cut)
    elif "multi" in method:
        signal_ds = ROOT.RooDataSet("signal_ds","signal_ds", signal_tree, ROOT.RooArgSet(var, max_abs_eta_var, mu1_eta, mu2_eta, ggh_pred_var, vbf_pred_var, dy_pred_var, tt_pred_var), cut)
    elif "BDT" in method:
        signal_ds = ROOT.RooDataSet("signal_ds","signal_ds", signal_tree, ROOT.RooArgSet(var, max_abs_eta_var, mu1_eta, mu2_eta, mva_var), cut)
    res = smodel.fitTo(signal_ds, ROOT.RooFit.Range("full"),ROOT.RooFit.Save(), ROOT.RooFit.Verbose(False), ROOT.RooFit.PrintLevel(-1000))
    res.Print()
    sigParamList = ["mean1", "mean2", "mean3", "width1", "width2", "width3", "mix1", "mix2"]
    for par in sigParamList:
        par_var = w.var("%s_%s"%(cat_name,par))
        par_var.setConstant(True)
    w.var("mu_res_beta").setRange(-5, 5)
    w.var("mu_scale_beta").setRange(-5, 5)
    w.var("mu_res_beta").setVal(0)
    w.var("mu_scale_beta").setVal(0)
    
    Import(w, smodel)
    return signal_rate

def add_sig_model_dcb(w, cat_name, input_path, cut, method, lumi):
    var = w.var("mass")
    var.setBins(5000)
    max_abs_eta_var = ROOT.RooRealVar("max_abs_eta_mu","Max abs(eta) of muons", 0, 2.4) 
    mu1_eta = ROOT.RooRealVar("mu1_eta","mu1_eta", -2.4, 2.4) 
    mu2_eta = ROOT.RooRealVar("mu2_eta","mu2_eta", -2.4, 2.4)
    if "binary" in method:
        sig_pred_var = ROOT.RooRealVar("sig_prediction", "sig_prediction", 0, 1)
        bkg_pred_var = ROOT.RooRealVar("bkg_prediction", "bkg_prediction", 0, 1)
        signal_tree = ROOT.TChain("tree_sig")
        signal_tree.Add(input_path)
        print "Loaded sig tree from "+input_path+" with %i entries."%signal_tree.GetEntries()    
    elif "multi" in method:
        ggh_pred_var = ROOT.RooRealVar("ggH_prediction", "ggH_prediction", 0, 1)
        vbf_pred_var = ROOT.RooRealVar("VBF_prediction", "VBF_prediction", 0, 1)
        dy_pred_var = ROOT.RooRealVar("DY_prediction", "DY_prediction", 0, 1)
        tt_pred_var = ROOT.RooRealVar("ttbar_prediction", "ttbar_prediction", 0, 1) 

        tree_list = ROOT.TList()

        ggh_tree = ROOT.TChain("tree_H2Mu_gg")
        ggh_tree.Add(input_path)
        ggh_tree_clone = ggh_tree.CloneTree()
        ggh_tree_clone.SetDirectory(0)
        tree_list.Add(ggh_tree_clone)


        vbf_tree = ROOT.TChain("tree_H2Mu_VBF")
        vbf_tree.Add(input_path)  
        vbf_tree_clone = vbf_tree.CloneTree()
        vbf_tree_clone.SetDirectory(0)  
        tree_list.Add(vbf_tree_clone)

        signal_tree = ROOT.TTree.MergeTrees(tree_list)
        print "Loaded ggH tree from "+input_path+" with %i entries."%ggh_tree.GetEntries()    
        print "Loaded VBF tree from "+input_path+" with %i entries."%vbf_tree.GetEntries()       
    elif "BDT" in method:
        mva_var = ROOT.RooRealVar("MVA", "MVA", -1, 1)
        signal_tree = ROOT.TChain("tree")
        signal_tree.Add(input_path)

    signal_tree.SetName("signal_tree")      
    signal_hist_name = "signal_%s"%cat_name
    signal_hist = ROOT.TH1D(signal_hist_name, signal_hist_name, 40, 110, 150)
    dummy = ROOT.TCanvas("dummy", "dummy", 800, 800)
    dummy.cd()
    signal_tree.Draw("mass>>%s"%(signal_hist_name), "(%s)*weight_over_lumi*%s"%(cut, lumi))
    dummy.Close()
    signal_rate = signal_hist.Integral()
    print signal_rate

    # ROOT.gROOT.ProcessLine(".L /home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/RooDCBShape.cxx")
    ROOT.gSystem.Load("/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/RooDCBShape_cxx.so")
    w.factory("RooDCBShape::%s_sig(mass, %s_mean[125,120,130], %s_sigma[2,0,5], %s_alphaL[2,0,25] , %s_alphaR[2,0,25], %s_nL[1.5,0,25], %s_nR[1.5,0,25])"%(cat_name,cat_name,cat_name,cat_name,cat_name,cat_name,cat_name))
    smodel = w.pdf("%s_sig"%cat_name)
    w.Print()
    if "binary" in method:
        signal_ds = ROOT.RooDataSet("signal_ds","signal_ds", signal_tree, ROOT.RooArgSet(var, max_abs_eta_var, mu1_eta, mu2_eta, sig_pred_var, bkg_pred_var), cut)
    elif "multi" in method:
        signal_ds = ROOT.RooDataSet("signal_ds","signal_ds", signal_tree, ROOT.RooArgSet(var, max_abs_eta_var, mu1_eta, mu2_eta, ggh_pred_var, vbf_pred_var, dy_pred_var, tt_pred_var), cut)
    elif "BDT" in method:
        signal_ds = ROOT.RooDataSet("signal_ds","signal_ds", signal_tree, ROOT.RooArgSet(var, max_abs_eta_var, mu1_eta, mu2_eta, mva_var), cut)
    res = smodel.fitTo(signal_ds, ROOT.RooFit.Range("full"),ROOT.RooFit.Save(), ROOT.RooFit.Verbose(False), ROOT.RooFit.PrintLevel(-1000))
    res.Print()
    sigParamList = ["mean", "sigma", "alphaL", "alphaR", "nL", "nR"]
    for par in sigParamList:
        par_var = w.var("%s_%s"%(cat_name,par))
        par_var.setConstant(True)
    Import(w, smodel)
    return signal_rate

def add_sig_model_dcb_with_nuisances(w, cat_name, input_path, cut, res_unc_val, scale_unc_val, method, lumi):
    var = w.var("mass")
    var.setBins(5000)
    max_abs_eta_var = ROOT.RooRealVar("max_abs_eta_mu","Max abs(eta) of muons", 0, 2.4) 
    mu1_eta = ROOT.RooRealVar("mu1_eta","mu1_eta", -2.4, 2.4) 
    mu2_eta = ROOT.RooRealVar("mu2_eta","mu2_eta", -2.4, 2.4)
    if "binary" in method:
        sig_pred_var = ROOT.RooRealVar("sig_prediction", "sig_prediction", 0, 1)
        bkg_pred_var = ROOT.RooRealVar("bkg_prediction", "bkg_prediction", 0, 1)
        signal_tree = ROOT.TChain("tree_sig")
        signal_tree.Add(input_path)
        print "Loaded sig tree from "+input_path+" with %i entries."%signal_tree.GetEntries()    
    elif "multi" in method:
        ggh_pred_var = ROOT.RooRealVar("ggH_prediction", "ggH_prediction", 0, 1)
        vbf_pred_var = ROOT.RooRealVar("VBF_prediction", "VBF_prediction", 0, 1)
        dy_pred_var = ROOT.RooRealVar("DY_prediction", "DY_prediction", 0, 1)
        tt_pred_var = ROOT.RooRealVar("ttbar_prediction", "ttbar_prediction", 0, 1) 

        tree_list = ROOT.TList()

        ggh_tree = ROOT.TChain("tree_H2Mu_gg")
        ggh_tree.Add(input_path)
        ggh_tree_clone = ggh_tree.CloneTree()
        ggh_tree_clone.SetDirectory(0)
        tree_list.Add(ggh_tree_clone)


        vbf_tree = ROOT.TChain("tree_H2Mu_VBF")
        vbf_tree.Add(input_path)  
        vbf_tree_clone = vbf_tree.CloneTree()
        vbf_tree_clone.SetDirectory(0)  
        tree_list.Add(vbf_tree_clone)

        signal_tree = ROOT.TTree.MergeTrees(tree_list)
        print "Loaded ggH tree from "+input_path+" with %i entries."%ggh_tree.GetEntries()    
        print "Loaded VBF tree from "+input_path+" with %i entries."%vbf_tree.GetEntries()       
    elif "BDT" in method:
        mva_var = ROOT.RooRealVar("MVA", "MVA", -1, 1)
        signal_tree = ROOT.TChain("tree")
        signal_tree.Add(input_path)

    signal_tree.SetName("signal_tree")  
    signal_hist_name = "signal_%s"%cat_name
    signal_hist = ROOT.TH1D(signal_hist_name, signal_hist_name, 40, 110, 150)
    dummy = ROOT.TCanvas("dummy", "dummy", 800, 800)
    dummy.cd()
    signal_tree.Draw("mass>>%s"%(signal_hist_name), "(%s)*weight_over_lumi*%s"%(cut, lumi))
    dummy.Close()
    signal_rate = signal_hist.Integral()
    print signal_rate

  
    w.factory("mu_res_beta [0, 0, 0]")
    w.factory("mu_scale_beta [0, 0, 0]")

    w.factory("mu_res_unc [%s, %s, %s]"%(res_unc_val, res_unc_val, res_unc_val))
    w.factory("mu_scale_unc [%s, %s, %s]"%(scale_unc_val, scale_unc_val, scale_unc_val))

    w.var("mu_res_unc").setConstant(True)
    w.var("mu_scale_unc").setConstant(True)
    w.factory("EXPR::%s_mean_times_nuis('%s_mean*(1 + mu_scale_unc*mu_scale_beta)',{%s_mean[125.0, 120., 130.],mu_scale_unc,mu_scale_beta})"%(cat_name,cat_name,cat_name))
    w.factory("EXPR::%s_sigma_times_nuis('%s_sigma*(1 + mu_res_unc*mu_res_beta)',{%s_sigma[2.0, 0., 5.0],mu_res_unc, mu_res_beta})"%(cat_name,cat_name,cat_name))

    ROOT.gSystem.Load("/home/dkondra/Hmumu_analysis/Hmumu_ML/cut_optimization/RooDCBShape_cxx.so")
    w.factory("RooDCBShape::%s_sig(mass, %s_mean_times_nuis, %s_sigma_times_nuis, %s_alphaL[2,0,25] , %s_alphaR[2,0,25], %s_nL[1.5,0,25], %s_nR[1.5,0,25])"%(cat_name,cat_name,cat_name,cat_name,cat_name,cat_name,cat_name))
    smodel = w.pdf("%s_sig"%cat_name)
    w.Print()
    if "binary" in method:
        signal_ds = ROOT.RooDataSet("signal_ds","signal_ds", signal_tree, ROOT.RooArgSet(var, max_abs_eta_var, mu1_eta, mu2_eta, sig_pred_var, bkg_pred_var), cut)
    elif "multi" in method:
        signal_ds = ROOT.RooDataSet("signal_ds","signal_ds", signal_tree, ROOT.RooArgSet(var, max_abs_eta_var, mu1_eta, mu2_eta, ggh_pred_var, vbf_pred_var, dy_pred_var, tt_pred_var), cut)
    elif "BDT" in method:
        signal_ds = ROOT.RooDataSet("signal_ds","signal_ds", signal_tree, ROOT.RooArgSet(var, max_abs_eta_var, mu1_eta, mu2_eta, mva_var), cut)
    res = smodel.fitTo(signal_ds, ROOT.RooFit.Range("full"),ROOT.RooFit.Save(), ROOT.RooFit.Verbose(False), ROOT.RooFit.PrintLevel(-1000))
    res.Print()
    sigParamList = ["mean", "sigma", "alphaL", "alphaR", "nL", "nR"]
    for par in sigParamList:
        par_var = w.var("%s_%s"%(cat_name,par))
        par_var.setConstant(True)

    w.var("mu_res_beta").setRange(-5, 5)
    w.var("mu_scale_beta").setRange(-5, 5)
    w.var("mu_res_beta").setVal(0)
    w.var("mu_scale_beta").setVal(0)    
    Import(w, smodel)
    return signal_rate

def add_bkg_model(w, cat_name, input_path, data_tree, cut, method):
    data = add_data(w, cat_name, input_path, data_tree, cut, method)
    var = w.var("mass")
    var.setBins(5000)
    var.setRange("left",110,120+0.1)
    var.setRange("right",130-0.1,150)
    # data = w.data("%s_data"%cat_name)
    w.factory("%s_a1 [1.66, 0.7, 2.1]"%cat_name)
    w.factory("%s_a2 [0.39, 0.30, 0.62]"%cat_name)
    w.factory("%s_a3 [-0.26, -0.40, -0.12]"%cat_name)
    w.factory("expr::%s_bwz_redux_f('(@1*(@0/100)+@2*(@0/100)^2)',{mass, %s_a2, %s_a3})"%(cat_name,cat_name,cat_name))
    w.factory("EXPR::%s_bkg('exp(@2)*(2.5)/(pow(@0-91.2,@1)+pow(2.5/2,@1))',{mass, %s_a1, %s_bwz_redux_f})"%(cat_name,cat_name,cat_name))
    fit_func = w.pdf('%s_bkg'%cat_name)
    r = fit_func.fitTo(data, ROOT.RooFit.Range("left,right"),ROOT.RooFit.Save(), ROOT.RooFit.Verbose(False), ROOT.RooFit.PrintLevel(-1000))
    # r.Print()
    integral_sb = fit_func.createIntegral(ROOT.RooArgSet(var), ROOT.RooFit.Range("left,right"))
    integral_full = fit_func.createIntegral(ROOT.RooArgSet(var), ROOT.RooFit.Range("full"))
    func_int_sb = integral_sb.getVal()
    func_int_full = integral_full.getVal()
    data_int_sb = data.sumEntries("1","left,right")
    data_int_full = data.sumEntries("1","full")
    bkg_rate = data_int_sb * (func_int_full/func_int_sb)
    print cut, data_int_full
    print "="*100
    return bkg_rate


def make_dnn_categories(categories, sig_input_path, data_input_path, data_tree, output_path, filename, statUnc=False, nuis=False, res_unc_val=0.1, scale_unc_val=0.0005, smodel='3gaus', method="", lumi=40000):
    # nCat = len(bins)-1
    combine_import = ""
    combine_bins = "bin         "
    combine_obs =  "observation "
    combine_bins_str = "bin        "
    combine_proc_str = "process    "
    combine_ipro_str = "process    "
    combine_rate_str = "rate       "
    if statUnc:
        combine_unc = "statUnc  lnN   "
    else:
        combine_unc = ""

    w = create_workspace()
    for cat_name, cut in categories.iteritems():

        print "Applying cut: ", cut

        if '3gaus' in smodel:

            if nuis:
                sig_rate = add_sig_model_with_nuisances(w, cat_name, sig_input_path, cut, res_unc_val, scale_unc_val, method, lumi)    
            else:
                sig_rate = add_sig_model(w, cat_name, sig_input_path, cut, method, lumi) 
        elif 'dcb' in smodel:
            if nuis:
                sig_rate = add_sig_model_dcb_with_nuisances(w, cat_name, sig_input_path, cut, res_unc_val, scale_unc_val, method, lumi)    
            else:
                sig_rate = add_sig_model_dcb(w, cat_name, sig_input_path, cut, method, lumi) 

        bkg_rate = add_bkg_model(w, cat_name, data_input_path, data_tree, cut, method)

        combine_import = combine_import+"shapes %s_bkg  %s %s.root w:%s_bkg\n"%(cat_name, cat_name, filename, cat_name)
        combine_import = combine_import+"shapes %s_sig  %s %s.root w:%s_sig\n"%(cat_name, cat_name, filename, cat_name)
        combine_import = combine_import+"shapes data_obs  %s %s.root w:%s_data\n"%(cat_name, filename, cat_name)

        combine_bins = combine_bins+cat_name+" "
        combine_obs = combine_obs+"-1   "

        combine_bins_str = combine_bins_str+ "{:14s}{:14s}".format(cat_name,cat_name)
        combine_proc_str = combine_proc_str+ "%s_sig      %s_bkg      "%(cat_name, cat_name)
        combine_ipro_str = combine_ipro_str+ "0             1             "
        combine_rate_str = combine_rate_str+ "{:<14f}{:<14f}".format(sig_rate, bkg_rate)
        if statUnc:
            combine_unc = combine_unc+ "{:<14f}{:<14f}".format(sqrt(sig_rate), sqrt(bkg_rate))


    combine_bins_str = combine_bins_str+"\n"
    combine_proc_str = combine_proc_str+"\n"
    combine_ipro_str = combine_ipro_str+"\n"
    combine_rate_str = combine_rate_str+"\n"
    # w.Print()
    workspace_file = ROOT.TFile.Open(output_path+filename+".root", "recreate")
    workspace_file.cd()
    w.Write()
    workspace_file.Close()

    return combine_import, combine_bins+"\n"+combine_obs+"\n", combine_bins_str+combine_proc_str+combine_ipro_str+combine_rate_str, combine_unc+"\n"


def create_datacard(categories, sig_in_path, data_in_path, data_tree, out_path, datacard_name, workspace_filename, statUnc=False, nuis=False, res_unc_val=0.1, scale_unc_val=0.0005, smodel='3gaus', method="", lumi=40000): 
    print "method:", method
    print "="*30
    print "Categories: "
    for key, value in categories.iteritems():
        print key, value
    print "="*30
    try:
        os.makedirs(out_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    import_str, bins_obs, cat_strings, unc_str = make_dnn_categories(categories, sig_in_path, data_in_path, data_tree, out_path, workspace_filename, statUnc=statUnc, nuis=nuis, res_unc_val=res_unc_val, scale_unc_val=scale_unc_val, smodel=smodel, method=method, lumi=lumi)
    out_file = open(out_path+datacard_name+".txt", "w")
    out_file.write("imax *\n")
    out_file.write("jmax *\n")
    out_file.write("kmax *\n")
    out_file.write("---------------\n")
    out_file.write(import_str)
    out_file.write("---------------\n")
    out_file.write(bins_obs)
    out_file.write("------------------------------\n")
    out_file.write(cat_strings)
    out_file.write("------------------------------\n")
    if statUnc:
        out_file.write(unc_str)
    if nuis:
        out_file.write("mu_res_beta    param    0    1.\n")
        out_file.write("mu_scale_beta    param    0    1.\n")
    out_file.close()

    