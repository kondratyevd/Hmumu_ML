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

def add_data(w, cat_number, input_path, cut):
    var = w.var("mass")
    var.setBins(5000)
    max_abs_eta_var = ROOT.RooRealVar("max_abs_eta_mu","Max abs(eta) of muons", 0, 2.4) 
    mu1_eta = ROOT.RooRealVar("mu1_eta","mu1_eta", -2.4, 2.4) 
    mu2_eta = ROOT.RooRealVar("mu2_eta","mu2_eta", -2.4, 2.4) 
    data_tree = ROOT.TChain("tree_Data")
    data_tree.Add(input_path+"/output_Data.root")  
    data_hist_name = "data_%i"%cat_number
    data_hist = ROOT.TH1D(data_hist_name, data_hist_name, 40, 110, 150)
    dummy = ROOT.TCanvas("dummy", "dummy", 800, 800)
    dummy.cd()
    data_tree.Draw("mass>>%s"%(data_hist_name), cut)
    dummy.Close()
    data = ROOT.RooDataSet("cat%i_data"%cat_number,"cat%i_data"%cat_number, data_tree, ROOT.RooArgSet(var, max_abs_eta_var, mu1_eta, mu2_eta), cut)
    Import(w, data)
    return w.data("cat%i_data"%cat_number)

def add_sig_model(w, cat_number, input_path, cut):
    var = w.var("mass")
    var.setBins(5000)
    max_abs_eta_var = ROOT.RooRealVar("max_abs_eta_mu","Max abs(eta) of muons", 0, 2.4) 
    mu1_eta = ROOT.RooRealVar("mu1_eta","mu1_eta", -2.4, 2.4) 
    mu2_eta = ROOT.RooRealVar("mu2_eta","mu2_eta", -2.4, 2.4) 
    signal_tree = ROOT.TChain("tree_H2Mu_gg")
    signal_tree.Add(input_path+"/output_train.root")  
    signal_hist_name = "signal_%i"%cat_number
    signal_hist = ROOT.TH1D(signal_hist_name, signal_hist_name, 40, 110, 150)
    dummy = ROOT.TCanvas("dummy", "dummy", 800, 800)
    dummy.cd()
    signal_tree.Draw("mass>>%s"%(signal_hist_name), "(%s)*weight"%(cut)) # only 80% of events were saved in "train" file, hence the weight
    dummy.Close()
    signal_rate = signal_hist.Integral()
    print signal_rate

    w.factory("cat%i_mixGG [0.5, 0.0, 1.0]"%cat_number)
    w.factory("cat%i_mixGG1 [0.5, 0.0, 1.0]"%cat_number)
    mixGG = w.var("cat%i_mixGG"%cat_number)
    mixGG1 = w.var("cat%i_mixGG1"%cat_number)
    w.factory("Gaussian::cat%i_gaus1(mass, cat%i_mean1[125., 120., 130.], cat%i_width1[1.0, 0.5, 5.0])"%(cat_number, cat_number, cat_number))
    w.factory("Gaussian::cat%i_gaus2(mass, cat%i_mean2[125., 120., 130.], cat%i_width2[5.0, 2.0, 10.])"%(cat_number, cat_number, cat_number))
    w.factory("Gaussian::cat%i_gaus3(mass, cat%i_mean3[125., 120., 130.], cat%i_width3[5.0, 1.0, 10.])"%(cat_number, cat_number, cat_number))
    gaus1 = w.pdf('cat%i_gaus1'%(cat_number))
    gaus2 = w.pdf('cat%i_gaus2'%(cat_number))
    gaus3 = w.pdf('cat%i_gaus3'%(cat_number))
    gaus12 = ROOT.RooAddPdf('cat%i_gaus12'%(cat_number), 'cat%i_gaus12'%(cat_number), gaus1, gaus2, mixGG)
    smodel = ROOT.RooAddPdf('cat%i_ggh'%cat_number, 'cat%i_ggh'%cat_number, gaus3, gaus12, mixGG1)
    # Import(w,smodel)
    w.Print()
    signal_ds = ROOT.RooDataSet("signal_ds","signal_ds", signal_tree, ROOT.RooArgSet(var, max_abs_eta_var, mu1_eta, mu2_eta), cut)
    res = smodel.fitTo(signal_ds, ROOT.RooFit.Range("full"),ROOT.RooFit.Save(), ROOT.RooFit.Verbose(False))
    res.Print()
    sigParamList = ["mean1", "mean2", "mean3", "width1", "width2", "width3", "mixGG", "mixGG1"]
    for par in sigParamList:
        par_var = w.var("cat%s_%s"%(cat_number,par))
        par_var.setConstant(True)
    Import(w, smodel)
    return signal_rate

def add_sig_model_with_nuisances(w, cat_number, input_path, cut):
    var = w.var("mass")
    var.setBins(5000)
    max_abs_eta_var = ROOT.RooRealVar("max_abs_eta_mu","Max abs(eta) of muons", 0, 2.4) 
    mu1_eta = ROOT.RooRealVar("mu1_eta","mu1_eta", -2.4, 2.4) 
    mu2_eta = ROOT.RooRealVar("mu2_eta","mu2_eta", -2.4, 2.4) 
    signal_tree = ROOT.TChain("tree_H2Mu_gg")
    signal_tree.Add(input_path+"/output_train.root")  
    signal_hist_name = "signal_%i"%cat_number
    signal_hist = ROOT.TH1D(signal_hist_name, signal_hist_name, 40, 110, 150)
    dummy = ROOT.TCanvas("dummy", "dummy", 800, 800)
    dummy.cd()
    signal_tree.Draw("mass>>%s"%(signal_hist_name), "(%s)*weight"%(cut)) # only 80% of events were saved in "train" file, hence the weight
    dummy.Close()
    signal_rate = signal_hist.Integral()
    print signal_rate

    w.factory("cat%i_mixGG [0.5, 0.0, 1.0]"%cat_number)
    w.factory("cat%i_mixGG1 [0.5, 0.0, 1.0]"%cat_number)

    # w.factory("cat%i_mu_res_beta [0, 0, 0]"%cat_number)
    # w.factory("cat%i_mu_scale_beta [0, 0, 0]"%cat_number)

    # w.factory("mu_res_beta [0, 0, 0]")
    w.factory("mu_res_beta [0, 0, 0]")
    w.factory("mu_scale_beta [0, 0, 0]")

    w.factory("cat%i_mu_res_unc [0.1, 0.1, 0.1]"%cat_number)
    w.factory("cat%i_mu_scale_unc [0.0005, 0.0005, 0.0005]"%cat_number)

    w.var("cat%i_mu_res_unc"%cat_number).setConstant(True)
    w.var("cat%i_mu_scale_unc"%cat_number).setConstant(True)

    mixGG = w.var("cat%i_mixGG"%cat_number)
    mixGG1 = w.var("cat%i_mixGG1"%cat_number)

    # w.factory("EXPR::cat%i_mean1_times_nuis('cat%i_mean1*(1 + cat%i_mu_scale_unc*mu_scale_beta)',{cat%i_mean1[125.0, 120., 130.],cat%i_mu_scale_unc,mu_scale_beta})"%(cat_number,cat_number,cat_number,cat_number,cat_number))
    # w.factory("EXPR::cat%i_mean2_times_nuis('cat%i_mean2*(1 + cat%i_mu_scale_unc*mu_scale_beta)',{cat%i_mean2[125.0, 120., 130.],cat%i_mu_scale_unc,mu_scale_beta})"%(cat_number,cat_number,cat_number,cat_number,cat_number))
    # w.factory("EXPR::cat%i_mean3_times_nuis('cat%i_mean3*(1 + cat%i_mu_scale_unc*mu_scale_beta)',{cat%i_mean3[125.0, 120., 130.],cat%i_mu_scale_unc,mu_scale_beta})"%(cat_number,cat_number,cat_number,cat_number,cat_number))

    # w.factory("expr::cat%i_deltaM21('cat%i_mean2-cat%i_mean1',{cat%i_mean2, cat%i_mean1})"%(cat_number,cat_number,cat_number,cat_number,cat_number))
    # w.factory("expr::cat%i_deltaM31('cat%i_mean3-cat%i_mean1',{cat%i_mean3, cat%i_mean1})"%(cat_number,cat_number,cat_number,cat_number,cat_number))

    # w.factory("EXPR::cat%i_mean2_final('cat%i_mean2_times_nuis + cat%i_mu_res_unc*mu_res_beta*cat%i_deltaM21',{cat%i_mean2_times_nuis, cat%i_mu_res_unc, mu_res_beta, cat%i_deltaM21})"%(cat_number,cat_number,cat_number,cat_number,cat_number,cat_number,cat_number))
    # w.factory("EXPR::cat%i_mean3_final('cat%i_mean3_times_nuis + cat%i_mu_res_unc*mu_res_beta*cat%i_deltaM31',{cat%i_mean3_times_nuis, cat%i_mu_res_unc, mu_res_beta, cat%i_deltaM31})"%(cat_number,cat_number,cat_number,cat_number,cat_number,cat_number,cat_number))

    w.factory("EXPR::cat%i_width1_times_nuis('cat%i_width1*(1 + cat%i_mu_res_unc*mu_res_beta)',{cat%i_width1[1.0, 0.5, 5.0],cat%i_mu_res_unc, mu_res_beta})"%(cat_number,cat_number,cat_number,cat_number,cat_number))
    # w.factory("EXPR::cat%i_width2_times_nuis('cat%i_width2*(1 + cat%i_mu_res_unc*mu_res_beta)',{cat%i_width2[5.0, 2.0, 10.],cat%i_mu_res_unc, mu_res_beta})"%(cat_number,cat_number,cat_number,cat_number,cat_number))
    # w.factory("EXPR::cat%i_width3_times_nuis('cat%i_width3*(1 + cat%i_mu_res_unc*mu_res_beta)',{cat%i_width3[5.0, 1.0, 10.],cat%i_mu_res_unc, mu_res_beta})"%(cat_number,cat_number,cat_number,cat_number,cat_number))

    # w.factory("Gaussian::cat%i_gaus1(mass, cat%i_mean1_times_nuis, cat%i_width1_times_nuis)"%(cat_number, cat_number, cat_number))
    # w.factory("Gaussian::cat%i_gaus2(mass, cat%i_mean2_final, cat%i_width2_times_nuis)"%(cat_number, cat_number, cat_number))
    # w.factory("Gaussian::cat%i_gaus3(mass, cat%i_mean3_final, cat%i_width3_times_nuis)"%(cat_number, cat_number, cat_number))
    # w.factory("Gaussian::cat%i_gaus2(mass, cat%i_mean2_times_nuis, cat%i_width2_times_nuis)"%(cat_number, cat_number, cat_number))
    # w.factory("Gaussian::cat%i_gaus3(mass, cat%i_mean3_times_nuis, cat%i_width3_times_nuis)"%(cat_number, cat_number, cat_number))

    w.factory("Gaussian::cat%i_gaus1(mass, cat%i_mean1[125.,120.,130.], cat%i_width1_times_nuis)"%(cat_number, cat_number, cat_number))
    w.factory("Gaussian::cat%i_gaus2(mass, cat%i_mean2[125.,120.,130.], cat%i_width2[5.,2.,10.])"%(cat_number, cat_number, cat_number))
    w.factory("Gaussian::cat%i_gaus3(mass, cat%i_mean3[125.,120.,130.], cat%i_width3[5.,1.,10.])"%(cat_number, cat_number, cat_number))

    gaus1 = w.pdf('cat%i_gaus1'%(cat_number))
    gaus2 = w.pdf('cat%i_gaus2'%(cat_number))
    gaus3 = w.pdf('cat%i_gaus3'%(cat_number))
    gaus12 = ROOT.RooAddPdf('cat%i_gaus12'%(cat_number), 'cat%i_gaus12'%(cat_number), gaus1, gaus2, mixGG)
    smodel = ROOT.RooAddPdf('cat%i_ggh'%cat_number, 'cat%i_ggh'%cat_number, gaus3, gaus12, mixGG1)
    # Import(w,smodel)
    w.Print()
    signal_ds = ROOT.RooDataSet("signal_ds","signal_ds", signal_tree, ROOT.RooArgSet(var, max_abs_eta_var, mu1_eta, mu2_eta), cut)
    res = smodel.fitTo(signal_ds, ROOT.RooFit.Range("full"),ROOT.RooFit.Save(), ROOT.RooFit.Verbose(False))
    res.Print()
    sigParamList = ["mean1", "mean2", "mean3", "width1", "width2", "width3", "mixGG", "mixGG1"]
    for par in sigParamList:
        par_var = w.var("cat%s_%s"%(cat_number,par))
        par_var.setConstant(True)
    w.var("mu_res_beta").setRange(-5, 5)
    w.var("mu_scale_beta").setRange(-5, 5)
    Import(w, smodel)
    return signal_rate

def add_bkg_model(w, cat_number, input_path, cut):
    data = add_data(w, cat_number, input_path, cut)
    var = w.var("mass")
    var.setBins(5000)
    var.setRange("left",110,120+0.1)
    var.setRange("right",130-0.1,150)
    # data = w.data("cat%i_data"%cat_number)
    w.factory("cat%i_a1 [1.66, 0.7, 2.1]"%cat_number)
    w.factory("cat%i_a2 [0.39, 0.30, 0.62]"%cat_number)
    w.factory("cat%i_a3 [-0.26, -0.40, -0.12]"%cat_number)
    w.factory("expr::cat%i_bwz_redux_f('(@1*(@0/100)+@2*(@0/100)^2)',{mass, cat%i_a2, cat%i_a3})"%(cat_number,cat_number,cat_number))
    w.factory("EXPR::cat%i_bkg('exp(@2)*(2.5)/(pow(@0-91.2,@1)+pow(2.5/2,@1))',{mass, cat%i_a1, cat%i_bwz_redux_f})"%(cat_number,cat_number,cat_number))
    fit_func = w.pdf('cat%i_bkg'%cat_number)
    r = fit_func.fitTo(data, ROOT.RooFit.Range("left,right"),ROOT.RooFit.Save(), ROOT.RooFit.Verbose(False))
    r.Print()
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


def make_eta_categories(bins, input_path, output_path, filename, statUnc=False, nuis=False):
    nCat = len(bins)-1
    cat_names = []
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
    for i, b in enumerate(bins):
        if i is len(bins)-1:
            break
        eta_min = bins[i]
        eta_max = bins[i+1]    
        name = "cat%i"%i
        cat_names.append(name)

        cut = "((max_abs_eta_mu>%.5f)&(max_abs_eta_mu<%.5f))"%(eta_min, eta_max)
        
        if nuis:
            sig_rate = add_sig_model_with_nuisances(w, i, input_path, cut)    
        else:
            sig_rate = add_sig_model(w, i, input_path, cut) 

        bkg_rate = add_bkg_model(w, i, input_path, cut)

        combine_import = combine_import+"shapes cat%i_bkg  cat%i %s.root w:cat%i_bkg\n"%(i,i,filename,i)
        combine_import = combine_import+"shapes cat%i_ggh  cat%i %s.root w:cat%i_ggh\n"%(i,i,filename,i)
        combine_import = combine_import+"shapes data_obs  cat%i %s.root w:cat%i_data\n"%(i,filename,i)

        combine_bins = combine_bins+name+" "
        combine_obs = combine_obs+"-1   "

        combine_bins_str = combine_bins_str+ "{:14s}{:14s}".format(name,name)
        combine_proc_str = combine_proc_str+ "cat%i_ggh      cat%i_bkg      "%(i,i)
        combine_ipro_str = combine_ipro_str+ "0             1             "
        combine_rate_str = combine_rate_str+ "{:<14f}{:<14f}".format(sig_rate, bkg_rate)
        if statUnc:
            combine_unc = combine_unc+ "{:<14f}{:<14f}".format(sqrt(sig_rate), sqrt(bkg_rate))


    combine_bins_str = combine_bins_str+"\n"
    combine_proc_str = combine_proc_str+"\n"
    combine_ipro_str = combine_ipro_str+"\n"
    combine_rate_str = combine_rate_str+"\n"
    w.Print()
    workspace_file = ROOT.TFile.Open(output_path+filename+".root", "recreate")
    workspace_file.cd()
    w.Write()
    workspace_file.Close()

    return combine_import, combine_bins+"\n"+combine_obs+"\n", combine_bins_str+combine_proc_str+combine_ipro_str+combine_rate_str, combine_unc+"\n"


def create_datacard(bins, in_path, out_path, name, workspace_filename, statUnc=False, nuis=False): 
    try:
        os.makedirs(out_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    import_str, bins_obs, cat_strings, unc_str = make_eta_categories(bins, in_path, out_path, workspace_filename, statUnc=statUnc, nuis=nuis)
    out_file = open(out_path+name+".txt", "w")
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

    print "Created datacard for bins ", bins