import ROOT
from math import sqrt
import os, sys, errno
Import = getattr(ROOT.RooWorkspace, 'import')

def create_workspace():
    var = ROOT.RooRealVar("hmass","Dilepton mass",110,150)     
    var.setBins(100)
    var.setRange("window",120,130)
    var.setRange("full",110,150)
    w = ROOT.RooWorkspace("w", False)
    Import(w, var)

    return w

def add_sig_model(w, cat_name, sig_path_list, cut):
    var = w.var("hmass")
    var.setBins(5000)

    bdtuf               = ROOT.RooRealVar("bdtuf", "bdtuf", -1, 1)
    bdtucsd_inclusive   = ROOT.RooRealVar("bdtucsd_inclusive", "bdtucsd_inclusive", -1, 1)
    bdtucsd_01jet       = ROOT.RooRealVar("bdtucsd_01jet", "bdtucsd_01jet", -1, 1)
    bdtucsd_2jet        = ROOT.RooRealVar("bdtucsd_2jet", "bdtucsd_2jet", -1, 1)
    njets               = ROOT.RooRealVar("njets", "njets", 0, 10)

    signal_tree = ROOT.TChain("tree")
    for path in sig_path_list:
        signal_tree.Add(path)
    signal_tree.SetName("signal_tree")

    signal_hist_name = "signal_%s"%cat_name
    signal_hist = ROOT.TH1D(signal_hist_name, signal_hist_name, 80, 115, 135)
    dummy = ROOT.TCanvas("dummy", "dummy", 800, 800)
    dummy.cd()
    signal_tree.Draw("hmass>>%s"%(signal_hist_name), "(%s)*weight"%(cut))
    dummy.Close()    

    sig_entries = signal_hist.GetEntries()
    signal_rate = signal_hist.Integral()
    print cut
    print "sig_entries = %f, sig_rate = %f"%(sig_entries, signal_rate)
    # if (signal_rate<1):
    #     return signal_rate, sig_entries

    w.factory("%s_mix1 [0.5, 0.0, 1.0]"%cat_name)
    w.factory("%s_mix2 [0.01, 0.0, 0.9]"%cat_name)
    mix1 = w.var("%s_mix1"%cat_name)
    mix2 = w.var("%s_mix2"%cat_name)
    w.factory("Gaussian::%s_gaus1(hmass, %s_mean1[125., 120., 130.], %s_width1[1.0, 0.5, 5.0])"%(cat_name, cat_name, cat_name))
    w.factory("Gaussian::%s_gaus2(hmass, %s_mean2[125., 115., 130.], %s_width2[5.0, 1.0, 10.])"%(cat_name, cat_name, cat_name))
    w.factory("Gaussian::%s_gaus3(hmass, %s_mean3[125., 115., 130.], %s_width3[5.0, 1.0, 10.])"%(cat_name, cat_name, cat_name))
    gaus1 = w.pdf('%s_gaus1'%(cat_name))
    gaus2 = w.pdf('%s_gaus2'%(cat_name))
    gaus3 = w.pdf('%s_gaus3'%(cat_name))
    smodel = ROOT.RooAddPdf('%s_sig'%cat_name, '%s_sig'%cat_name, ROOT.RooArgList(gaus1, gaus2, gaus3) , ROOT.RooArgList(mix1, mix2), ROOT.kTRUE)


    sig_binned = ROOT.RooDataHist("%s_sig_hist"%cat_name,"%s_sig_hist"%cat_name, ROOT.RooArgList(var), signal_hist)
    Import(w, sig_binned)
    # sig_binned.Print()
    cmdlist = ROOT.RooLinkedList()
    cmd0 = ROOT.RooFit.Range(120,130)
    cmd1 = ROOT.RooFit.Save()
    cmd2 = ROOT.RooFit.Verbose(False)
    cmd3 = ROOT.RooFit.PrintLevel(-1000)

    cmdlist.Add(cmd0)
    cmdlist.Add(cmd1)
    cmdlist.Add(cmd2)
    cmdlist.Add(cmd3)

    try:
        res = smodel.chi2FitTo(sig_binned, cmdlist)
        # res.Print()
    except:
        return 0, 0

    frame = var.frame()
    sig_binned.plotOn(frame, ROOT.RooFit.Name("%s_sig_hist"%cat_name))
    smodel.plotOn(frame, ROOT.RooFit.Name('%s_sig'%cat_name))

    chi2 = frame.chiSquare('%s_sig'%cat_name, "%s_sig_hist"%cat_name, 8)

    if chi2>10:
        res.Print()
        canv = ROOT.TCanvas("canv5", "canv5", 800, 800)
        canv.cd()
        frame.Draw()
        canv.Print("signal_fit.png")
        print cut
        print "Signal chi2/d.o.f: ", chi2
        return 0, 0

    sigParamList = ["mean1", "mean2", "mean3", "width1", "width2", "width3", "mix1", "mix2"]
    for par in sigParamList:
        par_var = w.var("%s_%s"%(cat_name,par))
        par_var.setConstant(True)
    Import(w, smodel)
    return signal_rate, sig_entries



def add_bkg_model(w, cat_name, bkg_path_list, cut):
    var = w.var("hmass")
    var.setBins(5000)


    bdtuf               = ROOT.RooRealVar("bdtuf", "bdtuf", -1, 1)
    bdtucsd_inclusive   = ROOT.RooRealVar("bdtucsd_inclusive", "bdtucsd_inclusive", -1, 1)
    bdtucsd_01jet       = ROOT.RooRealVar("bdtucsd_01jet", "bdtucsd_01jet", -1, 1)
    bdtucsd_2jet        = ROOT.RooRealVar("bdtucsd_2jet", "bdtucsd_2jet", -1, 1)
    njets               = ROOT.RooRealVar("njets", "njets", 0, 10)

    bkg_tree = ROOT.TChain("tree")
    for path in bkg_path_list:
        bkg_tree.Add(path)
    bkg_tree.SetName("bkg_tree")

    bkg_hist_name = "bkg_%s"%cat_name
    bkg_hist = ROOT.TH1D(bkg_hist_name, bkg_hist_name, 160, 110, 150)
    dummy = ROOT.TCanvas("dummy", "dummy", 800, 800)
    dummy.cd()
    bkg_tree.Draw("hmass>>%s"%(bkg_hist_name), "(%s)*weight"%(cut))
    dummy.Close()    

    bkg_entries = bkg_hist.GetEntries()
    bkg_rate = bkg_hist.Integral()
    # print cut
    # print "bkg_entries = %f, bkg_rate = %f"%(bkg_entries, bkg_rate)

    # if (bkg_entries<1000):
    #     return 0, bkg_entries


    w.factory("%s_a1 [1.66, 0.1, 2.1]"%cat_name)
    w.factory("%s_a2 [0.39, 0.30, 4]"%cat_name)
    w.factory("%s_a3 [-0.26, -2.40, -0.01]"%cat_name)
    w.factory("expr::%s_bwz_redux_f('(@1*(@0/100)+@2*(@0/100)^2)',{hmass, %s_a2, %s_a3})"%(cat_name,cat_name,cat_name))
    w.factory("EXPR::%s_bkg('exp(@2)*(2.5)/(pow(@0-91.2,@1)+pow(2.5/2,@1))',{hmass, %s_a1, %s_bwz_redux_f})"%(cat_name,cat_name,cat_name))
    fit_func = w.pdf('%s_bkg'%cat_name)
    

    bkg_binned = ROOT.RooDataHist("%s_bkg_hist"%cat_name,"%s_bkg_hist"%cat_name, ROOT.RooArgList(var), bkg_hist)
    Import(w, bkg_binned)
    # bkg_binned.Print()
    cmdlist = ROOT.RooLinkedList()
    cmd1 = ROOT.RooFit.Save()
    cmd2 = ROOT.RooFit.Verbose(False)
    cmd3 = ROOT.RooFit.PrintLevel(-1000)

    cmdlist.Add(cmd1)
    cmdlist.Add(cmd2)
    cmdlist.Add(cmd3)

    try:
        r = fit_func.chi2FitTo(bkg_binned, cmdlist)
    except:
        return 0, 0

    frame = var.frame()
    bkg_binned.plotOn(frame, ROOT.RooFit.Name("%s_bkg_hist"%cat_name))
    fit_func.plotOn(frame, ROOT.RooFit.Name('%s_bkg'%cat_name))

    chi2 = frame.chiSquare('%s_bkg'%cat_name, "%s_bkg_hist"%cat_name, 3)

    if chi2>10:
        r.Print()
        canv = ROOT.TCanvas("canv", "canv", 800, 800)
        canv.cd()
        frame.Draw()
        canv.Print("bkg_fit.png")
        print cut 
        print "Background chi2/d.o.f: ", chi2
        return 0, 0


    data_obs = ROOT.RooDataSet("%s_data"%cat_name,"%s_data"%cat_name, ROOT.RooArgSet(var, bdtuf, bdtucsd_inclusive, bdtucsd_01jet, bdtucsd_2jet, njets))
    Import(w, data_obs)

    return bkg_rate, bkg_entries


def make_categories_ucsd(categories, sig_path_list, bkg_path_list, output_path, filename):
    # nCat = len(bins)-1
    valid = True
    combine_import = ""
    combine_bins = "bin         "
    combine_obs =  "observation "
    combine_bins_str = "bin        "
    combine_proc_str = "process    "
    combine_ipro_str = "process    "
    combine_rate_str = "rate       "
    combine_unc = ""

    w = create_workspace()
    for cat_name, cut in categories.iteritems():

        # print "Applying cut: ", cut
        sig_rate, sig_entries = add_sig_model(w, cat_name, sig_path_list, cut) 
        bkg_rate, bkg_entries = add_bkg_model(w, cat_name, bkg_path_list, cut)

        # if (sig_rate<1) or (bkg_rate<1):
        #     valid = False

        combine_import = combine_import+"shapes %s_bkg  %s %s.root w:%s_bkg\n"%(cat_name, cat_name, filename, cat_name)
        combine_import = combine_import+"shapes %s_sig  %s %s.root w:%s_sig\n"%(cat_name, cat_name, filename, cat_name)
        combine_import = combine_import+"shapes data_obs  %s %s.root w:%s_data\n"%(cat_name, filename, cat_name)

        combine_bins = combine_bins+cat_name+" "
        combine_obs = combine_obs+"-1   "

        combine_bins_str = combine_bins_str+ "{:14s}{:14s}".format(cat_name,cat_name)
        combine_proc_str = combine_proc_str+ "%s_sig      %s_bkg      "%(cat_name, cat_name)
        combine_ipro_str = combine_ipro_str+ "0             1             "
        combine_rate_str = combine_rate_str+ "{:<14f}{:<14f}".format(sig_rate, bkg_rate)


    combine_bins_str = combine_bins_str+"\n"
    combine_proc_str = combine_proc_str+"\n"
    combine_ipro_str = combine_ipro_str+"\n"
    combine_rate_str = combine_rate_str+"\n"
    # w.Print()
    workspace_file = ROOT.TFile.Open(output_path+filename+".root", "recreate")
    workspace_file.cd()
    w.Write()
    workspace_file.Close()

    return combine_import, combine_bins+"\n"+combine_obs+"\n", combine_bins_str+combine_proc_str+combine_ipro_str+combine_rate_str, combine_unc+"\n", valid


def create_datacard_ucsd(categories, sig_path_list, bkg_path_list, out_path, datacard_name, workspace_filename): 

    try:
        os.makedirs(out_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    import_str, bins_obs, cat_strings, unc_str, valid = make_categories_ucsd(categories, sig_path_list, bkg_path_list, out_path, workspace_filename)
    
    if not valid:
        return False

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
    out_file.close()

    return True

    