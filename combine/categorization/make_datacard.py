# Goal:

# imax 1
# jmax 1
# kmax *
# ---------------
# shapes * * workspace.root w:$PROCESS
# ---------------
# bin bin1
# observation -1
# ------------------------------
# bin          bin1         bin1
# process      signal       background
# process      0            1

# # from unbinned fit
# rate        210.139180    450918.137735 

import ROOT
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

def add_data(w, cat_number, path, cut):
    var = w.var("mass")
    max_abs_eta_var = ROOT.RooRealVar("max_abs_eta_mu","Max abs(eta) of muons", 0, 2.4) 
    data_tree = ROOT.TChain("tree_Data")
    data_tree.Add(path+"/output_Data.root")  
    data_hist_name = "data_"+cut
    data_hist = ROOT.TH1D(data_hist_name, data_hist_name, 40, 110, 150)
    dummy = ROOT.TCanvas("dummy", "dummy", 800, 800)
    dummy.cd()
    data_tree.Draw("mass>>%s"%(data_hist_name), cut)
    dummy.Close()
    data = ROOT.RooDataSet("cat%i_data"%cat_number,"cat%i_data"%cat_number, data_tree, ROOT.RooArgSet(var, max_abs_eta_var), cut)
    Import(w, data)
    return w.data("cat%i_data"%cat_number)

def add_sig_model(w, cat_number, path, cut):
    var = w.var("mass")
    max_abs_eta_var = ROOT.RooRealVar("max_abs_eta_mu","Max abs(eta) of muons", 0, 2.4) 

    signal_tree = ROOT.TChain("tree_H2Mu_gg")
    # signal_tree.Add(path+"/output_test.root")  
    signal_tree.Add(path+"/output_train.root")  
    signal_hist_name = "signal_%i"%cat_number
    signal_hist = ROOT.TH1D(signal_hist_name, signal_hist_name, 40, 110, 150)
    dummy = ROOT.TCanvas("dummy", "dummy", 800, 800)
    dummy.cd()
    # signal_tree.Draw("mass>>%s"%(signal_hist_name), "(%s)*weight*5"%(cut)) # only 20% of events were saved in "test" file, hence the weight
    signal_tree.Draw("mass>>%s"%(signal_hist_name), "(%s)*weight*5/4"%(cut)) # only 80% of events were saved in "train" file, hence the weight
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
    Import(w,smodel)
    w.Print()
    signal_ds = ROOT.RooDataSet("signal_ds","signal_ds", signal_tree, ROOT.RooArgSet(var, max_abs_eta_var), cut)
    res = smodel.fitTo(signal_ds, ROOT.RooFit.Range("full"),ROOT.RooFit.Save(), ROOT.RooFit.Verbose(False))
    res.Print()
    sigParamList = ["mean1", "mean2", "mean3", "width1", "width2", "width3", "mixGG", "mixGG1"]
    for par in sigParamList:
        par_var = w.var("cat%s_%s"%(cat_number,par))
        par_var.setConstant(True)
    Import(w, smodel)

    return signal_rate

def add_bkg_model(w, cat_number, path, cut):
    data = add_data(w, cat_number, path, cut)
    var = w.var("mass")
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
    return bkg_rate

def get_rates(w, cat_number, eta_min, eta_max):
    # path = "output/Run_2018-12-19_14-25-02/Keras_multi/model_50_D2_25_D2_25_D2/root/"     # no index
    path = "output/Run_2019-01-18_14-34-07/Keras_multi/model_50_D2_25_D2_25_D2/root/"      # index '1'
      
    eta_cut = "((max_abs_eta_mu>%.5f)&(max_abs_eta_mu<%.5f))"%(eta_min, eta_max)
    sig_rate = add_sig_model(w, cat_number, path, eta_cut) 
    bkg_rate = add_bkg_model(w, cat_number, path, eta_cut)

    return sig_rate, bkg_rate

def make_eta_categories(bins, path, filename):
    nCat = len(bins)-1
    cat_names = []
    combine_import = ""
    combine_bins = "bin         "
    combine_obs =  "observation "
    combine_bins_str = "bin        "
    combine_proc_str = "process    "
    combine_ipro_str = "process    "
    combine_rate_str = "rate       "
    w = create_workspace()
    for i, b in enumerate(bins):
        if i is len(bins)-1:
            break
        eta_min = bins[i]
        eta_max = bins[i+1]    
        name = "cat%i"%i
        cat_names.append(name)

        rate_s, rate_b = get_rates(w, i, eta_min, eta_max)

        combine_import = combine_import+"shapes cat%i_bkg  cat%i %s.root w:cat%i_bkg\n"%(i,i,filename,i)
        combine_import = combine_import+"shapes cat%i_ggh  cat%i %s.root w:cat%i_ggh\n"%(i,i,filename,i)
        combine_import = combine_import+"shapes data_obs  cat%i %s.root w:cat%i_data\n"%(i,filename,i)

        combine_bins = combine_bins+name+" "
        combine_obs = combine_obs+"-1   "

        combine_bins_str = combine_bins_str+ "{:14s}{:14s}".format(name,name)
        combine_proc_str = combine_proc_str+ "cat%i_ggh      cat%i_bkg      "%(i,i)
        combine_ipro_str = combine_ipro_str+ "0             1             "
        combine_rate_str = combine_rate_str+ "{:<14f}{:<14f}".format(rate_s, rate_b)


    combine_bins_str = combine_bins_str+"\n"
    combine_proc_str = combine_proc_str+"\n"
    combine_ipro_str = combine_ipro_str+"\n"
    combine_rate_str = combine_rate_str+"\n"
    w.Print()
    workspace_file = ROOT.TFile.Open(path+filename+".root", "recreate")
    workspace_file.cd()
    w.Write()
    workspace_file.Close()

    return combine_import, combine_bins+"\n"+combine_obs+"\n", combine_bins_str+combine_proc_str+combine_ipro_str+combine_rate_str

def create_datacard(bins, path, name, workspace_filename): 
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    import_str, bins_obs, cat_strings = make_eta_categories(bins, path, workspace_filename)
    out_file = open(path+name+".txt", "w")
    out_file.write("imax *\n")
    out_file.write("jmax *\n")
    out_file.write("kmax *\n")
    out_file.write("---------------\n")
    out_file.write(import_str)
    out_file.write("---------------\n")
    out_file.write(bins_obs)
    out_file.write("------------------------------\n")
    out_file.write(cat_strings)
    out_file.close()

def plot_sig_evenly():
    significance = {
        '1'     : 0.553469,
        '2'     : 0.585567,
        '3'     : 0.593943,
        # '3'     : 0.597208, #"optimal"
        # '3'     : 0.5987,   #hig-17-019
        '4'     : 0.598493,
        '5'     : 0.600864,
        '6'     : 0.60118,
        '7'     : 0.568954,
        '8'     : 0.566162,
        '9'     : 0.561477,
        '10'    : 0.598896,
        '11'    : 0.580472,
        '12'    : 0.585927,
        '13'    : 0.584745,
        '14'    : 0.576766,
        '15'    : 0.600524,
        '16'    : 0.590952,
        '17'    : 0.598741,
        '18'    : 0.593598,
        '19'    : 0.596877,
        '20'    : 0.610964,
        '21'    : 0.589647,
        '22'    : 0.601112,
        '23'    : 0.610482,
        '24'    : 0.60994,
    }
    # significance1 = {
    #     '1'     :   0.552706,
    #     '2'     :   0.586123,
    #     '3'     :   0.593512,
    #     '4'     :   0.597389,
    #     '5'     :   0.601174,
    #     '6'     :   0.600333,
    #     '7'     :   0.596228,
    #     '8'     :   0.565726,
    #     '9'     :   0.658959,
    #     '10'    :   0.598229,
    #     '11'    :   0.575846,
    #     '12'    :   0.578905,
    #     '13'    :   0.598192,
    #     '14'    :   0.587813,
    #     '15'    :   0.567488,
    #     '16'    :   0.57834,
    #     '17'    :   0.592896,
    #     '18'    :   0.60844,
    #     '19'    :   0.586028,
    #     '20'    :   0.569959,
    #     '21'    :   0.583279,
    #     '22'    :   0.61001,
    #     '23'    :   0.586347,
    #     '24'    :   0.588394,
    # }
    graph = ROOT.TGraph()
    graph_p = ROOT.TGraph()
    graph1 = ROOT.TGraph()
    graph1_p = ROOT.TGraph()
    for i in range(24):
        graph.SetPoint(i, i+1, significance['%i'%(i+1)])
        graph_p.SetPoint(i, i+1, (significance['%i'%(i+1)]-significance['1'])/significance['1']*100)
        # graph1.SetPoint(i, i+1, significance1['%i'%(i+1)])
        # graph1_p.SetPoint(i, i+1, (significance1['%i'%(i+1)]-significance1['1'])/significance1['1']*100)
    graph.SetMarkerStyle(20)
    graph.SetMarkerSize(2)
    graph.SetLineWidth(2)
    graph_p.SetMarkerStyle(20)
    graph_p.SetMarkerSize(2)
    graph_p.SetLineWidth(2)    
    # graph1.SetMarkerStyle(20)
    # graph1.SetMarkerSize(2)
    # graph1.SetLineWidth(2)
    # graph1_p.SetMarkerStyle(20)
    # graph1_p.SetMarkerSize(2)
    # graph1_p.SetLineWidth(2)  
    # graph1.SetMarkerColor(ROOT.kRed)
    # graph1.SetLineColor(ROOT.kRed)
    # graph1_p.SetMarkerColor(ROOT.kRed)
    # graph1_p.SetLineColor(ROOT.kRed)

    # graph_o = ROOT.TGraph()
    # graph_op = ROOT.TGraph()
    # opt_sig = [
    #     [2, 0.585433],
    #     [2, 0.583011],
    #     [3, 0.596026],
    #     [3, 0.5987],
        # [4, 0.563011]
    # ]
    # for i,s in enumerate(opt_sig):
    #     graph_o.SetPoint(i, s[0], s[1])
    #     graph_op.SetPoint(i, s[0], (s[1]-significance['1'])/significance['1']*100)

    # graph_o.SetMarkerStyle(20)
    # graph_o.SetMarkerSize(2)
    # graph_o.SetLineWidth(2)
    # graph_op.SetMarkerStyle(20)
    # graph_op.SetMarkerSize(2)
    # graph_op.SetLineWidth(2)  
    # graph_o.SetMarkerColor(ROOT.kGreen)
    # graph_o.SetLineColor(ROOT.kGreen)
    # graph_op.SetMarkerColor(ROOT.kGreen)
    # graph_op.SetLineColor(ROOT.kGreen)


    canvas = ROOT.TCanvas("c", "c", 800, 800)
    canvas.cd()
    graph.Draw("apl")
    # graph1.Draw("plsame")
    # graph_o.Draw("psame")
    canvas.Print("combine/categorization/sig_plt.png")

    canvas = ROOT.TCanvas("c1", "c1", 800, 800)
    canvas.cd()
    graph_p.SetMaximum(15)
    graph_p.SetMinimum(0)
    graph_p.Draw("apl")
    # graph1_p.Draw("plsame")
    # graph_op.Draw("psame")
    canvas.Print("combine/categorization/sig_plt_percents.png")

def plot_2cat_scan():
    gr = ROOT.TGraph()
    gr1 = ROOT.TGraph()
    gr2 = ROOT.TGraph()
    gr3 = ROOT.TGraph()

    sign = [ #
        0.554437,
        0.516243,
        0.516894,
        0.562795,
        0.567076,
        0.571554,
        0.576514,
        0.581921,
        0.586033,
        0.586966,  #10
        0.587322,
        0.585567,
        0.585923,
        0.5856,
        0.589508,  #15
        0.58374,
        0.584785,
        0.541814,
        0.580755,
        0.577793,   #20
        0.575762,
        0.571004,
        0.562312
    ]
    sign1 = [ #1
        0.553628,
        0.515218,
        0.515502,
        0.560336,
        0.566407,
        0.570469,
        0.575749,
        0.581327,
        0.585433,
        0.586237,  #10
        0.586395,
        0.586123,
        0.58522,
        0.584909,
        0.586891,  #15
        0.583011,
        0.584645,
        0.584129,
        0.581442,
        0.57764,   #20
        0.574374,
        0.57041,
        0.561635
    ]



    sign2 = [ #test
        0.552966,
        0.561842,
        0.513707,
        0.561727,
        0.568971,
        0.569202,
        0.576269,
        0.581565,
        0.585576,
        0.587033,  #10
        0.587396,
        0.586626,
        0.585707,
        0.587236,
        0.589243,  #15
        0.582902,
        0.582451,
        0.584182,
        0.581898,
        0.57964,  #20
        0.573887,
        0.569479,
        0.562487
    ]

    sign3 = [ #test1
        0.555528,
        0.517786,
        0.56669,
        0.564121,
        0.569423,
        0.573305,
        0.577591,
        0.583426,
        0.588338,
        0.592604,  #10
        0.589984,
        0.589489,
        0.588423,
        0.590484,
        0.538182,  #15
        0.585608,
        0.586105,
        0.586364,
        0.584115,
        0.580525,  #20
        0.577304,
        0.573375,
        0.564528,
    ]

    for i in range(len(sign)):
        gr.SetPoint(i, (i+1)/10.0, (sign[i]-0.553469)/0.553469*100)
        gr1.SetPoint(i, (i+1)/10.0, (sign1[i]-0.552706)/0.552706*100)
        gr2.SetPoint(i, (i+1)/10.0, (sign2[i]-0.55153)/0.55153*100)
        gr3.SetPoint(i, (i+1)/10.0, (sign3[i]-0.554712)/0.554712*100)

    gr.SetMarkerStyle(20)
    gr.SetMarkerSize(2)
    gr.SetLineWidth(2)
    gr1.SetMarkerStyle(20)
    gr1.SetMarkerSize(2)
    gr1.SetLineWidth(2)    
    gr1.SetMarkerColor(ROOT.kRed)
    gr1.SetLineColor(ROOT.kRed)
    gr2.SetMarkerStyle(20)
    gr2.SetMarkerSize(2)
    gr2.SetLineWidth(2)    
    gr2.SetMarkerColor(ROOT.kGreen)
    gr2.SetLineColor(ROOT.kGreen)
    gr3.SetMarkerStyle(20)
    gr3.SetMarkerSize(2)
    gr3.SetLineWidth(2)    
    gr3.SetMarkerColor(ROOT.kBlue)
    gr3.SetLineColor(ROOT.kBlue)
    gr.GetXaxis().SetTitle("Rapidity cut")
    gr.GetYaxis().SetTitle("% gain in significance")
    gr.SetMinimum(0)
    gr.SetMaximum(10)
    gr.GetXaxis().SetRangeUser(0,2.4)
    canvas = ROOT.TCanvas("c", "c", 800, 800)
    canvas.cd()
    gr.Draw("apl")
    gr1.Draw("plsame")
    gr2.Draw("plsame")
    gr3.Draw("plsame")

    canvas.Print("combine/categorization/2cat_scan_2.png")

def plot_3cat_scan():
    gr = ROOT.TGraph()
    gr1 = ROOT.TGraph()
    gr2 = ROOT.TGraph()
    gr3 = ROOT.TGraph()

    sign = [
        0.557002, #10
        0.589961,
        0.592896,
        0.593644,
        0.584138,
        0.5967, #15
        0.596309,
        0.597208,
        0.598568,
        0.5987, 
        0.599409, #20
        0.571323,
        0.576055,
        0.591722
    ]
    sign1 = [
        0.555409, #10
        0.589778,
        0.594343,
        0.5929,
        0.595213,
        0.591885,#15
        0.595343,
        0.596026,
        0.601915,
        0.598848,
        0.598845, #20
        0.570285,
        0.575149,
        0.591722,
    ]
    sign_test = [
        0.556445, #10
        0.590442,
        0.592731,
        0.593267,
        0.594992,
        0.594525,#15
        0.595463,
        0.59643,
        0.598521,
        0.598884,
        0.599085, #20
        0.570522,
        0.597193,
        0.592889,
    ]
    sign_test1 = [
        0.559294, #10
        0.592393,
        0.595737,
        0.59541,
        0.597901,
        0.596895,#15
        0.598315,
        0.599259,
        0.601101,
        0.601233,
        0.601885, #20
        0.573869,
        0.603051,
        0.596896,
    ]
    for i in range(len(sign)):
        gr.SetPoint(i, (i+10)/10.0, (sign[i]-0.553469)/0.553469*100)
        gr1.SetPoint(i, (i+10)/10.0, (sign1[i]-0.552706)/0.552706*100)
        gr2.SetPoint(i, (i+10)/10.0, (sign_test[i]-0.55153)/0.55153*100)
        gr3.SetPoint(i, (i+10)/10.0, (sign_test1[i]-0.554712)/0.554712*100)

    gr.SetTitle("Fix one cut at 0.9")
    gr.SetMarkerStyle(20)
    gr.SetMarkerSize(2)
    gr.SetLineWidth(2)
    gr1.SetMarkerStyle(20)
    gr1.SetMarkerSize(2)
    gr1.SetLineWidth(2)    
    gr1.SetMarkerColor(ROOT.kRed)
    gr1.SetLineColor(ROOT.kRed)
    gr2.SetMarkerStyle(20)
    gr2.SetMarkerSize(2)
    gr2.SetLineWidth(2)    
    gr2.SetMarkerColor(ROOT.kGreen)
    gr2.SetLineColor(ROOT.kGreen)
    gr3.SetMarkerStyle(20)
    gr3.SetMarkerSize(2)
    gr3.SetLineWidth(2)    
    gr3.SetMarkerColor(ROOT.kBlue)
    gr3.SetLineColor(ROOT.kBlue)
    gr.GetXaxis().SetTitle("Rapidity cut")
    gr.GetYaxis().SetTitle("% gain in significance")
    gr.SetMinimum(0)
    gr.SetMaximum(10)
    # gr.GetXaxis().SetRangeUser(0,2.4)
    canvas = ROOT.TCanvas("c", "c", 800, 800)
    canvas.cd()
    gr.Draw("apl")
    gr1.Draw("plsame")
    gr2.Draw("plsame")
    gr3.Draw("plsame")
    # y = (0.585433-base)/base*100.0
    y=6
    line = ROOT.TLine(0.9,y,2.4,y)
    line.Draw("same")
    canvas.Print("combine/categorization/3cat_0p9_scan_2.png")

def plot_4cat_scan():
    gr = ROOT.TGraph()
    gr1 = ROOT.TGraph()
    gr2 = ROOT.TGraph()
    gr3 = ROOT.TGraph()

    sign = [
        0.603668, #10
        0.601165,
        0.602899,
        0.599513,
        0.566333,
        0.601409,#15
        0.584398,
        0.562898,
        0.567662,
        0.603055,
    ]
    sign1 = [
        0.567972, #10
        0.600637,
        0.602279,
        0.604818,
        0.565132,
        0.601168,#15
        0.599388,
        0.562019,
        0.567233,
        0.564117,
    ]
    sign_test = [
        0, #10
        0,
        0,
        0,
        0,
        0,#15
        0,
        0,
        0,
        0,
    ]
    sign_test1 = [
        0, #10
        0,
        0,
        0,
        0,
        0,#15
        0,
        0,
        0,
        0,
    ]
    for i in range(len(sign)):
        gr.SetPoint(i, (i+10)/10.0, (sign[i]-0.553469)/0.553469*100)
        gr1.SetPoint(i, (i+10)/10.0, (sign1[i]-0.552706)/0.552706*100)
        gr2.SetPoint(i, (i+10)/10.0, (sign_test[i]-0.55153)/0.55153*100)
        gr3.SetPoint(i, (i+10)/10.0, (sign_test1[i]-0.554712)/0.554712*100)

    gr.SetTitle("Fix cuts at 0.9, 2.0")
    gr.SetMarkerStyle(20)
    gr.SetMarkerSize(2)
    gr.SetLineWidth(2)
    gr1.SetMarkerStyle(20)
    gr1.SetMarkerSize(2)
    gr1.SetLineWidth(2)    
    gr1.SetMarkerColor(ROOT.kRed)
    gr1.SetLineColor(ROOT.kRed)
    gr2.SetMarkerStyle(20)
    gr2.SetMarkerSize(2)
    gr2.SetLineWidth(2)    
    gr2.SetMarkerColor(ROOT.kGreen)
    gr2.SetLineColor(ROOT.kGreen)
    gr3.SetMarkerStyle(20)
    gr3.SetMarkerSize(2)
    gr3.SetLineWidth(2)    
    gr3.SetMarkerColor(ROOT.kBlue)
    gr3.SetLineColor(ROOT.kBlue)
    gr.GetXaxis().SetTitle("Rapidity cut")
    gr.GetYaxis().SetTitle("% gain in significance")
    gr.SetMinimum(0)
    gr.SetMaximum(10)
    # gr.GetXaxis().SetRangeUser(0,2.4)
    canvas = ROOT.TCanvas("c", "c", 800, 800)
    canvas.cd()
    gr.Draw("apl")
    gr1.Draw("plsame")
    gr2.Draw("plsame")
    gr3.Draw("plsame")
    # y = (0.585433-base)/base*100.0
    y=8.35
    line = ROOT.TLine(0.9,y,2.0,y)
    line.Draw("same")
    canvas.Print("combine/categorization/4cat_0p9_2p0_scan.png")

# plot_sig_evenly()
# plot_2cat_scan()
# plot_3cat_scan()
# plot_4cat_scan()
# bins_list = [0, 0.8, 1.7, 2.4]
# create_datacard(bins_list, "combine/categorization/", "datacard", "workspace")
# create_datacard([0, 2.4], "combine/categorization/2cat_scan1/", "datacard_1cat", "workspace_1cat")
# create_datacard([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4], "combine/categorization/", "datacard_24cat", "workspace_24cat")
# create_datacard([0, 0.9, 1.9, 2.4], "combine/categorization/", "datacard_like2016", "workspace_like2016")
# create_datacard([0, 0.9, 2.4], "combine/categorization/", "datacard_opt2cat", "workspace_opt2cat")
# create_datacard([0, 1.6, 2.4], "combine/categorization/", "datacard_opt2cat1", "workspace_opt2cat1")
# create_datacard([0, 0.9, 1.7, 2.4], "combine/categorization/", "datacard_opt3cat", "workspace_opt3cat")
# create_datacard([0, 0.2, 1.6, 2.4], "combine/categorization/", "datacard_opt3cat1", "workspace_opt3cat1")
# create_datacard([0, 0.2, 0.9, 1.7, 2.4], "combine/categorization/", "datacard_opt4cat", "workspace_opt4cat")

# for i in range(25):
#     if not i:
#         continue

#     bins = [0]
#     for j in range(i):
#         bins.append((24*(j+1)/float(i))/10.0)
#     create_datacard(bins, "combine/categorization/evenly2/", "datacard_%icat"%i, "workspace_%icat"%i)
#     # print bins


# for i in range(23):
#     bins = [0, (i+1)/10.0, 2.4]
#     create_datacard(bins, "combine/categorization/2cat_scan/", "datacard_2cat_%i"%(i+1), "workspace_2cat_%i"%(i+1))


# for i in range(14):
#     print (i+10)/10.0
#     bins = [0, 0.9, (i+10)/10.0, 2.4]
#     create_datacard(bins, "combine/categorization/3cat_0p9_scan_test1/", "datacard_3cat_0p9_%i"%(i+10), "workspace_3cat_0p9_%i"%(i+10))

for i in range(11):
    # print (i+10)/10.0
    bins = [0, 0.9, 1.2, (i+13)/10.0, 2.4]
    create_datacard(bins, "combine/categorization/4cat_0p9_1p2_scan1/", "datacard_4cat_0p9_1p2_%i"%(i+10), "workspace_4cat_0p9_1p2_%i"%(i+10))
    # print bins