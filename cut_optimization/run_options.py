from make_datacards import create_datacard

def run_inclusive(in_dir, out_dir):
    create_datacard([0, 2.4], in_dir, out_dir, "datacard", "workspace")

def run_2cat_scan(in_dir, out_dir, nuis=False):
    for i in range(23):
        bins = [0, (i+1)/10.0, 2.4]
        create_datacard(bins, in_dir, out_dir, "datacard_2cat_%i"%(i+1), "workspace_2cat_%i"%(i+1), nuis=nuis)

def run_3rd_cut(in_dir, out_dir):
    second_cut_options = {
        # "1p8": 1.8,
        "1p9": 1.9,
        "2p0": 2.0,
        }
    scan_options = [
        "Bscan","Oscan", "Escan"
        ]
    for key, value in second_cut_options.iteritems():
        for scan in scan_options:
            if "O" in scan:
                for i in range(int((value - 1)*10)):
                    bins = [0, 0.9, (i+10)/10.0, value, 2.4]
                    create_datacard(bins, in_dir, out_dir+"_%s_%s/"%(key, scan), "datacard_0p9_%i_%s"%((i+10), key), "workspace_0p9_%i_%s"%((i+10), key))
            if "E" in scan:
                for i in range(23-int((value)*10)):
                    bins = [0, 0.9, value, i/10.0+value+0.1, 2.4]
                    create_datacard(bins, in_dir, out_dir+"_%s_%s/"%(key, scan), "datacard_0p9_%i_%s"%((i+1+value*10), key), "workspace_0p9_%i_%s"%((i+1+value*10), key))
            if "B" in scan:
                for i in range(8):
                    bins = [0, (i+1)/10.0 ,0.9, value, 2.4]
                    create_datacard(bins, in_dir, out_dir+"_%s_%s/"%(key, scan), "datacard_0p9_%i_%s"%((i+1), key), "workspace_0p9_%i_%s"%((i+1), key))
        
def full_3cat_scan(in_dir, out_dir):
    for i in range(24):
        for j in range(i):
            bins = [0.0, (j+1)/10.0, (i+1)/10.0, 2.4]
            create_datacard(bins, in_dir, out_dir, "datacard_3cat_%i_%i"%(j+1, i+1), "workspace_3cat_%i_%i"%(j+1, i+1), nuis=False)