#%%
import sys
import os
import importlib
#basedir=os.getcwd()
sys.path.insert(0, '/home/beams0/B304014/ptychosaxs/deconvolution/')
import deconvolution_JMM.deconvolve as dc
importlib.reload(dc)

if __name__ == '__main__':
    
    # # Setup device agnostic code
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using device: {device}")
    
    #dp_file='/home/beams/B304014/ptychosaxs/data/fly015/data_roibit_Ndp400_dp.hdf5'
    #probe_file='/home/beams/B304014/ptychosaxs/data/fly015/fly015_probe_N256.npy'
    dp_file='/home/beams/B304014/ptychosaxs/chansong/ckim_data585_766.mat'
    #235 42 235 42
    probe_file='/home/beams/B304014/ptychosaxs/chansong/ckim_data585_766.mat'
    recovered=dc.run(dp_file,probe_file)
    #%%
    dc.load_h5py_dp(dp_file)
# %%
