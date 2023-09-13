#%%
import sys
import os
# basedir=os.getcwd()
# sys.path.insert(0, basedir+'/ptychosaxs/deconvolution/')
import deconvolution_JMM.deconvolve as dc
#%%
if __name__ == '__main__':
    
    # # Setup device agnostic code
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using device: {device}")
    
    dp_file='/home/beams/B304014/ptychosaxs/data/fly015/data_roibit_Ndp400_dp.hdf5'
    probe_file='/home/beams/B304014/ptychosaxs/data/fly015/fly015_probe_N256.npy'
    recovered=dc.run(dp_file,probe_file)
    #%%
    dc.plot_q_radial_avg(recovered)
# %%
