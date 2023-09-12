#%%
import sys
import os
basedir=os.getcwd()
sys.path.insert(0, basedir+'/ptychosaxs/deconvolution/')

import deconvolution_JMM.deconvolve_batch_v1 as dc
#%%
if __name__ == '__main__':
    
    dp_file=basedir+'/ptychosaxs/data/fly015/data_roibit_Ndp400_dp.hdf5'
    probe_file=basedir+'/ptychosaxs/data/fly015/fly015_probe_N256.npy'
    dc.run(dp_file,probe_file)
# %%
