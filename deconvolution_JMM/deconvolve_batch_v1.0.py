import cupy as cp
import cv2
import matplotlib.pyplot as plt
from matplotlib import colors
from cupyx.scipy.signal import convolve2d as conv2
#from cupy.fft import fftn
from scipy.fftpack import fft2
import time
from time import perf_counter
from scipy.special import jv
import numpy as np
import h5py
from skimage.feature import peak_local_max
from skimage import img_as_float
from skimage import restoration
from scipy import ndimage as ndi
import azimuthal_avg as aa
from IPython.display import clear_output
from tqdm import tqdm





def load_h5py_dp(file):
    #file format for diffraction pattern data is hdf5 (hierarchical data format)
    with h5py.File(file,"r") as f:    
        # Print all root level object names (aka keys) 
        # these can be group or dataset names 
        print("Keys: %s" % f.keys())
        # get second object name/key
        a_group_key = list(f.keys())[1]
        sdd=f['detector_distance'][()][0]
        wavelength=f['lambda'][()][0]
        # print(f['ppX'][()].shape)
        # preferred methods to get dataset values:
        ds_obj = f[a_group_key]      # returns as a h5py dataset object
        ds_arr = f[a_group_key][()]  # returns as a numpy array
    return ds_arr,sdd,wavelength
        

    
def flip180(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr

def normal(array):
    array = array/cp.amax(array)*255
    array = cp.where(array < 0,  0, array)
    array = cp.where(array > 255, 255, array)
    array = array.astype(cp.int16)
    return array

def RL_deconvblind(img,PSF,iterations,verbose=False):
    #print('Calculating deconvolution...')
    img = img.astype(cp.float32)
    PSF = PSF.astype(cp.float32)
    
    #find minimum value in img excluding <=0 pixels
    a = np.nanmin(np.where(img<=0, np.nan, img))
    #replace <=0 values in image
    img = cp.where(img <= 0, a, img)
    init_img = img
    PSF_hat = flip180(PSF)
    for i in range(iterations):
        if verbose:
            print('Iteration: {}'.format(i+1))
        est_conv = conv2(init_img,PSF,'same', boundary='symm')
        relative_blur = (img / est_conv)
        error_est = conv2(relative_blur,PSF_hat, 'same',boundary='symm')
        init_img = init_img* error_est
    return init_img

def roi(image):
    result=0
    not_satisfied=True
    while not_satisfied:
        plt.figure()
        plt.imshow(image)#,norm=colors.LogNorm())
        plt.show()
        
        x,hx=(int(n) for n in input("Select region of interest (x) ").split())
        y,hy=(int(n) for n in input("Select region of interest (y) ").split())

        image_cropped = image[x:x+hx,y:y+hy]
    
        plt.imshow(image_cropped)
        plt.show()
        s=input("Satisfied? ")
        if s=='y':
            not_satisfied=False
            result=image_cropped
    return result

def FT_image(image):
    return np.abs(np.fft.fftshift(np.fft.fft2(image)))**2

def load_data(dp,psf):
    return dp,psf

def plotter(images,labels,log=False):
    # display n plots side by side
    n=len(images)
    fig, axes = plt.subplots(1, n, figsize=(8, 3))#, sharex=True, sharey=True)
    ax = axes.ravel()
    for i in range(0,n):
        if log:
            ax[i].imshow(images[i], cmap=plt.cm.gray,norm=colors.LogNorm())
        else:
            ax[i].imshow(images[i], cmap=plt.cm.gray)
        #ax[i].axis('off')
        ax[i].set_title(labels[i])
    plt.show()
    
def plotter_rgb(images,labels,log=False):
    # display n plots side by side
    n=len(images)
    fig, axes = plt.subplots(1, n, figsize=(8, 3))#, sharex=True, sharey=True)
    ax = axes.ravel()
    for i in range(0,n):
        if log:
            ax[i].imshow(images[i],norm=colors.LogNorm())
        else:
            ax[i].imshow(images[i])
        #ax[i].axis('off')
        ax[i].set_title(labels[i])
    plt.show()
    

def gamma_manip(image, gamma):
        result = np.uint8(cv2.pow(image / 255., gamma) * 255.)
        return result 
    
    
def peak_finder(dp_image_file):
    image=cv2.imread(dp_image_file)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #im = img_as_float(image)

    # image_max is the dilation of im with a 30*30 structuring element
    # It is used within peak_local_max function
    image_max = ndi.maximum_filter(image, size=30, mode='constant')
    # Comparison between image_max and im to find the coordinates of local maxima
    coordinates = peak_local_max(image, min_distance=10, threshold_abs=.01)
    plotter([image,image_max],['Original','Maximum filter'],log=True)
    # plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    # plt.show()
    # print("Peaks (coordinates)")
    # print(coordinates)
    # print('Peaks (q, theta)')
    peaks=peak_converter(coordinates)
    return coordinates#peaks


def peak_converter(peaks):
    center=200 #same for x and y
    q_hk=[]
    theta_hk=[]
    for peak in peaks:
        q_hk.append(np.sqrt((peak[0]-center)**2+(peak[1]-center)**2))
        theta_hk.append(np.arctan((center-peak[0])/(center-peak[1]))*180/np.pi)
        #print(q_hk,theta_hk)
    return q_hk,theta_hk

def sum_frames(dp_list,total_frames,live_plot=False):
    #diffraction patterns from hdf5_file
    #total=0
    total=np.zeros(dp_list[0].shape)
    #for dp in tqdm(dp_list[0:total_frames]):
    for dp in dp_list[0:total_frames]:
        if live_plot:
            clear_output(wait=True)
            plt.imshow(total,norm=colors.LogNorm())
            plt.show()
        total+=dp
        #total/=len(dp_list)
        #plt.pause(0.3)
    #plt.imshow(total,norm=colors.LogNorm())
    #plt.show()
    return total

def avg_frames(dp_list,total_frames,live_plot=False):
    #diffraction patterns from hdf5_file
    total=np.zeros(dp_list[0].shape)
    for dp in tqdm(dp_list[0:total_frames]):
        if live_plot:
            clear_output(wait=True)
            plt.imshow(total,norm=colors.LogNorm())
            plt.show()
        total+=dp
        #total/=len(dp_list)
        #plt.pause(0.3)
    #plt.imshow(total,norm=colors.LogNorm())
    #plt.show()
    return total/total_frames

device=3
with cp.cuda.Device(device):
    if __name__ == '__main__':
        directory='dps/fly015_temp/'
        #load in diffraction patterns (dps) data, including sample-to-detector distance (sdd) and xray wavelength (wavelength)
        dps,sdd,wavelength = load_h5py_dp('/home/beams/B304014/ptychosaxs/data/fly015/data_roibit_Ndp400_dp.hdf5')
        
        #load in point-spread function (psf) (the probe) data, *npy data file
        #the reconstructed probe from ptychography data
        psf_real = np.load('/home/beams/B304014/ptychosaxs/data/fly015/fly015_probe_N256.npy',allow_pickle=True)

        #plt.imshow(np.abs(psf_real)) #magnitude
        #plt.imshow(np.angle(psf_real)) #phase
        
        #intensity (magnitude**2) of the reconstructed probe (real space)
        psf_real_mag_2=np.abs(psf_real)**2
        
        #fourier transform of reconstructed probe (reciprocal space)
        psf=np.abs(np.fft.fftshift(np.fft.fft2(psf_real)))**2#,norm=colors.LogNorm())
        #cv2.imwrite(directory+'psf.png',psf)
        #psf=roi(psf) #FOR JUST CENTER FZP PATTERN 
                    #BEAM: x,y: 118 21
        psf=psf[118:139,118:139]
        #cv2.imwrite(directory+'psf.png',psf)
        
        
        count=0
        for dp in tqdm(dps[10:20]):
            #write dp to image file and load in the image
            cv2.imwrite('dp.png',dp)
        
            #process or done process ***
            dp_image = cv2.imread('dp.png')
            #dp_image=process_dp('dp.png')
        
            #convert images to gray scale
            #dp_image_gray=dp_image
            # if processed, comment out this line ***
            #dp_image_gray = cv2.cvtColor(dp_image, cv2.COLOR_BGR2GRAY)
        
        
            #deconvolute dp and PSF
            iterations =50
            ##convert to cupy array
            #dp_gray=cp.asarray(dp_image_gray)
            psf=cp.asarray(psf) 
            dp=cp.asarray(dp)
            
            #initialize GPU timer
            start_time=perf_counter()            
            #result = restoration.richardson_lucy(dp,psf,iterations)#,filter_epsilon=1)
            #result = restoration.wiener(dp_image_gray,psf,10000)
            #result = restoration.unsupervised_wiener(dp_image_gray,psf)[0]#,iterations)
            result = RL_deconvblind(dp, psf, iterations)
            
            # #get cupy arrays from GPU
            # result_norm=normal(result)
            # dp_gray_norm=normal(dp_gray)
            # psf_norm=normal(psf)
            # result_norm_cpu=result_norm.get()
            # dp_gray_norm_cpu=dp_gray_norm.get()
            # psf_norm_cpu=psf_norm.get()
            result_cpu=result.get()
            #dp_gray_cpu=dp_gray.get()
            dp_cpu=dp.get()
            psf_cpu=psf.get()
            # psf_cpu=psf
            # result_cpu=result
        
            #calculate time of deconvolution on GPU
            cp.cuda.Device(device).synchronize()
            stop_time = perf_counter( )
            time=str(round(stop_time-start_time,4))
            print("Computation time on GPU: "+time+" seconds")
        
            # plt.imshow(result_cpu,norm=colors.LogNorm())
            # plt.show()
            plotter_rgb([psf_cpu,dp_cpu,result_cpu],['psf','dp','recovered'],log=True)
            bkg=10 #to get rid of zeros
            plotter_rgb([psf_cpu+bkg,dp_cpu+bkg,result_cpu+bkg],['psf','dp','recovered'],log=True)
            # #save image
            # save='y'
            # if save == 'y':
            #      #out_filename="output_frame{}_iterations{}.png".format(frame,iterations)
            #      out_filename=directory+"output_frame{}_iterations{}_skW.png".format(count,iterations)
            #      cv2.imwrite(out_filename, result_cpu)
            #      print("Plots written to "+out_filename)
            # else:
            #      print("Plots not saved")
            
            count+=1
        
        #SHOW PROGRESSION OF DECONVOLUTION WITH MORE DP FRAMES
        iterations=50
        num_frames=len(dps)
        start=len(dps)-1
        psf=cp.asarray(psf)
        #for i in tqdm(range(start,len(dps[:num_frames]))):
        for i in range(start,len(dps[:num_frames])):
            dp=cp.asarray(avg_frames(dps[0:i],i))
            result = RL_deconvblind(dp, psf, iterations)
            result_cpu=result.get()
            clear_output(wait=True)
            plt.imshow(result_cpu,norm=colors.LogNorm())
            plt.show()
            #plt.pause(0.2)
        dp_cpu=dp.get()
        psf_cpu=psf.get()
        plotter_rgb([psf_cpu,dp_cpu,result_cpu],['psf','dp','recovered'],log=True)
        cv2.imwrite('dp.png',dp_cpu)
        np.save('dp.npy',dp_cpu)
        cv2.imwrite('recovered.png',result_cpu)
        np.save('recovered.npy',result_cpu)
        
