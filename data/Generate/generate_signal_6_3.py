# author:weishirui
# code version:2022-12-23
# description: generate virtual signal for following data gridding process.
# algorithm: 
# generate virtual signals with signal's pixel coordinate. (signal generate data)
# compute dec, ra of every pixel in datacube.
# gridding and generate final datacube.

# %%
# import:
import matplotlib.pyplot as plt
import math
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d
import numpy as np
import random
from astropy.io import fits
import os
import datetime
import pandas as pd

# %%
# definition:
freq_all = np.arange(65536) * 0.00762939 + 1000.00357628 # 15728
indfreq_HI = np.where((freq_all < 1420) & (freq_all > 1300))[0]
freq_HI = freq_all[indfreq_HI]

# const define:
width = 596
length = 786
# length-freq = 786*0.0076 = 5.9736 
# width-mjd = 596*0.05 = 29.8 > 20arcmin is OK


# %% 
# function:generate Poisson Noise--add random value from Gaussian Distribution. 
# Square of Every channel's random value is used to be sigma in Gaussian Distribution 
# params:
# data: 1D array
# mean: default 0
def PoissonNoise(data,mean):
    noise = np.zeros((1,data.shape[0]))
    for i in range(0,data.shape[0]):
        variance = data[i]**0.5
        noise[0,i] = np.random.normal(mean,variance,1)
    return noise

# %%
# function: add BeamShape
# params:
# width: axis = time
# decname: dec option ,default = Dec+4037
def timeGauss_pro(width,decname,beamnum,info):
    # match freq
    tr = np.arange(freq_all[0],freq_all[-1],10)
    freq_begin = 1420.406/(1+float(info['z']))
    indfreq = np.where((tr >freq_begin))[0][0]-1

    # choose beam:
    Gauss_shape = np.zeros((beamnum,width,2))
    # beams = ['M18','M17','M19','M16','M07','M08','M06','M02','M15','M01','M09','M05','M03','M14','M04','M10','M13','M11','M12']
    beam_index = [13,11,18,12,17,19,16,7,8,6,2,15,1,9,5,3,14,4,10,13,11,18,12,17]
    
    for i in range(0,len(beam_index)): 
        # there isn't M12 Response data:
        if beam_index[i]==12:
            beam_index[i] = random.choice([11,13]) # replace M12 with M11 or M13
            
        if beam_index[i] < 10:
            beam_info = '0'+str(beam_index[i])
        else:
            beam_info = str(beam_index[i])
        # 路径信息修改
        # 路径: 
        # version 1.0: /mnt/data66/home/weishirui/Documents/neural_network/signal_generate/BeamShape/
        # version 2.0: /mnt/data66/home/lupengjun/findsource/beam_shape/GuassResponseData_Dec+2715_M"+beam_c+"_ver2.0.fits"
        # version 3.0: /home/lupengjun/findsource/Dec+4037/GuassResponseData_Dec+4037_M01_ver2.0.fits
        Response_localpath = '/home/lupengjun/findsource/Dec+4037/GuassResponseData_'+decname+'_M'+beam_info+'_ver2.0.fits'
        hdu =  fits.open(Response_localpath)
        # match beam 
        shape_data = hdu[1].data # version1.0:shape = (2,50,348,2) # version2.0:shape = (2,50,596,2) # version3.0:shape = (2, 50, 596, 2)
        
        # polar:
        for j in range(0,2):
            Gauss_shape[i,:,j] = shape_data[j,indfreq,:,1]
    #draw1D("GaussResponse",np.arange(0,width,1),Gauss_shape[0,:,0])
    return Gauss_shape

# %%
# function: 1D Gaussian Smooth
# params:
# data-1D/2D
# sigma-std
def GaussianSmooth(data,sigma):
    if len(data.shape)==1:
        return gaussian_filter1d(data,sigma)
    else:
        s_data = np.zeros((data.shape))
        for i in range(0,data.shape[1]):
            s_data[:,i] = gaussian_filter1d(data[:,i],sigma)
        return s_data

# %%
# function: change scale:
# params:
# data: 2 polar (width,length,2)
# s_begin,s_end: scale
def change_scale(data,s_begin,s_end):
    scale = set()
    while len(scale)<2:
        scale.add(random.uniform(s_begin,s_end))
    scale = list(scale)
    data[:,:,0] = data[:,:,0]*scale[0]
    data[:,:,1] = data[:,:,1]*scale[1]
    return data

# %%
# function: choose a negative background
# collect to a fits will be better.
# can collect multiple beam data.
def background_choose(beamnum):
    decname = ['Dec+4037','Dec+4120','Dec+4059']
    negative_dir = '/home/weishirui/Documents/crafts_data/dataset/source_data/cali/center/'
    # choose a background:
    # choose dec:
    dec_opt = random.randint(0,len(decname)-1)
    dec_opt = 2 # only have Dec+4059 Negative
    hdu = fits.open(negative_dir+decname[dec_opt]+'_Negative_ver2.fits')
    # choose fits:
    fits_index = set()
    while len(fits_index)<beamnum:
        hdu_opt = random.randint(1,len(hdu)-1) # ignore 0
        fits_index.add(hdu_opt)
    # choose data:
    fits_index = list(fits_index)
    dataa = np.zeros((beamnum,width,length,2))
    for i in range(beamnum):
        beam_opt = random.randint(0,2)
        dataa[i] = hdu[fits_index[i]].data[beam_opt]
    return dataa

# %% 
# function: main function
# params:
    # signal_opt:"CRAFTS","ALFALFA"
    # index: signal model(1-15)
    # center: True/False
    # beamnum: 3/other
    # miss: True/False
    # u_hdu:unlimited hdu
    # c_hdu:center hdu
    # b_hdu:beam hdu
def generate_signal_pro(signal_opt,index,center,beamnum,miss,u_hdu,c_hdu,b_hdu):
    info = 0
    location_info = np.zeros((1,2)) # width,length
    rect_data = np.zeros((width,length,2))

    # crafts beam relative location
    beam_index = [18,17,19,16,7,8,6,2,15,1,9,5,3,14,4,10,13,11,12]
    # for data gridding, we need 24 beams includes upper and lower drift_dec
    all_beam_index = [13,11,18,12,17,19,16,7,8,6,2,15,1,9,5,3,14,4,10,13,11,18,12,17]

    # use CRAFTS data:
    if signal_opt=='CRAFTS':
        if center == False: # not center
            smooth_data = u_hdu[index].data[1]*1e11 # shape = (length,2)
            info = u_hdu[index].header
            # freq cut: default = center+-100
            f_begin,f_end = length//2-100,length//2+100
            # fr_begin = random.randint(0,length-200-1)
            fr_begin = random.randint(65,721-200-1) # since I have to choose 5MHz data, so the edges should be careful. 
            location_info[0][1] = fr_begin+100 
            for i in range(width):
                rect_data[i,fr_begin:fr_begin+200,:] = smooth_data[f_begin:f_end,:]
        else: # center
            rect_data = c_hdu[index].data[1]*1e11 # shape = (width,length,2)
            info = c_hdu[index].header
            location_info[0][1] = length//2
    
    # noise:
    noise_data = np.zeros((width,length,2))
    amplitude = random.randint(300,500)/10000
    for p in range(0,2):
        for i in range(0,width):
            noise_data[i,:,p] = rect_data[i,:,p]+amplitude*PoissonNoise(rect_data[i,:,p],0)
    noise_data[noise_data<0] = 0
    #draw2D("noise",noise_data[:,:,0])
    
    # change scale:
    scale_data = change_scale(noise_data,1.3,2.0) # shape = (width,length,2)
    
    # add Gauss:
    # calculate Gauss shape:
    if beamnum ==3:
        Gauss_shape = b_hdu[index].data # normalized shape = (3,width,2)
    else:
        Gauss_shape = timeGauss_pro(width,"Dec+4037",beamnum,info) # normalized shape = (beamnum,width,2)

    # put Gauss in time:
    Gauss_data = np.zeros((beamnum,width,length,2))
    # set present beams:
    present_choice = [c for c in range(0,beamnum)]
    if miss == False: # all present and beamnum==3
        # only pick up layer = 1 which exists signal absolutely.
        for f in range(length):
            for l in range(0,beamnum):
                Gauss_data[l,:,f,:] = scale_data[:,f,:]*Gauss_shape[l,:,:]
    else:
        # only 3 beams present.
        # choose central beams.
        present_num = 3 
        candidate_beams = [19,16,7,8,6,2,15,1,9,5,3,14]
    
        choose_index = beam_index.index(random.choice(candidate_beams))
        present_choice = beam_index[choose_index:choose_index+present_num]
        
        for f in range(length):
            for l in range(0,beamnum):
                if all_beam_index[l] in present_choice: # present beams:
                    Gauss_data[l,:,f,:] = scale_data[:,f,:]*Gauss_shape[l,:,:]

    # for i in range(0,beamnum):
    #     draw2D("BeamShape"+str(i),Gauss_data[i,:,:,1])
    
    # add background:
    # choose a background:
    bg_data = background_choose(beamnum)  # (beamnum, 596, 786, 2)
    #draw2D("background",bg_data[0,:,:,0])
    
    if center == True: # signal in time center
        cut_begin,cut_end = width//2-100,width//2+100
        bg_data[:,cut_begin:cut_end,:,:] += Gauss_data[:,cut_begin:cut_end,:,:]
        location_info[0][0] =  width//2
    else: 
        time_begin = random.randint(0,width-200)
        cut_begin,cut_end = width//2-100,width//2+100
        bg_data[:,time_begin:time_begin+200,:,:] += Gauss_data[:,cut_begin:cut_end,:,:]
        location_info[0][0] =  time_begin+100
        
    # location info with beam present info:
    location = np.zeros((beamnum,2))
    for i in range(0,beamnum):
        if all_beam_index[i] in present_choice:
            location[i] = location_info # (width,length)
        else:
            location[i] = np.array((-1,-1)) # miss signal

    return bg_data,location


# %%
"""
generate virtual signal with generate_signal_6_2.py in batches.
refractor code to speed up and improve memory performance.
"""
# function: script function
# params:
    # signal_opt:"CRAFTS","ALFALFA"
    # version:2
    # center: True/False
    # beamnum: 3/other
    # miss: True/False
    # batch_size: num
def generate_script(signal_opt,version,center,beamnum,miss,savepath): 
    u_hdu = fits.open('/home/weishirui/Documents/signal_generate/signal_shape/ver2/preprocess_data/unlimited_signalver'+str(version)+'_cali.fits')
    c_hdu = fits.open('/home/weishirui/Documents/signal_generate/signal_shape/ver2/preprocess_data/center_signalver'+str(version)+'_cali.fits')
    b_hdu = fits.open('/home/weishirui/Documents/signal_generate/BeamShape/ver2/signalver2_cali_BeamShape.fits')
    
    prim = fits.PrimaryHDU()
    prim.header['signal opt'] = signal_opt
    if center==True:
        prim.header['data source'] = os.path.basename('center_signalver'+str(version)+'_cali.fits')
    else:
        prim.header['data source'] = os.path.basename('unlimited_signalver'+str(version)+'_cali.fits')
    prim.header['center'] = center
    prim.header['beam num'] = beamnum
    prim.header['data miss'] = miss
    
    hdul = fits.HDUList([prim])
    for index in range(1,142):
    # for index in range(1,10):
        # valid:
        if (u_hdu[index].data[1]*1e11<1e-2).all():
            continue
        data = np.zeros((beamnum,width,length,2))
        location = np.zeros((beamnum,2))
        data,location = generate_signal_pro(signal_opt,index,center,beamnum,miss,u_hdu,c_hdu,b_hdu)
        hduu = fits.ImageHDU(data)
        hduu.header = u_hdu[index].header
        hduu.name = 'INDEX_'+str(index)
        hdul.append(hduu)
        hduuu = fits.ImageHDU(location,name= 'LocationInfo_'+str(index))
        hduuu.header['ax1'] = 'time'
        hduuu.header['ax2'] = 'freq'
        hdul.append(hduuu)
    hdul.writeto(savepath)



def main_process():
    # Virtual Data Generate:
    # config:
    VirtualData_config = {
        'signal_opt':"CRAFTS",
        'version':2,
        'center':False,
        'beamnum':24,
        'miss':True,
    }
    # savepath:
    curr_time = datetime.datetime.now()
    virtual_data_savepath = '/home/weishirui/Documents/crafts_data/dataset/Virtual_data/'+VirtualData_config['signal_opt']+"_version"+str(VirtualData_config['version'])+"_cali_"+str(curr_time.date())+'-'+str(curr_time.hour)+'-'+str(curr_time.minute)+".fits"
    generate_script(signal_opt = VirtualData_config['signal_opt'],version = VirtualData_config['version'],center = VirtualData_config['center'],beamnum = VirtualData_config['beamnum'],miss = VirtualData_config['miss'],savepath = virtual_data_savepath)
    
    print("Virtual data OK")
    
    
# main_process()


# %% 
# use current files to assign virtual signal coordinates:
def assign_coordinate(virtual_data_savepath,savepath):
    # layer - beam match:
    layer_beam = ['Dec+4120_M13','Dec+4120_M11','Dec+4059_M18','Dec+4120_M12','Dec+4059_M17','Dec+4059_M19',
        'Dec+4059_M16','Dec+4059_M07','Dec+4059_M08','Dec+4059_M06','Dec+4059_M02','Dec+4059_M15','Dec+4059_M01','Dec+4059_M09',
        'Dec+4059_M05','Dec+4059_M03','Dec+4059_M14','Dec+4059_M04','Dec+4059_M10','Dec+4059_M13','Dec+4059_M11','Dec+4037_M18','Dec+4059_M12','Dec+4037_M17']
    
    hdu = fits.open(virtual_data_savepath)

    prim = fits.PrimaryHDU()
    prim.header = hdu[0].header
    hdul = fits.HDUList([prim])

    for h in range(1,len(hdu),2):
        loc_data = hdu[h+1].data
        sig_loc = np.unique(np.where(loc_data!=-1)[0]).tolist()
        sig_beam = layer_beam[sig_loc[1]]
        sig_pixel = [int(loc_data[sig_loc[0],0]),int(loc_data[sig_loc[0],1])]
    
        # attach coordinate values:
        choose_file = random.randint(2,43)
        if choose_file<10:
            choose_file = '000'+str(choose_file)
        else:
            choose_file = '00'+str(choose_file)

        choose_mjd = random.randint(0,2047-width)

        coordinate_data = []
        for l in layer_beam:
            if l[:8] == 'Dec+4120':
                c_hdu = fits.open('/home/weishirui/Documents/crafts_data/dataset/source_data/coordinate/Dec+4120_10_05/Dec+4120_10_05_20210929_{}_{}_coordinate.fits'.format(l[-3:],choose_file))
            elif l[:8] == 'Dec+4059':
                c_hdu = fits.open('/home/weishirui/Documents/crafts_data/dataset/source_data/coordinate/Dec+4059_10_05/Dec+4059_10_05_20210930_{}_{}_coordinate.fits'.format(l[-3:],choose_file))
            elif l[:8] == 'Dec+4037':
                c_hdu = fits.open('/home/weishirui/Documents/crafts_data/dataset/source_data/coordinate/Dec+4037_10_05/Dec+4037_10_05_20211001_{}_{}_coordinate.fits'.format(l[-3:],choose_file))

            cn_data = np.zeros((3,width))
            cn_data[0] = c_hdu[1].data[choose_mjd:choose_mjd+width]['mjd']
            cn_data[1] = c_hdu[1].data[choose_mjd:choose_mjd+width]['ra(deg)']
            cn_data[2] = c_hdu[1].data[choose_mjd:choose_mjd+width]['dec(deg)']
            coordinate_data.append(cn_data)

        coordinate_data = np.array(coordinate_data)

        source_ra = coordinate_data[sig_loc[1],1,sig_pixel[0]]
        source_dec = coordinate_data[sig_loc[1],2,sig_pixel[0]]
        
        center_ra = coordinate_data[12,1,width//2]
        center_dec = coordinate_data[12,2,width//2]

        freq_index =random.randint(0,len(freq_HI)-length-1)
        freq_values = freq_HI[freq_index:freq_index+length]
        source_freq = freq_values[sig_pixel[1]]
        center_freq = freq_values[length//2]

        hduu = fits.ImageHDU(coordinate_data,name=hdu[h+1].name)
        hdu[h].header['source_ra'] = source_ra
        hdu[h].header['source_dec'] = source_dec
        hdu[h].header['source_freq'] = source_freq
        hdu[h].header['center_ra'] = center_ra
        hdu[h].header['center_dec'] = center_dec
        hdu[h].header['center_freq'] = center_freq
        hdu[h].header['beams'] = sig_beam

        hdul.append(hdu[h])
        hdul.append(hduu)
    
    hdul.writeto(savepath)


assign_coordinate('/home/weishirui/Documents/crafts_data/dataset/Virtual_data/CRAFTS_version2_cali_2022-12-24-20-32.fits','/home/weishirui/Documents/crafts_data/dataset/Virtual_data/CRAFTS_version2_cali_2022-12-24-20-32_cf.fits')
