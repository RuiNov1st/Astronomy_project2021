# origin attributions: 
"""
add Real distribution with tiny offset
with both polariazation
with new crafts data 
refractor code
signal model preprocess advanced.
can choose "center" or "random position"
beam number can over than 3
some beams don't contain any signal.
with location infomation
"""
# new attributions:
"""
transfer to calibration data
with regrid process
"""
# next version:
"""
add alfalfa signal model
"""
# wsr:Updated on April 18th,2022

# funtion example:
# main_process()

# import:
import matplotlib.pyplot as plt
import math
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
import random
from astropy.io import fits
import os
import datetime
import ra_trans_script_v5 as ra_trans

# definition:
freq_all = np.arange(65536) * 0.00762939 + 1000.00357628 # 15728
indfreq_HI = np.where((freq_all < 1420) & (freq_all > 1300))[0]
freq_HI = freq_all[indfreq_HI]
# const define:
width = 596
length = 786

# draw 2D image:
# params:
# title
# data:2D array
def draw2D(title,data):
    fig = plt.subplots(1, figsize=(15, 25))
    plt.title(str(title))
    plt.imshow(data, cmap='PuBu_r')
    plt.show()
    
# draw 1D image:
# params:
# title
# t-axis = 0 freq eg: t = freq[0:500]  (freq = np.arange(0,500,1))
# dara: 1D array
def draw1D(title,t,data):
    fig,ax = plt.subplots(1,1,figsize = (12,16))
    ax.plot(t,data,'k-')
    plt.title(title)
    plt.show()

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
    all_beam = [i for i in range(1,20)] # value = 1-19 index = 0-18  
    b_beam = random.randint(1,20-beamnum)-1
    beam_index = []
    beam_index = all_beam[b_beam:b_beam+beamnum]
    
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

# function: choose a negative background
# collect to a fits will be better.
# can collect multiple beam data.
def background_choose(beamnum):
    decname = ['Dec+4037','Dec+4120','Dec+4059']
    negative_dir = '/home/weishirui/Documents/crafts_data/dataset/source_data/cali/'
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
    # use CRAFTS data:
    if signal_opt=='CRAFTS':
        if center == False: # not center
            smooth_data = u_hdu[index].data[1]*1e11 # shape = (length,2)
            info = u_hdu[index].header
            # freq cut: default = center+-100
            f_begin,f_end = length//2-100,length//2+100
            fr_begin = random.randint(0,length-200-1)
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
        # some beams miss: maxium beam = 3
        present_num = random.randint(2,3)
        present_choice = set()
        while True: # strictly mutual exclusivity
            present_choice.add(random.randint(0,beamnum-1))
            if len(present_choice)==present_num:
                break
        for f in range(length):
            for l in range(0,beamnum):
                if l in present_choice: # present beams:
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
        if i in present_choice:
            location[i] = location_info # (width,length)
        else:
            location[i] = np.array((-1,-1)) # miss signal

    return bg_data,location


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
def generate_script(signal_opt,version,center,beamnum,miss,batch_size,savepath): 
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
    prim.header['batch size'] = batch_size
    hdul = fits.HDUList([prim])
    for index in range(1,142):
        # valid:
        if (u_hdu[index].data[1]*1e11<1e-2).all():
            continue
        data = np.zeros((batch_size,beamnum,width,length,2))
        location = np.zeros((batch_size,beamnum,2))
        for batch in range(0,batch_size):
            data[batch],location[batch] = generate_signal_pro(signal_opt,index,center,beamnum,miss,u_hdu,c_hdu,b_hdu)
        hduu = fits.ImageHDU(data,name= u_hdu[index].name+'_'+str(index))
        hduu.header = u_hdu[index].header
        hdul.append(hduu)
        hduuu = fits.ImageHDU(location,name= u_hdu[index].name+'__LocationInfo_'+str(index))
        hduuu.header['ax1'] = 'time'
        hduuu.header['ax2'] = 'freq'
        hdul.append(hduuu)
    hdul.writeto(savepath)

def main_process(RA = True):
    # Virtual Data Generate:
    # config:
    VirtualData_config = {
        'signal_opt':"CRAFTS",
        'version':2,
        'center':False,
        'beamnum':3,
        'miss':True,
        'batch_size':5
    }
    # savepath:
    curr_time = datetime.datetime.now()
    virtual_data_savepath = '/home/weishirui/Documents/crafts_data/dataset/Virtual_data/'+VirtualData_config['signal_opt']+"_version"+str(VirtualData_config['version'])+"_cali_"+str(curr_time.date())+'-'+str(curr_time.hour)+'-'+str(curr_time.minute)+".fits"
    generate_script(signal_opt = VirtualData_config['signal_opt'],version = VirtualData_config['version'],center = VirtualData_config['center'],beamnum = VirtualData_config['beamnum'],miss = VirtualData_config['miss'],batch_size = VirtualData_config['batch_size'],savepath = virtual_data_savepath)
    
    print("Virtual data OK")
    
    if not RA:
        return 
    # RA transfer
    # config:
    RATransfer_config = {
    'time_iniresl':3, # arcsec
    'time_newresl':10,
    'time_size':30, # arcmin
    'freq_iniresl':7.6, #kHz
    'freq_newresl':10
}
    
    # savepath:
    curr_time = datetime.datetime.now()
    RATransfer_savepath = '/home/weishirui/Documents/crafts_data/dataset/RAtransfer_dataset/virtual_data/'+VirtualData_config['signal_opt']+"_version"+str(VirtualData_config['version'])+"_cali_RATrans_"+str(curr_time.date())+'-'+str(curr_time.hour)+'-'+str(curr_time.minute)+".fits"
    ra_trans.main_process(filepath = virtual_data_savepath,newpath = RATransfer_savepath,config = RATransfer_config)
    
    print("RA Transfer OK")
    
main_process(RA=False)
