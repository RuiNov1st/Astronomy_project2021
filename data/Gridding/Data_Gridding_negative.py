# author:weishirui
# code version:2022-11-30
# description: apply Data Gridding to crafts data to cut specific negative Datacube for neural network.
# algorithm: 
# input:
# - /home/weishirui/Documents/crafts_data/dataset/source_data/coordinate/:dec & ra & mjd ; 
# - /home/weishirui/Documents/crafts_data/dataset/source_data/coordinate/beam_dec_ra.csv : CRAFTS's datasets dec & ra map;
# - /data07/ZD2021_1_2/:mjd & flux 
# - /home/gaoyunhao/Crafts/: calibration dataset: flux
# - /home/weishirui/Documents/crafts_data/dataset/source_data/coordinate_flux/Dec+4037_10_05/collection:mjd&flux&dec&ra
# - /home/lupengjun/findsource/signal_generate/mask_cube/:mask 
# CyGrid & HCGrid for single channel data gridding;
# HEGrid for multi-channel data gridding.


from astropy.io import fits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.interpolate import interp1d
import os
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from scipy.interpolate import griddata
import cygrid
import random
from scipy import interpolate
import copy
import gc
import sys
import psutil

dec_size = 20 # arcmin(full)
ra_size = 20 # arcmin(full)
freq_size = 2.5 # MHz, half
freq_resolution =  0.00762939 # MHz
freq_new_resolution = 0.1 # MHz

batch_idx = 3 # produce number

# %%
# init all the path using drift_dec_date params:
def path_init(drift_dec_date):
    global driftdec,driftdate,crafts_path,coordinate_path,mask_path,calibration_path
    global coordinate_flux_path,output_file,output_img_path
    global all_dec_path,HIsource_path

    # split name:
    driftdec = drift_dec_date[:14]
    driftdate = drift_dec_date[-8:]

    # input path:
    # crafts origin data:
    crafts_path = '/data07/ZD2021_1_2/{}/{}/'.format(driftdec,driftdate)
    # crafts all dataset dec information:
    all_dec_path = '/home/weishirui/Documents/crafts_data/dataset/source_data/coordinate/beam_dec_ra.csv'
    # coordinate path based on calibration dataset:
    coordinate_path = '/home/weishirui/Documents/crafts_data/dataset/source_data/coordinate/'
    # coordinate and flux dataset based on calibration dataset:
    coordinate_flux_path = '/home/weishirui/Documents/crafts_data/dataset/source_data/coordinate_flux/'
    # HI source information:
    HIsource_path = '/home/weishirui/Documents/crafts_data/dataset/source_data/HIsource_coordinate/crafts_{}_mjd.fits.gz'.format(drift_dec_date)
    # mask:
    mask_path = '/home/lupengjun/findsource/signal_generate/mask_cube/'
    # calibraion dataset: by gaoyunhao
    calibration_path = '/home/gaoyunhao/Crafts/'

    # output path:
    # gridding output path:
    output_file = '/home/weishirui/Documents/crafts_data/dataset/RAtransfer_dataset/gridding/negative/data/{}'.format(driftdec)
    # gridding img output path:
    output_img_path = '/home/weishirui/Documents/crafts_data/dataset/RAtransfer_dataset/gridding/negative/output_img/{}_{}'.format(driftdec,driftdate)

    if not os.path.exists(output_file):
        os.mkdir(output_file)
    if not os.path.exists(output_img_path):
        os.mkdir(output_img_path)

# %%
# Dataset prepare:
freq_all = np.arange(65536) * 0.00762939 + 1000.00357628
HIfreq = np.where((freq_all>1300) & (freq_all<1420))[0]
freq_all = freq_all[HIfreq]

# %%
# image kwargs:
imkw = dict(origin='lower', interpolation='nearest')
# matplotlib settings:
params = {
    'backend': 'pdf',
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'font.size': 10,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'font.family': 'serif',
    'figure.dpi': 75
}
rcParams.update(params)

# %%
# CyGrid
# set_header
def setup_header(mapcenter, mapsize, beamsize_fwhm):
    '''
    Produce a FITS header that contains the target field.
    '''
    
    # define target grid (via fits header according to WCS convention)
    # a good pixel size is a third of the FWHM of the PSF (avoids aliasing)
    
    # 2.9'/3 = 0.96' = 1 arcmin is OK
    pixsize = beamsize_fwhm / 3.
    # square will be fine.
    dnaxis1 = int(mapsize[0] / pixsize)
    dnaxis2 = int(mapsize[1] / pixsize)

    header = {
        'NAXIS': 2,
        'NAXIS1': dnaxis1,
        'NAXIS2': dnaxis2,
        'CTYPE1': 'RA---SIN',
        'CTYPE2': 'DEC--SIN',
        'CUNIT1': 'deg',
        'CUNIT2': 'deg',
        'CDELT1': -pixsize,
        'CDELT2': pixsize,
        'CRPIX1': dnaxis1 / 2.,
        'CRPIX2': dnaxis2 / 2.,
        'CRVAL1': mapcenter[0],
        'CRVAL2': mapcenter[1],
        }
    return header

# %%
class my_gridding():

    def __init__(self,mapsize,mapcenter,data,img):
        self.mapsize = mapsize 
        self.mapcenter = mapcenter
        self.beamsize_fwhm = 2.9/60  # 2.9arcmin
        self.lon_scale = np.cos(np.radians(mapcenter[1]))
        self.map_l = mapcenter[0]-1.1*mapsize[0]/2./self.lon_scale
        self.map_b = mapcenter[1]-1.1*mapsize[1]/2.
        self.map_r = mapcenter[0]+1.1*mapsize[0]/2./self.lon_scale
        self.map_t = mapcenter[1]+1.1*mapsize[1]/2.
        self.xcoords_a = data[0]
        self.ycoords_a = data[1]
        self.flux_a = data[2]
        self.img = img

    
    def get_input_data(self):
        # data to input:
        x_mapindex = set(np.where((self.xcoords_a>=self.map_l) & (self.xcoords_a<=self.map_r))[0])
        y_mapindex = set(np.where((self.ycoords_a>=self.map_b) & (self.ycoords_a<=self.map_t))[0])
        xy_mapindex = list(x_mapindex & y_mapindex)
        x_input = self.xcoords_a[xy_mapindex]
        y_input = self.ycoords_a[xy_mapindex]
        flux_input = self.flux_a[xy_mapindex]
        return x_input,y_input,flux_input
    
    
    def set_gridder(self):
        # gridder:
        self.target_header = setup_header(self.mapcenter, self.mapsize, self.beamsize_fwhm)
        # let's already define a WCS object for later use in our plots:
        self.target_wcs = WCS(self.target_header)
        self.gridder = cygrid.WcsGrid(self.target_header)

        # check multiprocess:
        self.gridder.set_num_threads(3) # limit multithreads.

        # kernel:
        kernelsize_fwhm = 0.6*2.9 / 60.  # degrees
        # kernelsize_fwhm = 2.5 / 60.
        # see https://en.wikipedia.org/wiki/Full_width_at_half_maximum
        kernelsize_sigma = kernelsize_fwhm / np.sqrt(8 * np.log(2))
        sphere_radius = 4. * kernelsize_sigma

        self.gridder.set_kernel(
            'gauss1d',
            (kernelsize_sigma,),
            sphere_radius,
            kernelsize_sigma / 2.
            )
    

    def start_gridding(self,source_name,channel):
        # setting:
        x_input,y_input,flux_input = self.get_input_data()

        if len(x_input)==0:
            return None,None,None

        if self.img:
            # show:
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax1.scatter(x_input,y_input,s=1)
            ax1.set_title("DEC*RA map")
            ax2 = fig.add_subplot(122)
            logdata = np.log10(flux_input)
            im = ax2.scatter(x_input,y_input,c=logdata,vmin =np.percentile(logdata,5) , vmax=np.percentile(logdata,90),cmap='PuBu_r')
            plt.colorbar(im,ax=ax2)
            plt.savefig(os.path.join(os.path.join(output_img_path,source_name),'{}_{}_input.jpg'.format(source_name,channel)))
            plt.close(fig)
            del logdata

        self.set_gridder()

        # grid and get map:
        self.gridder.grid(x_input, y_input, flux_input)
        cygrid_map = self.gridder.get_datacube()
        return cygrid_map,self.target_wcs,self.target_header

# %%
# compute mapsize for dec search:
def get_mapsize(mapsize,mapcenter):
    lon_scale = np.cos(np.radians(mapcenter[1]))
    map_l = mapcenter[0]-1.1*mapsize[0]/2./lon_scale
    map_b = mapcenter[1]-1.1*mapsize[1]/2.
    map_r = mapcenter[0]+1.1*mapsize[0]/2./lon_scale
    map_t = mapcenter[1]+1.1*mapsize[1]/2.

    return map_l,map_b,map_r,map_t

# %%
# get datafile whose coordinates are within map's range:
def get_datafile(beams,fullf,idx,idy,idz):
    # 3rd:collect data within ranges:
    df = pd.read_csv(all_dec_path,header=None,sep='\t')
    df.columns = ['drift_dec_date','beam','dec','ra_begin','ra_end']

    # get (idx,idy,idz)'s coordinate:
    with fits.open(os.path.join(os.path.join(coordinate_path,driftdec),driftdec+'_'+driftdate+'_'+beams[idx]+'_'+fullf+'_'+'coordinate.fits')) as coordinate_hdu:
        source_ra = coordinate_hdu[1].data['ra(deg)'][idy]
        source_dec = coordinate_hdu[1].data['dec(deg)'][idy]

    # compute range:
    map_l,map_b,map_r,map_t = get_mapsize((ra_size/60,dec_size/60),(source_ra,source_dec))
    # space expansion in order to avoid blank space:
    dec_space = 1/4*dec_size/60
    ra_space = 1/4*ra_size/60

    # search for dec:
    df_m = df[df['dec']>=map_b-dec_space] # bottom
    df_m = df_m[df_m['dec']<=map_t+dec_space] # top
    # search for ra:
    df_m = df_m[df_m['ra_begin']<=map_l-ra_space] # left
    df_m = df_m[df_m['ra_end']>=map_r+ra_space] # right
    
    # contain no dataset:
    if df_m.shape[0] == 0:
        print("Can not find proper data")
        return -1,-1,-1,-1,-1

    filename = df_m['drift_dec_date'].tolist()
    beamname = df_m['beam'].tolist()

    freq_cs = freq_all[idz]
    channel_list = np.where((freq_all>freq_cs-freq_size)&(freq_all<freq_cs+freq_size))[0]

    return filename,beamname,channel_list,(map_l,map_r,map_t,map_b),(source_ra,source_dec)

# %% 
# get cooridinate and mask_data
def get_negative_data(filename,beamname,channel_list,beams,fullf,map):
    map_l,map_r,map_t,map_b = map
    xcoords,ycoords,index,mask_data = {},{},{},[]

    for file in range(len(filename)):
        driftdec_t = filename[file][:14]
        beam_n = beamname[file]
        beam_idx = beams.index(beam_n)
        
        # get data within range:
        with fits.open(os.path.join(os.path.join(coordinate_path,driftdec_t),filename[file]+'_'+beam_n+'_'+fullf+'_'+'coordinate.fits')) as coordinate_hdu:
            c_data = coordinate_hdu[1].data

        # within this range:
        index_t = set(np.where(c_data['ra(deg)']>=map_l)[0])
        index_t = index_t & set(np.where(c_data['ra(deg)']<=map_r)[0])
        index_t = index_t & set(np.where(c_data['dec(deg)']>=map_b)[0])
        index_t = index_t & set(np.where(c_data['dec(deg)']<=map_t)[0])
        index_t = list(index_t)
        # save in dict manner:
        index['{}_{}'.format(filename[file],beam_n)] = index_t 
        xcoords['{}_{}'.format(filename[file],beam_n)] = c_data['ra(deg)'][index_t] 
        ycoords['{}_{}'.format(filename[file],beam_n)] = c_data['dec(deg)'][index_t]

        # get mask data:
        s_data,t_data = 0,0
        if 'Dec+4037' in driftdec_t:
            with fits.open(mask_path+driftdec_t+'/Signal_Mask_'+filename[file]+'_'+fullf+'.fits') as signal_mask_hduu:
                s_data = signal_mask_hduu[0].data
                t_data = signal_mask_hduu[0].data[beam_idx,:,:,0]
                t_data = t_data[index_t,:] # pick mjd first
                t_data = t_data[:,channel_list]
        else:
            with fits.open(mask_path+driftdec_t+'/Signal_Mask_'+filename[file]+'_'+beam_n+'_'+fullf+'.fits') as signal_mask_hduu:
                s_data = signal_mask_hduu[0].data
                t_data = signal_mask_hduu[0].data[:,:,0]
                t_data = t_data[index_t,:] # pick mjd first
                t_data = t_data[:,channel_list] # pick channel
                
        
        xlen = 298
        ylen = 393
        # make up for the edge setting in signal mask producing process
        if beam_idx == 0 or beam_idx==18:
            t_data[:,:] = 1 
        xlen_in = [i for i in range(len(index_t)) if index_t[i]<xlen or index_t[i]>s_data.shape[1]-xlen]
        t_data[xlen_in,:] = 1
        ylen_in = [i for i in range(len(channel_list)) if channel_list[i]<ylen or channel_list[i]>s_data.shape[2]-ylen]
        t_data[:,ylen_in] = 1

        mask_data.extend(t_data)

        del s_data
        
        del c_data 
        gc.collect()


    return xcoords,ycoords,index,mask_data

# check mask:
def check_mask(mask_data):
    # 4th: check whether contains any signal:
    mask_data = np.array(mask_data)
    threshold = 0.001 # can tolerate pieces of signal
    # too much signal
    if len(np.where(mask_data==0)[0])>len(mask_data)*threshold:
        print("contain signal! Should be discard.")
        print(len(np.where(mask_data==0)[0]),len(mask_data)*threshold)
        return False
    
    return True

# %%
# get flux data
def get_dataset(filename,beamname,channel_list,xcoords,ycoords,index,beams,fullf):
    # 5th:get flux data and gridding:
    xcoords_l,ycoords_l,flux_l0,flux_l1 = [],[],[],[]
    for file in range(len(filename)):
        driftdec_t = filename[file][:14]
        calibration_pathh = os.path.join(calibration_path,'{}_cube/'.format(driftdec_t[:8]))
        beam_n = beamname[file]
        beam_idx = beams.index(beam_n)

        # polaration
        for j in range(2): 
            # flux data:
            flux_path = os.path.join(calibration_pathh,'{}_{}_{}.fits'.format(driftdec_t,fullf,j)) # calibaration dataset by gaoyunhao 
            with fits.open(flux_path) as cali_hdu:
                cali_data = cali_hdu[0].data # shape = (19,2048,15728)
                cali_data_t = cali_data[beam_idx,:,:]
                cali_data_t = cali_data_t[index['{}_{}'.format(filename[file],beam_n)],:] # pick mjd first
                cali_data_t = cali_data_t[:,channel_list]
                if j==0:
                    flux_l0.extend(cali_data_t)
                else:
                    flux_l1.extend(cali_data_t)
            
        xcoords_l.extend(xcoords['{}_{}'.format(filename[file],beam_n)])
        ycoords_l.extend(ycoords['{}_{}'.format(filename[file],beam_n)])

    xcoords_a = np.array(xcoords_l)
    ycoords_a=  np.array(ycoords_l)
    flux_l0 = np.expand_dims(np.array(flux_l0),axis=2)
    flux_l1 = np.expand_dims(np.array(flux_l1),axis=2)
    flux_a = np.concatenate((flux_l0,flux_l1),axis=2) # (:,:,2)
    print(xcoords_a.shape,flux_a.shape)

    del xcoords_l,ycoords_l,flux_l0,flux_l1
    
    return xcoords_a,ycoords_a,flux_a

def set_header_datacube(header,channel_num,freq_cs):
    new_header = {
        'NAXIS': 4,
        'NAXIS1': header['naxis1'],
        'NAXIS2': header['naxis2'],
        'NAXIS3': channel_num,  # need dummy spectral axis
        'NAXIS4': 2, # polar
        'CTYPE1': 'RA---SIN',
        'CTYPE2': 'DEC--SIN',
        'CTYPE3': 'FREQ',
        'CTYPE4': 'STOKES',
        'CUNIT1': 'deg',
        'CUNIT2': 'deg',
        'CUNIT3': 'GHz',
        'CDELT1':header['cdelt1'],
        'CDELT2':header['cdelt2'],
        'CDELT3':0.00762939/1e3, # GHz
        'CDELT4':-1.0,
        'CRVAL1':header['crval1'],
        'CRVAL2':header['crval2'],
        'CRVAL3':freq_cs/1e3,
        'CRVAL4':-5.0, # polarization code at reference pixel (XX)
        'CRPIX1':header['crpix1'],
        'CRPIX2':header['crpix2'],
        'CRPIX3':channel_num/2, 
        'CRPIX4':1.0 # polarization code reference pixel
    }
    return new_header

# %%
# produce datacube:
def get_datacube(source_name):
    gridding_data = []
    for j in range(2):
        gridding_data_s = []
        with fits.open(os.path.join(os.path.join(output_file,source_name),'{}_{}.fits'.format(source_name,j))) as hdu:
            for i in range(1,len(hdu)):
                gridding_data_s.append(hdu[i].data)
            header = hdu[len(hdu)//2].header

        
        gridding_data.append(np.array(gridding_data_s))# (channel,dec,ra)
        
    gridding_data = np.array(gridding_data) # (polar,channel,dec,ra)
    freq_cs = header['freq']
    new_header = set_header_datacube(header,gridding_data.shape[1],freq_cs)

    prim = fits.PrimaryHDU()
    prim.header['Info'] = '{} Datacube'.format(source_name)
    hdul = fits.HDUList([prim])

    hduu = fits.ImageHDU(gridding_data)
    for key in new_header.keys():
        hduu.header[key] = new_header[key]

    hdul.append(hduu)
    hdul.writeto(os.path.join(os.path.join(output_file,source_name),source_name+'_datacube.fits'),overwrite=True)

    print("source {} datacube finished!".format(source_name))

    del gridding_data
    del hdul
    gc.collect()

    # delete single polar data:
    try:
        for j in range(2):
            os.remove(os.path.join(os.path.join(output_file,source_name),'{}_{}.fits'.format(source_name,j)))
    except Exception as e:
        print("delete error: {}".format(e))

# %%
def negative_datacube(drift_dec_date,negative_example_num):
    """
    params:drift_dec_date: indicates dec which source belongs to and the matching file's name.
    params:negative_example_num:amount of negative datacube should be generated.
    """
    # file init:
    path_init(drift_dec_date)

    beams = ['M18','M17','M19','M16','M07','M08','M06','M02','M15','M01','M09','M05','M03','M14','M04','M10','M13','M11','M12']
    

    for num in range(negative_example_num):
        # 1st: Get Mask Array:
        # select a numf(M02-M43)
        numf = random.randint(2,43)
        fullf = ''
        if numf<10:
            fullf = '000'+str(numf)
        else:
            fullf = '00'+str(numf)
        
        # get signal_mask
        signal_mask_hdu = fits.open(mask_path+driftdec+'/Signal_Mask_'+driftdec+'_'+driftdate+'_'+fullf+'.fits')
        mask = signal_mask_hdu[0].data # shape:(19, 2048, 15728, 2)
        signal_mask_hdu.close()

        # 2nd: Get A Non-signal Coordinate:
        # get non-zero place: 0-signal,1-non-signal
        # to find an appropriate coordiate, impose some constrains on it:
        # 1. beam which be selected should locate nearby center. We only select central 7 beams
        # 2. mjd range whose coordinates set should be large enough to contain data whose ra in ra_size.
        # 3. freq range should large enough to contain data in freq_size 
        ra_step = 0.05 # arcmin
        mjd_unit = int(ra_size/2/ra_step)
        freq_unit = int(freq_size/freq_resolution)
        mask_m = mask[6:-6,mjd_unit:-1*mjd_unit,freq_unit:-1*freq_unit,0]
        ind = np.nonzero(mask_m) # shape = (3,)

        print("ind computation finished!")

        # every numf produce 10 datacube:
        for nnum in range(10):

            while True:
                # select a coordinate:
                ids = np.random.randint(ind[0].shape[0])
                idx = 6+ind[0][ids] # beam
                idy = mjd_unit+ind[1][ids] # mjd
                idz = freq_unit+ind[2][ids] # freq

                print(idx,idy,idz)

                filename,beamname,channel_list,map,source = get_datafile(beams,fullf,idx,idy,idz)
                
                # don't have proper data: select again.
                if not isinstance(filename, list):
                    continue
                print("get filename within map range finished!")

                # get coordinate  & mask data:
                xcoords,ycoords,index,mask_data = get_negative_data(filename,beamname,channel_list,beams,fullf,map)

                # check mask data: 
                # Contain signal:select again
                if not check_mask(mask_data):
                    continue

                del mask_data
                print("mask data check pass!")
                
                # get flux data:
                xcoords_a,ycoords_a,flux_a = get_dataset(filename,beamname,channel_list,xcoords,ycoords,index,beams,fullf)
                
                del xcoords,ycoords,index
                gc.collect()
                print("gridding data collect finished!")
               
                # polartion:
                for j in range(2):
                    # gridding process:
                    prim = fits.PrimaryHDU()
                    hdul = fits.HDUList([prim])

                    # iterate channel:
                    for c_idx in range(len(channel_list)):
                        flux = flux_a[:,c_idx,j]

                        # only draw central channel:
                        img_flag = False
                        if c_idx==len(channel_list)//2:
                            img_flag = True
                        
                        source_name = '{}_{}_{}_{}'.format(driftdec,batch_idx,num,nnum)
                        # mkdir
                        if not os.path.exists(os.path.join(output_img_path,source_name)):
                            os.mkdir(os.path.join(output_img_path,source_name))


                        # gridding:
                        mg = my_gridding((ra_size/60,dec_size/60),(source[0],source[1]),(xcoords_a,ycoords_a,flux),img_flag)
                        cygrid_map,target_wcs,target_header = mg.start_gridding(source_name,c_idx)
                    
                        if not isinstance(cygrid_map, np.ndarray):
                            continue
                        
                        # nan:
                        nan_idx = np.where(np.isnan(cygrid_map)==True)
                        if len(nan_idx[0])<=0.01*cygrid_map.shape[0]*cygrid_map.shape[1]:
                            for i in range(len(nan_idx[0])):
                                cygrid_map[nan_idx[0][i],nan_idx[1][i]] = np.nanmean(cygrid[nan_idx[0][i],:])
                        else:
                            print("nan!")
                            continue

                        if not os.path.exists(os.path.join(output_file,source_name)):
                            os.mkdir(os.path.join(output_file,source_name))
                        
                       
                        # save fits
                        hduu = fits.ImageHDU(data=cygrid_map)
                        hduu.header['index'] = c_idx
                        for key,value in target_header.items():
                            hduu.header[key] = value
                        hduu.header['freq'] = freq_all[channel_list[c_idx]]
                        hdul.append(hduu)

                        
                        if img_flag:
                            # show:
                            fig = plt.figure(figsize=(14,7))
                            # cygrid map:
                            ax1 = fig.add_subplot(111, projection=target_wcs.celestial)
                            logdata= np.log10(cygrid_map)
                            im = ax1.imshow(logdata,vmin=np.percentile(logdata,5), vmax=np.percentile(logdata,90),**imkw,cmap='PuBu_r')
                            ax1.set_title("Name = {}, RA = {}, DEC = {}, channel = {}".format(source_name,source[0],source[1],freq_all[channel_list[c_idx]]))
                            lon, lat = ax1.coords
                            lon.set_axislabel('R.A. [deg]')
                            lat.set_axislabel('Dec [deg]')
                            plt.colorbar(im,ax=ax1)
                            plt.savefig(os.path.join(os.path.join(output_img_path,source_name),'{}_{}.jpg'.format(source_name,c_idx)))
                            plt.close(fig)
                            print("{} finish".format(channel_list[c_idx]))
                            del logdata
                        
                        del cygrid_map
                        del mg
                        gc.collect()

                        if c_idx % 50 == 0:
                            print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
                        
                    
                    hdul.writeto(os.path.join(os.path.join(output_file,source_name),'{}_{}.fits'.format(source_name,j)),overwrite=True)
                    print("{} {} finish=======√".format(source_name,j))


                # release:
                gc.collect()
                del xcoords_a,ycoords_a,flux_a
                
                # produce datacube:
                get_datacube(source_name)
                break

                
if __name__ == '__main__':
    negative_datacube('Dec+4037_10_05_20211001',10)

