# %%
# author:weishirui
# code version:2022-12-23
# description: apply Data Gridding to virtual data to cut Datacube for neural network.
# algorithm: 
# input:
# - /home/weishirui/Documents/crafts_data/dataset/Virtual_data: data

# CyGrid & HCGrid for single channel data gridding;
# HEGrid for multi-channel data gridding.

# %%
from astropy.io import fits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
from astropy.wcs import WCS
import cygrid
import gc
import psutil

dec_size = 20 # arcmin(full)
ra_size = 20 # arcmin(full)
freq_size = 2.5 # MHz, half
freq_resolution =  0.00762939 # MHz
freq_new_resolution = 0.1 # MHz

# %%
# init all the path using drift_dec_date params:
def path_init(virtual_name):
    global virtual_path,output_file,output_img_path
    
    # input path:
    # virtual data:
    virtual_path = '/home/weishirui/Documents/crafts_data/dataset/Virtual_data/{}'.format(virtual_name)
    
    # output path:
    # gridding output path:
    output_file = '/home/weishirui/Documents/crafts_data/dataset/RAtransfer_dataset/virtual_data/datacube/{}'.format(virtual_name)
    # gridding img output path:
    output_img_path = '/home/weishirui/Documents/crafts_data/dataset/RAtransfer_dataset/virtual_data/output_img/{}'.format(virtual_name)

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
            plt.savefig(os.path.join(os.path.join(output_img_path,source_name),'{}_{}_invert_input.jpg'.format(source_name,channel)))
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
    hdul.writeto(os.path.join(os.path.join(output_file,source_name),source_name+'.fits'),overwrite=True)

    print("source {} datacube finished!".format(source_name))

    del gridding_data
    del hdul,hduu
    gc.collect()

    # delete single polar data:
    try:
        for j in range(2):
            os.remove(os.path.join(os.path.join(output_file,source_name),'{}_{}.fits'.format(source_name,j)))
    except Exception as e:
        print("delete error: {}".format(e))


# %% 
# source examples:
def source_datacube(virtual_name):
    """
    params:drift_dec_date: indicates dec which source belongs to and the matching file's name.
    """
    # file init:
    path_init(virtual_name)
    virtual_name_date = virtual_name.split('_')[-2]
    
    # load virtual data:
    v_hdu = fits.open(virtual_path)
    
    for v in range(1,len(v_hdu),2):
        # select a signal:
        flux_data = v_hdu[v].data[:,:,65:721,:] # select 5MHz
        xcoords = v_hdu[v+1].data[:,1,:]
        ycoords = v_hdu[v+1].data[:,2,:]
        # squeeze:
        xcoords_a = xcoords.flatten()
        ycoords_a = ycoords.flatten()

        center_ra = v_hdu[v].header['center_ra']
        center_dec = v_hdu[v].header['center_dec']
        center_freq = v_hdu[v].header['center_freq']

        source_name = virtual_name_date+'_'+v_hdu[v].name
       
        print("{} {} {} begin:".format(source_name,center_ra,center_dec))


        if not os.path.exists(os.path.join(output_img_path,source_name)):
            os.mkdir(os.path.join(output_img_path,source_name))
        
        # don't produce the same data
        if os.path.exists(os.path.join(os.path.join(output_file,source_name),source_name+'.fits')):
            continue
    
        p_flag = False

        for p in range(2): # polar
            prim = fits.PrimaryHDU()
            hdul = fits.HDUList([prim])
    
            # iterate channel:
            for c_idx in range(flux_data.shape[2]):
                flux = flux_data[:,:,c_idx,p]
                # squeeze:
                flux = flux.flatten()
                
                img_flag = False
                # only draw central channel:
                if c_idx==flux_data.shape[2]//2:
                    img_flag = True

                # gridding:
                mg = my_gridding((ra_size/60,dec_size/60),(center_ra,center_dec),(xcoords_a,ycoords_a,flux),img_flag)
                cygrid_map,target_wcs,target_header = mg.start_gridding(source_name,c_idx)

                if not isinstance(cygrid_map, np.ndarray):
                    p_flag = True
                    break
                
                # nan:
                nan_idx = np.where(np.isnan(cygrid_map)==True)
                if len(nan_idx[0])<=0.01*cygrid_map.shape[0]*cygrid_map.shape[1]:
                    for i in range(len(nan_idx[0])):
                        cygrid_map[nan_idx[0][i],nan_idx[1][i]] = np.nanmean(cygrid[nan_idx[0][i],:])
                else:
                    print("nan!")
                    p_flag = True
                    break
                

                if not os.path.exists(os.path.join(output_file,source_name)):
                    os.mkdir(os.path.join(output_file,source_name))
                
                # save fits
                hduu = fits.ImageHDU(data=cygrid_map)
                hduu.header['index'] = c_idx
                for key,value in target_header.items():
                    hduu.header[key] = value
                hduu.header['freq'] = center_freq
                hdul.append(hduu)

                if img_flag:
                    # show:
                    fig = plt.figure(figsize=(14,7))
                    # cygrid map:
                    ax1 = fig.add_subplot(111, projection=target_wcs.celestial)
                    logdata= np.log10(cygrid_map)
                    im = ax1.imshow(logdata,vmin=np.percentile(logdata,5), vmax=np.percentile(logdata,90),**imkw,cmap='PuBu_r')
                    ax1.set_title("Name = {}, RA = {}, DEC = {}, channel = {}".format(source_name,center_ra,center_dec,center_freq))
                    lon, lat = ax1.coords
                    lon.set_axislabel('R.A. [deg]')
                    lat.set_axislabel('Dec [deg]')
                    plt.colorbar(im,ax=ax1)
                    plt.savefig(os.path.join(os.path.join(output_img_path,source_name),'{}_{}_invert.jpg'.format(source_name,c_idx)))
                    plt.close(fig)
                    print("{} finish".format(c_idx))
                    del logdata
                
                del cygrid_map
                del mg
                gc.collect()

                if c_idx % 50 == 0:
                    print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
                
            if p_flag == False:    
                hdul.writeto(os.path.join(os.path.join(output_file,source_name),'{}_{}.fits'.format(source_name,p)),overwrite=True)
                print("{} finish=======√".format(source_name))

            del hdul
        # release:
        del xcoords,ycoords,xcoords_a,ycoords_a,flux_data
        gc.collect()
        
        if p_flag == False:
            # produce datacube:
            get_datacube(source_name)

        

# compress frequency:
def transfer_freq(data):
    # data.shape = (channel,dec,ra)
    times = int(freq_new_resolution/freq_resolution)
    # only compress not interpolation
    if times<1:
        print("Can't Compress! Please adjust your resolution.")
        return 

    # regrid function
    # average to new resolution:
    length = data.shape[0]
    frequency_left = len([j for j in range(length//2+times//2-1,0,int(-1*times))])-1
    frequency_right = len([j for j in range(length//2+times//2,length,times)])-1
    m_data = np.zeros((frequency_left+frequency_right,data.shape[1],data.shape[2]))
    left_data = []
    right_data = [] 
    
    # average operation from center to avoid source offset.
    for d in range(data.shape[1]):
        for r in range(data.shape[2]):
            # left:
            left_index = [j for j in range(length//2+times//2-1,0,int(-1*times))]
            left_index.reverse()
            left_indexx = []
            for j in range(len(left_index)-1):
                left_indexx.append([x for x in range(left_index[j],left_index[j+1])])
            left_data = data[left_indexx,d,r]
            left_data = np.mean(left_data,axis=1)
        
            # right:
            right_index = [j for j in range(length//2+times//2,length,times)]
            right_indexx = []
            for j in range(len(right_index)-1):
                right_indexx.append([x for x in range(right_index[j],right_index[j+1])])
            right_data = data[right_indexx,d,r]
            right_data = np.mean(right_data,axis=1)
            
            m_data[:frequency_left,d,r] = left_data.copy()
            m_data[frequency_left:,d,r] = right_data.copy()

    return m_data


# %%
if __name__ == '__main__':
    source_datacube('CRAFTS_version2_cali_2022-12-24-20-32_cf.fits')