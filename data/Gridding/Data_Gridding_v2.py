# %%
# author:weishirui
# code version:2022-12-02
# description: apply Data Gridding to crafts data to cut Datacube for neural network.
# algorithm: 
# input:
# - /home/weishirui/Documents/crafts_data/dataset/source_data/coordinate/:dec & ra & mjd ; 
# - /home/weishirui/Documents/crafts_data/dataset/source_data/coordinate/beam_dec_ra.csv : CRAFTS's datasets dec & ra map;
# - /data07/ZD2021_1_2/:mjd & flux 
# - /home/gaoyunhao/Crafts/: calibration dataset: flux
# - /home/weishirui/Documents/crafts_data/dataset/source_data/coordinate_flux/Dec+4037_10_05/collection:mjd&flux&dec&ra
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
import threading

dec_size = 20 # arcmin(full)
ra_size = 20 # arcmin(full)
freq_size = 2.5 # MHz, half
freq_resolution =  0.00762939 # MHz
freq_new_resolution = 0.1 # MHz

level_choose = 'D'

# %%
# init all the path using drift_dec_date params:
def path_init(drift_dec_date):
    global driftdec,driftdate,crafts_path,coordinate_path,calibration_path
    global coordinate_flux_path,output_file,output_img_path,origin_path 
    global all_dec_path,HIsource_path,all_file_info_path

    # split name:
    driftdec = drift_dec_date[:14]
    driftdate = drift_dec_date[-8:]

    # input path:
    # crafts origin data:
    crafts_path = '/data07/ZD2021_1_2/{}/{}/'.format(driftdec,driftdate)
    # crafts all dataset dec information:
    all_dec_path = '/home/weishirui/Documents/crafts_data/dataset/source_data/coordinate/beam_dec_ra.csv'
    # crafts all dataset beams file dec ra information:(now only contains Dec+4037 Dec+4059 Dec+4120)
    all_file_info_path = '/home/weishirui/Documents/crafts_data/dataset/source_data/coordinate/beam_file_dec_ra.csv'

    # coordinate path based on calibration dataset:
    coordinate_path = '/home/weishirui/Documents/crafts_data/dataset/source_data/coordinate/'
    # coordinate and flux dataset based on calibration dataset:
    coordinate_flux_path = '/home/weishirui/Documents/crafts_data/dataset/source_data/coordinate_flux/'
    # HI source information:
    HIsource_path = '/home/weishirui/Documents/crafts_data/dataset/source_data/HIsource_coordinate/crafts_{}_mjd.fits.gz'.format(drift_dec_date)
    # calibraion dataset: by gaoyunhao
    calibration_path = '/home/gaoyunhao/Crafts/'

    # output path:
    # gridding output path:
    output_file = '/home/weishirui/Documents/crafts_data/dataset/RAtransfer_dataset/gridding/positive/data/{}'.format(driftdec)
    # gridding img output path:
    output_img_path = '/home/weishirui/Documents/crafts_data/dataset/RAtransfer_dataset/gridding/positive/output_img/{}_{}'.format(driftdec,driftdate)
    # origin path:
    origin_path =  os.path.join(output_file, 'origin')
    

    if not os.path.exists(output_file):
        os.mkdir(output_file)
    if not os.path.exists(origin_path):
        os.mkdir(origin_path)
    output_file = os.path.join(output_file,level_choose)
    if not os.path.exists(output_file):
        os.mkdir(output_file)

    if not os.path.exists(output_img_path):
        os.mkdir(output_img_path)
    output_img_path = os.path.join(output_img_path,level_choose)
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

# get origin dataset:
def get_origin_dataset(origin_flux0, origin_flux1,dec_order):
    dec_order = sorted(dec_order,key = lambda x:x[1])
    flux_0 = []
    flux_1 = []
    min_shape = np.inf
    for i in dec_order:
        if i in origin_flux0.keys():
            min_shape = min(min_shape,origin_flux0[i].shape[0])
    for i in dec_order:
        if i in origin_flux0.keys():
            flux_0.append(origin_flux0[i][:min_shape,:])
            flux_1.append(origin_flux1[i][:min_shape,:])
    
    flux_0 = np.expand_dims(np.array(flux_0),axis=3)
    flux_1 = np.expand_dims(np.array(flux_1),axis=3)
    flux = np.concatenate((flux_0,flux_1),axis=3) # (:,:,:,2)
    flux = np.array(flux)

    return flux

   
# get dataset:
# not only from one dec but other dec to make up for the gaps.
def get_dataset(source_ra,source_dec,channel):
    beams = ['M18','M17','M19','M16','M07','M08','M06','M02','M15','M01','M09','M05','M03','M14','M04','M10','M13','M11','M12']

    # compute mapsize:
    map_l,map_b,map_r,map_t = get_mapsize((ra_size/60,dec_size/60),(source_ra,source_dec))

    # search for decs and beams included:
    df = pd.read_csv(all_dec_path,header=None,sep='\t')
    df.columns = ['drift_dec_date','beam','dec','ra_begin','ra_end']
    
    # space expansion in order to avoid blank space:
    ra_space = 1/4*ra_size/60
    dec_space = 1/4*dec_size/60

    # search for dec:
    df = df[df['dec']>=map_b-dec_space] # bottom
    df = df[df['dec']<=map_t+dec_space] # top
    # search for ra:
    df = df[df['ra_begin']<=map_l-ra_space] # left
    df = df[df['ra_end']>=map_r+ra_space] # right

    
    # contain no dataset:
    if df.shape[0] == 0:
        print("Can not find data in map")
        return -1,-1,-1,-1

    # read data:
    filename = df['drift_dec_date'].tolist()
    beamname = df['beam'].tolist()

    # sort dec by file&beam
    df['file_beam'] = df.apply((lambda x:x['drift_dec_date']+'_'+x['beam']),axis=1)
    df_order = df.sort_values(by='dec',inplace=False)
    dec_order = dict(zip(df_order['file_beam'].tolist(),[i for i in range(df_order.shape[0])]))


    # cluster together:
    file_beam = dict()
    for file in range(len(filename)):
        if not (filename[file] in file_beam.keys()):
            file_beam[filename[file]] = []
        
        file_beam[filename[file]].append(beamname[file])

    # release:
    del df

    df =  pd.read_csv(all_file_info_path,header=None,sep='\t')
    df.columns = ['drift_dec_date','beam','index','dec','ra_begin','ra_end']

    xcoords_l,ycoords_l,flux_l0,flux_l1 = [],[],[],[]
    origin_flux0 = {}
    origin_flux1 = {}

    for file,beam_n in file_beam.items():
        mjd_index = {}

        driftdec_t = file[:14]
        beam_idx = sorted([beams.index(i) for i in beam_n]) # sort by asc
        calibration_pathh = os.path.join(calibration_path,'{}_cube/'.format(driftdec_t[:8]))
        
        # select fullf:
        dff = df[df['drift_dec_date']==file]
        dff = dff[dff['beam'].isin(beam_n)]
        # dff = dff[dff['ra_begin']<=map_l-ra_space]
        # dff = dff[dff['ra_end']>=map_r+ra_space]
        # think about it:
        dff = dff[dff['ra_begin']<=map_r+ra_space] # consider signals which lie across two files.
        dff = dff[dff['ra_end']>=map_l-ra_space]
        
        full_list = sorted(list(set(dff['index'].tolist())))

        if len(full_list)==0:
            print("len(full_list)==0")
            return -1,-1,-1,-1

        # dec & ra data:
        for full in full_list:
            if full == 1 or full==44:
                continue
            fullf = ''
            if full<10:
                fullf = '000'+str(full)
            else:
                fullf = '00'+str(full)
            
            for b_s in beam_idx:
                with fits.open(os.path.join(os.path.join(coordinate_path,driftdec_t),file+'_'+beams[b_s]+'_'+fullf+'_'+'coordinate.fits')) as coordinate_hdu:
                    c_data = coordinate_hdu[1].data # shape = (2048,)
                    xcoords_l.extend(c_data['ra(deg)'])
                    ycoords_l.extend(c_data['dec(deg)'])

                # within this range:
                index_t = set(np.where(c_data['ra(deg)']>=map_l)[0])
                index_t = index_t & set(np.where(c_data['ra(deg)']<=map_r)[0])
                index_t = index_t & set(np.where(c_data['dec(deg)']>=map_b)[0])
                index_t = index_t & set(np.where(c_data['dec(deg)']<=map_t)[0])
                if not fullf in mjd_index.keys():
                    mjd_index[fullf] = set()
                mjd_index[fullf] = mjd_index[fullf] | index_t
                
            mjd_index[fullf] = sorted(list(mjd_index[fullf]))
            mjd_index[fullf] = [mjd_index[fullf][i] for i in range(0,len(mjd_index[fullf])-1) if mjd_index[fullf][i+1]-mjd_index[fullf][i] ==1]

        # polaration
        for j in range(2): 
            for full in full_list:
                if full == 1 or full==44:
                    continue
                fullf = ''
                if full<10:
                    fullf = '000'+str(full)
                else:
                    fullf = '00'+str(full)
                    
                # flux data:
                flux_path = os.path.join(calibration_pathh,'{}_{}_{}.fits'.format(driftdec_t,fullf,j)) # calibaration dataset by gaoyunhao 
                
                if not os.path.exists(flux_path):
                    print("Can not find flux_path")
                    return -1,-1,-1,-1

                with fits.open(flux_path) as cali_hdu:
                    cali_data = cali_hdu[0].data # shape = (19,2048,15728)
                    for b_s in beam_idx:
                        cali_data_t = cali_data[b_s,:,:]
                        cali_data_t = cali_data_t[:,channel]
                        if len(cali_data_t.shape)!=2:
                            print(b_s,len(channel)) 
                        if j==0:
                            # collect origin dataset:
                            if '{}_{}'.format(file,beams[b_s]) in origin_flux0.keys():
                                origin_flux0['{}_{}'.format(file,beams[b_s])] = np.concatenate((origin_flux0['{}_{}'.format(file,beams[b_s])],cali_data_t[mjd_index[fullf],:]),axis=0)
                            else:
                                origin_flux0['{}_{}'.format(file,beams[b_s])] = cali_data_t[mjd_index[fullf],:]

                            flux_l0.extend(cali_data_t)
                        else:
                            flux_l1.extend(cali_data_t)
                            # collect origin dataset:
                            if '{}_{}'.format(file,beams[b_s]) in origin_flux1.keys():
                                origin_flux1['{}_{}'.format(file,beams[b_s])] = np.concatenate((origin_flux1['{}_{}'.format(file,beams[b_s])],cali_data_t[mjd_index[fullf],:]),axis=0)
                            else:
                                origin_flux1['{}_{}'.format(file,beams[b_s])] = cali_data_t[mjd_index[fullf],:]

    if len(flux_l0) == 0:
        print("flux == 0")
        return -1,-1,-1,-1
        

    xcoords_a = np.array(xcoords_l)
    ycoords_a=  np.array(ycoords_l)
    flux_l0 = np.expand_dims(np.array(flux_l0),axis=2)
    flux_l1 = np.expand_dims(np.array(flux_l1),axis=2)
    flux_a = np.concatenate((flux_l0,flux_l1),axis=2) # (:,:,2)
    
    # origin flux data:

    origin_flux_a = get_origin_dataset(origin_flux0,origin_flux1,dec_order)
    

    del xcoords_l,ycoords_l,flux_l0,flux_l1,origin_flux0,origin_flux1


    return xcoords_a,ycoords_a,flux_a,origin_flux_a

# %%
# read source file: select some obvious sources to show.(tentative work)
def source_select():
    dec_num = {'Dec+4037':0,'Dec-0619':1,'Dec+4059':2,'Dec+4120':3}
    beams = ['M18','M17','M19','M16','M07','M08','M06','M02','M15','M01','M09','M05','M03','M14','M04','M10','M13','M11','M12']
    df = pd.read_excel('/home/lupengjun/findsource/第二批人工.xlsx',sheet_name=dec_num[driftdec.split('_')[0]],engine = 'openpyxl')
    
    # remove sources which have been generated before.
    exists_source = os.listdir(output_file)
    exists_source = [int(s[3:]) for s in exists_source]

    # only select A signal for img:
    A_source = list(set(df[(df['2D评分']==level_choose) & (df['1D评分']==level_choose)]['信号'].tolist()))
    A_source = sorted([int(i) for i in A_source])
    print(A_source)
    A_source = np.array(sorted(list(set(A_source)-set(exists_source)))) # remove existing sources.
    print(A_source)
    # to-do-list:
    # use all levels' signals to generate training datasets.

    with fits.open(HIsource_path) as source_hdu:
        all_source = list(source_hdu[1].data['name'])
        all_source = np.array([int(i[3:]) for i in all_source]) # skip 'NSA'

    # select A:
    source_name = []
    source_dec = []
    source_ra = []
    source_mjd = []
    source_freq = []
    for s in A_source:
        if s in all_source:
            t_index = np.where(np.isin(all_source,s)==True)[0]
            # more than one beam:
            # name,ra,dec are same. mjd is different. select the center one.
            source_name.append(source_hdu[1].data['name'][t_index[0]])
            if(int(source_hdu[1].data['name'][t_index[0]][3:]) != s):
                print(s)
            source_dec.append(source_hdu[1].data['dec'][t_index[0]])
            source_ra.append(source_hdu[1].data['ra'][t_index[0]])
            source_freq.append(1420.406/(1+source_hdu[1].data['z'][t_index[0]]))

            if len(t_index)>1:
                source_beams = source_hdu[1].data['beam'][t_index].tolist()
                beam_order = [beams.index(i) for i in source_beams] # dec order of beams
                beams_dict = dict(zip(beam_order,source_beams)) 
                beams_sort = sorted(beams_dict.items(),key = lambda x:x[0]) # sort by order
                t_indexx = source_beams.index(beams_sort[len(t_index)//2][1]) # select center beams
                source_mjd.append(source_hdu[1].data['mjd'][t_index[t_indexx]]) # get the matching mjd
            else:
                source_mjd.append(source_hdu[1].data['mjd'][t_index])
        else:
            continue

        
    # release:
    del df
    
    return source_name,source_dec,source_ra,source_freq,source_mjd

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
    del hdul
    gc.collect()

    # delete single polar data:
    try:
        for j in range(2):
            os.remove(os.path.join(os.path.join(output_file,source_name),'{}_{}.fits'.format(source_name,j)))
    except Exception as e:
        print("delete error: {}".format(e))


# %% 
# source examples:
def source_datacube(drift_dec_date):
    """
    params:drift_dec_date: indicates dec which source belongs to and the matching file's name.
    """
    # file init:
    path_init(drift_dec_date)

    # select source:
    source_name,source_dec,source_ra,source_freq,_ = source_select()

    # gridding
    for s_idx in range(len(source_dec)):

        print("{} {} {} begin:".format(source_name[s_idx],source_dec[s_idx],source_ra[s_idx]))

        freq_cs = source_freq[s_idx]
        channel_list = np.where((freq_all>freq_cs-freq_size)&(freq_all<freq_cs+freq_size))[0]
        # channel_list = [channel_list[len(channel_list)//2]]

        if len(channel_list)==0:
            print("can't find proper channel!")
            continue

        # get coordinate & flux data:
        xcoords_a,ycoords_a,flux_a,origin_flux_a = get_dataset(source_ra[s_idx],source_dec[s_idx],channel_list)
        # there isnot proper data for gridding.
        if not type(xcoords_a) is np.ndarray:
            continue
        
        prim = fits.PrimaryHDU()
        hdul = fits.HDUList([prim])
        hduu = fits.ImageHDU(data=origin_flux_a)
        hdul.append(hduu)
        hdul.writeto(os.path.join(origin_path,'{}_origin.fits'.format(source_name[s_idx])),overwrite=True)
        
        if not os.path.exists(os.path.join(output_img_path,source_name[s_idx])):
            os.mkdir(os.path.join(output_img_path,source_name[s_idx]))

        p_flag = False
        for p in range(2): # polar
            prim = fits.PrimaryHDU()
            hdul = fits.HDUList([prim])
            # iterate channel:
            for c_idx in range(len(channel_list)):
                flux = flux_a[:,c_idx,p]

                img_flag = False
                # only draw central channel:
                if c_idx==len(channel_list)//2:
                    img_flag = True

                # gridding:
                mg = my_gridding((ra_size/60,dec_size/60),(source_ra[s_idx],source_dec[s_idx]),(xcoords_a,ycoords_a,flux),img_flag)
                cygrid_map,target_wcs,target_header = mg.start_gridding(source_name[s_idx],c_idx)

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

                if not os.path.exists(os.path.join(output_file,source_name[s_idx])):
                    os.mkdir(os.path.join(output_file,source_name[s_idx]))
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
                    ax1.set_title("Name = {}, RA = {}, DEC = {}, channel = {}".format(source_name[s_idx],source_ra[s_idx],source_dec[s_idx],freq_all[channel_list[c_idx]]))
                    lon, lat = ax1.coords
                    lon.set_axislabel('R.A. [deg]')
                    lat.set_axislabel('Dec [deg]')
                    plt.colorbar(im,ax=ax1)
                    plt.savefig(os.path.join(os.path.join(output_img_path,source_name[s_idx]),'{}_{}_invert.jpg'.format(source_name[s_idx],c_idx)))
                    plt.close(fig)
                    print("{} finish".format(channel_list[c_idx]))
                    del logdata
                
                del cygrid_map
                del mg
                gc.collect()

                if c_idx % 50 == 0:
                    print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
                
            if p_flag == False:    
                hdul.writeto(os.path.join(os.path.join(output_file,source_name[s_idx]),'{}_{}.fits'.format(source_name[s_idx],p)),overwrite=True)
                print("{} finish=======√".format(source_name[s_idx]))

            del hdul
        # release:
        del xcoords_a,ycoords_a,flux_a
        gc.collect()
        
        if p_flag == False:
            # produce datacube:
            get_datacube(source_name[s_idx])

        

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
    source_datacube('Dec+4120_10_05_20210929')