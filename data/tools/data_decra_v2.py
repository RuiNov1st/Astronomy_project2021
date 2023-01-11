# %%
"""
author:weishirui
code version:2023-01-10
description: 
compute every record's dec & ra coordinate in crafts dataset.
update(10.21): using gaoyunhao's dataset which have been calilbration already. Have to compute delta mjd of every beam based on M01.
update(11.5): not only just consider one dec, but compute all dec.
update(11.13): crude estimate of dec should be replaced by the dec with mjd.
update(11.28): generate a file that records the relationships between dec, beam and fileindex 's ra and dec.
update(1.3): modify to accomodate the 3rd batch datasets.
update(1.10): record mjd begin & mjd end into a csv file.
algorithm: 
input:
- /data07/position:dec & ra & mjd ; 
- /data07/ZD2021_1_2/:mjd ;
- /home/gaoyunhao/Crafts/: calibration dataset: flux ;
interpolation mjd to align
"""

# %%
from astropy.io import fits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors
import os

# %%
# get dec list:
def get_filelist():
    declist = os.listdir('/data07/ZD2021_1_2/')
    declist = [i for i in declist if len(i) == 14]
    # get dec date:
    decdate = []
    drift_dec_date = []
    for i in declist:
        decdate_s = os.listdir(os.path.join('/data07/ZD2021_1_2/',i))
        decdate.append(decdate_s)
        for j in decdate_s:
            drift_dec_date.append(i+'_'+j)
    
    return drift_dec_date

# %%
# compute other beams' ra & dec:
# algorithm: from crossmatch.py by lupengjun
beams0 = {'beam_name':['M01','M02','M03','M04','M05','M06','M07',
                        'M08','M09','M10','M11','M12','M13','M14',
                        'M15','M16','M17','M18','M19'],
            'beam_offset_ra':[0.,5.74,2.88,-2.86,-5.74,-2.88,2.86,
                            11.5,8.61,5.75,0.018,-5.71,-8.6,-11.5,
                            -8.63,-5.77,-0.0181,5.73,8.61],
            'beam_offset_dec':[0.,0.00811,-4.97,-4.98,-0.0127,4.97,4.98,
                            0.0116,-4.96,-9.93,-9.94,-9.96,-4.99,-0.03,
                            4.95,9.93,9.94,9.95,4.98]   }
pi = 3.1415926535
bra = np.array(beams0['beam_offset_ra'])
bdec = np.array(beams0['beam_offset_dec'])
bang = np.arctan(bdec/bra)/pi * 180.0
ind2 = np.where(bra<0)[0]
ind4 = np.where((bra>0) & (bdec<0))[0]
bang[ind2] = bang[ind2] + 180.
bang[ind4] = bang[ind4] + 360.
bang[0]=0.
###########
bang_rot = bang + 23.4
###########
beamdist = np.sqrt(bra**2 + bdec**2)
beams = {'beam_name':beams0['beam_name'],                  'beam_offset_ra':list(beamdist*np.cos(bang_rot/180.*pi)),                 'beam_offset_dec':list(beamdist*np.sin(bang_rot/180.*pi))}

# %%
# get M01's dec & ra in every mjd record:
def get_position(drift_dec_date):
    position_path = '/data07/position/{}.fits.gz'.format(drift_dec_date)
    if os.path.exists(position_path):
        with fits.open(position_path) as pos_hdu:
            pos_data = pos_hdu[1].data
            # data:
            pos_mjd = pos_data['mjd']
            pos_ra = pos_data['ra_deg']
            pos_dec = pos_data['dec_deg']

            return pos_mjd,pos_ra,pos_dec
    else:
        print("{} doesn't exist!".format(position_path))

# %%
# get M01's mjd & ra & dec (average dec):
def M01_baseline(pos_mjd,pos_ra,pos_dec,drift_dec_date):

    # interpolate function:
    ra_fun = interp1d(pos_mjd,pos_ra,kind='linear',fill_value='extrapolate')
    dec_fun = interp1d(pos_mjd,pos_dec,kind='cubic',fill_value='extrapolate') # exist bias

    # split name:
    driftdec = drift_dec_date[:14]
    driftdate = drift_dec_date[-8:]

    # compute differet beams' mjd gap based on M01
    # delta_mjd of every beam transfers from a constant value to an array related to mjd.
    delta_mjd = {}
    # delta ra:
    dic = {'M18': -6.180555555555555e-05, 'M17': 0.0001886574074074074, 'M19': -0.0002813888888888889, 'M16': 0.0004391666666666667, 'M07': -3.0925925925925924e-05, 'M08': -0.0005009722222222222, 'M06': 0.00021958333333333335, 'M02': -0.00025050925925925924, 'M15': 0.0004700925925925926, 'M01': -0.0, 'M09': -0.0004700925925925926, 'M05': 0.00025050925925925924, 'M03': -0.00021958333333333335, 'M14': 0.0005009722222222222, 'M04': 3.0925925925925924e-05, 'M10': -0.0004391666666666667, 'M13': 0.0002813888888888889, 'M11': -0.00018877314814814814, 'M12': 6.180555555555555e-05}
    
    # current dec(crude):
    # dec = int(int(driftdec[4:8])/100)+(int(driftdec[4:8])/100-int(int(driftdec[4:8])/100))*100/60 # (deg)
    

    for i in dic.keys():
        delta_mjd[i] = (dic['M01']-dic[i])/np.cos(pos_dec/180*np.pi) # an array, same shape as pos_dec

    # M01 baseline:
    crafts_mjd = []
    M01_ra = []
    M01_dec = []
    for i in range(1,45):
        # record index:
        index = ''
        if i<10:
            index = '000'+str(i)
        else:
            index = '00'+str(i)
        # get mjd:
        M01_path = '/data07/ZD2021_1_2/{}/{}/{}_arcdrift-M01_W_{}.fits'.format(driftdec,driftdate,driftdec,index)
        with fits.open(M01_path) as M01_hdu:
            M01_data = M01_hdu[1].data
            M01_mjd = M01_data['utobs']
            # interpolate:
            M01_ra_s = ra_fun(M01_mjd) # s->single
            M01_dec_s = dec_fun(M01_mjd)
            crafts_mjd.extend(M01_mjd)
            M01_ra.extend(M01_ra_s)
            M01_dec.extend(M01_dec_s)
            
    # change to ndarray
    crafts_mjd = np.array(crafts_mjd)
    M01_ra = np.array(M01_ra)
    M01_dec = np.array(M01_dec)

    return ra_fun,dec_fun,crafts_mjd,M01_ra,M01_dec

# %%
# get M01's mjd & ra & dec (average dec):
def M01_baseline_sim(pos_mjd,pos_ra,pos_dec,drift_dec_date,M01_path_o = '/data15/ZD2020_1_2/Dec-0011_04_05/20200903'):

    # interpolate function:
    ra_fun = interp1d(pos_mjd,pos_ra,kind='linear',fill_value='extrapolate')
    dec_fun = interp1d(pos_mjd,pos_dec,kind='cubic',fill_value='extrapolate') # exist bias

    # split name:
    driftdec = drift_dec_date[:14]
    driftdate = drift_dec_date[-8:]

    # M01 baseline:
    crafts_mjd = []
    
    # get mjd:
    # M01_path = '/data07/ZD2021_1_2/{}/{}/{}_arcdrift-M01_W_0002.fits'.format(driftdec,driftdate,driftdec)
    M01_path = os.path.join(M01_path_o,'{}_arcdrift-M01_W_0002.fits'.format(driftdec))
    with fits.open(M01_path) as M01_hdu:
        M01_data = M01_hdu[1].data
        M01_mjd = M01_data['utobs']
        M01_mjd_begin = M01_mjd[0]
        mjd_step = np.mean(np.array(M01_mjd[1:])-np.array(M01_mjd[:-1]))
        
        crafts_mjd = [M01_mjd_begin+i*mjd_step for i in range(0,2048*42)]

        print(M01_mjd_begin,mjd_step,len(crafts_mjd))

    # change to ndarray
    crafts_mjd = np.array(crafts_mjd)

    return ra_fun,dec_fun,crafts_mjd

# %%
def get_beam(ra_fun,dec_fun,drift_dec_date,crafts_mjd,pos_dec):

    # split name:
    driftdec = drift_dec_date[:14]
    driftdate = drift_dec_date[-8:]

    # compute differet beams' mjd gap based on M01
    # delta_mjd of every beam transfers from a constant value to an array related to mjd.
    delta_mjd = {} 
    # delta ra:
    dic = {'M18': -6.180555555555555e-05, 'M17': 0.0001886574074074074, 'M19': -0.0002813888888888889, 'M16': 0.0004391666666666667, 'M07': -3.0925925925925924e-05, 'M08': -0.0005009722222222222, 'M06': 0.00021958333333333335, 'M02': -0.00025050925925925924, 'M15': 0.0004700925925925926, 'M01': -0.0, 'M09': -0.0004700925925925926, 'M05': 0.00025050925925925924, 'M03': -0.00021958333333333335, 'M14': 0.0005009722222222222, 'M04': 3.0925925925925924e-05, 'M10': -0.0004391666666666667, 'M13': 0.0002813888888888889, 'M11': -0.00018877314814814814, 'M12': 6.180555555555555e-05}
    
    # current dec(crude):
    #dec = int(int(driftdec[4:8])/100)+(int(driftdec[4:8])/100-int(int(driftdec[4:8])/100))*100/60 # (deg)
    
    for i in dic.keys():
        delta_mjd[i] = (dic['M01']-dic[i])/np.cos(pos_dec/180*np.pi)


    all_mjd = {} # include all beams' mjd
    all_coords_ra = {} # include all beams' ra
    all_coords_dec = {} # include all beams' dec

    all_mjd['M01'] = crafts_mjd

    # compute every beam's mjd based on delta_mjd
     # if Mx earlier than M01, then mjd will be substracted.
    # else mjd will be added.
    for m in dic.keys():
        all_mjd[m] = all_mjd['M01']-delta_mjd[m] # substract an array, same shape as all_mjd

        # interpolate:
        # now just compute M01's ra & dec in every mjd:
        all_coords_ra[m] = np.array(ra_fun(all_mjd[m]))
        all_coords_dec[m] = np.array(dec_fun(all_mjd[m]))
        
        # now transform into other beams:
        m_index = beams0['beam_name'].index(m)
        
        all_coords_ra[m] = all_coords_ra[m]+beams['beam_offset_ra'][m_index]/60./np.cos(all_coords_dec[m]/180*np.pi) # all_coordis_dec[m] has not been changed yet.
        all_coords_dec[m] = all_coords_dec[m]+beams['beam_offset_dec'][m_index]/60.
    
    return all_mjd,all_coords_dec,all_coords_ra

# %%
def get_beam_sim(ra_fun,dec_fun,drift_dec_date,crafts_mjd):

    # split name:
    driftdec = drift_dec_date[:14]
    driftdate = drift_dec_date[-8:]

    # compute differet beams' mjd gap based on M01
    # delta_mjd of every beam transfers from a constant value to an array related to mjd.
    delta_mjd = {}
    # delta ra:
    dic = {'M18': -6.180555555555555e-05, 'M17': 0.0001886574074074074, 'M19': -0.0002813888888888889, 'M16': 0.0004391666666666667, 'M07': -3.0925925925925924e-05, 'M08': -0.0005009722222222222, 'M06': 0.00021958333333333335, 'M02': -0.00025050925925925924, 'M15': 0.0004700925925925926, 'M01': -0.0, 'M09': -0.0004700925925925926, 'M05': 0.00025050925925925924, 'M03': -0.00021958333333333335, 'M14': 0.0005009722222222222, 'M04': 3.0925925925925924e-05, 'M10': -0.0004391666666666667, 'M13': 0.0002813888888888889, 'M11': -0.00018877314814814814, 'M12': 6.180555555555555e-05}
    
    all_mjd = {} # include all beams' mjd
    all_coords_ra = {} # include all beams' ra
    all_coords_dec = {} # include all beams' dec

    all_mjd['M01'] = crafts_mjd
    all_coords_dec['M01'] = np.array(dec_fun(all_mjd['M01']))

    # current dec(crude):
    # dec = int(int(driftdec[4:8])/100)+(int(driftdec[4:8])/100-int(int(driftdec[4:8])/100))*100/60 # (deg)
    for i in dic.keys():
        delta_mjd[i] = (dic['M01']-dic[i])/np.cos(all_coords_dec['M01']/180*np.pi)

    # compute every beam's mjd based on delta_mjd
     # if Mx earlier than M01, then mjd will be substracted.
    # else mjd will be added.
    for m in dic.keys():
        all_mjd[m] = all_mjd['M01']-delta_mjd[m]

        # interpolate:
        # now just compute M01's ra & dec in every mjd:
        all_coords_ra[m] = np.array(ra_fun(all_mjd[m]))
        all_coords_dec[m] = np.array(dec_fun(all_mjd[m]))
        
        # now transform into other beams:
        m_index = beams0['beam_name'].index(m)
        
        all_coords_ra[m] = all_coords_ra[m]+beams['beam_offset_ra'][m_index]/60./np.cos(all_coords_dec[m]/180*np.pi) # all_coordis_dec[m] has not been changed yet.
        all_coords_dec[m] = all_coords_dec[m]+beams['beam_offset_dec'][m_index]/60.

        # ra:begin and end:
        all_coords_ra[m] = [all_coords_ra[m][0],all_coords_ra[m][-1]]
        # dec: average
        # avoid adjustment:
        length = len(all_coords_dec[m])
        begin = int(0.1*length)
        end = int(0.9*length)
        all_coords_dec[m] = np.mean(all_coords_dec[m][begin:end])


    return all_coords_dec,all_coords_ra

# %%
def save_dec_ra(drift_dec_date,all_coords_dec,all_coords_ra):
    prim = fits.PrimaryHDU()
    prim.header['Info'] = "beams'location in a Dec"

    dic = {'M18': -6.180555555555555e-05, 'M17': 0.0001886574074074074, 'M19': -0.0002813888888888889, 'M16': 0.0004391666666666667, 'M07': -3.0925925925925924e-05, 'M08': -0.0005009722222222222, 'M06': 0.00021958333333333335, 'M02': -0.00025050925925925924, 'M15': 0.0004700925925925926, 'M01': -0.0, 'M09': -0.0004700925925925926, 'M05': 0.00025050925925925924, 'M03': -0.00021958333333333335, 'M14': 0.0005009722222222222, 'M04': 3.0925925925925924e-05, 'M10': -0.0004391666666666667, 'M13': 0.0002813888888888889, 'M11': -0.00018877314814814814, 'M12': 6.180555555555555e-05}

    f = open('/home/weishirui/Documents/crafts_data/dataset/source_data/coordinate/beam_dec_ra.csv','a')
    for b in dic.keys():
        f.write('{}\t{}\t{}\t{}\t{}\n'.format(drift_dec_date,b,all_coords_dec[b],all_coords_ra[b][0],all_coords_ra[b][1]))
    
    f.close()

# %%
# from data_decra.py
# don't add up all mjds together, but seperate in W01-W44's manner.
def calibration(pos_mjd,pos_ra,pos_dec,drift_dec_date,M01_path_o = '/data15/ZD2020_1_2/Dec-0011_04_05/20200903' ):
    
    # interpolate function:
    ra_fun = interp1d(pos_mjd,pos_ra,kind='linear',fill_value='extrapolate')
    dec_fun = interp1d(pos_mjd,pos_dec,kind='cubic',fill_value='extrapolate') # exist bias

    # split name:
    driftdec = drift_dec_date[:14]
    driftdate = drift_dec_date[-8:]
    output_path = '/home/weishirui/Documents/crafts_data/dataset/source_data/coordinate/{}/'.format(driftdec)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # compute differet beams' mjd gap based on M01
    # delta_mjd of every beam transfers from a constant value to an array related to mjd.
    delta_mjd = {}
    # delta ra:
    dic = {'M18': -6.180555555555555e-05, 'M17': 0.0001886574074074074, 'M19': -0.0002813888888888889, 'M16': 0.0004391666666666667, 'M07': -3.0925925925925924e-05, 'M08': -0.0005009722222222222, 'M06': 0.00021958333333333335, 'M02': -0.00025050925925925924, 'M15': 0.0004700925925925926, 'M01': -0.0, 'M09': -0.0004700925925925926, 'M05': 0.00025050925925925924, 'M03': -0.00021958333333333335, 'M14': 0.0005009722222222222, 'M04': 3.0925925925925924e-05, 'M10': -0.0004391666666666667, 'M13': 0.0002813888888888889, 'M11': -0.00018877314814814814, 'M12': 6.180555555555555e-05}
    
    
    # draft a document to record the relationship:
    f = open('/home/weishirui/Documents/crafts_data/dataset/source_data/coordinate/beam_file_dec_ra.csv','a')
    
    for i in range(1,45):
        # record index:
        index = ''
        if i<10:
            index = '000'+str(i)
        else:
            index = '00'+str(i)
        
        all_mjd = {} # include all beams' mjd
        all_coords_ra = {} # include all beams' ra
        all_coords_dec = {} # include all beams' dec

        # get mjd:
        # M01_path = '/data07/ZD2021_1_2/{}/{}/{}_arcdrift-M01_W_{}.fits'.format(driftdec,driftdate,driftdec,index)
        # since raw flux datasets are not from the same path.
        M01_path = os.path.join(M01_path_o,'{}_arcdrift-M01_W_{}.fits'.format(driftdec,index))
        with fits.open(M01_path) as M01_hdu:
            M01_data = M01_hdu[1].data

            M01_mjd = M01_data['utobs']
            all_mjd['M01'] = np.array(M01_mjd)
        
        all_coords_dec['M01'] = np.array(dec_fun(all_mjd['M01']))
        for i in dic.keys():
            delta_mjd[i] = (dic['M01']-dic[i])/np.cos(all_coords_dec['M01']/180*np.pi) # an array, same shape as pos_dec
        
        # compute every beam's mjd based on delta_mjd
        # if Mx earlier than M01, then mjd will be substracted.
        # else mjd will be added.
        for m in dic.keys():
            all_mjd[m] = all_mjd['M01']-delta_mjd[m]

            # interpolate:
            # now just compute M01's ra & dec in every mjd:
            all_coords_ra[m] = np.array(ra_fun(all_mjd[m]))
            all_coords_dec[m] = np.array(dec_fun(all_mjd[m]))
            
            # now transform into other beams:
            m_index = beams0['beam_name'].index(m)
            
            all_coords_ra[m] = all_coords_ra[m]+beams['beam_offset_ra'][m_index]/60./np.cos(all_coords_dec[m]/180*np.pi) # all_coordis_dec[m] has not been changed yet.
            all_coords_dec[m] = all_coords_dec[m]+beams['beam_offset_dec'][m_index]/60.


            # write to fits:
            bname = [m]*len(all_mjd[m])
            hdu = fits.BinTableHDU.from_columns(
                [fits.Column(name = 'beam', format = '3A', array = bname),
                fits.Column(name = 'mjd', format = 'D', array = all_mjd[m]),
                fits.Column(name = 'ra(deg)', format = 'D', array = all_coords_ra[m]),
                fits.Column(name = 'dec(deg)', format = 'D', array = all_coords_dec[m])])
            
            # dec-date-beam-index(W01-44)
            hdu.writeto(os.path.join(output_path,'{}_{}_{}_{}_coordinate.fits'.format(driftdec,driftdate,m,index)),overwrite=True)

            # write into document:
            f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(drift_dec_date,m,index, np.mean(all_coords_dec[m]),np.min(all_coords_ra[m]),np.max(all_coords_ra[m]))) 
            
            print('{}_{}_{}_{}_coordinate finish ======== √'.format(driftdec,driftdate,m,index) )

        del all_mjd,all_coords_ra,all_coords_dec
            

    f.close()


# %%
# simple calibration: only for mjd_begin & mjd end
def calibration_sim(pos_mjd,pos_ra,pos_dec,drift_dec_date,M01_path_o = '/data15/ZD2020_1_2/Dec-0011_04_05/20200903' ):
    
    # interpolate function:
    ra_fun = interp1d(pos_mjd,pos_ra,kind='linear',fill_value='extrapolate')
    dec_fun = interp1d(pos_mjd,pos_dec,kind='cubic',fill_value='extrapolate') # exist bias

    # split name:
    driftdec = drift_dec_date[:14]
    driftdate = drift_dec_date[-8:]
    
    # compute differet beams' mjd gap based on M01
    # delta_mjd of every beam transfers from a constant value to an array related to mjd.
    delta_mjd = {}
    # delta ra:
    dic = {'M18': -6.180555555555555e-05, 'M17': 0.0001886574074074074, 'M19': -0.0002813888888888889, 'M16': 0.0004391666666666667, 'M07': -3.0925925925925924e-05, 'M08': -0.0005009722222222222, 'M06': 0.00021958333333333335, 'M02': -0.00025050925925925924, 'M15': 0.0004700925925925926, 'M01': -0.0, 'M09': -0.0004700925925925926, 'M05': 0.00025050925925925924, 'M03': -0.00021958333333333335, 'M14': 0.0005009722222222222, 'M04': 3.0925925925925924e-05, 'M10': -0.0004391666666666667, 'M13': 0.0002813888888888889, 'M11': -0.00018877314814814814, 'M12': 6.180555555555555e-05}
    
    
    # draft a document to record the relationship:
    f = open('/home/weishirui/Documents/crafts_data/dataset/source_data/coordinate/beam_file_dec_ra_mjd.csv','a')
    
    for i in range(1,45):
        # record index:
        index = ''
        if i<10:
            index = '000'+str(i)
        else:
            index = '00'+str(i)
        
        all_mjd = {} # include all beams' mjd
        all_coords_ra = {} # include all beams' ra
        all_coords_dec = {} # include all beams' dec

        # get mjd:
        # M01_path = '/data07/ZD2021_1_2/{}/{}/{}_arcdrift-M01_W_{}.fits'.format(driftdec,driftdate,driftdec,index)
        # since raw flux datasets are not from the same path.
        M01_path = os.path.join(M01_path_o,'{}_arcdrift-M01_W_{}.fits'.format(driftdec,index))
        with fits.open(M01_path) as M01_hdu:
            M01_data = M01_hdu[1].data

            M01_mjd = M01_data['utobs']
            all_mjd['M01'] = np.array(M01_mjd)
        
        all_coords_dec['M01'] = np.array(dec_fun(all_mjd['M01']))
        for i in dic.keys():
            delta_mjd[i] = (dic['M01']-dic[i])/np.cos(all_coords_dec['M01']/180*np.pi) # an array, same shape as pos_dec
        
        # compute every beam's mjd based on delta_mjd
        # if Mx earlier than M01, then mjd will be substracted.
        # else mjd will be added.
        for m in dic.keys():
            all_mjd[m] = all_mjd['M01']-delta_mjd[m]

            # interpolate:
            # now just compute M01's ra & dec in every mjd:
            all_coords_ra[m] = np.array(ra_fun(all_mjd[m]))
            all_coords_dec[m] = np.array(dec_fun(all_mjd[m]))
            
            # now transform into other beams:
            m_index = beams0['beam_name'].index(m)
            
            all_coords_ra[m] = all_coords_ra[m]+beams['beam_offset_ra'][m_index]/60./np.cos(all_coords_dec[m]/180*np.pi) # all_coordis_dec[m] has not been changed yet.
            all_coords_dec[m] = all_coords_dec[m]+beams['beam_offset_dec'][m_index]/60.

            # write into document:
            f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(drift_dec_date,m,index, np.mean(all_coords_dec[m]),np.min(all_coords_ra[m]),np.max(all_coords_ra[m]),np.min(all_mjd[m]),np.max(all_mjd[m]))) 
            
            print('{}_{}_{}_{}_coordinate finish ======== √'.format(driftdec,driftdate,m,index) )

        del all_mjd,all_coords_ra,all_coords_dec
            

    f.close()


# %%
# def main():
#     # filelist = get_filelist()
#     filelist = ['Dec+4939_04_05_20211215']
#     raw_data = ['/data07/ZD2021_1_2/Dec+4939_04_05/20211215']
#     for f in range(len(filelist)):
#         pos_mjd,pos_ra,pos_dec = get_position(filelist[f])
#         ra_fun,dec_fun,crafts_mjd = M01_baseline_sim(pos_mjd,pos_ra,pos_dec,filelist[f],raw_data[f])
#         all_coords_dec,all_coords_ra = get_beam_sim(ra_fun,dec_fun,filelist[f],crafts_mjd)
#         save_dec_ra(filelist[f],all_coords_dec,all_coords_ra)

#         print("{} finished ======√".format(filelist[f]))

def main():
    filelist = ['Dec+4037_10_05_20211001']
    raw_data = ['/data07/ZD2021_1_2/Dec+4037_10_05/20211001']
    for f in range(len(filelist)):
        pos_mjd,pos_ra,pos_dec = get_position(filelist[f])
        # calibration(pos_mjd,pos_ra,pos_dec,filelist[f],raw_data[f])
        calibration_sim(pos_mjd,pos_ra,pos_dec,filelist[f],raw_data[f])

# %% 
def draw():
    df = pd.read_csv('/home/weishirui/Documents/crafts_data/dataset/source_data/coordinate/beam_dec_ra.csv',sep='\t',header=None)
    df.columns = ['drift_dec_date','beam','dec','ra_begin','ra_end']

    colorlist=["mediumspringgreen","aquamarine","turquoise","lightseagreen","mediumturquoise","azure","lightcyan","paleturquoise","teal","darkcyan","c","cyan","aqua","darkturquoise","cadetblue","powderblue","lightblue","deepskyblue","skyblue","lightskyblue","steelblue","aliceblue","dodgerblue"]

    beam_name = ['M01','M02','M03','M04','M05','M06','M07',
                        'M08','M09','M10','M11','M12','M13','M14',
                        'M15','M16','M17','M18','M19']
    # draw
    fig = plt.figure(figsize=(100,100))
    group = df.groupby(by='beam')
    for b,g in group:
        for _,row in g.iterrows():
            x = np.arange(row['ra_begin'],row['ra_end'])
            plt.plot(x,[row['dec'] for i in range(len(x))],c=colorlist[beam_name.index(b)])

    # plt.show()
    plt.savefig('/home/weishirui/Documents/crafts_data/dataset/source_data/coordinate/beam_info.jpg')

main()
# draw()