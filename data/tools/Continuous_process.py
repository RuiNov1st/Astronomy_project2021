# %%
"""
author:weishirui
code version:2023-01-10
description: 
Compute every Continuous Sources' ra and record their names, dec
algorithm: 
input:
- /home/lupengjun/findsource/fine_search/beam_data_maxr_0_1.5/ : ContinuousSpe
- /home/weishirui/Documents/crafts_data/dataset/source_data/coordinate/beam_file_dec_ra_mjd.csv: mjd info
"""

# %%
import pandas as pd
import numpy as np
from astropy.io import fits
import os
from scipy.interpolate import interp1d



def Continuous_record(drift_dec_date = 'Dec+4037_10_05_20211001'):
    driftdec = drift_dec_date[:14]
    driftdate = drift_dec_date[-8:]

    Continuous_Spec_path = '/home/lupengjun/findsource/fine_search/beam_data_maxr_0_1.5/{}'.format(drift_dec_date)
    Continuous_Spec_output_path = '/home/weishirui/Documents/crafts_data/dataset/source_data/Continuous'
    coordinate_path = '/home/weishirui/Documents/crafts_data/dataset/source_data/coordinate/'
    beam_mjd_info_path = '/home/weishirui/Documents/crafts_data/dataset/source_data/coordinate/beam_file_dec_ra_mjd.csv'
    
    mjd_df = pd.read_csv(beam_mjd_info_path,header = None,sep = '\t')
    mjd_df.columns = ['drift_dec_date','beam','file','dec','ra_begin','ra_end','mjd_begin','mjd_end']
    mjd_df = mjd_df[mjd_df['drift_dec_date']==drift_dec_date] # select drift_dec_date

    beams = ['M18','M17','M19','M16','M07','M08','M06','M02','M15','M01','M09','M05','M03','M14','M04','M10','M13','M11','M12']
    source_mjd = []
    source_ra = []
    source_dec = []
    source_name = []
    source_beam = [] # just for evaluate
    source_file = [] # just for evaluate
    for b in beams:
        con_hdu = fits.open(os.path.join(Continuous_Spec_path,'ContinuousSpec_{}_{}_ver4.0.fits'.format(driftdec,b)))  
        mjd_df_b = mjd_df[mjd_df['beam'] == b] # select beam

        for c_i in range(1,len(con_hdu),3):
            # get mjd:
            conSpe_mjd = con_hdu[c_i].header['MJD']
            mjd_df_b_m = mjd_df_b[mjd_df_b['mjd_begin']<=conSpe_mjd]
            mjd_df_b_m = mjd_df_b_m[mjd_df_b_m['mjd_end']>=conSpe_mjd]

            # select file:
            if len(mjd_df_b_m) != 1 or  mjd_df_b_m['file'].tolist()[0] in ['0001','0044']:
                continue

            # get ra:
            file_idx = mjd_df_b_m['file'].tolist()[0]
            if file_idx<10:
                file_idx = '000'+str(file_idx)
            else:
                file_idx = '00'+str(file_idx)
            with fits.open(os.path.join(coordinate_path,'{}/{}_{}_{}_coordinate.fits'.format(driftdec,drift_dec_date,b,file_idx)))as coordinate_hdu:
                mjd_data = np.array(coordinate_hdu[1].data['mjd'])
                ra_data = np.array(coordinate_hdu[1].data['ra(deg)'])

            # interpolate:
            ra_fun = interp1d(mjd_data,ra_data,kind='linear',fill_value='extrapolate')
            conSpe_ra = ra_fun(conSpe_mjd)

            # get dec:
            conSpe_dec = con_hdu[c_i].header['DEC']
            # get name:
            conSpe_name = con_hdu[c_i].header['SOURCE']+'_'+con_hdu[c_i].header['NUM']

            source_mjd.append(conSpe_mjd)
            source_ra.append(conSpe_ra)
            source_dec.append(conSpe_dec)
            source_name.append(conSpe_name)
            source_beam.append(b)
            source_file.append(file_idx)

            print('{} {} finish==========√'.format(b,conSpe_name))
            
    
    # write to fits:
    hdu = fits.BinTableHDU.from_columns(
            [fits.Column(name = 'mjd', format = 'D', array = np.array(source_mjd)),
            fits.Column(name = 'ra(deg)', format = 'D', array = np.array(source_ra)),
            fits.Column(name = 'dec(deg)', format = 'D', array = np.array(source_dec)),
            fits.Column(name = 'name', format = '15A', array = np.array(source_name)),
            fits.Column(name = 'beam', format = '3A', array = np.array(source_beam)),
            fits.Column(name = 'file', format = '4A', array = np.array(source_file)),
            ])
        
    # dec-date-beam-index(W01-44)
    hdu.writeto(os.path.join(Continuous_Spec_output_path,'{}_ContinuousSpec.fits'.format(drift_dec_date)),overwrite=True)

# %%
if __name__ == '__main__':
    Continuous_record('Dec+4037_10_05_20211001')
