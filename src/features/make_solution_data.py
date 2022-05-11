from tqdm import tqdm
import pandas as pd
import numpy as np

import sys
sys.path.insert(0,'./notebooks')
import magnav
# Be sure to be at MagNav root directory

for n in tqdm(range(2,6)):
    
    df = pd.read_hdf(r'./data/interim/Flt_data.h5', key=f'Flt100{n}')
    
    # Get square pattern data
    flight_number = 2
    df_pattern    = pd.read_hdf(r'./data/interim/Flt_data.h5',key=f'Flt100{flight_number}')
    mask_1 = (df_pattern.LINE == 1002.02)
    tl_sq = df_pattern[mask_1] # Square Tolles-Lawson pattern
    
    # Apply Tolles Lawson correction
    fs      = 10.0
    lowcut  = 0.1
    highcut = 0.9
    filt    = ['Butterworth',4]

    calcIGRF = df['DCMAG1']-df['IGRFMAG1'] 
    
    A = magnav.create_TL_A(df['FLUXB_X'],df['FLUXB_Y'],df['FLUXB_Z'])
    
    TL_coef_3 = magnav.create_TL_coef(tl_sq['FLUXB_X'],tl_sq['FLUXB_Y'],tl_sq['FLUXB_Z'],tl_sq['UNCOMPMAG3'],
                                  lowcut=lowcut,highcut=highcut,fs=fs,filter_params=filt)
    TL_coef_4 = magnav.create_TL_coef(tl_sq['FLUXB_X'],tl_sq['FLUXB_Y'],tl_sq['FLUXB_Z'],tl_sq['UNCOMPMAG4'],
                                  lowcut=lowcut,highcut=highcut,fs=fs,filter_params=filt)
    TL_coef_5 = magnav.create_TL_coef(tl_sq['FLUXB_X'],tl_sq['FLUXB_Y'],tl_sq['FLUXB_Z'],tl_sq['UNCOMPMAG5'],
                                  lowcut=lowcut,highcut=highcut,fs=fs,filter_params=filt)
    
    df['TL_comp_mag3_sq'] = np.reshape(df['UNCOMPMAG3'].tolist(),(-1,1))-np.dot(A,TL_coef_3)+np.mean(np.dot(A,TL_coef_3))
    df['TL_comp_mag4_sq'] = np.reshape(df['UNCOMPMAG4'].tolist(),(-1,1))-np.dot(A,TL_coef_4)+np.mean(np.dot(A,TL_coef_4))
    df['TL_comp_mag5_sq'] = np.reshape(df['UNCOMPMAG5'].tolist(),(-1,1))-np.dot(A,TL_coef_5)+np.mean(np.dot(A,TL_coef_5))
    
    df['TL_comp_mag3_sq'] = df['TL_comp_mag3_sq']-df['DIURNAL']-calcIGRF
    df['TL_comp_mag4_sq'] = df['TL_comp_mag4_sq']-df['DIURNAL']-calcIGRF
    df['TL_comp_mag5_sq'] = df['TL_comp_mag5_sq']-df['DIURNAL']-calcIGRF
    
    # Get selected features
    df_sol = df[['FLUXB_TOT','FLUXB_X','FLUXC_TOT','FLUXC_Y','FLUXD_Y','FLUXD_Z','TL_comp_mag3_sq',
                 'UNCOMPMAG3','TL_comp_mag4_sq','TL_comp_mag5_sq','V_CABT','LINE','IGRFMAG1']]
    
    # export to HDF5
    df_sol.to_hdf(f'./data/interim/Sol_dataset.h5',key=f'Flt100{n}')
    # export to csv
    df_sol.to_csv(f'./data/interim/Sol_dataset_csv/Flt100{n}.csv')