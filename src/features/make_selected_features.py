#!/usr/bin/env python3

# Be sure to be at MagNav root directory

from tqdm import tqdm
import pandas as pd
import numpy as np
import datetime
import os

import sys
sys.path.insert(0,'src')
import magnav

def apply_IGRF_DIURNAL_corrections(df,diurnal=True,igrf=True):

    mag_measurements = np.array(['TL_comp_mag4_cl','TL_comp_mag5_cl'])
    COR_df = df
    
    # Diurnal cor
    if diurnal == True:
        COR_df[mag_measurements] = COR_df[mag_measurements]-np.reshape(COR_df['DIURNAL'].values,[-1,1])

    # IGRF cor
    lat  = COR_df['LAT']
    lon  = COR_df['LONG']
    h    = COR_df['BARO']*1e-3 # Kilometers above WGS84 ellipsoid
    date = datetime.datetime(2020, 6, 29) # Date on which the flights were made
    Be, Bn, Bu = magnav.igrf(lon,lat,h,date)

    if igrf == True:
        COR_df[mag_measurements] = COR_df[mag_measurements]-np.reshape(np.sqrt(Be**2+Bn**2+Bu**2)[0],[-1,1])

    return COR_df


if __name__ == "__main__":

    print('Python script to transform raw data to slected features.(If you have any doubt, please refer to the notebook n°1)\nChoice of  corrections : ')
    diurnal_choice = int(input("Apply Diurnal correction ? (1 for True, 0 for False)\n"))
    igrf_choice = int(input("Apply IGRF correction ? (1 for True, 0 for False)\n"))
    
    name = 'DownSelected_Dataset'
    
    if os.path.isfile(f'data/processed/{name}.h5'):
        os.remove(f'data/processed/{name}.h5')
        
    folder_path = f'data/processed/{name}_csv'
    if not(os.path.isdir(folder_path)):
        os.mkdir(folder_path)
        
    for n in tqdm(range(2,8)):

        # Get raw data
        df = pd.read_hdf(r'data/interim/Flt_data.h5', key=f'Flt100{n}')


        #---Tolles-Lawson compensation---#

        # Get cloverleaf pattern data
        flight_number = 2
        df_pattern    = pd.read_hdf(r'data/interim/Flt_data.h5',key=f'Flt100{flight_number}')
        mask  = (df_pattern.LINE == 1002.20)
        tl_cl = df_pattern[mask] # Cloverleaf Tolles-Lawson pattern

        # filter parameters
        fs      = 10.0
        lowcut  = 0.1
        highcut = 0.9
        filt    = ['Butterworth',4]

        # A matrix of Tolles-Lawson
        A = magnav.create_TL_A(df['FLUXB_X'],df['FLUXB_Y'],df['FLUXB_Z'])

        # Tolles Lawson coefficients computation
        TL_coef_4 = magnav.create_TL_coef(tl_cl['FLUXB_X'],tl_cl['FLUXB_Y'],tl_cl['FLUXB_Z'],tl_cl['UNCOMPMAG4'],
                                      lowcut=lowcut,highcut=highcut,fs=fs,filter_params=filt, ridge=0.025)
        TL_coef_5 = magnav.create_TL_coef(tl_cl['FLUXB_X'],tl_cl['FLUXB_Y'],tl_cl['FLUXB_Z'],tl_cl['UNCOMPMAG5'],
                                      lowcut=lowcut,highcut=highcut,fs=fs,filter_params=filt, ridge=0.025)

        # Magnetometers correction
        df['TL_comp_mag4_cl'] = np.reshape(df['UNCOMPMAG4'].tolist(),(-1,1))-np.dot(A,TL_coef_4)+np.mean(np.dot(A,TL_coef_4))
        df['TL_comp_mag5_cl'] = np.reshape(df['UNCOMPMAG5'].tolist(),(-1,1))-np.dot(A,TL_coef_5)+np.mean(np.dot(A,TL_coef_5))

        # IGRF and diurnal correction
        COR_df = apply_IGRF_DIURNAL_corrections(df,diurnal=diurnal_choice,igrf=igrf_choice)
        
        # Selected features
        features = ['TL_comp_mag5_cl','TL_comp_mag4_cl','V_BAT1','V_BAT2','INS_VEL_N','INS_VEL_W','INS_VEL_V','BARO','CUR_IHTR',
                    'CUR_ACLo','CUR_ACHi','CUR_TANK','CUR_FLAP','V_BLOCK','PITCH','ROLL','AZIMUTH','LINE','IGRFMAG1','COMPMAG1']

        # export to HDF5
        COR_df[features].to_hdf(f'data/processed/{name}.h5',key=f'Flt100{n}')
        # export to csv
        COR_df[features].to_csv(f'data/processed/{name}_csv/Flt100{n}.csv')
        
    print('Done')