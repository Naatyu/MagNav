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
import ppigrf

def apply_IGRF_DIURNAL_corrections(df,diurnal=True,igrf=True):
    
    mag_measurements = np.array(['TL_comp_mag2_cl','TL_comp_mag3_cl','TL_comp_mag4_cl','TL_comp_mag5_cl'])
    COR_df = df
    
    # Diurnal cor
    if diurnal == True:
        COR_df[mag_measurements] = COR_df[mag_measurements]-np.reshape(COR_df['DIURNAL'].values,[-1,1])
    
    # IGRF cor
    lat  = COR_df['LAT']
    lon  = COR_df['LONG']
    h    = COR_df['BARO']*1e-3 # Kilometers above WGS84 ellipsoid
    date = datetime.datetime(2020, 6, 29) # Date on which the flights were made
    Be, Bn, Bu = ppigrf.igrf(lon,lat,h,date)
    
    if igrf == True:
        COR_df[mag_measurements] = COR_df[mag_measurements]-np.reshape(np.sqrt(Be**2+Bn**2+Bu**2)[0],[-1,1])

    return COR_df

def add_A_terms(flux_X,flux_Y,flux_Z,df,letter):
    
    A = magnav.create_TL_A(flux_X,flux_Y,flux_Z)
    
    # Permanent terms
    df[f'TL_flux{letter}_X'] = A[:,0]
    df[f'TL_flux{letter}_Y'] = A[:,1]
    df[f'TL_flux{letter}_Z'] = A[:,2]

    # Induced terms
    df[f'TL_flux{letter}_XX'] = A[:,3]
    df[f'TL_flux{letter}_XY'] = A[:,4]
    df[f'TL_flux{letter}_XZ'] = A[:,5]
    df[f'TL_flux{letter}_YY'] = A[:,6]
    df[f'TL_flux{letter}_YZ'] = A[:,7]
    df[f'TL_flux{letter}_ZZ'] = A[:,8]

    # Eddy terms
    df[f'TL_flux{letter}_XXdot'] = A[:,9]
    df[f'TL_flux{letter}_XYdot'] = A[:,10]
    df[f'TL_flux{letter}_XZdot'] = A[:,11]
    df[f'TL_flux{letter}_YXdot'] = A[:,12]
    df[f'TL_flux{letter}_YYdot'] = A[:,13]
    df[f'TL_flux{letter}_YZdot'] = A[:,14]
    df[f'TL_flux{letter}_ZXdot'] = A[:,15]
    df[f'TL_flux{letter}_ZYdot'] = A[:,16]
    df[f'TL_flux{letter}_ZZdot'] = A[:,17]
    
    return df

def add_BodytoNav_terms(Pitch,Yaw,Roll,df):
    # Matrix transition from body to Nav
    c_p = np.cos(Pitch) #p = pitch, y = yaw, r = roll, c = cos, s = sin
    s_p = np.sin(Pitch)
    c_y = np.cos(Yaw)
    s_y = np.sin(Yaw)
    c_r = np.cos(Roll)
    s_r = np.sin(Roll)

    df['TM1'] = c_p*c_y
    df['TM2'] = s_r*s_p*c_y-c_r*s_y
    df['TM3'] = c_r*s_p*c_y+s_r*s_y
    df['TM4'] = c_p*s_y
    df['TM5'] = s_r*s_p*s_y+c_r*c_y
    df['TM6'] = c_r*s_p*s_y-s_r*c_y
    df['TM7'] = -s_p
    df['TM8'] = s_r*c_p
    df['TM9'] = c_r*c_p
    
    return df

if __name__ == "__main__":

    print('Python script to transform raw data to corrected + new features.(If you have any doubt, please refer to the notebook n°1)\nChoice of  corrections : ')
    diurnal_choice = int(input("Apply Diurnal correction ? (1 for True, 0 for False)\n"))
    igrf_choice = int(input("Apply IGRF correction ? (1 for True, 0 for False)\n"))

    if os.path.isfile('data/processed/Full_Dataset.h5'):
        os.remove('data/processed/Full_Dataset.h5')
        
    folder_path = f'data/processed/Full_Dataset_csv'
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
        TL_coef_2 = magnav.create_TL_coef(tl_cl['FLUXB_X'],tl_cl['FLUXB_Y'],tl_cl['FLUXB_Z'],tl_cl['UNCOMPMAG2'],
                                      lowcut=lowcut,highcut=highcut,fs=fs,filter_params=filt)
        TL_coef_3 = magnav.create_TL_coef(tl_cl['FLUXB_X'],tl_cl['FLUXB_Y'],tl_cl['FLUXB_Z'],tl_cl['UNCOMPMAG3'],
                                      lowcut=lowcut,highcut=highcut,fs=fs,filter_params=filt)
        TL_coef_4 = magnav.create_TL_coef(tl_cl['FLUXB_X'],tl_cl['FLUXB_Y'],tl_cl['FLUXB_Z'],tl_cl['UNCOMPMAG4'],
                                      lowcut=lowcut,highcut=highcut,fs=fs,filter_params=filt)
        TL_coef_5 = magnav.create_TL_coef(tl_cl['FLUXB_X'],tl_cl['FLUXB_Y'],tl_cl['FLUXB_Z'],tl_cl['UNCOMPMAG5'],
                                      lowcut=lowcut,highcut=highcut,fs=fs,filter_params=filt)

        # Magnetometers correction
        df['TL_comp_mag2_cl'] = np.reshape(df['UNCOMPMAG2'].tolist(),(-1,1))-np.dot(A,TL_coef_2)+np.mean(np.dot(A,TL_coef_2))
        df['TL_comp_mag3_cl'] = np.reshape(df['UNCOMPMAG3'].tolist(),(-1,1))-np.dot(A,TL_coef_3)+np.mean(np.dot(A,TL_coef_3))
        df['TL_comp_mag4_cl'] = np.reshape(df['UNCOMPMAG4'].tolist(),(-1,1))-np.dot(A,TL_coef_4)+np.mean(np.dot(A,TL_coef_4))
        df['TL_comp_mag5_cl'] = np.reshape(df['UNCOMPMAG5'].tolist(),(-1,1))-np.dot(A,TL_coef_5)+np.mean(np.dot(A,TL_coef_5))

        # IGRF and diurnal correction
        COR_df = apply_IGRF_DIURNAL_corrections(df,diurnal=diurnal_choice,igrf=igrf_choice)

        # Add A TL terms
        add_A_terms(df['FLUXB_X'],df['FLUXB_Y'],df['FLUXB_Z'],COR_df,'B')
        add_A_terms(df['FLUXC_X'],df['FLUXC_Y'],df['FLUXC_Z'],COR_df,'C')
        add_A_terms(df['FLUXD_X'],df['FLUXD_Y'],df['FLUXD_Z'],COR_df,'D')
        
        # Add Cosines Matrix terms
        add_BodytoNav_terms(df['PITCH'],df['AZIMUTH'],df['ROLL'],COR_df)

        # export to HDF5
        COR_df.to_hdf(f'data/processed/Full_Dataset.h5',key=f'Flt100{n}')
        # export to csv
        COR_df.to_csv(f'data/processed/Full_Dataset_csv/Flt100{n}.csv')
        
    print('Done')