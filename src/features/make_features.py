from tqdm import tqdm
import pandas as pd
import numpy as np
import datetime

import sys
sys.path.insert(0,'notebooks')
import magnav
import ppigrf

# Be sure to be at MagNav root directory

def apply_corrections(df,diurnal=True,igrf=True):
    
    mag_measurements = np.array(['TL_comp_mag5_cl','TL_comp_mag3_cl'])
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


print('Python script to transform raw data to corrected features for model training.(If you have any doubt, please refer to the notebook n°1)\nChoice of corrections : ')
diurnal_choice = int(input("Apply Diurnal correction ? (1 for True, 0 for False)\n"))
igrf_choice = int(input("Apply IGRF correction ? (1 for True, 0 for False)\n"))

    
for n in tqdm(range(2,6)):
    
    # Get raw data
    df = pd.read_hdf(r'data/interim/Flt_data.h5', key=f'Flt100{n}')
    
    
    #---Tolles-Lawson compensation---#
    
    # Get cloverleaf pattern data
    flight_number = 2
    df_pattern    = pd.read_hdf(r'data/interim/Flt_data.h5',key=f'Flt100{flight_number}')
    mask  = (df_pattern.LINE == 1002.20)
    tl_cl = df_pattern[mask] # Square Tolles-Lawson pattern
    
    # filter parameters
    fs      = 10.0
    lowcut  = 0.1
    highcut = 0.9
    filt    = ['Butterworth',4]
    
    # A matrix of Tolles-Lawson
    A = magnav.create_TL_A(df['FLUXB_X'],df['FLUXB_Y'],df['FLUXB_Z'])
    
    # Tolles Lawson coefficients computation
    TL_coef_3 = magnav.create_TL_coef(tl_cl['FLUXB_X'],tl_cl['FLUXB_Y'],tl_cl['FLUXB_Z'],tl_cl['UNCOMPMAG3'],
                                  lowcut=lowcut,highcut=highcut,fs=fs,filter_params=filt)
    TL_coef_5 = magnav.create_TL_coef(tl_cl['FLUXB_X'],tl_cl['FLUXB_Y'],tl_cl['FLUXB_Z'],tl_cl['UNCOMPMAG5'],
                                  lowcut=lowcut,highcut=highcut,fs=fs,filter_params=filt)
    
    # Magnetometers correction
    df['TL_comp_mag3_cl'] = np.reshape(df['UNCOMPMAG3'].tolist(),(-1,1))-np.dot(A,TL_coef_3)+np.mean(np.dot(A,TL_coef_3))
    df['TL_comp_mag5_cl'] = np.reshape(df['UNCOMPMAG5'].tolist(),(-1,1))-np.dot(A,TL_coef_5)+np.mean(np.dot(A,TL_coef_5))
    
    # IGRF and diurnal correction
    COR_df = apply_corrections(df,diurnal=diurnal_choice,igrf=igrf_choice)
    
    # Get selected features
    features = ['TL_comp_mag3_cl','TL_comp_mag5_cl','V_BAT1','V_BAT2','TOPO',
                'INS_VEL_N','INS_VEL_W','INS_VEL_V','BARO','CUR_IHTR','PITCH','ROLL','AZIMUTH','LINE','IGRFMAG1']
    
    # export to HDF5
    COR_df[features].to_hdf(f'data/interim/dataset.h5',key=f'Flt100{n}')
    # export to csv
    COR_df[features].to_csv(f'data/interim/dataset_csv/Flt100{n}.csv')
print('Done')