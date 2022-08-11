import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy import signal
import time
import random
import copy


#-------------------------#
#----General Functions----#
#-------------------------#


def sampling_frequency(df):
    """
    Calculates the sampling frequency of a pandas DataFrame indexed in time.

    Arguments:
    - `df` : pandas DataFrame

    Returns:
    - `samp_freq` : sampling frequency of the dataset
    """
    
    duration  = df.index[-1]-df.index[0]
    samp_freq = np.shape(df)[0]/duration
    
    return samp_freq

def to_hms(seconds):
    """
    Transforms seconds to hours:minutes:seconds format.

    Arguments:
    - `seconds` : number in seconds that will be converted

    Returns:
    - `hms_format` : seconds converted to h:m:s format
    """
    
    hms_format = time.strftime("%Hh:%Mm:%Ss",time.gmtime(seconds))
    
    return hms_format

def get_random_color():
    """
    Return a random color in HEX format.

    Arguments:
    - None

    Returns:
    - `rand_col` : random color in HEX format 
    """
    
    rand_col = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    
    return rand_col

def rmse(y_pred, y_true, subtract_mean=True):
    """
    rmse(y_true, y_pred, subtract_mean=True)

    Calculate root-mean-square error.

    Arguments:
    - `y_true` : ground truth target values
    - `y_pred` : estimated target values
    - `substract_mean` : (optional) center values to 0 mean

    Returns:
    - `err` : root-mean-squared error
    """
    
    if subtract_mean:
        err = np.sqrt(mean_squared_error(y_true - y_true.mean(), 
                                         y_pred - y_pred.mean()))
    else:
        err = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return err


#---------------------------------#
#----Digital filters functions----#
#---------------------------------#


def create_butter_filter(lowcut, highcut, fs, order=4):
    """
    Design an Nth-order digital Butterworth filter.

    Arguments:
    - `lowcut, highcut` : critical frequencies of the filter [Hz]
    - `fs`              : sampling frequency [Hz]
    - `order`           : (optional) order of the Butterworth filter

    Returns:
    - `sos` : second-order representation of the IIR filter
    """
    
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    
    # band-pass
    if ((lowcut > 0) & (lowcut < highcut)) & (highcut < fs/2):
        sos = signal.butter(order,[low,high],analog=False,btype='band',output='sos')
    
    # low-pass
    elif ((lowcut <= 0) | (lowcut >= fs/2)) & (highcut < fs/2):
        sos = signal.butter(order,highcut,analog=False,btype='lowpass',output='sos')
    
    # high-pass
    elif (lowcut > 0) & ((highcut <=0 ) | (highcut >= fs/2)):
        sos = signal.butter(order,lowcut,analog=False,btype='highpass',output='sos')
    
    return sos

def apply_butter_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply an Nth-order digital Butterworth filter.

    Arguments:
    - `data`            : the data to which the filter should be applied
    - `lowcut, highcut` : critical frequencies of the filter [Hz]
    - `fs`              : sampling frequency [Hz]
    - `order`           : (optional) order of the Butterworth filter

    Returns:
    - `y` : filtered data
    """
    
    sos = create_butter_filter(lowcut,highcut,fs,order=order)
    y   = signal.sosfiltfilt(sos,data)
    
    return y

def create_firwin_filter(lowcut, highcut, fs, ntaps=255, window='hamming'):
    """
    Design an FIR filter using the window method.

    Arguments:
    - `lowcut, highcut` : critical frequencies of the filter [Hz]
    - `fs`              : sampling frequency [Hz]
    - `ntaps`           : (optional) length of the filter (number of coefficinets), must be odd if it is a pass-band
    - `window`          : (optional) Desired window to use (boxcar, triang, blackman, hamming, hann, ...) see scipy.signal.get_window

    Returns:
    - `taps` : coefficients of length numtaps FIR filter
    """
    
    nyq = 0.5*fs
    
    # band-pass
    if ((lowcut > 0) & (lowcut < highcut)) & (highcut < fs/2):
        taps = signal.firwin(ntaps,[lowcut,highcut],nyq=nyq,pass_zero='bandpass',window=window,scale=False)
    
    # low-pass
    elif ((lowcut <= 0) | (lowcut >= fs/2)) & (highcut < fs/2):
        taps = signal.firwin(ntaps,highcut,nyq=nyq,pass_zero='lowpass',window=window,scale=False)
    
    # high-pass
    elif (lowcut > 0) & ((highcut <=0 ) | (highcut >= fs/2)):
        taps = signal.firwin(ntaps,lowcut,nyq=nyq,pass_zero='highpass',window=window,scale=False)
    
    return taps

def apply_firwin_filter(data, lowcut, highcut, fs, ntaps=255, window='hamming'):
    """
    Apply an FIR filter using the window method.

    Arguments:
    - `data` : the data to which the filter should be applied
    - `lowcut, highcut` : critical frequencies of the filter [Hz]
    - `fs` : sampling frequency [Hz]
    - `ntaps` : (optional) length of the filter (number of coefficinets), must be odd if it is a pass-band
    - `window` : (optional) Desired window to use (boxcar, triang, blackman, hamming, hann, ...) see scipy.signal.get_window

    Returns:
    - `y` : filtered data
    """
    
    taps = create_firwin_filter(lowcut,highcut,fs,ntaps,window=window)
    y    = signal.filtfilt(taps,1,data)
    
    return y


#-------------------------------#
#----Tolles-Lawson Functions----#
#-------------------------------#


def create_TL_A(Bx, By, Bz, add_induced=True, add_eddy=True, Bt_scale=50000):
    """
    Create Tolles-Lawson A matrix using vector magnetometer measurements.

    Arguments:
    - `Bx, By, Bz` : vector magnetometer measurements
    - `add_induced, add_eddy` : (optional) add induced and/or eddy terms to Tolles-Lawson A matrix.
    - `Bt_scale` : (optional) scaling factor for induced and eddy current terms

    Returns:
    - `A` : Tolles-Lawson A matrix
    """

    Bt = np.sqrt(Bx**2 + By**2 + Bz**2)
    s  = Bt / Bt_scale # scale
    cosX, cosY, cosZ = Bx/Bt, By/Bt, Bz/Bt
    cosX_dot = np.gradient(cosX)
    cosY_dot = np.gradient(cosY)
    cosZ_dot = np.gradient(cosZ)        

    # (3) permanent moment
    A = np.column_stack((cosX, cosY, cosZ))

    # (6) induced moment
    if add_induced:
        A_ind = np.column_stack((s*cosX*cosX,
                                 s*cosX*cosY,
                                 s*cosX*cosZ,
                                 s*cosY*cosY,
                                 s*cosY*cosZ,
                                 s*cosZ*cosZ))
        A = np.column_stack((A, A_ind))

    # (9) eddy current
    if add_eddy:        
        A_edd = np.column_stack((s*cosX*cosX_dot,
                                 s*cosX*cosY_dot,
                                 s*cosX*cosZ_dot,
                                 s*cosY*cosX_dot,
                                 s*cosY*cosY_dot,
                                 s*cosY*cosZ_dot,
                                 s*cosZ*cosX_dot,
                                 s*cosZ*cosY_dot,
                                 s*cosZ*cosZ_dot))
        A = np.column_stack((A, A_edd))

    return A

def create_TL_coef(Bx, By, Bz, meas, add_induced=True, add_eddy=True, lowcut=0.1, highcut=0.9, fs=10.0, filter_params=['Butterworth',4], ridge=None, Bt_scale=50000):
    """
    Create Tolles-Lawson coefficients using vector and scalar magnetometer 
    measurements and a bandpass, low-pass or high-pass filter.

    Arguments:
    - `Bx, By, Bz`: vector magnetometer measurements
    - `meas`: scalar magnetometer measurements
    - `add_induced, add_eddy` : (optional) Add induced and/or eddy terms to Tolles-Lawson A matrix.
    - `lowcut, highcut` : (optional) critical frequencies of the filter [Hz]
    - `fs`: (optional) sampling frequency [Hz]
    - `filter_params` : (optional) ['Butterworth',4] ['firwin',255,'hamming'] 'None'
    - `ridge` : (optional) Ridge parameter for ridge regression. Disabled by default.
    - `Bt_scale` : (optional) scaling factor for induced and eddy current terms

    Returns:
    - `TL_coef`: Tolles-Lawson coefficients
    """
    
    # apply filter to scalar measurements
    if filter_params[0] == 'Butterworth':
        meas_f = apply_butter_filter(meas,lowcut,highcut,fs,order=filter_params[1])
    elif filter_params[0] == 'firwin':
        meas_f = apply_firwin_filter(meas,lowcut,highcut,fs,ntaps=filter_params[1],window=filter_params[2])
    elif filter_params == 'None':
        meas_f = meas
    
    if meas_f.ndim != 2:
        meas_f = np.reshape(meas_f.tolist(),(-1,1))
    
    # create Tolles-Lawson A matrix
    A = create_TL_A(Bx,By,Bz,add_induced=add_induced,add_eddy=add_eddy,Bt_scale=Bt_scale)
    
    # filter each column of A
    A_f = copy.deepcopy(A)
    for i in range(np.shape(A)[1]):
        if filter_params[0] == 'Butterworth':
            A_f[:,i] = apply_butter_filter(A[:,i],lowcut,highcut,fs,order=filter_params[1])
        elif filter_params[0] == 'firwin':
            A_f[:,i] = apply_firwin_filter(A[:,i],lowcut,highcut,fs,ntaps=filter_params[1],window=filter_params[2])
        elif filter_params == 'None':
            pass
    
    # trim first/last 20 elements due to filter artifacts
    trim = 20
    A_f_t = A_f[trim:np.shape(A_f)[0]-trim,:]
    meas_f_t = meas_f[trim:np.shape(meas_f)[0]-trim,:]
    
    # get Tolles-Lawson coefficients
    if ridge == None:
        TL_coef = np.linalg.lstsq(A_f_t, meas_f_t, rcond=None)[0]
    else:
        TL_coef = np.linalg.inv(A_f_t.T.dot(A_f_t)+ridge*np.eye(np.shape(A_f_t)[1])).dot(A_f_t.T).dot(meas_f_t)
    
    return TL_coef