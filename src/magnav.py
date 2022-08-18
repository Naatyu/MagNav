import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error
from scipy import signal, interpolate
import time
import random
import copy
import h5py
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'data')


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


#------------------------------#
#----Anomaly maps Functions----#
#------------------------------#


# Tools for using magnetic anomaly maps

# The ChallMagMap.upward fonction is a pure python version of : https://github.com/MIT-AI-Accelerator/MagNav.jl/blob/master/src/fft_maps.jl
# TODO: Add a Reference for this fonction, maybe : William J. Hinze, Ralph R.B. Von Frese, and Afif H. Saad. Gravity and magnetic
# exploration: Principles, practices, and applications. 2010 ?

# TODO: add vector_fft(map_in, dx, dy, D, I) fonction : Get magnetic anomaly map vector components using declination and inclination.
# cf. : https://github.com/MIT-AI-Accelerator/MagNav.jl/blob/c04d850768b2801f441ff7880ae6f827858103c0/src/fft_maps.jl#L26


def _map_spacing(x):
    return np.abs(x[-1] - x[0]) / (len(x) - 1)


def _get_ki(nx, ny, di):
    dk = 2 * np.pi / (nx * di)
    mn = np.mod(nx, 2)
    k = dk * np.arange((-nx + mn) / 2, (nx + mn) / 2)
    return np.tile(k, (ny, 1))


def _get_k(xe, de,  xn, dn):
    Ne, Nn = len(xe), len(xn)
    ke_m = _get_ki(Ne, Nn, de)
    ke = np.fft.ifftshift(ke_m)
    kn_m = _get_ki(Nn, Ne, dn).T
    kn = np.fft.ifftshift(kn_m)
    k = np.sqrt(ke_m**2 + kn_m**2)
    k = np.fft.ifftshift(k)
    return k, ke, kn


def  upward_map(map, xe, de,  xn, dn, dz):       
    # upward continuation function for shifting magnetic anomaly maps
    k, kx, ky = _get_k(xe, de,  xn, dn)
    H = np.exp(-k * dz)
    map_upward = np.real(np.fft.ifft2(np.fft.fft2(map) * H))
    return map_upward

class ChallMagMap:
    source = 'Challenge problem Magnetic Anomaly Map'

    def __init__(self, file_name):
        self.file_name = file_name
        # import map data
        f = h5py.File(file_name, 'r')
        self.alt = f['alt'][()]
        self.xe = f['xx'][:]  # longitude
        self.xn = f['yy'][:]  # latitude
        data = f['map'][:].T
        map_data = np.where(data<-100000, 0, data)
        self.map = map_data
        self.dn = _map_spacing(self.xn)
        self.de = _map_spacing(self.xe)

    def __repr__(self):
        return f'{self.source}, file: {self.file_name}'
     
    def upward(self, alt_upward):
        # upward continuation function for shifting magnetic anomaly maps
        self.alt_upward = alt_upward
        dz = alt_upward - self.alt
        if dz <= 0:
            raise ValueError(
                f'alt_upward must be greater than or equal to alt_map ({self.alt}m)')
        xe, de = self.xe, self.de
        xn, dn = self.xn, self.dn
        self.map_upward = upward_map(self.map, xe, de, xn, dn, dz)

    def interpolate(self, xei, xni, at_alt_upward=False):
        # TODO : Add bands checks
        if at_alt_upward:
            try:
                Z = self.map_upward
            except AttributeError:
                raise ValueError('Upward the map first')
        else:
            Z = self.map
        interp = interpolate.RectBivariateSpline(self.xn, self.xe, Z)
        return interp.ev(xni, xei)

    def plot(self, ax=None, at_alt_upward=False, plot_city=False):
        if ax is None:
            ax = plt.gca()
        if at_alt_upward:
            try:
                Z = self.map_upward
            except AttributeError:
                raise ValueError('Upward the map first')
        else:
            Z = self.map
        levels = np.linspace(self.map.min(), self.map.max(), 100)
        X, Y = np.meshgrid(self.xe, self.xn)
        cs = ax.contourf(X, Y, Z, levels=levels, cmap=plt.cm.turbo)
        cbar = plt.colorbar(cs, ax=ax, shrink=0.9)
        cbar.ax.set_ylabel('Magnetic anomaly [nT]')
        ax.set_xlabel('Xe UTM ZONE 18N [m]')
        ax.set_ylabel('Xn UTM ZONE 18N [m]')
        if plot_city:
            ax.text(368457, 5036186, r'Renfrew', fontsize=10)
            ax.text(445455, 5030014, r'Ottawa', fontsize=10)
        return ax

    def vector_fft(self, D, I):
        
        (Ny,Nx) = np.shape(self.map)
        (s,u,v) = _get_k(self.xe, self.de, self.xn, self.dn)
        
        l = np.cos((np.radians(I)))*np.cos((np.radians(D)))
        m = np.cos((np.radians(I)))*np.sin((np.radians(I)))
        n = np.sin((np.radians(I)))
        
        F = np.fft.fft(self.map)
        
        Hx = 1j*u / (1j*(u*l+m*v)+n*s)
        Hy = 1j*v / (1j*(u*l+m*v)+n*s)
        Hz = s    / (1j*(u*l+m*v)+n*s)
        
        Hx[1,1] = 1
        Hy[1,1] = 1
        Hz[1,1] = 1
        
        Bx = np.real(np.fft.ifft(Hx*F))
        By = np.real(np.fft.ifft(Hy*F))
        Bz = np.real(np.fft.ifft(Hz*F))
        
        return Bx, By, Bz


#----------------------#
#----IGRF Functions----#
#-------- -------------#


d2r = np.pi/180

basepath = os.path.dirname(__file__)
shc_fn = 'data/external/IGRF13.shc' # Default shc file

# Geomagnetic reference radius:
RE = 6371.2 # km

# World Geodetic System 84 parameters:
WGS84_e2 = 0.00669437999014
WGS84_a  = 6378.137 # km



def is_leapyear(year):
    """ Check for leapyear (handles arrays and preserves shape)

    """

    # if array:
    if type(year) is np.ndarray:
        out = np.full_like(year, False, dtype = bool)

        out[ year % 4   == 0] = True
        out[ year % 100 == 0] = False
        out[ year % 400 == 0] = True

        return out

    # if scalar:
    if year % 400 == 0:
        return True

    if year % 100 == 0:
        return False

    if year % 4 == 0:
        return True

    else:
        return False


def yearfrac_to_datetime(fracyear):
    """ 
    Convert fraction of year to datetime 

    Parameters
    ----------
    fracyear : iterable
        Date(s) in decimal year. E.g., 2021-03-28 is 2021.2377
        Must be an array, list or similar.

    Returns
    -------
    datetimes : array
        Array of datetimes
    """

    year = np.uint16(fracyear) # truncate fracyear to get year
    # use pandas TimedeltaIndex to represent time since beginning of year: 
    delta_year = pd.TimedeltaIndex((fracyear - year)*(365 + is_leapyear(year)), unit = 'D')
    # and DatetimeIndex to represent beginning of years:
    start_year = pd.DatetimeIndex(list(map(str, year)))
 
    # adding them produces the datetime:
    return (start_year + delta_year).to_pydatetime()


def get_legendre(theta, keys):
    """ 
    Calculate Schmidt semi-normalized associated Legendre functions

    Calculations based on recursive algorithm found in "Spacecraft Attitude Determination and Control" by James Richard Wertz
    
    Parameters
    ----------
    theta : array
        Array of colatitudes in degrees
    keys: iterable
        list of spherical harmnoic degree and order, tuple (n, m) for each 
        term in the expansion

    Returns
    -------
    P : array 
        Array of Legendre functions, with shape (theta.size, len(keys)). 
    dP : array
        Array of dP/dtheta, with shape (theta.size, len(keys))
    """

    # get maximum N and maximum M:
    n, m = np.array([k for k in keys]).T
    nmax, mmax = np.max(n), np.max(m)

    theta = theta.flatten()[:, np.newaxis]

    P = {}
    dP = {}
    sinth = np.sin(d2r*theta)
    costh = np.cos(d2r*theta)

    # Initialize Schmidt normalization
    S = {}
    S[0, 0] = 1.

    # initialize the functions:
    for n in range(nmax +1):
        for m in range(nmax + 1):
            P[n, m] = np.zeros_like(theta, dtype = np.float64)
            dP[n, m] = np.zeros_like(theta, dtype = np.float64)

    P[0, 0] = np.ones_like(theta, dtype = np.float64)
    for n in range(1, nmax +1):
        for m in range(0, min([n + 1, mmax + 1])):
            # do the legendre polynomials and derivatives
            if n == m:
                P[n, n]  = sinth * P[n - 1, m - 1]
                dP[n, n] = sinth * dP[n - 1, m - 1] + costh * P[n - 1, n - 1]
            else:

                if n == 1:
                    Knm = 0.
                    P[n, m]  = costh * P[n -1, m]
                    dP[n, m] = costh * dP[n - 1, m] - sinth * P[n - 1, m]

                elif n > 1:
                    Knm = ((n - 1)**2 - m**2) / ((2*n - 1)*(2*n - 3))
                    P[n, m]  = costh * P[n -1, m] - Knm*P[n - 2, m]
                    dP[n, m] = costh * dP[n - 1, m] - sinth * P[n - 1, m] - Knm * dP[n - 2, m]

            # compute Schmidt normalization
            if m == 0:
                S[n, 0] = S[n - 1, 0] * (2.*n - 1)/n
            else:
                S[n, m] = S[n, m - 1] * np.sqrt((n - m + 1)*(int(m == 1) + 1.)/(n + m))


    # now apply Schmidt normalization
    for n in range(1, nmax + 1):
        for m in range(0, min([n + 1, mmax + 1])):
            P[n, m]  *= S[n, m]
            dP[n, m] *= S[n, m]

    Pmat  = np.hstack(tuple(P[key] for key in keys))
    dPmat = np.hstack(tuple(dP[key] for key in keys)) 
    return Pmat, dPmat    


def read_shc(filename = shc_fn):
    """ 
    Read .shc (spherical harmonic coefficient) file

    The function produces data frames that have time as index and 
    spherical harmonic degree and order as columns. In the case of IGRF,
    the times will correspond to the different models 5 years apart

    Parameters
    ----------
    filename : string
        filename of .shc file

    Returns
    -------
    g : DataFrame
        pandas DataFrame of gauss coefficients for cos terms.
    h : DataFrame
        pandas DataFrame of gauss coefficients for sin terms.

    Note
    ----
    This code has no special treatment of "secular variation" coefficients. 
    Instead, the SV coefficients of IGRF should be used to make gauss 
    coefficients. This must be done prior to this code (when making the 
    .shc file).
    """

    header = 2
    coeffdict = {}
    with open(filename, 'r') as f:
        for line in f.readlines():

            if line.startswith('#'): # this is a header that we don't read
                continue

            if header == 2: # read parameters (could be skipped...)
                N_MIN, N_MAX, NTIMES, SP_ORDER, N_STEPS = list(map(int, line.split()[:5]))
                header -= 1
                continue

            if header == 1: # read years
                times = yearfrac_to_datetime(list(map(float, line.split())) )
                header -= 1
                continue

            key = tuple(map(int, line.split()[:2]))
            coeffdict[key] = np.array(list(map(float, line.split()[2:])))

    g = {key:coeffdict[key] for key in coeffdict.keys() if key[1] >= 0}
    h = {(key[0], -key[1]):coeffdict[key] for key in coeffdict.keys() if key[1] < 0 }
    for key in [k for k in g.keys() if k[1] == 0]: # add zero coefficients for m = 0 in h dictionary
        h[key] = 0

    # this must be true:
    assert len(g.keys()) == len(h.keys())

    gdf = pd.DataFrame(g, index = times)
    hdf = pd.DataFrame(h, index = times)

    # make sure that the column keys of h are in same order as in g:
    hdf = hdf[gdf.columns]

    return gdf, hdf



def geod2geoc(gdlat, height, Bn, Bu):
    """
    Convert from geocentric to geodetic coordinates

    Example:
    --------
    theta, r, B_th, B_r = geod2lat(gdlat, height, Bn, Bu)

    Parameters
    ----------
    gdlat : array
        Geodetic latitude [degrees]
    h : array
        Height above ellipsoid [km]
    Bn : array
        Vector in northward direction, relative to ellipsoid
    Bu : array
        Vector in upward direction, relative to ellipsoid

    Returns
    -------
    theta : array
        Colatitudes [degrees]
    r : array
        Radius [km]
    B_th : array
        Vector component in theta direction
    B_r : array
        Vector component in radial direction
    """

    a = WGS84_a
    b = a*np.sqrt(1 - WGS84_e2)

    sin_alpha_2 = np.sin(gdlat*d2r)**2
    cos_alpha_2 = np.cos(gdlat*d2r)**2

    # calculate geocentric latitude and radius
    tmp = height * np.sqrt(a**2 * cos_alpha_2 + b**2 * sin_alpha_2)
    beta = np.arctan((tmp + b**2)/(tmp + a**2) * np.tan(gdlat * d2r))
    theta = np.pi/2 - beta
    r = np.sqrt(height**2 + 2 * tmp + a**2 * (1 - (1 - (b/a)**4) * sin_alpha_2) / (1 - (1 - (b/a)**2) * sin_alpha_2))

    # calculate geocentric components
    psi  =  np.sin(gdlat*d2r) * np.sin(theta) - np.cos(gdlat*d2r) * np.cos(theta)
    
    B_r  = -np.sin(psi) * Bn + np.cos(psi) * Bu
    B_th = -np.cos(psi) * Bn - np.sin(psi) * Bu

    theta = theta/d2r

    return theta, r, B_th, B_r
 
def geoc2geod(theta, r, B_th, B_r):
    """
    Convert from geodetic to geocentric coordinates

    Based on Matlab code by Nils Olsen, DTU

    Example:
    --------
    gdlat, height, Bn, Bu = geod2lat(theta, r, B_th, B_r)

    Parameters
    ----------
    theta : array
        Colatitudes [degrees]
    r : array
        Radius [km]
    B_th : array
        Vector component in theta direction
    B_r : array
        Vector component in radial direction

    Returns
    -------
    gdlat : array
        Geodetic latitude [degrees]
    h : array
        Height above ellipsoid [km]
    Bn : array
        Vector in northward direction, relative to ellipsoid
    Bu : array
        Vector in upward direction, relative to ellipsoid
    """
    
    a = WGS84_a
    b = a*np.sqrt(1 - WGS84_e2)

    E2 = 1.-(b/a)**2
    E4 = E2*E2
    E6 = E4*E2
    E8 = E4*E4
    OME2REQ = (1.-E2)*a
    A21 =     (512.*E2 + 128.*E4 + 60.*E6 + 35.*E8)/1024.
    A22 =     (                        E6 +     E8)/  32.
    A23 = -3.*(                     4.*E6 +  3.*E8)/ 256.
    A41 =    -(           64.*E4 + 48.*E6 + 35.*E8)/1024.
    A42 =     (            4.*E4 +  2.*E6 +     E8)/  16.
    A43 =                                   15.*E8 / 256.
    A44 =                                      -E8 /  16.
    A61 =  3.*(                     4.*E6 +  5.*E8)/1024.
    A62 = -3.*(                        E6 +     E8)/  32.
    A63 = 35.*(                     4.*E6 +  3.*E8)/ 768.
    A81 =                                   -5.*E8 /2048.
    A82 =                                   64.*E8 /2048.
    A83 =                                 -252.*E8 /2048.
    A84 =                                  320.*E8 /2048.
    
    GCLAT = (90-theta)
    SCL = np.sin(GCLAT * d2r)
    
    RI = a/r
    A2 = RI*(A21 + RI * (A22 + RI* A23))
    A4 = RI*(A41 + RI * (A42 + RI*(A43+RI*A44)))
    A6 = RI*(A61 + RI * (A62 + RI* A63))
    A8 = RI*(A81 + RI * (A82 + RI*(A83+RI*A84)))
    
    CCL = np.sqrt(1-SCL**2)
    S2CL = 2.*SCL  * CCL
    C2CL = 2.*CCL  * CCL-1.
    S4CL = 2.*S2CL * C2CL
    C4CL = 2.*C2CL * C2CL-1.
    S8CL = 2.*S4CL * C4CL
    S6CL = S2CL * C4CL + C2CL * S4CL
    
    DLTCL = S2CL * A2 + S4CL * A4 + S6CL * A6 + S8CL * A8
    gdlat = DLTCL + GCLAT * d2r
    height = r * np.cos(DLTCL)- a * np.sqrt(1 -  E2 * np.sin(gdlat) ** 2)


    # magnetic components 
    psi = np.sin(gdlat) * np.sin(theta*d2r) - np.cos(gdlat) * np.cos(theta*d2r)
    Bn = -np.cos(psi) * B_th - np.sin(psi) * B_r 
    Bu = -np.sin(psi) * B_th + np.cos(psi) * B_r 

    gdlat = gdlat / d2r

    return gdlat, height, Bn, Bu



def igrf_gc(r, theta, phi, date, coeff_fn = shc_fn):
    """
    Calculate IGRF model components

    Input and output in geocentric coordinates

    Broadcasting rules apply for coordinate arrays, and the
    combined shape will be preserved. The dates are kept out
    of the broadcasting, so that the output will have shape
    (N, ...) where N is the number of dates, and ... represents
    the combined shape of the coordinates. If you pass scalars,
    the output will be arrays of shape (1, 1)
    
    Parameters
    ----------
    r : array
        radius [km] of IGRF calculation
    theta : array
        colatitude [deg] of IGRF calculation
    phi : array
        longitude [deg], positive east, of IGRF claculation
    date : date(s)
        one or more dates to evaluate IGRF coefficients
    coeff_fn : string, optional
        filename of .shc file. Default is latest IGRF

    Return
    ------
    Br : array
        Magnetic field [nT] in radial direction
    Btheta : array
        Magnetic field [nT] in theta direction (south on an
        Earth-centered sphere with radius r)
    Bphi : array
        Magnetic field [nT] in eastward direction
    """

    # read coefficient file:
    g, h = read_shc(coeff_fn)

    if not hasattr(date, '__iter__'):
        date = np.array([date])
    else:
        date = np.array(date)

    if np.any(date > g.index[-1]) or np.any(date < g.index[0]):
        print('Warning: You provided date(s) not covered by coefficient file \n({} to {})'.format(
              g.index[0].date(), g.index[-1].date()))

    # convert input to arrays in case they aren't
    r, theta, phi = tuple(map(lambda x: np.array(x, ndmin = 1), [r, theta, phi]))

    # get coordinate arrays to same size and shape
    shape = np.broadcast_shapes(r.shape, theta.shape, phi.shape)
    r, theta, phi = map(lambda x: np.broadcast_to(x, shape)   , [r, theta, phi])
    r, theta, phi = map(lambda x: x.flatten().reshape((-1 ,1)), [r, theta, phi]) # column vectors

    # make row vectors of wave numbers n and m:
    n, m = np.array([k for k in g.columns]).T
    n, m = n.reshape((1, -1)), m.reshape((1, -1))

    # get maximum N and maximum M:
    N, M = np.max(n), np.max(m)

    # get the legendre functions
    P, dP = get_legendre(theta, g.keys())

    # Append coefficients at desired times (skip if index is already in coefficient data frame):
    index = g.index.union(date)

    g = g.reindex(index).groupby(index).first() # reindex and skip duplicates
    h = h.reindex(index).groupby(index).first() # reindex and skip duplicates

    # interpolate and collect the coefficients at desired times:
    g = g.interpolate(method = 'time').loc[date, :]
    h = h.interpolate(method = 'time').loc[date, :]

    # compute cosmlon and sinmlon:
    cosmphi = np.cos(phi * d2r * m) # shape (n_coords x n_model_params/2)
    sinmphi = np.sin(phi * d2r * m)

    # make versions of n and m that are repeated twice
    nn, mm = np.tile(n, 2), np.tile(m, 2)

    # calculate Br:
    G  = (RE / r) ** (nn + 2) * (nn + 1) * np.hstack((P * cosmphi, P * sinmphi))
    Br = G.dot(np.hstack((g.values, h.values)).T).T # shape (n_times, n_coords)

    # calculate Btheta:
    G  = -(RE / r) ** (nn + 1) * np.hstack((dP * cosmphi, dP * sinmphi)) \
         * RE / r
    Btheta = G.dot(np.hstack((g.values, h.values)).T).T # shape (n_times, n_coords)

    # calculate Bphi:
    G  = -(RE / r) ** (nn + 1) * mm * np.hstack((-P * sinmphi, P * cosmphi)) \
         * RE / r / np.sin(theta * d2r)
    Bphi = G.dot(np.hstack((g.values, h.values)).T).T # shape (n_times, n_coords)

    # reshape and return
    outshape = tuple([Bphi.shape[0]] + list(shape))
    return Br.reshape(outshape), Btheta.reshape(outshape), Bphi.reshape(outshape)


def igrf(lon, lat, h, date, coeff_fn = shc_fn):
    """
    Calculate IGRF model components

    Input and output in geodetic coordinates

    Broadcasting rules apply for coordinate arrays, and the
    combined shape will be preserved. The dates are kept out
    of the broadcasting, so that the output will have shape
    (N, ...) where N is the number of dates, and ... represents
    the combined shape of the coordinates. If you pass scalars,
    the output will be arrays of shape (1, 1)
    
    Parameters
    ----------
    lon : array
        longitude [deg], postiive east, of IGRF calculation
    lat : array
        geodetic latitude [deg] of IGRF calculation
    h : array
        height [km] above ellipsoid for IGRF calculation
    date : date(s)
        one or more dates to evaluate IGRF coefficients
    coeff_fn : string, optional
        filename of .shc file. Default is latest IGRF

    Return
    ------
    Be : array
        Magnetic field [nT] in eastward direction
    Bn : array
        Magnetic field [nT] in northward direction, relative to
        ellipsoid
    Bu : array
        Magnetic field [nT] in upward direction, relative to 
        ellipsoid
    """

    # convert input to arrays and cast to same shape:
    lon, lat, h = tuple(map(lambda x: np.array(x, ndmin = 1), [lon, lat, h]))
    shape = np.broadcast_shapes(lon.shape, lat.shape, h.shape)
    lon, lat, h = map(lambda x: np.broadcast_to(x, shape), [lon, lat, h])
    lon, lat, h = map(lambda x: x.flatten(), [lon, lat, h])

    # convert to geocentric:
    theta, r, _, __ = geod2geoc(lat, h, h, h)
    phi = lon

    # calculate geocentric components of IGRF:
    Br, Btheta, Bphi = igrf_gc(r, theta, phi, date, coeff_fn = coeff_fn)
    Be = Bphi

    # convert output to geodetic
    lat_, h_, Bn, Bu = geoc2geod(theta, r, Btheta, Br)

    # return with shapes implied by input
    outshape = tuple([Be.shape[0]] + list(shape))
    return Be.reshape(outshape), Bn.reshape(outshape), Bu.reshape(outshape)