"""
Tools for using magnetic anomaly maps

The ChallMagMap.upward fonction is a pure python version of : https://github.com/MIT-AI-Accelerator/MagNav.jl/blob/master/src/fft_maps.jl
TODO: Add a Reference for this fonction, maybe : William J. Hinze, Ralph R.B. Von Frese, and Afif H. Saad. Gravity and magnetic
exploration: Principles, practices, and applications. 2010 ?

TODO: add vector_fft(map_in, dx, dy, D, I) fonction : Get magnetic anomaly map vector components using declination and inclination.
cf. : https://github.com/MIT-AI-Accelerator/MagNav.jl/blob/c04d850768b2801f441ff7880ae6f827858103c0/src/fft_maps.jl#L26
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


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