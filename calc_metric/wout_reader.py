#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 13:21:07 2021

@author: michael
"""

import numpy as np
from netCDF4 import Dataset


class read_wout:
    """
    A class to read wout netCDF files from VMEC

    ...

    Attributes
    ----------
    pathName : str
        Full path to the wout file.

    Methods
    -------
    transForm_2D_sSec(self, s_val, u_dom, v_dom, ampKeys):
        Performs 2D Fourier transform along one flux surface on specified keys

    Raises
    ------
    IOError
        wout file specified does not exist
    """

    def __init__(self, pathName):

        try:
            rootgrp = Dataset(pathName, 'r')
        except IOError:
            raise IOError("Path does not exist : "+pathName)

        self.R_major = rootgrp['/Rmajor_p'][:]

        self.ns = rootgrp['/ns'][0]
        self.s_grid = np.linspace(0, 1, self.ns)
        self.ds = 1. / (self.ns-1)

        self.xm = rootgrp['/xm'][:]
        self.xn = rootgrp['/xn'][:]
        self.md = len(self.xm)

        self.xm_nyq = rootgrp['/xm_nyq'][:]
        self.xn_nyq = rootgrp['/xn_nyq'][:]
        self.md_nyq = len(self.xm_nyq)

        self.fourierAmps = {'R': rootgrp['/rmnc'][:, :],
                            'Z': rootgrp['/zmns'][:, :],
                            'Jacobian': rootgrp['/gmnc'][:, :],
                            'Bu': rootgrp['/bsubumnc'][:, :],
                            'Bv': rootgrp['/bsubvmnc'][:, :],
                            'Bs': rootgrp['/bsubsmns'][:, :],
                            'Bmod': rootgrp['/bmnc'][:, :]}

        self.fourierAmps['dR_ds'] = np.gradient(rootgrp['/rmnc'][:, :], self.ds, axis=0)
        self.fourierAmps['dR_du'] = -rootgrp['/rmnc'][:, :]*self.xm
        self.fourierAmps['dR_dv'] = rootgrp['/rmnc'][:, :]*self.xn

        self.fourierAmps['dZ_ds'] = np.gradient(rootgrp['/zmns'][:, :], self.ds, axis=0)
        self.fourierAmps['dZ_du'] = rootgrp['/zmns'][:, :]*self.xm
        self.fourierAmps['dZ_dv'] = -rootgrp['/zmns'][:, :]*self.xn

        self.fourierAmps['dBmod_ds'] = np.gradient(rootgrp['/bmnc'][:, :], self.ds, axis=0)
        self.fourierAmps['dBmod_du'] = -rootgrp['/bmnc'][:, :] * self.xm_nyq
        self.fourierAmps['dBmod_dv'] = rootgrp['/bmnc'][:, :] * self.xn_nyq

        self.fourierAmps['dBs_du'] = rootgrp['/bsubsmns'][:, :] * self.xm_nyq
        self.fourierAmps['dBs_dv'] = -rootgrp['/bsubsmns'][:, :] * self.xn_nyq

        self.fourierAmps['dBu_ds'] = np.gradient(rootgrp['/bsubumnc'][:, :], self.ds, axis=0)
        self.fourierAmps['dBu_dv'] = rootgrp['/bsubumnc'][:, :] * self.xn_nyq

        self.fourierAmps['dBv_ds'] = np.gradient(rootgrp['/bsubvmnc'][:, :], self.ds, axis=0)
        self.fourierAmps['dBv_du'] = -rootgrp['/bsubvmnc'][:, :] * self.xm_nyq

        self.cosine_keys = ['R', 'Jacobian', 'dR_ds', 'dZ_du', 'dZ_dv']
        self.sine_keys = ['Z', 'Lambda', 'dR_du', 'dR_dv', 'dZ_ds']

        self.cosine_nyq_keys = ['Bu', 'Bv', 'Bmod', 'dBs_du', 'dBs_dv',
                                'dBu_ds', 'dBv_ds', 'dBmod_ds']
        self.sine_nyq_keys = ['Bs', 'dBu_dv', 'dBv_du', 'dBmod_du', 'dBmod_dv']

        rootgrp.close()

    def transForm_3D(self, s_dom, u_dom, v_dom, ampKeys):
        """ Performs 3D Fourier transform on specified keys

        Parameters
        ----------
        s_dom : array
            radial domain on which to perform Fourier transform
        u_dom : array
            poloidal domain on which to perform Fourier transform
        v_dom : array
            toroidal domain on which to perform Fourier transform
        ampKeys : list
            keys specifying Fourier amplitudes to be transformed

        Raises
        ------
        NameError
            at least on key in ampKeys is not an available Fourier amplitude.
        """
        s_idx_beg = np.argmin(np.abs(self.s_grid - s_dom[0]))
        s_idx_end = 1 + np.argmin(np.abs(self.s_grid - s_dom[-1]))

        self.s_num = int(s_dom.shape[0])
        self.u_num = int(u_dom.shape[0])
        self.v_num = int(v_dom.shape[0])

        self.s_dom = s_dom
        self.u_dom = u_dom
        self.v_dom = v_dom

        pol, tor = np.meshgrid(self.u_dom, self.v_dom)

        pol_xm = np.dot(self.xm.reshape(self.md, 1), pol.reshape(1, self.v_num * self.u_num))
        tor_xn = np.dot(self.xn.reshape(self.md, 1), tor.reshape(1, self.v_num * self.u_num))

        cos_mu_nv = np.cos(pol_xm - tor_xn)
        sin_mu_nv = np.sin(pol_xm - tor_xn)

        pol_nyq_xm = np.dot(self.xm_nyq.reshape(self.md_nyq, 1), pol.reshape(1, self.v_num * self.u_num))
        tor_nyq_xn = np.dot(self.xn_nyq.reshape(self.md_nyq, 1), tor.reshape(1, self.v_num * self.u_num))

        cos_nyq_mu_nv = np.cos(pol_nyq_xm - tor_nyq_xn)
        sin_nyq_mu_nv = np.sin(pol_nyq_xm - tor_nyq_xn)

        self.invFourAmps = {}
        for key in ampKeys:

            if any(ikey == key for ikey in self.cosine_keys):
                self.invFourAmps[key] = np.dot(self.fourierAmps[key][s_idx_beg:s_idx_end], cos_mu_nv).reshape(self.s_num, self.v_num, self.u_num)

            elif any(ikey == key for ikey in self.sine_keys):
                self.invFourAmps[key] = np.dot(self.fourierAmps[key][s_idx_beg:s_idx_end], sin_mu_nv).reshape(self.s_num, self.v_num, self.u_num)

            elif any(ikey == key for ikey in self.cosine_nyq_keys):
                self.invFourAmps[key] = np.dot(self.fourierAmps[key][s_idx_beg:s_idx_end], cos_nyq_mu_nv).reshape(self.s_num, self.v_num, self.u_num)

            elif any(ikey == key for ikey in self.sine_nyq_keys):
                self.invFourAmps[key] = np.dot(self.fourierAmps[key][s_idx_beg:s_idx_end], sin_nyq_mu_nv).reshape(self.s_num, self.v_num, self.u_num)

            else:
                raise NameError('key = {} : is not available'.format(key))

    def transForm_2D_sSec(self, s_val, u_dom, v_dom, ampKeys):
        """ Performs 2D Fourier transform along one flux surface on specified keys

        Parameters
        ----------
        s_val : float
            effective radius of flux surface on which Fourier tronsfrom will
            be performed
        u_dom : array
            poloidal domain on which to perform Fourier transform
        v_dom : array
            toroidal domain on which to perform Fourier transform
        ampKeys : list
            keys specifying Fourier amplitudes to be transformed

        Raises
        ------
        NameError
            at least on key in ampKeys is not an available Fourier amplitude.
        """
        self.s_num = 1
        self.u_num = int(u_dom.shape[0])
        self.v_num = int(v_dom.shape[0])

        self.u_dom = u_dom
        self.v_dom = v_dom

        s_idx = np.argmin(np.abs(self.s_grid - s_val))

        pol, tor = np.meshgrid(self.u_dom, self.v_dom)

        pol_xm = np.dot(self.xm.reshape(self.md, 1), pol.reshape(1, self.v_num * self.u_num))
        tor_xn = np.dot(self.xn.reshape(self.md, 1), tor.reshape(1, self.v_num * self.u_num))

        cos_mu_nv = np.cos(pol_xm - tor_xn)
        sin_mu_nv = np.sin(pol_xm - tor_xn)

        pol_nyq_xm = np.dot(self.xm_nyq.reshape(self.md_nyq, 1), pol.reshape(1, self.v_num * self.u_num))
        tor_nyq_xn = np.dot(self.xn_nyq.reshape(self.md_nyq, 1), tor.reshape(1, self.v_num * self.u_num))

        cos_nyq_mu_nv = np.cos(pol_nyq_xm - tor_nyq_xn)
        sin_nyq_mu_nv = np.sin(pol_nyq_xm - tor_nyq_xn)

        self.invFourAmps = {}
        for key in ampKeys:

            if any(ikey == key for ikey in self.cosine_keys):
                self.invFourAmps[key] = np.dot(self.fourierAmps[key][s_idx], cos_mu_nv).reshape(self.v_num, self.u_num)

            elif any(ikey == key for ikey in self.sine_keys):
                self.invFourAmps[key] = np.dot(self.fourierAmps[key][s_idx], sin_mu_nv).reshape(self.v_num, self.u_num)

            elif any(ikey == key for ikey in self.cosine_nyq_keys):
                self.invFourAmps[key] = np.dot(self.fourierAmps[key][s_idx], cos_nyq_mu_nv).reshape(self.v_num, self.u_num)

            elif any(ikey == key for ikey in self.sine_nyq_keys):
                self.invFourAmps[key] = np.dot(self.fourierAmps[key][s_idx], sin_nyq_mu_nv).reshape(self.v_num, self.u_num)

            else:
                raise NameError('key = {} : is not available'.format(key))


if __name__ == "__main__":
    import os

    path = os.path.join('/home', 'michael', 'Desktop', 'Research',
                        'stellarators', 'HSX', 'Epsilon_Valley', 'valley_100',
                        'wout_files', 'wout_HSX_0-1-111.nc')

    wout = read_wout(path)

    s_dom = np.linspace(0, 1, 128)
    begIdx = np.argmin(np.abs(s_dom - .25**2))
    endIdx = 1 + np.argmin(np.abs(s_dom - .75**2))
    s_dom = s_dom[begIdx:endIdx]

    upts = 61
    vpts = 4 * upts
    u_dom = np.linspace(0, 2*np.pi, upts)
    v_dom = np.linspace(0, 2*np.pi, vpts)

    keys = ['R', 'Z', 'Jacobian', 'dR_ds', 'dR_du', 'dR_dv', 'dZ_ds', 'dZ_du',
            'dZ_dv', 'Bmod', 'dBmod_ds', 'dBmod_du', 'dBmod_dv', 'Bs', 'Bu',
            'Bv', 'dBs_du', 'dBs_dv', 'dBu_ds', 'dBu_dv', 'dBv_ds', 'dBv_du']

    wout.transForm_3D(s_dom, u_dom, v_dom, keys)
    dBv_du = wout.invFourAmps['dBv_du']
