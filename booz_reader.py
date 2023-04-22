# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 10:53:09 2020

@author: micha
"""

import os

import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import interp1d


class read_booz():
    """
    A class to read boozmn netCDF files from STELLOPT

    ...

    Attributes
    ----------
    booz_path : str
        Full path to the boozmn file.

    Methods
    -------
    qhs_metric(s_grid)
        Calculate the root sum square of the symmetry breaking
        modes in the boozer spectrum.

    Raises
    ------
    IOError
        wout file specified does not exist
    """

    def __init__(self, booz_path):

        if not os.path.isfile(booz_path):
            raise IOError("Path does not exist : "+booz_path)

        rootgrp = Dataset(booz_path, 'r')

        ns = rootgrp['/ns_b'][0] - 1
        nsurf = rootgrp['/rmnc_b'][()].shape[0]

        self.ns = ns - (ns - nsurf)
        self.ds = 1. / ns
        self.s_full = np.linspace(0.5 * self.ds, 1 - 0.5 * self.ds, ns)

        s_grid_beg = 0.5 * self.ds
        s_grid_end = 1 - (ns - nsurf + 0.5) * self.ds
        self.s_grid = np.linspace(s_grid_beg, s_grid_end, self.ns)

        self.iota = interp1d(self.s_full, rootgrp['/iota_b'][1::])

        self.mpol = rootgrp['/mboz_b'][0]
        self.ntor = rootgrp['/nboz_b'][0]

        self.xm = rootgrp['/ixm_b'][:]
        self.xn = rootgrp['/ixn_b'][:]
        self.modes = self.xm.shape[0]

        # self.Bmod = rootgrp['/bmnc_b'][:,:]
        self.fourierAmps = {'R': interp1d(self.s_grid, rootgrp['/rmnc_b'][:, :], axis=0),
                            'Z': interp1d(self.s_grid, rootgrp['/zmns_b'][:, :], axis=0),
                            'P': interp1d(self.s_grid, rootgrp['/pmns_b'][:, :], axis=0),
                            'Jacobian': interp1d(self.s_grid, rootgrp['/gmn_b'][:, :], axis=0),
                            'Bmod': interp1d(self.s_grid, rootgrp['/bmnc_b'][:, :], axis=0)}

        self.cosine_keys = ['R', 'Jacobian', 'Bmod']
        self.sine_keys = ['Z', 'P']

        rootgrp.close()

    def qhs_metric(self, s_grid):
        """ Calculate the root sum square of the symmetry breaking
        modes in the boozer spectrum.
        """
        num_of_modes = self.modes
        Bmn = np.zeros(len(s_grid))
        Bmod_model = self.fourierAmps['Bmod']

        for i in range(num_of_modes):
            if self.xm[i] == 0 and self.xn[i] == 0:
                B_00 = Bmod_model(s_grid)[:, i]
            elif self.xm[i] == 0 and self.xn[i] != 0:
                Bmn = Bmn + Bmod_model(s_grid)[:, i]**2
            else:
                sym_chk = self.xn[i]/self.xm[i]
                if not sym_chk == 4:
                    Bmn = Bmn + Bmod_model(s_grid)[:, i]**2

        Q_met = np.sqrt(Bmn)/B_00
        Q_full_volume=np.trapz(Q_met, s_grid)/(s_grid[-1] - s_grid[0])
        index=np.argmin(np.abs(s_grid-0.5))
        Q_seq0p5=Q_met[index]
        return Q_full_volume,Q_seq0p5
    
    def s_hat_metric(self, s_grid):
        s_hat = -(1/self.iota(0.5))*(np.gradient(self.iota(s_grid), s_grid))[60]
        return s_hat
    
    def qhs_metric_plot(self,s_grid):
        num_of_modes = self.modes
        Bmn = np.zeros(len(s_grid))
        Bmod_model = self.fourierAmps['Bmod']

        for i in range(num_of_modes):
            if self.xm[i] == 0 and self.xn[i] == 0:
                B_00 = Bmod_model(s_grid)[:, i]
            elif self.xm[i] == 0 and self.xn[i] != 0:
                Bmn = Bmn + Bmod_model(s_grid)[:, i]**2
            else:
                sym_chk = self.xn[i]/self.xm[i]
                if not sym_chk == 4:
                    Bmn = Bmn + Bmod_model(s_grid)[:, i]**2

        Q_met = np.sqrt(Bmn)/B_00
        return Q_met, s_grid