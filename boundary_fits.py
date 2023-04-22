#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 14:30:46 2021

@author: michael
"""

import numpy as np


class boundaryFits():

    def __init__(self, wout, ma_vec):
        # Flux Grid #
        self.s_dom = wout.s_dom
        self.v_dom = wout.v_dom
        self.u_dom = wout.u_dom

        # Flux Grid Points #
        self.s_num = wout.s_num
        self.v_num = wout.v_num
        self.u_num = wout.u_num

        # Define Magnetic Axis #
        self.ma_vec = ma_vec

        ma_vec_3D = np.tile(self.ma_vec, self.u_num).reshape(self.v_num, self.u_num, 3)
        ma_vec_3D = np.tile(ma_vec_3D, self.s_num).reshape(self.v_num, self.u_num, self.s_num, 3)
        ma_vec_3D = np.swapaxes(ma_vec_3D, 1, 2)
        self.ma_vec_3D = np.swapaxes(ma_vec_3D, 0, 1)

        # Get Plasma Boundary #
        self.R = wout.invFourAmps['R']
        self.Z = wout.invFourAmps['Z']

        # Calculate Center Line #
        R_major = wout.R_major
        self.cl_vec = np.array([R_major, 0, 0])

        # Calculate Parallel and Perpendiculat Directions to Shafranov Shift #
        k_par = self.ma_vec - self.cl_vec
        k_par = k_par / np.repeat(np.linalg.norm(k_par, axis=1), 3).reshape(k_par.shape)
        k_perp = -np.cross(k_par, np.array([0, 0, 1]))

        k_par_3D = np.tile(k_par, self.u_num).reshape(self.v_num, self.u_num, 3)
        k_par_3D = np.tile(k_par_3D, self.s_num).reshape(self.v_num, self.u_num, self.s_num, 3)
        k_par_3D = np.swapaxes(k_par_3D, 1, 2)
        self.k_par_3D = np.swapaxes(k_par_3D, 0, 1)

        k_perp_3D = np.tile(k_perp, self.u_num).reshape(self.v_num, self.u_num, 3)
        k_perp_3D = np.tile(k_perp_3D, self.s_num).reshape(self.v_num, self.u_num, self.s_num, 3)
        k_perp_3D = np.swapaxes(k_perp_3D, 1, 2)
        self.k_perp_3D = np.swapaxes(k_perp_3D, 0, 1)

        # Transform Coordinates to Magnetic Axis #
        R_vec = np.stack((self.R, self.Z, np.zeros((self.s_num, self.v_num, self.u_num))), axis=3)
        self.r_vec = R_vec - self.ma_vec_3D

        # Normalizations #
        self.u_norm = 1. / self.u_dom[-1]
        self.v_norm = 1. / self.v_dom[-1]
        self.s_norm = 1. / np.trapz(np.ones(self.s_dom.shape), self.s_dom)

    def calc_delta(self):
        """
        Claculate the toroidal average delta parameter.
        """
        delta = np.linalg.norm(self.ma_vec - self.cl_vec, axis=1)
        return np.trapz(delta, self.v_dom) * self.v_norm

    def calc_rho(self):
        """
        Claculate the toroidal average rho parameter.
        """
        r_vec_norm = np.linalg.norm(self.r_vec, axis=3)
        rho = np.trapz(r_vec_norm, self.u_dom, axis=2) * self.u_norm
        rho = np.trapz(rho, self.v_dom, axis=1) * self.v_norm
        return np.trapz(rho, self.s_dom) * self.s_norm

    def calc_kappa(self):
        """
        Claculate the toroidal average kappa parameter.
        """
        kappa_perp = np.abs(np.sum(self.r_vec * self.k_perp_3D, axis=3))
        kappa_par = np.abs(np.sum(self.r_vec * self.k_par_3D, axis=3))

        # Calculate Poloidal Averages #
        kappa_perp_avg = np.trapz(kappa_perp, self.u_dom, axis=2)
        kappa_par_avg = np.trapz(kappa_par, self.u_dom, axis=2)

        # Calculate Kappa Ratio #
        kappa = kappa_perp_avg / kappa_par_avg
        kappa = np.trapz(kappa, self.v_dom) * self.v_norm
        return np.trapz(kappa, self.s_dom) * self.s_norm


if __name__ == "__main__":
    import os
    import wout_reader as wr

    path = os.path.join('/home', 'michael', 'Desktop', 'Research',
                        'stellarators', 'HSX', 'Epsilon_Valley',
                        'valley_100', 'wout_files', 'wout_HSX_0-1-111.nc')

    wout = wr.read_wout(path)

    s_dom = np.linspace(0, 1, 128)
    begIdx = np.argmin(np.abs(s_dom - .25**2))
    endIdx = 1 + np.argmin(np.abs(s_dom - .75**2))
    s_dom = s_dom[begIdx:endIdx]

    upts = 101
    vpts = upts
    u_dom = np.linspace(0, 2*np.pi, upts)
    v_dom = np.linspace(0, .25*np.pi, vpts)

    keys = ['R', 'Z']

    wout.transForm_2D_sSec(0, u_dom, v_dom, keys)
    ma_vec = np.stack((wout.invFourAmps['R'][:, 0], wout.invFourAmps['Z'][:, 0], np.zeros(vpts)), axis=1)

    wout.transForm_3D(s_dom, u_dom, v_dom, keys)
    bfit = boundaryFits(wout, ma_vec)

    delta = bfit.calc_delta()
    rho = bfit.calc_rho()
    kappa = bfit.calc_kappa()

    print('Delta = {0}\nRho = {1}\nKappa = {2}'.format(delta, rho, kappa))
