#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 13:45:23 2021

@author: michael
"""

import numpy as np


class BcurveTools():

    def __init__(self, wout):
        # Flux Grid #
        self.s_dom = wout.s_dom
        self.v_dom = wout.v_dom
        self.u_dom = wout.u_dom

        # Flux Grid Points #
        self.s_num = wout.s_num
        self.v_num = wout.v_num
        self.u_num = wout.u_num

        # Wout File #
        self.wout = wout

        # Plasma Boundary #
        self.R_coord = wout.invFourAmps['R']
        self.Z_coord = wout.invFourAmps['Z']

        # Jacobian #
        self.jacob = wout.invFourAmps['Jacobian']
        self.jacob_inv = 1. / self.jacob

        # Magnetic Field #
        self.B = wout.invFourAmps['Bmod']
        self.B_inv = 1. / self.B

        # Magnetic Unit Vector Componenets #
        self.bs = wout.invFourAmps['Bs'] * self.B_inv
        self.bu = wout.invFourAmps['Bu'] * self.B_inv
        self.bv = wout.invFourAmps['Bv'] * self.B_inv

        # Magnetic Unit Vector Derivatives #
        self.dbs_du = self.B_inv * (wout.invFourAmps['dBs_du'] - self.bs * wout.invFourAmps['dBmod_du'])
        self.dbs_dv = self.B_inv * (wout.invFourAmps['dBs_dv'] - self.bs * wout.invFourAmps['dBmod_dv'])

        self.dbu_ds = self.B_inv * (wout.invFourAmps['dBu_ds'] - self.bu * wout.invFourAmps['dBmod_ds'])
        self.dbu_dv = self.B_inv * (wout.invFourAmps['dBu_dv'] - self.bu * wout.invFourAmps['dBmod_dv'])

        self.dbv_ds = self.B_inv * (wout.invFourAmps['dBv_ds'] - self.bv * wout.invFourAmps['dBmod_ds'])
        self.dbv_du = self.B_inv * (wout.invFourAmps['dBv_du'] - self.bv * wout.invFourAmps['dBmod_du'])

        # Magnetic Field Strength Derivatives #
        self.dB_ds = wout.invFourAmps['dBmod_ds']
        self.dB_du = wout.invFourAmps['dBmod_du']
        self.dB_dv = wout.invFourAmps['dBmod_dv']

        # Covariant Metric Components #
        self.gss = wout.invFourAmps['dR_ds']**2 + wout.invFourAmps['dZ_ds']**2
        self.guu = wout.invFourAmps['dR_du']**2 + wout.invFourAmps['dZ_du']**2
        self.guv = wout.invFourAmps['dR_du'] * wout.invFourAmps['dR_dv'] + wout.invFourAmps['dZ_du'] * wout.invFourAmps['dZ_dv']
        self.gvv = wout.invFourAmps['dR_dv']**2 + wout.invFourAmps['dZ_dv']**2 + self.R_coord**2

        # Contravariant Metric Components #
        det_Gxx_inv = 1. / (self.R_coord * self.R_coord * ((wout.invFourAmps['dR_du'] * wout.invFourAmps['dZ_ds'] - wout.invFourAmps['dR_ds'] * wout.invFourAmps['dZ_du'])**2))
        self.g_su = det_Gxx_inv * ((wout.invFourAmps['dR_ds'] * wout.invFourAmps['dZ_dv'] - wout.invFourAmps['dR_dv'] * wout.invFourAmps['dZ_ds']) * (wout.invFourAmps['dR_dv'] * wout.invFourAmps['dZ_du'] - wout.invFourAmps['dR_du'] * wout.invFourAmps['dZ_dv']) - self.R_coord * self.R_coord * (wout.invFourAmps['dZ_ds'] * wout.invFourAmps['dZ_du'] + wout.invFourAmps['dR_ds'] * wout.invFourAmps['dR_du']))
        self.g_sv = det_Gxx_inv * ((wout.invFourAmps['dR_du'] * wout.invFourAmps['dZ_dv'] - wout.invFourAmps['dR_dv'] * wout.invFourAmps['dZ_du']) * (wout.invFourAmps['dR_ds'] * wout.invFourAmps['dZ_du'] - wout.invFourAmps['dR_du'] * wout.invFourAmps['dZ_ds']))
        self.g_uu = det_Gxx_inv * ((wout.invFourAmps['dR_ds'] * wout.invFourAmps['dZ_dv'] - wout.invFourAmps['dR_dv'] * wout.invFourAmps['dZ_ds'])**2 + self.R_coord * self.R_coord * (wout.invFourAmps['dZ_ds']**2 + wout.invFourAmps['dR_ds']**2))
        self.g_uv = det_Gxx_inv * ((wout.invFourAmps['dR_ds'] * wout.invFourAmps['dZ_dv'] - wout.invFourAmps['dR_dv'] * wout.invFourAmps['dZ_ds']) * (wout.invFourAmps['dR_du'] * wout.invFourAmps['dZ_ds'] - wout.invFourAmps['dR_ds'] * wout.invFourAmps['dZ_du']))
        self.g_vv = det_Gxx_inv * (wout.invFourAmps['dR_ds'] * wout.invFourAmps['dZ_du'] - wout.invFourAmps['dR_du'] * wout.invFourAmps['dZ_ds'])**2

        # Calculate Curvature #
        self.k_norm = (self.g_sv*self.bs + self.g_uv*self.bu + self.g_vv*self.bv)*(self.dbs_dv - self.dbv_ds) - (self.g_su*self.bs + self.g_uu*self.bu + self.g_uv*self.bv)*(self.dbu_ds - self.dbs_du)

        # Calculate Surface Area and Differential #
        self.surf_area_norm = 1. / np.trapz(np.trapz(self.jacob, self.u_dom, axis=2), self.v_dom, axis=1)

        # Calculate PsiN Domain Normalization #
        self.s_norm = 1. / np.trapz(np.ones(self.s_dom.shape), self.s_dom)

    def calc_F_kappa(self):
        """ Calculate the F_kappa metric on a single flux surface or across all
        flux surfaces. This calculation is defined as
        \mathcal{F}_{\kappa} = \oint ( (B - \langle B\rangle) / \langle B\rangle ) \kappa_{norm} da
        """
        B_avg = np.trapz(np.trapz(self.B * self.jacob, self.u_dom, axis=2), self.v_dom, axis=1) * self.surf_area_norm
        B_avg = np.tile(B_avg, self.u_num).reshape(self.u_num, self.s_num).T
        B_avg = np.tile(B_avg, self.v_num).reshape(self.s_num, self.v_num, self.u_num)

        F_kappa_integ = ((self.B - B_avg) / B_avg) * self.k_norm * self.jacob
        F_kappa = np.trapz(np.trapz(F_kappa_integ, self.u_dom, axis=2), self.v_dom, axis=1) * self.surf_area_norm
        return np.trapz(F_kappa, self.s_dom) * self.s_norm

    def calc_G_ss(self):
        """ Calculate the G_ss metric on a single flux surface or across all
        flux surfaces. This calculation is defined as
        \mathcal{G}_{ss} = \oint g_ss \kappa_{norm} da
        """
        G_ss_integ = self.gss * self.k_norm * self.jacob
        G_ss = np.trapz(np.trapz(G_ss_integ, self.u_dom, axis=2), self.v_dom, axis=1) * self.surf_area_norm
        return np.trapz(G_ss, self.s_dom) * self.s_norm

    def calc_avgB(self):
        """ Calculate the average magnetic field strength on a flux surface.
        Then take the integral average across flux surfaces.
        """
        B_avg_integ = self.B * self.jacob
        B_avg = np.trapz(np.trapz(B_avg_integ, self.u_dom, axis=2), self.v_dom, axis=1) * self.surf_area_norm
        return np.trapz(B_avg, self.s_dom) * self.s_norm


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

    upts = 61
    vpts = 4 * upts
    u_dom = np.linspace(0, 2*np.pi, upts, endpoint=False)
    v_dom = np.linspace(0, 2*np.pi, vpts, endpoint=False)

    keys = ['R', 'Z', 'Jacobian', 'dR_ds', 'dR_du', 'dR_dv', 'dZ_ds', 'dZ_du',
            'dZ_dv', 'Bmod', 'dBmod_ds', 'dBmod_du', 'dBmod_dv', 'Bs', 'Bu',
            'Bv', 'dBs_du', 'dBs_dv', 'dBu_ds', 'dBu_dv', 'dBv_ds', 'dBv_du']

    wout.transForm_3D(s_dom, u_dom, v_dom, keys)
    curB = BcurveTools(wout)

    F_kappa = curB.calc_F_kappa()
    G_ss = curB.calc_G_ss()
    B_avg = curB.calc_avgB()

    print(B_avg)
    print(F_kappa / .00479385)
    print(G_ss / .000298151)
