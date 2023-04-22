#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 17:38:53 2021

@author: michael
"""

import os
import numpy as np

import wout_reader as wr
import neo_reader as nr
import booz_reader as br

import curveB_tools as cb
import boundary_fits as bf


def run(direct, location):
    # Base Directory #
    base_dirc = os.path.join(location, direct)

    # Metric Dump #
    dump_path = os.path.join(base_dirc, 'metric_output.txt')

    # Define File Paths #
    wout_path = os.path.join(base_dirc, 'wout_HSX_aux_opt0.nc')
    booz_path = os.path.join(base_dirc, 'boozmn_HSX_aux_opt0.nc')
    neo_path = os.path.join(base_dirc, 'neo_out.HSX_aux.00000')

    # Instantiate Data Objects #
    wout = wr.read_wout(wout_path)
    booz = br.read_booz(booz_path)
    neo = nr.read_neo(neo_path, wout_path)

    # Define Flux Surface Grid #
    upts = 61
    vpts = 4 * upts
    u_dom = np.linspace(0, 2*np.pi, upts, endpoint=False)
    v_dom = np.linspace(0, 2*np.pi, vpts, endpoint=False)

    # Calculate Variables Throughout Domain #
    keys = ['R', 'Z', 'Jacobian', 'dR_ds', 'dR_du', 'dR_dv', 'dZ_ds', 'dZ_du',
            'dZ_dv', 'Bmod', 'dBmod_ds', 'dBmod_du', 'dBmod_dv', 'Bs', 'Bu', 'Bv',
            'dBs_du', 'dBs_dv', 'dBu_ds', 'dBu_dv', 'dBv_ds', 'dBv_du']
    wout.transForm_3D(wout.s_grid, u_dom, v_dom, keys)

    # Calculate Flux Surface Averaged Metrics #
    curB = cb.BcurveTools(wout)

    B_avg = curB.calc_avgB()
    F_kappa = curB.calc_F_kappa()
    G_ss = curB.calc_G_ss()

    # Calculate Radial Average Epsilon Effective #
    eps_eff = neo.average_epsilon()

    # Calculate Radial Average QHS Metric # # Calculate Radial QHS Metric at Q_s = 0.5  #
    Q = booz.qhs_metric(booz.s_grid)
    qhs = Q[0]
    Q_seq0p5 = Q[1]
    
    # Calculate Magnetic Shear at r/a = 0.5 #
    s_hat = booz.s_hat_metric(booz.s_grid)
    
    # Define Flux Surface Grid #
    upts = 61
    vpts = upts
    u_dom = np.linspace(0, 2*np.pi, upts)
    v_dom = np.linspace(0, .25*np.pi, vpts)

    # Find Magnetic Axis #
    wout.transForm_2D_sSec(0, u_dom, v_dom, ['R', 'Z'])
    ma_vec = np.stack(
        (wout.invFourAmps['R'][:, 0], wout.invFourAmps['Z'][:, 0], np.zeros(vpts)), axis=1)

    # Calculate Boundary Throughout Radial Domain #
    wout.transForm_3D(wout.s_grid, u_dom, v_dom, ['R', 'Z'])

    # Calculate Geometry Parameters #
    bfit = bf.boundaryFits(wout, ma_vec)
    delta = bfit.calc_delta()
    rho = bfit.calc_rho()
    kappa = bfit.calc_kappa()

    # Output Data #
    with open(dump_path, 'w') as my_file:
        my_file.write(
            'Eps_eff QHS B_avg F_kappa G_ss kappa delta rho Q_seq0p5 s_hat\n')
        my_file.write('{0} {1} {2} {3} {4} {5} {6} {7} {8} {9}\n'.format(
            eps_eff, qhs, B_avg, F_kappa, G_ss, kappa, delta, rho, Q_seq0p5, s_hat))
