#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 13:15:23 2023

@author: smiskey
"""

import os
import numpy as np

import wout_reader as wr
import neo_reader as nr

import curveB_tools as cb
import boundary_fits as bf

import matplotlib.pyplot as plt

def neo_plot(direct, location):
    # Base Directory #
    base_dirc = os.path.join(location, direct)

    # Metric Dump #
    dump_path = os.path.join(base_dirc, 'metric_output.txt')

    # Define File Paths #
    wout_path = os.path.join(base_dirc, 'wout_HSX_aux_opt0.nc')
    booz_path = os.path.join(base_dirc, 'boozmn_HSX_aux_opt0.nc')
    neo_path = os.path.join(base_dirc, 'neo_out.HSX_aux.00000')
    
    #Read NEO
    neo = nr.read_neo(neo_path, wout_path)
    
    return neo.plot_epsilon()

eps_eff_2, s_dom_2 = neo_plot("job_2","/media/smiskey/USB/main_coil_0/set_Predicted_data_files")    
eps_eff_7, s_dom_7 = neo_plot("job_7","/media/smiskey/USB/main_coil_0/set_Predicted_data_files")    
eps_eff_QHS, s_dom_QHS = neo_plot("job_364","/media/smiskey/USB/main_coil_0/set_3_data_files")        


plt.scatter(s_dom_2,(eps_eff_2)**(3/2), color = "red", s=0.5)
plt.scatter(s_dom_7,(eps_eff_7)**(3/2), color = "blue", s=0.5)
plt.scatter(s_dom_QHS,(eps_eff_QHS)**(3/2), color = "black", s=0.5)

plt.xlabel("r/a")
plt.ylabel(r'$\mathcal{E}_{eff}$')

location = 0 # For the best location
legend_drawn_flag = True
plt.legend(["Ideal Config 2", "Ideal Config 7", "QHS"], loc=0, frameon=legend_drawn_flag)
