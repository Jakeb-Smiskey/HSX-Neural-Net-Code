#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 17:31:07 2023

@author: smiskey
"""
import os
import h5py
import numpy as np

os.chdir("/media/smiskey/USB/main_coil_0/{}".format("set_prediction"))
new = np.zeros(shape=(1000000, 3)) 
n = 0
for i in new:
 	i[1] = 1000000
 	i[2] = n
 	n = n+1
input_data = os.path.join('/home/smiskey/Documents/HSX/HSX_Configs/auxCoil_Configs', 'auxStates{}.txt'.format("Prediction")) #Define auxState number
data = np.loadtxt(input_data, dtype = float)
new_data = np.block([new, data])
with h5py.File("coildata.h5fd", 'w') as f:
	f.create_dataset('dataset', data = new_data)
