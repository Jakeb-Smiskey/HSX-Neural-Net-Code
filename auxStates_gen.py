# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 15:28:45 2020

@author: micha
"""

import numpy as np
import os

def inp(percent):
	npts = 3
	Npts = npts**6

	coil_base = np.linspace(-(float(percent)/100), (float(percent)/100), npts)
#states = np.array([0.,0.,0.,0.,0.,0.]).reshape(1,6)

	states = np.empty((Npts, 6))

	for i, c1 in enumerate(coil_base):
    		for j, c2 in enumerate(coil_base):
        		for k, c3 in enumerate(coil_base):
           			 for l, c4 in enumerate(coil_base):
               	 			for m, c5 in enumerate(coil_base):
                    				for n, c6 in enumerate(coil_base):
                       					 idx = i * npts**5 + j * npts**4 + k * npts**3 + l * npts**2 + m * npts + n
                       					 states[idx] = np.array([c1, c2, c3, c4, c5, c6])
#'''
#for i, c in enumerate(coil_base):
#    states[i] = np.full(6, c)

#states = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
#'''
	output = open('auxStates{}.txt'.format(percent), 'w')
	for state in states:
    		output.write( ' '.join(['%.2f' % (s) for s in state]) + '\n' )
	output.close()
	auxConfigs = "/home/smiskey/Documents/HSX/HSX_Configs/auxCoil_Configs/"
	command = "mv auxStates{}.txt ".format(percent) + auxConfigs
	os.system(command)
percent = input("Enter Aux Coil percent as whole number: ")
inp(percent)
