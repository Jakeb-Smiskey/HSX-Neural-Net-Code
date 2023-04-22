#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 12:33:21 2022

@author: smiskey
"""
import numpy as np
import random
import os

states = np.empty((1000000,6))
coil_base = np.linspace(-.1, .1, 3)

for i in range(len(states)):
    for p in range(6):
        rng = np.random.default_rng()
        states[i][p] = coil_base[rng.integers(3)]
    states[i] = random.uniform(1,0)*states[i]
print(states)
percent = "Prediction"
output = open('auxStates{}.txt'.format(percent), 'w')
for state in states:
    output.write( ' '.join(['%.3f' % (s) for s in state]) + '\n' )
output.close()
auxConfigs = "/home/smiskey/Documents/HSX/HSX_Configs/auxCoil_Configs/"
command = "mv auxStates{}.txt ".format(percent) + auxConfigs
os.system(command)
