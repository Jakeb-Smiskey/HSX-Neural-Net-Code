# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 13:19:34 2020

@author: smiskey
"""

import os
import numpy as np

import functions as fun


exp = 'HSX'
main_state = np.array([1., 1., 1., 1., 1., 1.])

basePath = '/home/smiskey/Documents/HSX'
dirConfigs = os.path.join(basePath, exp+'_Configs')
dirOut = os.path.join('{}_Configs'.format(exp), 'queued_out')

# Import Base Input File
file_name = os.path.join(dirConfigs, 'coil_data', 'input.{}_aux'.format(exp))

f = open(file_name, 'r')
lines = f.readlines()
f.close()


# Import Auxiliary Coil States
def aux_file(num):
	auxStates = os.path.join(dirConfigs, 'auxCoil_Configs', 'auxStates{}.txt'.format(num))
	aux_states = fun.readStates(auxStates)
	return aux_states

state = input("Enter auxState number: ")
aux_states = aux_file(state)
conNum = state
name = 'main_coil_{}'.format(conNum)
command= 'mkdir ' + dirConfigs + "/" +  name
os.system(command)
config_dir = os.path.join(dirConfigs, name)
setNum = 0

# Construct Job MetaData
mult = 14
base_crnt = -10722.0
main_crnts = base_crnt * main_state
aux_crnts = mult * base_crnt * aux_states

fun.writeMetaFile(config_dir, setNum, main_state, aux_states)


# Construct Job Input Files
dirc_names = []
for c, crnt in enumerate(aux_crnts):
    dirc_name = 'job_{}'.format(c)
    dirc_names.append(dirc_name)
    new_dirc = os.path.join(dirConfigs,"jobs{}".format(state), dirc_name)
    os.makedirs(new_dirc)

    crnt_str = ' '.join(['{}'.format(i) for i in crnt])

    new_lines = lines
    new_lines[22] = '  EXTCUR =  -10722.0 '+crnt_str+'\n'

    file_name_new = os.path.join(new_dirc, 'input.HSX_aux')

    f = open(file_name_new, 'w')
    for l in new_lines:
        f.write(l)
    f.close()


