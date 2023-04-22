#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 13:57:42 2023

@author: smiskey
"""


import os
import h5py
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from machine_functions import LoadData
from matplotlib.patches import ConnectionPatch

np.random.seed(3145)

os.chdir('/media/smiskey/USB/main_coil_0/set_prediction')  # Change location
with h5py.File("TestPredict.h5fd", 'r') as f:
    ls = list(f.keys())
    data = f.get(ls[0])
    dataset1 = np.array(data)
coildf = pd.DataFrame(np.array(dataset1))
okeys = ["Main ID", "Set ID", "Job ID", "Aux Coil 1", "Aux Coil 2", "Aux Coil 3", "Aux Coil 4",
         "Aux Coil 5", "Aux Coil 6", "Q", "kappa", "Eps_eff", "delta_iota_H", "s_hat"]
for i in range(len(okeys)):
    coildf = coildf.rename(columns={i: okeys[i]})
# This is for plotting purposes
coildf = coildf.sort_values(by=["Eps_eff"], ascending=False)


"Plotting Eps_eff vs Kappa"
Eps_eff = np.array(coildf['Eps_eff'])**(3/2)
Kappa = np.array(coildf['kappa'])
Q = np.array(coildf['Q'])
s = np.array(coildf["s_hat"])
iota = coildf[coildf["delta_iota_H"]==1]

Q_sorted = iota.sort_values(by = ["Q"], ascending=True)

rand_aux = LoadData("/media/smiskey/USB/main_coil_0/set_3").open_aux()
rand_metric = LoadData("/media/smiskey/USB/main_coil_0/set_3").open_metric()
print(rand_aux)
print(rand_metric["Eps_eff"])
#rand_metric = (rand_metric[["Eps_eff","QHS","kappa","s_hat","delta_iota_H"]])
#subset_df = rand_metric.iloc[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
#print(rand_aux.iloc[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]])
sub_m = rand_metric.iloc[[2,7]]
sub_aux = rand_aux.iloc[[2,7]]
print(sub_m)
#rand_metric = pd.concat([subset_df,sub_m],ignore_index=True)
#rand_metric = rand_metric.rename(columns={"QHS": "Q"})
#subset_df = subset_df.rename(columns={"QHS": "Q"})

#print(subset_df)


os.chdir('/media/smiskey/USB/main_coil_0/set_prediction')  # Change location
data_m = pd.read_csv("Q_rand_data.txt", sep="\t")
df_1 = pd.DataFrame(data_m)
data_m = pd.read_csv("rand_data.txt", sep="\t")
#df_2 = pd.DataFrame(data_m)
#df_2 =  pd.concat([df_1,df_2], ignore_index=True)
#metric_s = df_2[["Eps_eff", "Q","kappa", "s_hat","delta_iota_H"]]
#print(df_1)
#print(df_1)
#print(subset_df)
##relative_error = (abs(rand_metric-metric_s)/rand_metric*100)
#print(sum(relative_error["s_hat"])/len(relative_error["Eps_eff"]))
#print(relative_error["delta_iota_H"])

#ideal = rand_aux.iloc[[2,7]]
#ideal_m = rand_metric.iloc[[2,7]]
#ideal = pd.concat([ideal,ideal_m], axis = 1, ignore_index=True)
#ideal = ideal.join(ideal_m)
#print(ideal)
# Q_rand = Q_sorted.sample(n=15)
# data_coils = Q_rand[["Aux Coil 1", "Aux Coil 2", "Aux Coil 3", "Aux Coil 4", "Aux Coil 5", "Aux Coil 6"]]
# Q_rand.to_csv("Q_rand_data.txt", sep='\t', index=False)
# data_coils.to_csv("Q_rand_aux_data.txt", sep='\t', index=False)

# rand_data = coildf.sample(n=50)
# rand_aux = rand_data[["Aux Coil 1", "Aux Coil 2", "Aux Coil 3", "Aux Coil 4", "Aux Coil 5", "Aux Coil 6"]]
# rand_data.to_csv("rand_data.txt", sep='\t', index=False)
# rand_aux.to_csv("rand_aux_data.txt", sep='\t', index=False)

# print(Q_rand)
# print(rand_data)

#data = pd.concat([Q_sorted.head(),Q_sorted.tail()])

############################################################################

# os.chdir('/media/smiskey/USB/main_coil_0/set_Q_low') #Change location
# np.savetxt("Q_low_data.txt",data,fmt="%f",delimiter=" ")

# os.chdir("/home/smiskey/Documents/HSX/HSX_Configs/auxCoil_Configs/")
# data_aux = data.drop(["Job ID", "delta_iota_H","Eps_eff","kappa","Q","s_hat", "Main ID", "Set ID"], axis=1)
# np.savetxt('auxStatesQ_low.txt',np.array(data_aux), fmt='%f' )
#########################################################################






