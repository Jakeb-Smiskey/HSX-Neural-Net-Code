#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 16:24:39 2023

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

os.chdir('/media/smiskey/USB/main_coil_0/set_prediction')  # Change location
with h5py.File("TestPredict.h5fd", 'r') as f:
    ls = list(f.keys())
    data = f.get(ls[0])
    dataset1 = np.array(data)
coildf = pd.DataFrame(np.array(dataset1))
okeys = ["Main ID", "Set ID", "Job ID", "Aux Coil 1", "Aux Coil 2", "Aux Coil 3", "Aux Coil 4",
         "Aux Coil 5", "Aux Coil 6", "Q", "kappa", "Eps_eff", "delta_iota_H","s_hat"]
for i in range(len(okeys)):
    coildf = coildf.rename(columns={i: okeys[i]})
# This is for plotting purposes
coildf = coildf.sort_values(by=["Eps_eff"], ascending=False)

coils10 = LoadData("/media/smiskey/USB/main_coil_0/set_10").open_aux()
coils8 = LoadData("/media/smiskey/USB/main_coil_0/set_8").open_aux()
coils6 = LoadData("/media/smiskey/USB/main_coil_0/set_6").open_aux()
metric10 = LoadData("/media/smiskey/USB/main_coil_0/set_10").open_metric()
metric8 = LoadData("/media/smiskey/USB/main_coil_0/set_8").open_metric()
metric6 = LoadData("/media/smiskey/USB/main_coil_0/set_6").open_metric()
coils3 = LoadData("/media/smiskey/USB/main_coil_0/set_3").open_aux()
metric3 = LoadData("/media/smiskey/USB/main_coil_0/set_3").open_metric()
coils1 = LoadData("/media/smiskey/USB/main_coil_0/set_1").open_aux()
metric1 = LoadData("/media/smiskey/USB/main_coil_0/set_1").open_metric()

coils10[["Q", "kappa", "Eps_eff", "delta_iota_H", "s_hat"]
        ] = metric10[["QHS", "kappa", "Eps_eff", "delta_iota_H", "s_hat"]]
coils8[["Q", "kappa", "Eps_eff", "delta_iota_H", "s_hat"]
       ] = metric8[["QHS", "kappa", "Eps_eff", "delta_iota_H", "s_hat"]]
coils6[["Q", "kappa", "Eps_eff", "delta_iota_H", "s_hat"]
       ] = metric6[["QHS", "kappa", "Eps_eff", "delta_iota_H", "s_hat"]]
coils3[["Q", "kappa", "Eps_eff", "delta_iota_H", "s_hat"]
       ] = metric3[["QHS", "kappa", "Eps_eff", "delta_iota_H", "s_hat"]]
coils1[["Q", "kappa", "Eps_eff", "delta_iota_H", "s_hat"]
       ] = metric1[["QHS", "kappa", "Eps_eff", "delta_iota_H", "s_hat"]]

Q_train = np.array(pd.concat(
    [coils10["Q"], coils8["Q"], coils6["Q"], coils3["Q"], coils1["Q"]], ignore_index=True))
K_train = np.array(pd.concat([coils10["kappa"], coils8["kappa"],
                   coils6["kappa"], coils3["kappa"], coils1["kappa"]], ignore_index=True))
Eps_train = np.array(pd.concat([coils10["Eps_eff"], coils8["Eps_eff"], coils6["Eps_eff"],
                     coils3["Eps_eff"], coils1["Eps_eff"]], ignore_index=True))**(3/2)
delta_train = np.array(pd.concat([coils10["delta_iota_H"], coils8["delta_iota_H"],
                       coils6["delta_iota_H"], coils3["delta_iota_H"], coils1["delta_iota_H"]], ignore_index=True))
s_train = np.array(pd.concat([coils10["s_hat"], coils8["s_hat"],
                     coils6["s_hat"], coils3["s_hat"]], ignore_index=True))
#Create the Data
Eps_eff = np.array(coildf['Eps_eff'])**(3/2)
Kappa = np.array(coildf['kappa'])
Q = np.array(coildf['Q'])
s_hat = np.array(coildf["s_hat"])
iota_1 = coildf[coildf['delta_iota_H']==0]
K_iota = np.array(iota_1["kappa"])
Q_iota = np.array(iota_1["Q"])

rand_aux = LoadData("/media/smiskey/USB/main_coil_0/set_predicted").open_aux()
rand_metric = LoadData("/media/smiskey/USB/main_coil_0/set_predicted").open_metric()
sub_m = rand_metric.iloc[[2,7]]
sub_aux = rand_aux.iloc[[2,7]]

sub_m = sub_m[["Eps_eff","kappa","QHS","delta_iota_H"]]
print(sub_m)


#Setting a Condition for Data



#Plotting the Data
plt.close('all')

font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 16}

mpl.rc('font', **font)
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['lines.linewidth'] = 2

fig = plt.figure(layout = "constrained", figsize=(8.4,8))
spec = gs.GridSpec(3, 1, wspace=0.1, figure=fig)

ax1 = fig.add_subplot(spec[0])
ax2 = fig.add_subplot(spec[1])
ax3 = fig.add_subplot(spec[2])

Q_low, Q_hgh = 0.9, 2
K_low, K_hgh = 0.98, 1.02
s = ax1.scatter(Kappa, Q, c=Eps_eff, s=1,
                norm=LogNorm(), marker='o', cmap='viridis_r')
ax1.scatter(K_iota, Q_iota, c = "grey", s = 0.1, alpha = 0.2, marker = "o" )
ax1.scatter([1.], [1.], c='k', s=200, marker='*', edgecolor='w')
ax1.plot([K_low, K_hgh], [Q_low, Q_low], c='k', ls='--')
ax1.plot([K_low, K_hgh], [Q_hgh, Q_hgh], c='k', ls='--')
ax1.plot([K_low, K_low], [Q_low, Q_hgh], c='k', ls='--')
ax1.plot([K_hgh, K_hgh], [Q_low, Q_hgh], c='k', ls='--')
cmap1 = fig.colorbar(s, ax=ax1)
cmap1.ax.set_ylabel(r'$\mathcal{E} / \mathcal{E}^*$')

s2 = ax2.scatter(Kappa, Q, c=Eps_eff, s=1, norm=LogNorm(), cmap='viridis_r')
ax2.scatter(K_iota, Q_iota, c = "grey",  s = 0.1, alpha = 0.2, marker = "o" )
ax2.scatter([1.], [1.], c='k', s=200, marker='*', edgecolor='w')
# Draw Connections #
con1 = ConnectionPatch(xyA=(K_low, Q_low), xyB=(K_low, Q_hgh),
                        coordsA='data', coordsB='data', axesA=ax1, axesB=ax2, color='k')
ax1.add_artist(con1)
con2 = ConnectionPatch(xyA=(K_hgh, Q_low), xyB=(K_hgh, Q_hgh),
                        coordsA='data', coordsB='data', axesA=ax1, axesB=ax2, color='k')
ax1.add_artist(con2)
ax2.plot([.995, 1.005], [0.90, 0.90], c='k', ls='--')
ax2.plot([.995, 1.005], [1.1, 1.1], c='k', ls='--')
ax2.plot([.995, .995], [0.90, 1.1], c='k', ls='--')
ax2.plot([1.005, 1.005], [0.90, 1.1], c='k', ls='--')
ax2.set_ylim(Q_low, Q_hgh)
ax2.set_xlim(K_low, K_hgh)
cmap1 = fig.colorbar(s2, ax=ax2)
cmap1.ax.set_ylabel(r'$\mathcal{E} / \mathcal{E}^*$')
s3=ax3.scatter(Kappa, Q, c=Eps_eff, s=1, norm=LogNorm(), cmap='viridis_r')
ax3.scatter(K_iota, Q_iota, c = "grey",  s = 0.1, alpha = 0.2, marker = "o"  )
ax3.scatter([1.], [1.], c='k', s=200, marker='*', edgecolor='w')

#Add Locations of Important Configurations
ax2.scatter([1.009553], [1.293576], c='k', s=50, marker='D', edgecolor='w')
ax2.scatter([1.011075], [1.481992], c='k', s=50, marker='h', edgecolor='w')

# Draw Connections #
con3 = ConnectionPatch(xyA=(.995, .90), xyB=(.995, 1.1),
                        coordsA='data', coordsB='data', axesA=ax2, axesB=ax3, color='k')
ax2.add_artist(con3)
con4 = ConnectionPatch(xyA=(1.005, .90), xyB=(1.005, 1.1),
                        coordsA='data', coordsB='data', axesA=ax2, axesB=ax3, color='k')
ax2.add_artist(con4)
cmap1 = fig.colorbar(s3, ax=ax3)
cmap1.ax.set_ylabel(r'$\mathcal{E} / \mathcal{E}^*$')

ax1.text(1.075, 2.5, '(a)')
ax2.text(1.015, 1.0, '(b)')
ax3.text(1.004, .97, '(c)')

ax3.set_ylim(0.90, 1.1)
ax3.set_xlim(.995, 1.005)

ax1.set_ylabel(r'$\mathcal{Q} / \mathcal{Q}^*$')
ax2.set_ylabel(r'$\mathcal{Q} / \mathcal{Q}^*$')
ax3.set_ylabel(r'$\mathcal{Q} / \mathcal{Q}^*$')
ax3.set_xlabel(r'$\mathcal{K} / \mathcal{K}^*$')

# plt.show()

#plt.close("all")


# fig2 = plt.figure()
# bx = fig2.add_subplot()
# c1 = bx.scatter(K_train, Q_train, c=Eps_train, s=1, norm=LogNorm(), marker="o", cmap='plasma')
# cmap2 = fig2.colorbar(c1, ax=bx)
# bx.scatter([1.], [1.], c='k', s=200, marker='*', edgecolor='w')

# cmap2.ax.set_ylabel(r'$\mathcal{E train} / \mathcal{E}^*$')
# bx.set_ylabel(r'$\mathcal{Q} / \mathcal{Q}^*$')
# bx.set_xlabel(r'$\mathcal{K} / \mathcal{K}^*$')

# fig3 = plt.figure()
# cx = fig3.add_subplot()
# d1 = cx.scatter(Kappa, Q, c=Eps_eff, norm = LogNorm(), s=1, marker='o', cmap='viridis_r')
# cmap3 = fig3.colorbar(d1, ax=cx)
# cx.scatter([1.], [1.], c='k', s=200, marker='*', edgecolor='w')
# cmap3.ax.set_ylabel(r'$\mathcal{E} / \mathcal{E}^*$')
# cx.set_ylabel(r'$\mathcal{Q} / \mathcal{Q}^*$')
# cx.set_xlabel(r'$\mathcal{K} / \mathcal{K}^*$')


#fig4 = plt.figure()
#cx = fig4.add_subplot()
#d1 = cx.scatter(s_hat, Q, s=.5, marker='o')
#cmap3 = fig3.colorbar(d1, ax=cx)
#cmap3.ax.set_ylabel(r'$\mathcal{Q} / \mathcal{Q}^*$')
#cx.set_ylabel(r'$\mathcal{Q} / \mathcal{Q}^*$')
#cx.set_xlabel(r'$\mathcal{S} / \mathcal{S}^*$')
#z = np.polyfit(Kappa, s_hat, 1)
#p = np.poly1d(z)
#plt.plot(Kappa, p(Kappa), "black")


plt.show()
