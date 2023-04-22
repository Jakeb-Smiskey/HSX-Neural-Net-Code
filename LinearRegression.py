#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 16:36:48 2023

@author: smiskey
"""

import os
from machine_functions import LoadData
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

np.random.seed(3146)

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
coilsprediction = LoadData(
    "/media/smiskey/USB/main_coil_0/set_prediction").open_aux()

METRIC_NAME = ["QHS", "kappa", "Eps_eff", "delta_iota_H","s_hat"]
keys = ["Aux Coil 1", "Aux Coil 2",
        "Aux Coil 3", "Aux Coil 4", "Aux Coil 5", "Aux Coil 6"]
Aux3 = coils3[keys]
Aux6 = coils6[keys]
Aux8 = coils8[keys]
Aux10 = coils10[keys]
Aux1 = coils1[keys]
m3 = metric3[METRIC_NAME]
m6 = metric6[METRIC_NAME]
m8 = metric8[METRIC_NAME]
m10 = metric10[METRIC_NAME]
m1 = metric1[METRIC_NAME]
X_input = pd.concat([Aux3, Aux6, Aux8, Aux10, Aux1],
                    ignore_index=True)  # Input X values
Y_input = pd.concat([m3, m6, m8, m10, m1],
                    ignore_index=True)  # Input Y values
X_train, X_test, Y_train, Y_test = train_test_split(
    X_input, Y_input, test_size=0.2, random_state=42)


reg = LinearRegression().fit(X_train,Y_train["QHS"])
predict = reg.predict(X_test)
score_Q = r2_score(Y_test["QHS"],predict)

# Q Plot #
plt.scatter(Y_test["QHS"], predict, color="blue")
plt.plot(Y_test["QHS"], Y_test["QHS"], color="black")
plt.xlabel("Q True data")
plt.ylabel("Q Prediction data")
plt.title("Q True vs Prediction")
plt.text(x=min(Y_test["QHS"]), y=max(predict), s="R$^2$: " + str(
    '%.5f' % score_Q), horizontalalignment='left', verticalalignment='center')
plt.show()

