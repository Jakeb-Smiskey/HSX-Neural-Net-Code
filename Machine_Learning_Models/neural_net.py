#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 15:29:15 2023

@author: smiskey
"""

import os
from machine_functions import LoadData
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import r2_score, confusion_matrix

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


PRED = False

# Testing the Accuracy of the NN
if PRED is False:
    METRIC_NAME = ["QHS", "kappa", "Eps_eff", "delta_iota_H", "s_hat"]
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
    # Create the training and test data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_input, Y_input, test_size=0.2, random_state=42)

    # Creating NN
    mlpr = MLPRegressor(max_iter=100000, solver="lbfgs")  # Regression NN
    mlpc = MLPClassifier(max_iter=10000, activation="relu",
                         solver="adam", hidden_layer_sizes=(200, 2))

    # Fitting NN
    model_one = mlpr.fit(
        X_train, Y_train[["QHS", "kappa", "Eps_eff", "s_hat"]])
    model_delta = mlpc.fit(X_train, Y_train["delta_iota_H"])

    # Predicting NN
    predict_one = model_one.predict(X_test)
    predict_delta = model_delta.predict(X_test)

    # R^2 scores for each model
    delta_score = model_delta.score(X_test, Y_test["delta_iota_H"])
    score_one = model_one.score(
        X_test, Y_test[["QHS", "kappa", "Eps_eff", "s_hat"]])
    score_Q = r2_score(Y_test["QHS"], predict_one[:, 0])
    score_K = r2_score(Y_test["kappa"], predict_one[:, 1])
    score_E = r2_score(Y_test["Eps_eff"], predict_one[:, 2])
    score_s = r2_score(Y_test["s_hat"], predict_one[:, 3])
    total_score = (score_Q+score_K+score_E+delta_score+score_s)/5
    print("QHS,Kappa,Eps_eff Score: ", score_Q, score_K, score_E, score_s)
    print("Delta_iota_H Score: ", delta_score)

    # Delta_iota_H Plot #
    cm = confusion_matrix(Y_test["delta_iota_H"], predict_delta)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix for $\mathcal{\Delta\iota}$')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Predicted 0', 'Predicted 1'], rotation=45)
    plt.yticks(tick_marks, ['True 0', 'True 1'])
    plt.tight_layout()
    plt.ylabel('True $\mathcal{\Delta\iota}$')
    plt.xlabel('Predicted $\mathcal{\Delta\iota}$')
    plt.text(x=min(Y_test["delta_iota_H"])-.4, y=max(predict_delta), s="R$^2$: " + str(
        '%.5f' % delta_score), horizontalalignment='left', verticalalignment='center')
    plt.show()

    # Q Plot #
    plt.scatter(Y_test["QHS"], predict_one[:, 0], color="blue")
    plt.plot(Y_test["QHS"], Y_test["QHS"], color="black")
    plt.xlabel("Q True data")
    plt.ylabel("Q Prediction data")
    plt.title("Q True vs Prediction")
    plt.text(x=min(Y_test["QHS"]), y=max(predict_one[:, 0]), s="R$^2$: " + str(
        '%.5f' % score_Q), horizontalalignment='left', verticalalignment='center')
    plt.show()

    # Kappa Plot #
    plt.scatter(Y_test["kappa"], predict_one[:, 1], color="red")
    plt.plot(Y_test["kappa"], Y_test["kappa"], color="black")
    plt.xlabel("\u039A True data")
    plt.ylabel("\u039A Prediction data")
    plt.title("\u039A True vs Prediction")
    plt.text(x=min(Y_test["kappa"]), y=max(predict_one[:, 1]), s="R$^2$: " + str(
        '%.5f' % score_K), horizontalalignment='left', verticalalignment='center')
    plt.show()

    #Eps_eff Plot #
    plt.scatter(Y_test["Eps_eff"], predict_one[:, 2], color="green")
    plt.plot(Y_test["Eps_eff"], Y_test["Eps_eff"], color="black")
    plt.xlabel("$\u03B5_{eff}$ True data")
    plt.ylabel("$\u03B5_{eff}$Prediction data")
    plt.title("$\u03B5_{eff}$ True vs Prediction")
    plt.text(x=min(Y_test["Eps_eff"]), y=max(predict_one[:, 2]), s="R$^2$: " + str(
        '%.5f' % score_E), horizontalalignment='left', verticalalignment='center')
    plt.show()

    #s_hat Plot #
    plt.scatter(Y_test["s_hat"], predict_one[:, 3], color="purple")
    plt.plot(Y_test["s_hat"], Y_test["s_hat"], color="black")
    plt.xlabel("$\\hat{s}$ True data")
    plt.ylabel("$\\hat{s}$ Prediction data")
    plt.title("$\\hat{s}$ True vs Prediction")
    plt.text(x=min(Y_test["s_hat"]), y=max(predict_one[:, 3]), s="R$^2$: " + str(
        '%.5f' % score_s), horizontalalignment='left', verticalalignment='center')
    plt.show()

    # Table of Scores #
    data = {
        "Metric": [r'$\mathcal{Q} / \mathcal{Q}^*$', r'$\mathcal{K} / \mathcal{K}^*$', "$\u03B5_{eff} / \u03B5^*$ ", r"$\\hat{s}$/$\\hat{s}^*$", r"$\mathcal{\Delta\iota}$", "Total Score"],
        "Score": [score_Q, score_K, score_E, score_s, delta_score, total_score]
    }
    df = pd.DataFrame(data)
    df.set_index("Metric", inplace=True)
    print(df)

# Predicting Novel Configurations with a Neural Network
if PRED is True:
    # Location where code runs
    os.chdir("/media/smiskey/USB/main_coil_0/set_prediction")
    keys = ["Aux Coil 1", "Aux Coil 2", "Aux Coil 3",
            "Aux Coil 4", "Aux Coil 5", "Aux Coil 6"]
    METRIC_NAME = ["QHS", "kappa", "Eps_eff", "delta_iota_H", "s_hat"]
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
    AuxPredict = coilsprediction[keys]
    X_input = pd.concat([Aux3, Aux6, Aux8, Aux10, Aux1],
                        ignore_index=True)  # Input X values
    Y_input = pd.concat([m3, m6, m8, m10, m1],
                        ignore_index=True)  # Input Y values
    Y_input["QHS"] = np.log(Y_input["QHS"])
    #Neural Networks#
    mlpr = MLPRegressor(max_iter=100000, solver="lbfgs")  # Regression NN
    mlpc = MLPClassifier(max_iter=10000, activation="relu",
                         solver="adam", hidden_layer_sizes=(200, 2))  # Classification NN
    for i in METRIC_NAME:
        #Fitting the Neural Net#
        print(i)
        if i == "delta_iota_H":
            model_delta = mlpc.fit(X_input, Y_input[i])
            predict_delta = model_delta.predict(AuxPredict)
            coilsprediction[i] = predict_delta
        else:
            model_one = mlpr.fit(X_input, Y_input[i])
            predict_one = model_one.predict(AuxPredict)
            coilsprediction[i] = predict_one

    print(coilsprediction)
    # Saves the pandas dataframe as an H5 file
    with h5py.File("TestPredict.h5fd", 'w') as f:
        f.create_dataset('dataset', data=np.array(coilsprediction))
