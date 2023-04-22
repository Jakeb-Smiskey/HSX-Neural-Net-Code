#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 15:32:06 2023

@author: smiskey
"""

import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor, MLPClassifier


class LoadData:
    """
    word
    """

    def __init__(self, directory):
        self.directory = directory

    def open_aux(self):
        '''

        Returns
        -------
        coildf : TYPE
            DESCRIPTION.

        '''
        os.chdir(self.directory)
        with h5py.File("coildata.h5fd", 'r') as file:
            lists = list(file.keys())
            data = file.get(lists[0])
            dataset1 = np.array(data)
        coildf = pd.DataFrame(np.array(dataset1))
        keys = ["Main ID", "Set ID", "Job ID", "Aux Coil 1", "Aux Coil 2",
                "Aux Coil 3", "Aux Coil 4", "Aux Coil 5", "Aux Coil 6"]
        num = len(keys)
        for i in range(num):
            coildf = coildf.rename(columns={i: keys[i]})
        return coildf

    def open_metric(self):
        '''


        Returns
        -------
        metricdf : TYPE
            DESCRIPTION.

        '''
        os.chdir(self.directory)
        with h5py.File("metricdata.h5fd", 'r') as file:
            lists = list(file.keys())
            data = file.get(lists[0])
            dataset1 = np.array(data)
        metricdf = pd.DataFrame(np.array(dataset1))
        keys = ["Main ID", "Set ID", "Job ID", 'Eps_eff', 'QHS', ' B_avg', 'F_kappa',
                'G_ss', 'kappa', 'delta', 'rho',"Q_seq0p5","s_hat", 'iota_4/4', 'iota_8/7', 'delta_iota', 'delta_iota_H']
        num = len(keys)
        for i in range(num):
            metricdf = metricdf.rename(columns={i: keys[i]})
        return metricdf


class RFModel:
    """
    words
    """

    def __init__(self, x_input, y_input, test_size=0.33, n_estimators=1000):
        self.x_input = x_input
        self.y_input = y_input
        self.test_size = test_size
        self.n_estimators = n_estimators

    def validate(self, test=0, real=0):
        '''

        Parameters
        ----------
        test : TYPE, optional
            DESCRIPTION. The default is 0.
        real : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        TYPE
            DESCRIPTION.
        predict : TYPE
            DESCRIPTION.
        score : TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.

        '''
        if isinstance(test, int) and isinstance(real, int):
            x_train, x_test, y_train, y_test = train_test_split(
                self.x_input, self.y_input, test_size=self.test_size, random_state=67, shuffle=True)
            # defines the random forest regressor
            clf = RandomForestRegressor(n_estimators=self.n_estimators)
            model = clf.fit(x_train, np.ravel(y_train, order="F"))
            predict = model.predict(x_test)
            score = model.score(x_test, y_test)
            print("Model Training Score", score)
            scorecv = cross_val_score(clf, self.x_input,
                                      np.ravel(self.y_input, order="F"), cv=5, scoring="r2")
            print("Cross val mean score", scorecv.mean())
            print("Cross val std", scorecv.std())
            return y_test, predict, score, scorecv.mean()
        # defines the random forest regressor
        clf = RandomForestRegressor(n_estimators=self.n_estimators)
        model = clf.fit(self.x_input, np.ravel(self.y_input, order="F"))
        predict = model.predict(test)
        score = model.score(test, real)
        print("Model Training Score", score)
        scorecv = cross_val_score(clf, self.x_input, np.ravel(
            self.y_input, order="F"), cv=5, scoring="r2")
        print("Cross val mean score", scorecv.mean())
        print("Cross val std", scorecv.std())
        return real, predict, score, scorecv.mean()

    def predict(self, test, real):
        '''

        Parameters
        ----------
        test : TYPE
            DESCRIPTION.

        Returns
        -------
        predict : TYPE
            DESCRIPTION.

        '''
        clf = RandomForestRegressor(
            n_estimators=self.n_estimators)  # defines the random forest regressor
        model = clf.fit(self.x_input, np.ravel(self.y_input, order="F"))
        predict = model.predict(test)
        score = model.score(test, real)
        return real, predict, score


class NNModel:
    """
    words
    """

    def __init__(self, x_input, y_input):
        """


        Parameters
        ----------
        x_input : TYPE
            DESCRIPTION.
        y_input : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.x_input = x_input
        self.y_input = y_input

    def MLP(self, test, real=None, cross_val = False, pred = True):
        """


        Parameters
        ----------
        test : TYPE
            DESCRIPTION.
        real : TYPE
            DESCRIPTION.

        Returns
        -------
        predict : TYPE
            DESCRIPTION.
        score : TYPE
            DESCRIPTION.

        """
        if pred is True and cross_val is False:
            mpl = MLPRegressor(max_iter = 100000, solver = "lbfgs")
            model = mpl.fit(self.x_input, np.ravel(self.y_input, order="F"))
            predict = model.predict(test)
            # print(model.score(test,real))
            return predict

        if cross_val is False and pred is False:
            mpl = MLPRegressor(max_iter = 100000, solver = "lbfgs")
            model = mpl.fit(self.x_input, np.ravel(self.y_input, order="F"))
            predict = model.predict(test)
            score = model.score(test, real)
            return real, predict, score
        
        if cross_val is True and pred is False:
            mpl = MLPRegressor(max_iter = 10000)
            model = mpl.fit(self.x_input, np.ravel(self.y_input, order="F"))
            predict = model.predict(test)
            score = model.score(test, real)
            scorecv = cross_val_score(mpl, self.x_input,
                                      np.ravel(self.y_input, order="F"), cv=5, scoring="r2")
            print("Cross val mean score", scorecv.mean())
            print("Cross val std", scorecv.std())
        if cross_val is True and pred is True:
            print("Error: Both cross_val and pred can not be true")


    def Classifier(self, test, real=None, cross_val = False, pred = True):
        if cross_val is False and pred is False:
           mpl = MLPClassifier(max_iter = 10000, activation="relu", solver = "adam", hidden_layer_sizes=(200,2) )
           model = mpl.fit(self.x_input, np.ravel(self.y_input, order="F"))
           predict = model.predict(test)
           score = model.score(test, real)
           scorecv = cross_val_score(mpl, self.x_input,
                                  np.ravel(self.y_input, order="F"), cv=10, scoring="r2")
           print(scorecv)
           return real, predict, score
        if cross_val is False and pred is True:
            mpl = MLPClassifier(max_iter = 10000, activation="relu", solver = "adam", hidden_layer_sizes=(200,2) )
            model = mpl.fit(self.x_input, np.ravel(self.y_input, order="F"))
            predict = model.predict(test)
            return predict
        
def plot(x_input, y_input, score, metric, color):
    '''


    Parameters
    ----------
    x_input : TYPE
        DESCRIPTION.
    y_input : TYPE
        DESCRIPTION.
    metric : TYPE
        DESCRIPTION.
    color : TYPE
        DESCRIPTION.
    score : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    plt.figure()
    plt.scatter(
        x=x_input,
        y=y_input,
        color=color)
    plt.plot(x_input, x_input, color="black")
    uni = ["\u03B5", "\u039A", "Q"]  # Eps_eff, Kappa, Q
    if metric == "Eps_eff":
        metric = "$\u03B5_{eff}$"
    if metric == "kappa":
        metric = uni[1]
    if metric == "QHS":
        metric = uni[2]
    if metric == "delta_iota_H":
        metric = r"$\Delta\iota_H$"
    plt.xlabel(f"{metric} True data")
    plt.ylabel(f"{metric} Prediction data")
    plt.title(f"{metric} True vs Prediction")
    plt.text(x = min(x_input), y = max(y_input), s="R$^2$: "+ str('%.5f' % score), horizontalalignment='left', verticalalignment='center')
