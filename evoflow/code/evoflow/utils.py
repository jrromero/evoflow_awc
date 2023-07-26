# -*- coding: utf-8 -*-

#    Rafael Barbudo Lunar, PhD Student
#    Knowledge and Discovery Systems (KDIS)
#    University of Cordoba, Spain

import numpy as np
import os
import pandas as pd
from scipy.io import arff
import sklearn
from sklearn.ensemble import VotingClassifier
from skmultilearn.dataset import load_from_arff
import unicodedata


def is_number(str_num):
    """
    Check whether a string encodes a number

    Parameters
    ----------
    str_num: str
      String that might encodes a number

    Returns
    -------
    True or False depening on whether 'str_num' encodes a number
    """
    try:
        float(str_num)
        return True
    except ValueError:
        pass

    try:
        unicodedata.numeric(str_num)
        return True
    except (TypeError, ValueError):
        pass

    return False


def load_dense_arff(filename):
    """
    Load an arff dataset as a numpy array

    Parameters
    ----------
    filename: str
      Path of the file containing the dataset

    Returns
    -------
    The data itself (X), the labels (y) and metadata
    of the dataset
    """
    data, meta = arff.loadarff(filename)
    attr_names = meta.names()

    X = data[attr_names[:-1]]
    X = np.asarray(X.tolist())
    y = data[attr_names[-1]]

    return X, y, meta


def load_sparse_arff(filename):
    """
    Load a sparse arff dataset as a numpy array

    Parameters
    ----------
    filename: str
      Path of the file containing the dataset

    Returns
    -------
    The data itself (X) and the labels (y)
    """
    X, y = load_from_arff(filename, label_count=1,
                          label_location="end",
                          load_sparse=True)

    X = X.astype(np.float)
    X = X.toarray()
    y = y.toarray()

    return X, y


def preprocess_data(X, attr_meta):
    """
    Preprocess the dataset to be usable by EvoFlow

    Parameters
    ----------
    X: ndarray
      The array data-type representing the samples.
    attr_meta:
      Data about the attributes of X

    Returns
    -------
    The preprocessed dataset.
    """
    X = pd.DataFrame(X, columns=attr_meta.names()[:-1])
    X_num = pd.DataFrame()
    X_cat = pd.DataFrame()

    for index, col in enumerate(X.columns):
        if attr_meta.types()[index] == 'numeric':
            X_num[col] = X[col].astype(np.float64)
        else:
            X_cat[col] = X[col].str.decode("utf-8")

    if not X_cat.empty:
        X_cat = pd.get_dummies(X_cat)

    return pd.concat([X_num, X_cat], axis=1).values


def log_elite(automl, X_train, y_train, X_test, y_test):

    # Compute and save the statistics of the best individual
    best_ind = automl.elite.best_ind()
    best_ind.pipeline.fit(X_train, y_train)
    y_pred = best_ind.pipeline.predict(X_test)
    performance_dic = compute_performance(y_test, y_pred)

    with open(os.path.join(automl.outdir, "best_ind.txt"), "a") as myfile:
        myfile.write(str(y_pred.tolist()) + '\n')
        myfile.write(str(performance_dic) + '\n')
        myfile.write(str(best_ind) + '\n')

    # Compute and save the statistics of the final ensemble
    best_fitness = best_ind.fitness.values[0]
    weights = [ind.fitness.values[0]/best_fitness for ind in automl.elite]
    estimators = [(str(idx), est.pipeline) for idx, est in enumerate(automl.elite)]
    vt = VotingClassifier(estimators, voting="hard", weights=weights)
    vt.fit(X_train, y_train)
    y_pred = vt.predict(X_test)
    performance_dic = compute_performance(y_test, y_pred)

    with open(os.path.join(automl.outdir, "best_ensemble.txt"), "a") as myfile:
        myfile.write(str(y_pred.tolist()) + '\n')
        myfile.write(str(performance_dic) + '\n')
        for estimator in estimators:
            myfile.write(str(estimator) + '\n')


def compute_performance(y_test, y_pred):

    return {'accuracy_score': sklearn.metrics.accuracy_score(y_test, y_pred),
            'balanced_accuracy_score': sklearn.metrics.balanced_accuracy_score(y_test, y_pred),
            'macro_precision_score': sklearn.metrics.precision_score(y_test, y_pred, average="macro"),
            'macro_recall_score': sklearn.metrics.recall_score(y_test, y_pred, average="macro"),
            'macro_f1_score': sklearn.metrics.f1_score(y_test, y_pred, average="macro"),
            'micro_precision_score': sklearn.metrics.precision_score(y_test, y_pred, average="micro"),
            'micro_recall_score': sklearn.metrics.recall_score(y_test, y_pred, average="micro"),
            'micro_f1_score': sklearn.metrics.f1_score(y_test, y_pred, average="micro")}
