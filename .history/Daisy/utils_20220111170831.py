from data import NumericalField, CategoricalField, Iterator
from data import Dataset
from synthesizer import VGAN_generator, VGAN_discriminator
from synthesizer import LGAN_generator, LGAN_discriminator, LSTM_discriminator
from synthesizer import DCGAN_generator, DCGAN_discriminator
from synthesizer import V_Train
from random import choice
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import torch

import json

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import random
import math


def to_df(data,dataset):
    samples = data.reshape(data.shape[0], -1)
    samples = samples[:,:dataset.dim]
    samples = samples.cpu()
    sample_table = dataset.reverse(samples.detach().numpy())
    df = pd.DataFrame(sample_table,columns=dataset.columns)
    return df








def compute_kl(real, pred):
    return torch.sum((torch.log(pred + 1e-4) - torch.log(real + 1e-4)) * pred)

def KL_Loss(x_fake, x_real, col_type, col_dim):
    kl = 0.0
    sta = 0
    end = 0
    for i in range(len(col_type)):
        dim = col_dim[i]
        sta = end
        end = sta+dim
        fakex = x_fake[:,sta:end]
        realx = x_real[:,sta:end]
        if col_type[i] == "gmm":
            fake2 = fakex[:,1:]
            real2 = realx[:,1:]
            # column sum
            dist = torch.sum(fake2, dim=0)

            dist = dist / torch.sum(dist)
            real = torch.sum(real2, dim=0)
            real = real / torch.sum(real)
            kl += compute_kl(real, dist)
        else:
            dist = torch.sum(fakex, dim=0)
            dist = dist / torch.sum(dist)
            
            real = torch.sum(realx, dim=0)
            real = real / torch.sum(real)
            
            kl += compute_kl(real, dist)
    return kl


def mean_Loss(x_fake, x_real, col_type, col_dim):
    mean = 0.0
    sta = 0
    end = 0
    for i in range(len(col_type)):
        dim = col_dim[i]
        sta = end
        end = sta+dim
        fakex = x_fake[:,sta:end]
        realx = x_real[:,sta:end]
        if col_type[i] == "gmm":
            fake2 = fakex[:,1:]
            real2 = realx[:,1:]
            dist = torch.mean(fake2, dim=0)
            dist = dist / torch.sum(dist)
            real = torch.mean(real2, dim=0)
            real = real / torch.sum(real)
            mean += torch.sum(abs(real - dist))
        else:
            dist = torch.mean(fakex, dim=0)
            dist = dist / torch.sum(dist)
            
            real = torch.mean(realx, dim=0)
            real = real / torch.sum(real)
            mean += torch.sum(abs(real - dist))
    return mean


def plot_pdf(data,label,bin=10):
    count, bins_count = np.histogram(data, bins=10)
    pdf = count / sum(count)

    plt.plot(bins_count[1:], pdf, label=label)

def plot_cdf(data,label,bin=10):
    count, bins_count = np.histogram(data, bins=10)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    plt.plot(bins_count[1:], cdf, label=label)

def make_compare_plot(datasets,col_name, function, names, title):
    for i in range(len(datasets)):
        data = datasets[i]
        function(data[col_name],names[i])
    plt.title(title)
    plt.legend()
    plt.show()


def convert_type(data,columns):
    data = data[columns].astype('category')
    for col in columns:
        data[col] = data[col].cat.codes
    return data


def make_prediction(response, response_type):
    train_data_y = training_data[response].astype("float64")
    train_data_X = training_data.drop(columns=[response,"label","fnlwgt"]).astype("float64")

    test_data_X = test.drop(columns=[response,"label","fnlwgt"]).astype("float64")
    test_data_y = test[response].astype("float64")


    train_data_sample_X = data_sample.drop(columns=[response,"label","fnlwgt"]).astype("float64")
    train_data_sample_y = data_sample[response].astype("float64")


    # ITS-Linear
    train_its_weak_X = its_weak.drop(columns=[response,"label","fnlwgt"]).astype("float64")
    train_its_weak_y = its_weak[response].astype("float64")

    # origin ITS Loss
    train_its_X = its.drop(columns=[response,"label","fnlwgt"]).astype("float64")
    train_its_y = its[response].astype("float64")


    # ITS-complex
    train_its_complex_X = its_complex.drop(columns=[response,"label","fnlwgt"]).astype("float64")
    train_its_complex_y = its_complex[response].astype("float64")

    # VAE baseline
    train_VAE_X = baseline_VAE.drop(columns=[response,"label","fnlwgt"]).astype("float64")
    train_VAE_y = baseline_VAE[response].astype("float64")

    result = []

    if response_type == "clf":

        clf = xgb.XGBClassifier(eval_metric='mlogloss')

        clf.fit(train_data_X, train_data_y)
        #result.append(cross_val_score(clf, test_data_X,test_data_y, cv=3).mean())
        result.append(clf.score(test_data_X,test_data_y))

        clf.fit(train_data_sample_X, train_data_sample_y)
        result.append(clf.score(test_data_X,test_data_y))

        clf.fit(train_its_weak_X, train_its_weak_y)
        result.append(clf.score(test_data_X,test_data_y))

        clf.fit(train_its_X, train_its_y)
        result.append(clf.score(test_data_X,test_data_y))

        clf.fit(train_its_complex_X, train_its_complex_y)
        result.append(clf.score(test_data_X,test_data_y))

        try:
            clf.fit(train_VAE_X, train_VAE_y)
            result.append(clf.score(test_data_X,test_data_y))
        except:
            result.append(np.nan)


    else:
        clf = xgb.XGBRegressor(eval_metric='mlogloss')

        clf.fit(train_data_X, train_data_y)
        result.append(clf.score(test_data_X,test_data_y))

        clf.fit(train_data_sample_X, train_data_sample_y)
        result.append(clf.score(test_data_X,test_data_y))

        clf.fit(train_its_weak_X, train_its_weak_y)
        result.append(clf.score(test_data_X,test_data_y))

        clf.fit(train_its_X, train_its_y)
        result.append(clf.score(test_data_X,test_data_y))

        clf.fit(train_its_complex_X, train_its_complex_y)
        result.append(clf.score(test_data_X,test_data_y))

        try:
            clf.fit(train_VAE_X, train_VAE_y)
            result.append(clf.score(test_data_X,test_data_y))
        except:
            result.append(np.nan)

    return result