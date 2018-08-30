import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from sklearn.preprocessing import MinMaxScaler
from fancyimpute import MICE

def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

def showBoxPlot(data):
    fig, ax = plt.subplots()
    ax.boxplot(data)
    plt.show()

def scaleMinMax(data):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return data

def miceImputation(data):
    imp = MICE()
    if (pd.DataFrame(data).isnull().values.any() == True):
        data = imp.complete(data)
    return data

def show_histogram(data):
    plt.hist(data, normed=True, bins=30)
    plt.show()

def get_higly_correlated_matrix(data):
    return  data.corr().abs()