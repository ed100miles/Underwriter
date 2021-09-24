import csv
import matplotlib.pyplot as plt
import matplotlib
from numpy.lib.function_base import average
import pandas as pd
from pathlib import Path
import numpy as np

# Import df to analyse:
from prepare_data import df

# General distributions:

def show_gen_distributions(df):
    df.hist(bins=50)
    df.hist(bins=50)
    plt.show()
# show_gen_distributions(df)

# Check Correlations:

def show_pos_neg_correlation():
    num_column_list = [column for column in df.columns if df[column].dtype == int or df[column].dtype == float]
    print(df.corr()['CLAIM_FLAG'].sort_values(ascending=False))
    plt.matshow(df.corr(), cmap=plt.cm.RdYlBu, norm=matplotlib.colors.CenteredNorm())
    plt.xticks(range(0,len(num_column_list)), labels=num_column_list, rotation='vertical')
    plt.yticks(range(0,len(num_column_list)), labels=num_column_list, rotation='horizontal')
    plt.colorbar()
    plt.show()
# show_pos_neg_correlation()

def show_abs_correlation():
    num_column_list = [column for column in df.columns if df[column].dtype == int or df[column].dtype == float]
    print(df.corr()['CLAIM_FLAG'].abs().sort_values(ascending=False))
    plt.matshow(df.corr().abs(), cmap=plt.cm.Greens)
    plt.xticks(range(0,len(num_column_list)), labels=num_column_list, rotation='vertical')  
    plt.yticks(range(0,len(num_column_list)), labels=num_column_list, rotation='horizontal')
    plt.colorbar()
    plt.show()
show_abs_correlation()

# df.describe()

