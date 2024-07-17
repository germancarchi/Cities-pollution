import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency
from scipy.stats import f_oneway

"""def replace_tempo_values(x):
    if "?" in x:
        return np.nan
    else:
        return x"""

def calculate_statiscal_descriptors(df, col):
    mean = np.mean(df[col])
    std = np.std(df[col])
    return mean, std


def remove_outliers(x, mean, std):
    if x < mean - 3 * std:
        return mean - 3 * std
    elif x > mean + 3 * std:
        return mean + 3 * std
    else:
        return x
    

def chi2_test(data, feature, target):
    table = pd.crosstab(data[feature], data[target])
    stat, p, dof, expected = chi2_contingency(table)
    return p


def anova_test(data, feature, target):
    groups = data.groupby(target)[feature].apply(list)
    stat, p = f_oneway(*groups)
    return p

def fill_null_with_mean(col, df):
    mean = np.mean(df[col])
    df[col] = df[col].fillna(mean)
    return df