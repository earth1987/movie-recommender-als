"""Custom code that gets used across more than notebook"""
import pandas as pd
from random import sample
import scipy
import numpy as np
from scipy.stats import f_oneway

def perc_func(df, column):
    """Outputs the proportional breakdown of values for a given column in a dataframe."""
    summary = pd.DataFrame(
        dict(count=df[column].value_counts(),
             perc=round(df[column].value_counts(normalize=True)*100,1)
            )
    )
    # summary.loc['All'] = [df[column].value_counts().sum(), 
    #                       df[column].value_counts(normalize=True).sum()*100]

    return summary

def retained_func(x, measure):
    """Calculate retention outcome counts and percentages for a given measure. 
    Where x is the raw results DataFrame and measure is either retention_1 or retention_7
    """
    df = x.groupby(by=['version', measure])['userid'].count().rename('count')
    df.index = df.index.rename(['version', 'outcome'])
    df = df.reset_index(level='outcome').join(
        x.groupby(by=['version'])['userid'].count().rename('total'))
    df['perc'] = round(df['count'] / df['total'] * 100, 2)
    df = df.reset_index().pivot(index='outcome', columns='version', values='perc')
    df['diff'] = df['gate_30'] - df['gate_40']
    return df


def perm_func(x, n_A, n_B):
    """Compute difference in metric after reconstruction of A/B from shuffled x"""
    n = n_A + n_B
    x = x.sample(frac = 1) # shuffle
    idx_A = set(sample(range(n), n_A))
    idx_B = set(range(n)) - idx_A
    return x.loc[list(idx_A)].mean() - x.loc[list(idx_B)].mean()


def cramers_v(vars, df):
    """Compute cramers v between categorical variables for a given dataframe df.
    This calculation assumes each dataframe element consists of a (var1, var2) tuple.
    """
    var1, var2 = vars
    contigency_table = pd.crosstab(index=df[var1],
                                   columns=df[var2])
    X2 = scipy.stats.chi2_contingency(contigency_table)
    chi_stat = X2[0]
    N = len(df)
    min_dim = (min(contigency_table.shape)-1)
    return np.sqrt((chi_stat/N) / min_dim)


def oneway_anova(vars, df):
    """Compute one way ANOVA between a categorical and numeric variable for a given dataframe df.
    This calculation assumes each dataframe element consists of a (categorical var, numerical var) tuple.
    """
    cat, num = vars
    res = f_oneway(*tuple(
        [df[num].groupby(df[cat]).get_group(val) for val in df[cat].unique()]))
    return res.pvalue


def point_bs(vars, df):
    """Compute point biserial correlation between a dichotomous categorical variable and a numeric variable for a given dataframe df.
    This calculation assumes each dataframe element consists of a (categorical var, numerical var) tuple.
    """
    var1, var2 = vars
    res = scipy.stats.pointbiserialr(df[var1], df[var2])
    return res.statistic