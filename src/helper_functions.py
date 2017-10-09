# standard imports
import glob
import os

# third party imports
import pandas as pd
import scipy.stats as sp

def load_dataset():

    path = 'data'
    all_files = glob.glob(os.path.join(path, "*.csv"))

    df_from_each_file = (pd.read_csv(f, converters={'category':CustomParser, 'profile':CustomParser}) for f in all_files)
    concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)
    return concatenated_df

def ttest_by(vals, by):
    '''Compute a t-test on a column based on an indicator for which sample the values are in.'''
    vals1 = vals[by]
    vals2 = vals[-by]

    return sp.stats.ttest_ind(vals1, vals2)

def kstest_by(vals, by):
    '''Compute a Kolmogorov-Smirnov on a column based on an indicator for which sample the values are in.'''
    vals1 = vals[by]
    vals2 = vals[-by]

    return sp.stats.ks_2samp(vals1, vals2)

def CustomParser(data):
    import json
    j1 = json.loads(data)
    return j1
