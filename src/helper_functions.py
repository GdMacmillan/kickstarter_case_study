# standard imports
import glob
import os

# third party imports
import pandas as pd

def load_dataset():

    path = 'data'
    all_files = glob.glob(os.path.join(path, "*.csv"))

    df_from_each_file = (pd.read_csv(f) for f in all_files)
    concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)
    return concatenated_df
