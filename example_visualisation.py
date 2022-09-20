import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from urllib import request
from eeg_viz import plot_eeg_dataset


# Helper function to rename columns
def reformat_name(name):
    '''
    reformat from XX.X.band.x.channel to band.channel
    '''
    band, _, channel = name[5:].split(sep='.')
    return f'{band}.{channel}'


# Download dataset
url = 'https://osf.io/download/d7ye5/'
dst_path = 'dataset.csv'
if not os.path.exists(dst_path):
    request.urlretrieve(url, dst_path)

# Read dataset
df = pd.read_csv(dst_path)

# Read channels info
channels_fp = 'EEG_channels.csv'
channels = pd.read_csv(channels_fp)
channels.set_index('channel', inplace=True)

# Pre-process data for visualisation
# Get only disorders data and band-channel PSDs
mis = df.isna().sum()
sep_col = mis[mis == df.shape[0]].index[0]
df = df.loc[:, 'main.disorder':sep_col].drop(sep_col, axis=1)
# Rename column names
reformat_vect = np.vectorize(reformat_name)
new_colnames = np.concatenate((df.columns[:2], reformat_vect(df.columns[2:])))
df.set_axis(new_colnames, axis=1, inplace=True)
# mean powers per main disorder
main_mean = df.groupby('main.disorder').mean().reset_index()
# list of bands
bands = ['delta', 'theta', 'alpha', 'beta', 'highbeta', 'gamma']
# convert from wide to long
main_mean = pd.wide_to_long(main_mean, bands, ['main.disorder'], 'channel', sep='.', suffix=r'\w+')
# Visualising
conds = ['Healthy control',
         'Schizophrenia',
         'Mood disorder',
         'Anxiety disorder',
         'Obsessive compulsive disorder',
         'Addictive disorder',
         'Trauma and stress related disorder']
conds_labs = [x.replace('disorder', '') for x in conds]
fig, subfigs = plot_eeg_dataset(main_mean, channels, conditions_ordered=conds, condition_labels=conds_labs,
                                gwidth=2.5, cb_pos=(0.85, 0.1))
fig.savefig('plot.png', bbox_inches='tight')
