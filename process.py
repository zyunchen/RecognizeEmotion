from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.io import loadmat
import mne
from sklearn.preprocessing import PolynomialFeatures 



# Parameters
fs = 600       # Hz
duration = 3   # s
n_channels = 62
ch_names = ['MEG_' + str(i_ch + 1) for i_ch in range(n_channels)]
ch_types = ['mag']  * n_channels


samplesfile = loadmat('seed/eeg_raw_data/1/10_20151014.mat')
print(samplesfile.keys())
samples = samplesfile['tyc_eeg3']  #extract numpy array from the dict
sfreq = 256
info = mne.create_info(ch_names,sfreq)
raw = mne.io.RawArray(samples,info)

#a band-pass filter between 1 and 75 Hz was applied to filter the unrelated artifacts
raw_new = raw.copy().filter(1,75,'all')

# downsampling to 200HZ
raw_downsampled = raw_new.copy().resample(sfreq=200)


raw_downsampled.plot()
print('数据集的形状为', raw_new.get_data().shape)
print('number of channels', raw_new.info.get('nchan'))


