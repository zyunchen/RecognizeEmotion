from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  # plotting
import numpy as np  # linear algebra
import os  # accessing directory structure
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.io import loadmat
from scipy.stats import differential_entropy, norm
import mne
import sys
from sklearn.preprocessing import PolynomialFeatures

# Parameters
fs = 600  # Hz
duration = 3  # s
n_channels = 62
ch_names = ['MEG_' + str(i_ch + 1) for i_ch in range(n_channels)]
ch_types = ['mag'] * n_channels

sfreq = 256

def load_data(path):
    samplesfile = loadmat(path)  # 'seed/eeg_raw_data/1/10_20151014.mat'

    datas = []
    limit = 16  # The first 16 tries for training data.
    for key in samplesfile:
        if key.startswith("_"): continue
        if limit <= 0: break
        limit -= 1
        print("load data: ", key)
        samples = samplesfile[key]

        info = mne.create_info(ch_names, sfreq)
        raw = mne.io.RawArray(samples, info)

        # a band-pass filter between 1 and 75 Hz was applied to filter the unrelated artifacts
        raw_new = raw.copy().filter(1, 75, 'all') #TODO: why copy?

        # downsampling to 200HZ
        raw_downsampled = raw_new.copy().resample(sfreq=200) #TODO: why copy?

        datas.append(raw_downsampled)

    return datas


def compute_psd_de(raw_array, picks, fre):
    # compute psd
    psd = raw_array.compute_psd(picks = picks, fmin=fre[0], fmax=fre[1])  # ['MEG_1','MEG_2']

    # compute differential entropy
    de = differential_entropy(raw_array.copy().filter(l_freq = fre[0], h_freq = fre[1], picks = picks).get_data())  # .get_data(picks=picks)
    return psd, de


picks2 = ['MEG_24', 'MEG_32']  # T7 and T8
picks4 = ['MEG_24', 'MEG_32', 'MEG_15', 'MEG_23']  # T7 and T8 and FT7 and FT8
picks6 = ['MEG_24', 'MEG_32', 'MEG_15', 'MEG_23', 'MEG_33', 'MEG_41']  # T7 and T8 and FT7 and FT8 and TP7 and TP8
picks62 = "all"
picks = [picks2, picks4, picks6, picks62]
fres = [(1, 4), (4, 8), (8, 14), (14, 31),
        (31, 50)]  # delta: 1-4Hz; theta: 4-8Hz; alpha: 8-14Hz; beta: 14-31 Hz; and gamma: 31-50Hz。

# paths = ['seed/eeg_raw_data/1/10_20151014.mat']
paths = [
"/1/4_20151111.mat",
"/1/9_20151028.mat",
"/1/8_20151103.mat",
"/1/10_20151014.mat",
"/1/2_20150915.mat",
"/1/11_20150916.mat",
"/1/7_20150715.mat",
"/1/6_20150507.mat",
"/1/15_20150508.mat",
"/1/12_20150725.mat",
"/1/14_20151205.mat",
"/1/3_20150919.mat",
"/1/5_20160406.mat",
"/1/13_20151115.mat",
"/1/1_20160518.mat",
"/3/3_20151101.mat",
"/3/11_20151011.mat",
"/3/2_20151012.mat",
"/3/1_20161126.mat",
"/3/8_20151117.mat",
"/3/5_20160420.mat",
"/3/13_20161130.mat",
"/3/6_20150512.mat",
"/3/10_20151023.mat",
"/3/14_20151215.mat",
"/3/15_20150527.mat",
"/3/7_20150721.mat",
"/3/12_20150807.mat",
"/3/4_20151123.mat",
"/3/9_20151209.mat",
"/2/13_20151125.mat",
"/2/15_20150514.mat",
"/2/14_20151208.mat",
"/2/1_20161125.mat",
"/2/6_20150511.mat",
"/2/7_20150717.mat",
"/2/8_20151110.mat",
"/2/10_20151021.mat",
"/2/9_20151119.mat",
"/2/2_20150920.mat",
"/2/11_20150921.mat",
"/2/5_20160413.mat",
"/2/3_20151018.mat",
"/2/12_20150804.mat",
"/2/4_20151118.mat",
]

psds = []
des = []


for path in paths:
    for data in load_data("seed/eeg_raw_data" + path):
        for pick in picks:
            for fre in fres:
                psd, de = compute_psd_de(data, pick, fre)
                psds.append(psd)
                des.append(de)

print("outcome psds is ",psds)
print("outcome des is ",des)


