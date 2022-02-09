from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import struct


def import_HK_data(file_path, skip=0):
    file_list = [f for f in listdir(file_path) if isfile(join(file_path, f))]

    file_data = []
    for file in range(skip, len(file_list)):
        file_data.append(pd.read_csv(file_path + '/' + file_list[file], delimiter='\s+', header=0))

    HK_data = {}

    for i in range(len(list(file_data[0]))):
        data_lines = []
        for k in range(len(file_data)):
            data_lines.append(file_data[k][list(file_data[0])[i]])
        HK_data[list(file_data[0])[i]] = np.concatenate((data_lines))

    return HK_data, list(HK_data)


def import_binary_LIF_cts(file_path, dtype='>i2', skip=0):
    file_list = [f for f in listdir(file_path) if isfile(join(file_path, f))]

    channel_format = {'sig_counts': 0, 'ref_counts': 1, 'seed_LD_current': 2
        , 'laser_pwr_PT0': [3, 4], 'time_ms': [7, 8], 'seed_LD_mode': 9}
    binary_data = {}
    channel_count = 10

    for dict_key in channel_format.items():
        binary_data[dict_key[0]] = []

    for file in range(skip, len(file_list)):
        data = np.fromfile(file_path + '/' + file_list[file], dtype=dtype)

        frames = np.array(data)
        decimate_arr = [frames[idx::channel_count] for idx in range(channel_count)]

        for channel, channel_ID in channel_format.items():
            if type(0) == type(channel_ID):
                binary_data[channel] = np.concatenate((binary_data[channel], decimate_arr[channel_ID]))
            if type([]) == type(channel_ID):
                hi = decimate_arr[channel_ID[0]] * 65536
                lo = np.where(decimate_arr[channel_ID[1]] < 0
                              , decimate_arr[channel_ID[1]] + 65536, decimate_arr[channel_ID[1]])
                binary_data[channel] = np.concatenate((binary_data[channel], lo + hi))

    binary_data['sig_counts_norm'] = binary_data['sig_counts'] / (binary_data['laser_pwr_PT0'] / 100000)

    return binary_data
