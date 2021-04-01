import os
import numpy as np
import h5py
import json
import pandas as pd

PIXELS_PER_IMAGE = 549 ** 2
def calculate_file_freq(labels):

    file_freq = np.zeros(6)
    for i in range(6):
        file_freq[i] = np.count_nonzero(labels == i)

    return file_freq




def calculate_MFB_stage(file_list,stage):

    filename_list = [os.path.basename(i) for i in file_list]
    stage_freq = np.zeros((6))
    parent_folder = os.path.dirname(file_list[0])
    if stage == 0:
        # fmask is at index 13 in spectral_data
        label_stats_file = os.path.join(parent_folder,'label_stats.csv')
    else:
        label_stats_file = os.path.join(parent_folder,'label_stats_stage{}.csv'.format(stage))
    label_stats_df = pd.read_csv(label_stats_file)
    stage_stats = label_stats_df[label_stats_df['FILENAME'].isin(filename_list)]

    # [['FILENAME', 'NONE_F',
    #   'CLEAR_F', 'CLOUD_F',
    #   'SHADOW_F', 'ICE_F', 'WATER_F', ]]
    stage_freq = np.array([stage_stats['NONE_F'].sum(),
                       stage_stats['CLEAR_F'].sum(),
                       stage_stats['CLOUD_F'].sum(),
                       stage_stats['SHADOW_F'].sum(),
                       stage_stats['ICE_F'].sum(),
                       stage_stats['WATER_F'].sum()],dtype=float)/PIXELS_PER_IMAGE





    # for file in file_list:
    #     with h5py.File(file, 'r') as hf:
    #         labels = hf.get('data')[:]
    #         if 13 >= labels.shape[-1]:
    #             raise NotImplementedError('Stage labels not found for stage {}'.format(stage))
    #         labels = labels[:, :, 13]
    #
    #         stage_freq += calculate_file_freq(labels)
    np.set_printoptions(suppress=True, precision=2)



    return stage_freq,stage_stats.shape[0]

