"""Functions used for generating the label
statistics csv files to be used during the training. """

import os, sys
import glob
import numpy as np
import csv
import h5py
import argparse
import logging
from sklearn.metrics import confusion_matrix
logger = logging.getLogger('__name__')


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
    )


def get_args(argv=None):
    """Parses the arguments entered by the user."""

    parser = argparse.ArgumentParser(description="Gives stats of H5 files ")
    parser.add_argument('-p', '--path', help='Path of H5 files',
                        default=os.getcwd())
    parser.add_argument('-m', '--mode', default='train',
                        help='mode for data use')

    return parser.parse_args(argv)




np.set_printoptions(precision=4, suppress=True)


def count_label(label_mask):
    uniques, counts = np.unique(label_mask, return_counts=True)
    label_dict = {i: 0 for i in range(6)}
    for unique, countr in zip(uniques, counts):
        label_dict[unique] = countr
    return label_dict


def get_label_info(data, mode):
    fmask = data[:, :, 13]

    fmask_count = count_label(fmask)
    if mode == 'train':
        true_count = None
        conf_mat = np.zeros((6, 6))
    else:
        true_label = data[:, :, 14]

        true_count = count_label(true_label)
        true_count[0] = 0

        true_label_filtered = true_label[true_label != 0].reshape(-1, )
        fmask_filtered = fmask[true_label != 0].reshape(-1, )

        conf_mat = confusion_matrix(true_label_filtered, fmask_filtered,
                                    labels=[0, 1, 2, 3, 4, 5])

    return fmask_count, true_count, conf_mat


def get_band_info(data):
    bands = data[:, :, :13]
    band_mean = np.mean(bands, axis=(0, 1))
    band_std = np.std(bands, axis=(0, 1))
    band_min = np.min(bands, axis=(0, 1))
    band_max = np.max(bands, axis=(0, 1))

    return [band_mean, band_std, band_min, band_max]


def make_label_entry(filename, fmask_count, val_count, label_csv_path):
    """Generates csv entry for each row of the label stats file"""
    fields = ['FILENAME', 'NONE_F', 'CLEAR_F',
              'CLOUD_F', 'SHADOW_F', 'ICE_F',
              'WATER_F']
    values = [filename]
    values.extend(fmask_count.values())
    if val_count is not None:
        fields.extend(['NONE_T', 'CLEAR_T',
                       'CLOUD_T', 'SHADOW_T', 'ICE_T',
                       'WATER_T'])
        values.extend(val_count.values())
    values_dict = dict(zip(fields, values))

    if os.path.exists(label_csv_path):
        mode = 'a'
    else:
        mode = 'w'
    with open(label_csv_path, mode=mode) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        if mode == 'w':
            writer.writeheader()
        writer.writerow(values_dict)


def make_band_entry(filename, stats, band_csv_path):
    N_BANDS = 13
    fields = ['FILENAME', ]
    for stat_name in ['mean', 'std', 'min', 'max']:
        fields.extend(['{}_{}'.format(stat_name, i) for i in range(N_BANDS)])
    values = [filename]
    for i in range(len(stats)):
        values.extend(stats[i].tolist())

    values_dict = dict(zip(fields, values))

    if os.path.exists(band_csv_path):
        mode = 'a'
    else:
        mode = 'w'
    with open(band_csv_path, mode=mode) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        if mode == 'w':
            writer.writeheader()
        writer.writerow(values_dict)


def main_csv(mode, path):
    current_path = os.path.abspath(path)
    logger.info('Running dataset stats for {} in {} mode'.format(current_path, mode))

    H5_files = glob.glob(os.path.join(current_path, "*.h5"))
    label_csv_path = os.path.join(current_path, 'label_stats.csv')
    band_csv_path = os.path.join(current_path, 'band_stats.csv')
    if os.path.exists(label_csv_path):
        os.remove(label_csv_path)
    if mode == 'train' and os.path.exists(band_csv_path):
        os.remove(band_csv_path)
    stats = np.zeros((6))
    conf_mat = np.zeros((6, 6))

    for h5_file_path in H5_files:
        with h5py.File(h5_file_path, 'r') as hf:
            spectral_image = hf.get('data')[:]
            filename = os.path.basename(h5_file_path)
            fmask_count, true_count, conf_ = get_label_info(spectral_image,
                                                            mode)

            if mode == 'test':
                conf_mat += conf_
                for i in true_count:
                    stats[i] += true_count[i]
            else:
                for i in fmask_count:
                    stats[i] += fmask_count[i]
            make_label_entry(filename, fmask_count, true_count, label_csv_path)

    if mode == 'test':

        if np.sum(conf_mat[1:, 1:]):
            logger.info('Confusion matrix comparing FMask and true labels \n{}'.format(conf_mat.astype(int)))
            fmask_correct = np.sum(conf_mat.diagonal()) / np.sum(conf_mat)
            logger.info('Fmask Agreement with labels=> {:.3%}'.format(fmask_correct))
        else:
            logger.info('No FMask labels found')

        out_string = 'NONE:{:.2%} \t CLEAR:{:.2%} \t CLOUD:{:.2%} \t' \
                     'SHADOW:{:.2%} \t ICE:{:.2%} \t ' \
                     'WATER:{:.2%}'.format(*stats / np.sum(stats))
        logger.info('Test Class Distribution => {}'.format(out_string))


    else:
        out_string = 'NONE:{:.2%} \t CLEAR:{:.2%} \t CLOUD:{:.2%} \t' \
                     'SHADOW:{:.2%} \t ICE:{:.2%} \t ' \
                     'WATER:{:.2%}'.format(*stats / np.sum(stats))
        logger.info('Train Class Distribution => {}'.format(out_string))





if __name__ == '__main__':
    setup_logger()
    args = get_args()
    main_csv(mode=args.mode, path=args.path)
