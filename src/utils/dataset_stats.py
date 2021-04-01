"""Functions used for generating the band statistics and the fmask label
statistics csv files to be used during the training. """

import csv
import glob
import os

import h5py
import numpy as np


def _write_to_csv(csv_path, fields, values_dict):
    """Writes the given values_dict to csv file.
    If the file does not already exist, a header entry is also added using the
    fields argument."""

    if os.path.exists(csv_path):
        mode = 'a'
    else:
        mode = 'w'
    with open(csv_path, mode=mode) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        if mode == 'w':
            writer.writeheader()
        writer.writerow(values_dict)


def _make_label_entry(filename, fmask_count, label_csv_path):
    """Generates csv entry for each row of the label stats file"""
    fields = ['FILENAME', 'NONE_F', 'CLEAR_F',
              'CLOUD_F', 'SHADOW_F', 'ICE_F',
              'WATER_F']
    values = [filename]
    values.extend(fmask_count)
    values_dict = dict(zip(fields, values))
    _write_to_csv(label_csv_path, fields, values_dict)


def _count_label(label_mask):
    """Counts the number pixels for each class label"""
    labels, counts = np.unique(label_mask, return_counts=True)
    count_array=np.zeros(6,dtype=np.uint64)
    for label, countr in zip(labels, counts):
        count_array[label] = countr
    return count_array


def _get_band_info(data):
    """Computes statistics for each band"""
    bands = data[:, :, :13]
    band_mean = np.mean(bands, axis=(0, 1))
    band_std = np.std(bands, axis=(0, 1))
    band_min = np.min(bands, axis=(0, 1))
    band_max = np.max(bands, axis=(0, 1))

    return [band_mean, band_std, band_min, band_max]


def _make_band_entry(filename, stats, band_csv_path):
    """Generates csv entry for each row of the band stats file"""
    N_BANDS = 13
    fields = ['FILENAME', ]
    for stat_name in ['mean', 'std', 'min', 'max']:
        fields.extend(['{}_{}'.format(stat_name, i) for i in range(N_BANDS)])
    values = [filename]
    for i in range(len(stats)):
        values.extend(stats[i].tolist())

    values_dict = dict(zip(fields, values))
    _write_to_csv(band_csv_path, fields, values_dict)


def save_stats(h5_path):
    """Main function that generates the statistics csv files for all the h5 files
    in the given path """
    current_path = os.path.abspath(h5_path)

    H5_files = glob.glob(os.path.join(current_path, "*.h5"))
    label_csv_path = os.path.join(current_path, 'label_stats.csv')
    band_csv_path = os.path.join(current_path, 'band_stats.csv')

    if os.path.exists(label_csv_path):
        os.remove(label_csv_path)
    if os.path.exists(band_csv_path):
        os.remove(band_csv_path)

    total_fmask_count = np.zeros(6, dtype=np.uint64)
    full_dataset_stats = {
        'mean': np.zeros(13),
        'std': np.zeros(13),
        'min': 500000 * np.ones(13),
        'max': np.zeros(13),
        'count': 0
    }

    for h5_file_path in H5_files:
        with h5py.File(h5_file_path, 'r') as hf:
            spectral_image = hf.get('data')[:]
        filename = os.path.basename(h5_file_path)
        fmask_count = _count_label(spectral_image[:, :, 13])

        band_stats = _get_band_info(spectral_image)

        _make_band_entry(filename, band_stats, band_csv_path)
        _make_label_entry(filename, fmask_count, label_csv_path)

        total_fmask_count += fmask_count
        full_dataset_stats['mean'] += band_stats[0]
        full_dataset_stats['std'] += np.square(band_stats[1])
        full_dataset_stats['min'] = np.minimum(band_stats[2],
                                            full_dataset_stats['min'])
        full_dataset_stats['max'] = np.maximum(band_stats[3],
                                            full_dataset_stats['max'])
        full_dataset_stats['count'] += 1


    # Aggregate complete dataset statistics
    full_band_list = [full_dataset_stats['mean'] / full_dataset_stats['count'],
                      np.sqrt(full_dataset_stats['std'] / full_dataset_stats['count']),
                      full_dataset_stats['min'], full_dataset_stats['max']]

    _make_band_entry('COMPLETE', full_band_list, band_csv_path)


if __name__ == '__main__':
    pass
