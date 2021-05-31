import argparse
import glob
import logging
import os
import random
from tempfile import TemporaryDirectory

import h5py
import numpy as np
from cv2 import resize, INTER_NEAREST, INTER_CUBIC
from rasterio import open as raster_open
from rasterio import windows

from utils.dataset_stats import main_csv
from utils.dir_paths import TRAIN_SAFE_PATH, VALID_SAFE_PATH, PRED_PATH

logger = logging.getLogger(__name__)


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
    )


def get_args(argv=None):
    """Parses the arguments entered by the user."""

    parser = argparse.ArgumentParser(description='Splits SAFE files into h5 '
                                                 'files ')
    parser.add_argument('-p', '--path', help='Path of containing safe folders',
                        default=None)
    parser.add_argument('-m', '--mode', default='train',
                        help='mode for data use')

    return parser.parse_args(argv)


def get_data_size(network_input_size=256, border_width=1):
    # Effective sub-scene size in 20m resolution by subtracting the border
    # width on both sides.
    size_20m = network_input_size - (border_width * 2)

    size_10m = 2 * size_20m

    # To rescale to 20m resolution set Rescale size to half of window size.
    rescale_size = int(size_10m / 2)

    return rescale_size, size_10m


def get_img_paths(safe_path, mode):
    """Returns a sorted list of image files for a given safe folder path. The
    FMASK will be included and the true color image will be excluded from
    the list."""

    # get all Sentinel2 filenames
    img_paths = sorted(
        glob.glob(os.path.join(safe_path, '**', 'IMG_DATA', '*.jp2'),
                  recursive=True))
    # remove true color image
    img_paths = list(filter(lambda x: not x.endswith('TCI.jp2'), img_paths))
    # get fmask
    fmask_path = glob.glob(
        os.path.join(safe_path, '**', 'IMG_DATA', '*FMASK.tif'),
        recursive=True)
    # raise error if fmask not found in train mode
    if len(fmask_path) == 1:
        img_paths.extend(fmask_path)
    elif mode == 'train':
        raise FileNotFoundError('FMask file not found in format *FMASK.tif')
    else:
        # if fmask is not available in test or predict, we create a zero
        # array instead
        logger.warning(
            'No FMASK file found for {}'.format(os.path.basename(safe_path)))
        nofile_path = 'NOFILE' + os.path.basename(img_paths[0]).replace(
            'B01.jp2',
            'FMASK.tif')
        img_paths.append(nofile_path)

    # For validation mode check if true label is available and get its path
    if mode != 'train':
        label_path = glob.glob(
            os.path.join(safe_path, '**', 'IMG_DATA', '*LABELS.tif'),
            recursive=True)
        # raise error if fmask not found in train mode
        if len(label_path) == 1:
            img_paths.extend(label_path)
        else:
            logger.warning(
                'No labelfile found for {}'.format(os.path.basename(safe_path)))
            nofile_path = 'NOFILE' + os.path.basename(img_paths[0]).replace(
                'B01.jp2',
                'LABELS.tif')
            img_paths.append(nofile_path)

    return img_paths


def split_img(img_path, temp_dirpath, overlap=0, network_input_size=256):
    rescale_size, size_10m = get_data_size(network_input_size)
    if img_path.startswith('NOFILE'):

        n_tiles = len(glob.glob(temp_dirpath + '/*_B05.jp2_PATCH*'))
        window_data = np.zeros((rescale_size, rescale_size), dtype=np.uint16)
        for n_patch in range(n_tiles):
            out_path = os.path.join(temp_dirpath,
                                    img_path.replace('NOFILE', '')
                                    + '_B05.jp2_PATCH{}'.format(n_patch))
            np.save(out_path, window_data)
        return None
    else:
        band = raster_open(img_path)
        window_length = size_10m * 10
        window_length1 = size_10m * (10 - overlap)

        top_bound = band.bounds.top - window_length
        right_bound = band.bounds.right - window_length

        # Take the metadata from band 5
        if 'B05' in img_path:
            patch_meta = {}
        else:
            patch_meta = None
        a = 0
        b = 0
        n_patch = 0
        while True:
            left = min(band.bounds.left + (b * window_length1), right_bound)
            bottom = min(band.bounds.bottom + (a * window_length1), top_bound)
            top = bottom + window_length
            right = left + window_length

            polygon_window = windows.from_bounds(left=left,
                                                 bottom=bottom,
                                                 right=right,
                                                 top=top,
                                                 transform=band.transform)

            window_data = band.read(1, window=polygon_window)
            window_transform = windows.transform(polygon_window,
                                                 band.transform)

            if patch_meta is not None:
                meta = {}
                meta['width'] = window_data.shape[1]
                meta['height'] = window_data.shape[0]
                meta['bounds'] = (left, bottom, right, top)
                patch_meta[n_patch] = meta

            if 'LABEL' in img_path or 'FMASK' in img_path:
                label_interpolation = True
            else:
                label_interpolation = False
            window_data, window_transform = fix_window_size(window_data,
                                                            window_transform,
                                                            rescale_size,
                                                            label_interpolation)

            out_path = os.path.join(temp_dirpath,
                                    os.path.basename(
                                        img_path) + '_PATCH{}'.format(
                                        n_patch))
            np.save(out_path, window_data)

            if bottom == top_bound:
                a = 0
                if left == right_bound:
                    break
                else:
                    b += 1
            else:
                a += 1
            n_patch += 1

    return patch_meta


def resize_window(new_size, window_data, window_transform,
                  label_interpolation=False):
    old = window_data.shape[0]
    if label_interpolation:
        interpolation = INTER_NEAREST
    else:
        interpolation = INTER_CUBIC
    window_data = resize(window_data, dsize=(new_size, new_size),
                         interpolation=interpolation)

    # scale image transform
    window_transform = window_transform * window_transform.scale(
        (old / new_size),
        (old / new_size)
    )
    return window_data, window_transform


def fix_window_size(window_data, window_transform, rescale_size,
                    label_interpolation=False):
    if window_data.shape[0] != rescale_size:
        window_data, window_transform = resize_window(rescale_size,
                                                      window_data,
                                                      window_transform,
                                                      label_interpolation)

    return window_data, window_transform


def make_patch(safe_file_list, mode, network_input_size=256):
    rescale_size, size_10m = get_data_size(network_input_size)
    total_files = len(safe_file_list)
    for file_n, safe_path in enumerate(safe_file_list):

        logger.info('[{}/{}] Splitting {}'.format(file_n + 1,
                                                  total_files,
                                                  os.path.basename(safe_path)))

        img_paths = get_img_paths(safe_path, mode)
        if mode == 'predict':
            out_folder_path = safe_path + '_H5'
            safe_filename = os.path.basename(safe_path)
            overlap = 2
        else:
            [parent_path, safe_filename] = safe_path.rsplit('/', 1)
            out_folder_path = parent_path + '_H5'
            overlap = 0
        os.makedirs(out_folder_path, exist_ok=True)

        metadata_dict = {}

        B05_path = [pth for pth in img_paths if 'B05' in pth][0]
        B05_band = raster_open(B05_path)
        metadata = B05_band.meta.copy()
        metadata.update({
            'driver': 'GTiff',
            'dtype': 'uint8'
        })

        metadata_dict['parent'] = metadata
        del B05_band

        with TemporaryDirectory() as td:
            for n_i, img_path in enumerate(img_paths):
                patch_meta = split_img(img_path, td, overlap,
                                       network_input_size)
                if patch_meta:
                    metadata_dict = {**metadata_dict, **patch_meta}
            total_tiles = len(metadata_dict) - 1
            for n_patch in range(total_tiles):
                spectral_data = np.zeros(
                    (rescale_size, rescale_size, len(img_paths)),
                    dtype=np.uint16)
                out_file_name = safe_filename.replace('.SAFE',
                                                      '_PATCH{}.h5'.format(
                                                          n_patch))

                split_img_paths = sorted(
                    glob.glob(
                        os.path.join(td, '*_PATCH{}.npy'.format(n_patch))))
                for idx, split_img_path in enumerate(split_img_paths):
                    window = np.load(split_img_path)
                    spectral_data[:, :, idx] = window

                    labels_file = {'test': 'LABELS.tif', 'train': 'FMASK.tif'}

                    if (mode in labels_file and labels_file[
                        mode] in split_img_path):
                        if np.all(window == 0):
                            out_file_name += '.NIL'
                if not out_file_name.endswith('h5.NIL'):
                    out_file_path = os.path.join(out_folder_path, out_file_name)
                    with h5py.File(out_file_path, 'w') as hf:
                        hf.create_dataset('data', data=spectral_data, )

        if mode == 'predict':
            out_file_name = safe_filename.replace('.SAFE', '_META.npy')
            out_file_path = os.path.join(out_folder_path, out_file_name)
            np.save(out_file_path, metadata_dict)

    if mode != 'predict':
        main_csv(mode=mode, path=out_folder_path)


if __name__ == '__main__':

    # Set a constant seed for reproducibility
    random.seed(42)
    setup_logger()

    args = get_args()

    # When path argument is not given, set path using default paths from  
    # utils/dir_paths.py 
    if not args.path:
        if args.mode == 'train':
            args.path = TRAIN_SAFE_PATH
        elif args.mode == 'test':
            args.path = VALID_SAFE_PATH
        elif args.mode == 'predict':
            args.path = PRED_PATH
        else:
            NotImplementedError()

    safe_parent_path = os.path.abspath(args.path)

    logger.info('Processing files from {}'.format(safe_parent_path))
    safe_file_list = glob.glob(os.path.join(safe_parent_path, '*.SAFE'))

    if len(safe_file_list) == 0:
        # When no safe file is found
        raise FileNotFoundError('Provide path of folder containing safe files '
                                'using --path argument.')

    make_patch(safe_file_list, args.mode)
