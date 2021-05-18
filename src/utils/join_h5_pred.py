import logging
from itertools import product
from datetime import datetime

from matplotlib import pyplot as plt
import rasterio
import numpy as np
import h5py
import os,glob
import argparse
import cv2
from rasterio import windows

from utils.dir_paths import EXP_DATA_PATH

logger = logging.getLogger('__name__')

def get_args(argv=None):
    """Parses the arguments entered by the user."""

    parser = argparse.ArgumentParser(description="Joins h5 files ")
    parser.add_argument('-p', '--path', help='Path of H5')
    parser.add_argument('-o', '--output', help='Path of H5',default=None)


    return parser.parse_args(argv)



def join_files(h5_folder,output = None,exp=None):
    file_list = glob.glob(os.path.join(h5_folder, '*.h5'))
    file_list = list(set(i.rsplit("_", 1)[0] for i in file_list))

    for parent in file_list:
        window_dict = np.load(parent + '_META.npy', allow_pickle=True).item()
        if output and exp:
            postfix = '_LABELS_{}_{}.tif'.format(exp,datetime.now().strftime("%m_%d_%H_%M"))
            destination = os.path.join(output,os.path.basename(parent)+postfix)
        elif output:
            destination = os.path.join(output,os.path.basename(parent)+'_LABELS.tif')
        else:
            destination = parent + '_LABELS.tif'
        logger.info('Joining file in {}'.format(destination))

        metadata = window_dict['parent']
        with rasterio.open(destination, 'w', **metadata) as dst:
            dst.write(np.zeros((5490, 5490), dtype=np.uint8), 1)

        patch_per_tile = int(np.sqrt(len(window_dict)))
        for n_patch, (a, b) in enumerate(product(range(patch_per_tile), repeat=2)):
            # if n_patch>18:
            #     continue
            file_name = parent + '_PATCH{}.h5'.format(n_patch)
            with h5py.File(file_name, 'r') as hf:
                fmask = hf.get('data')[:][:, :, 14]

            west, south, east, north = window_dict[n_patch]['bounds']
            wind = rasterio.windows.from_bounds(west, south, east, north, transform=metadata['transform'])

            with rasterio.open(destination, 'r+', **metadata) as mosaic_raster:
                window_data = mosaic_raster.read(1, window=wind)
                if (a % (patch_per_tile - 1) != 0) and (b % (patch_per_tile - 1) != 0):

                    window_data[3:-3,3:-3] = fmask[3:-3,3:-3]
                    fmask = window_data

                elif a % (patch_per_tile - 1) != 0:
                    window_data[3:-3,:] = fmask[3:-3,:]
                    fmask = window_data

                elif b % (patch_per_tile - 1) != 0:
                    window_data[:, 3:-3] = fmask[:, 3:-3]
                    fmask = window_data

                # fmask[fmask == 0] = 5
                mosaic_raster.write(fmask.astype(np.uint8), 1, window=wind)



if __name__ == '__main__':
    args = get_args()
    join_files(args.path,args.output)


