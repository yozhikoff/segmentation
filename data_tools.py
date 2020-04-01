import numpy as np
import os
from pathlib import Path
from distutils import dir_util
import subprocess
import cv2 as cv
from PIL import Image
from matplotlib import pyplot as plt

Image.MAX_IMAGE_PIXELS = None


def get_x_and_y(name):
    x, y = os.path.splitext(name)[0].split('_')[-2:]
    return int(x), int(y)


def split_image(img, x_tiles_cnt=None, y_tiles_cnt=None, x_tile_size=None, y_tile_size=None, base='img'):
    '''
    Splits an image array to smaller tiles for further segmentation.
    Specify tiles count OR tiles size. 
    X axis means the arr.shape[1] coordinate, be careful!
    Tile names are used to restore the initial image after segmentation.

    Parameters
    ----------
    img : numpy ndarray
        The input image.
    x_tiles_cnt : integer
        Number of tiles along the x axis of img.
    y_tiles_cnt : integer
        Number of tiles along the y axis of img.
    x_tiles_size : integer
        Size of tile along x axis.
    y_tiles_size : integer
        Size of tile along y axis.
    base : str
        Base for tile names.

    Returns
    -------
    tiles : list
        List of tiles.
    tile_names : list
        List of tile names.
    '''
    if (x_tile_size is not None) and (y_tile_size is not None):
        x_tiles_cnt = img.shape[1] // x_tile_size
        y_tiles_cnt = img.shape[0] // y_tile_size

    if (x_tiles_cnt is not None) and (y_tiles_cnt is not None):
        x_ticks = np.linspace(0, img.shape[1], x_tiles_cnt + 1).astype(int)
        y_ticks = np.linspace(0, img.shape[0], y_tiles_cnt + 1).astype(int)

    else:
        raise Exception('Specify tiles count OR tiles size.')

    tiles = []
    tile_names = []

    for x_num, x in enumerate(zip(x_ticks[:-1], x_ticks[1:])):
        for y_num, y in enumerate(zip(y_ticks[:-1], y_ticks[1:])):
            tiles.append(img[y[0]:y[1], x[0]:x[1]])
            tile_names.append(f'{base}_{x_num}_{y_num}')
    return tiles, tile_names


def prepare_test_data(tiles, tile_names, base_dir, force=False):
    '''
    Saves data in the proper way.
    
    Parameters
    ----------
    tiles : list
        List of tiles.
    tile names : list
        List of tile names.
    base_dir : str
        Full path to base directory, 'full/path/../data_test' in normal case.
    force : bool
        Rewrite existing files

    Returns
    -------
    None
    '''

    base_dir = Path(base_dir)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    if not force and len(os.listdir(base_dir)) > 0:
        raise ValueError(f'base_dir {base_dir} is not empty, use force=True option if you want to rewrite files')
    elif len(os.listdir(base_dir)):
        dir_util.remove_tree(str(network_dir / 'predictions'))
        os.makedirs(base_dir)

    for tile, name in zip(tiles, tile_names):
        if tile.max() <= 1:
            tile = (tile * 255).astype(np.uint8)
        os.mkdir(base_dir / name)
        os.mkdir(base_dir / name / 'images')
        cv.imwrite(str(base_dir / name / 'images' / f'{name}.png'), tile)


def restore_image(work_dir, tiff=False):
    '''
    Restores the initial image.
    
        Parameters
    ----------
    work_dir : str
        Full path to directory with files.
    tiff : bool
        Is the target image a multilayer tiff or not

    Returns
    -------
    img : numpy ndarray
        Initial image
    
    '''

    work_dir = Path(work_dir)

    file_names = sorted(os.listdir(work_dir), key=lambda x: get_x_and_y(x)[::-1])
    coords = np.array([get_x_and_y(n) for n in file_names])
    x_max, y_max = coords.max(axis=0)

    if tiff:
        tiles = {}
        max_number = int(0)
        for n in file_names:
            tmp = cv.imread(str(work_dir / n), -1)
            tmp = (tmp + max_number) * (tmp > 0)
            max_number = tmp.max()
            tiles[get_x_and_y(n)] = tmp.copy()
    else:
        tiles = {get_x_and_y(n): cv.imread(str(work_dir / n), -1) for n in file_names}

    long_tiles = []

    for y in range(y_max + 1):
        long_tiles.append([])
        for x in range(x_max + 1):
            long_tiles[-1].append(tiles[(x, y)])

    long_tiles = [np.hstack(i) for i in long_tiles]

    return np.vstack(long_tiles)


def perform_segmentation(full_img_path, sample_dir, network_dir, force=False):
    network_dir = Path(network_dir)
    try:
        full_img = cv.imread(full_img_path, -1)
    except cv.error:
        full_img = plt.imread(full_img_path)
    tiles, tile_names = split_image(img=full_img, x_tile_size=1000, y_tile_size=1000)
    prepare_test_data(tiles, tile_names, sample_dir, force=force)

    try:
        dir_util.remove_tree(str(network_dir / 'data_test'))
    except:
        pass
    os.mkdir(str(network_dir / 'data_test'))
    dir_util.copy_tree(sample_dir, str(network_dir / 'data_test'));

    try:
        dir_util.remove_tree(str(network_dir / 'predictions'))
    except:
        pass

    try:
        dir_util.remove_tree(str(network_dir / 'albu/results_test'))
    except:
        pass

    subprocess.run(f"cd {network_dir} && bash 'predict_test.sh'", shell=True)
    dir_util.copy_tree(str(network_dir / 'predictions'), str(Path(sample_dir)) + '_segmented');