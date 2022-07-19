import os
from multiprocessing import Pool as ProcessPool

import cv2 as cv
import numpy as np
import pandas as pd
import scipy.ndimage
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


def get_x_and_y(name):
    x, y = os.path.splitext(name)[0].split('_')[-2:]
    return int(x), int(y)


def flatten_list(l):
    return [item for sublist in l for item in sublist]


class NucleiFeatures:

    def position(self, img, orig, **kwargs):
        x, y = scipy.ndimage.measurements.center_of_mass(img)
        x = x + self.x_min + kwargs['x_tile'] * img.shape[0]
        y = y + self.y_min + kwargs['y_tile'] * img.shape[1]
        return [x, y]

    def ellips(self, img, orig, **kwargs):
        try:
            cont = cv.findContours(img.astype(np.uint8).T.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[1][0][:, 0, :]
            ellipse_center, axles, angle = cv.fitEllipse(cont)
            x, y = ellipse_center
            x = x + self.x_min + kwargs['x_tile'] * img.shape[0]
            y = y + self.y_min + kwargs['y_tile'] * img.shape[1]
            if axles[1] > 100:
                axles = (30, 30)
            return [*axles, x, y, angle]
        except:
            return [0] * 5

    def size(self, img, orig, **kwargs):
        return [img.sum()]

    def color(self, img, orig, **kwargs):
        cell_pixels = orig[img]
        return [*cell_pixels.mean(axis=0), *cell_pixels.std(axis=0)]

    def __init__(self, tif_folder, png_folder, features, x_min=0, y_min=0):
        self.tif_folder = tif_folder
        self.png_folder = png_folder
        self.computed_features = None
        self.feature_dict = {'position': (self.position, ['x', 'y']),
                             'size': (self.size, ['size']),
                             'ellipse': (
                                 self.ellips, ['first_axis', 'second_axis', 'ellipse_x', 'ellipse_y', 'ellipse_angle']),
                             'color': (
                                 self.color,
                                 ['Blue_mean', 'Red_mean', 'Green_mean', 'Blue_std', 'Red_std', 'Green_std']),
                             'color_gray': (
                                 self.color,
                                 ['Color_mean', 'Color_std'])
                             }
        if features == 'all':
            self.features = self.feature_dict.keys()
        else:
            self.features = features

        self.x_min = x_min
        self.y_min = y_min

    @property
    def feature_names(self):
        names = []
        for f in self.features:
            names += self.feature_dict[f][1]
        return names

    def compute(self):
        self.computed_features = []
        base_names = [os.path.splitext(i)[0] for i in os.listdir(self.tif_folder)]
        for filename in tqdm(base_names):
            img = cv.imread(f'{self.tif_folder}/{filename}.tif', -1)
            orig = cv.imread(f'{self.png_folder}/{filename}/images/{filename}.png', 1)

            img = np.rot90(img, k=3)
            img = np.flip(img, axis=1)
            orig = np.rot90(orig, k=3)
            orig = np.flip(orig, axis=1)
            for i in range(1, img.max()):
                tmp_img = (img == i)
                tmp = []
                x_tile, y_tile = get_x_and_y(filename)
                for f in self.features:
                    tmp += self.feature_dict[f][0](tmp_img, orig, x_tile=x_tile, y_tile=y_tile)
                self.computed_features.append(tmp)
        return self

    def compute_multipricess(self, n_workers=10):

        orig_list = []
        img_list = []
        filenames = []
        base_names = [os.path.splitext(i)[0] for i in os.listdir(self.tif_folder)]
        for filename in base_names:
            img = cv.imread(f'{self.tif_folder}/{filename}.tif', -1)
            orig = cv.imread(f'{self.png_folder}/{filename}/images/{filename}.png', 1)

            img = np.rot90(img, k=3)
            img = np.flip(img, axis=1)
            orig = np.rot90(orig, k=3)
            orig = np.flip(orig, axis=1)

            orig_list.append(orig)
            img_list.append(img)
            filenames.append(filename)

        global compute_one

        def compute_one(data):
            computed_features = []
            img = data[0]
            orig = data[1]
            filename = data[2]

            for i in range(1, img.max()):
                tmp_img = (img == i)
                tmp = []
                x_tile, y_tile = get_x_and_y(filename)
                for f in self.features:
                    tmp += self.feature_dict[f][0](tmp_img, orig, x_tile=x_tile, y_tile=y_tile)
                computed_features.append(tmp)
            return computed_features

        with ProcessPool(n_workers) as p:
            result = list(tqdm(p.imap(compute_one, zip(img_list, orig_list, filenames)), total=len(orig_list)))
            self.computed_features = flatten_list(result)

        return self

    def df(self):
        if self.computed_features is None:
            self.compute()
        df = pd.DataFrame(self.computed_features, columns=self.feature_names)
        return df

class FastNucleiFeatures:
    """
    Class for fast nuclear feuture calculations using sparse matrices. Memory intesive but speed up factor is ~1000x
    """
    def __init__(self, seg, orig, features='all', x_min=0, y_min=0, fast=True):
        """
        Class constructor, takes seg (predicted enumerated nuclei tiff) and orig (original image for colour calcluations)
        """
        
        self.seg = seg
        self.orig = orig
        self.coo = coo_matrix(seg)
        self.computed_features = None
        self.fast = fast
        if self.fast:
            self.indices = {i+1:ind for i,ind in enumerate(self.get_indices_sparse(self.seg))}
        self.feature_dict = {'position': (self.position, ['x', 'y']),
                             'size': (self.size, ['size']),
                             'ellipse': (
                                 self.ellips, ['first_axis', 'second_axis', 'ellipse_x', 'ellipse_y', 'ellipse_angle']),
                             'color': (
                                 self.color,
                                 ['Blue_mean', 'Red_mean', 'Green_mean', 'Blue_std', 'Red_std', 'Green_std'])
                             }
        if features == 'all':
            self.features = self.feature_dict.keys()
        else:
            self.features = features
        self.x_min = x_min
        self.y_min = y_min

    def position(self, img, orig, **kwargs):
        x, y = scipy.ndimage.measurements.center_of_mass(img)
        x = x + self.x_min + kwargs['x_nucl']
        y = y + self.y_min + kwargs['y_nucl']
        return [x, y]

    def ellips(self, img, orig, **kwargs):
        try:
            cont = cv.findContours(img.astype(np.uint8).T.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[1][0][:, 0, :]
            ellipse_center, axles, angle = cv.fitEllipse(cont)
            x, y = ellipse_center
            x = x + self.x_min + kwargs['x_nucl'] 
            y = y + self.y_min + kwargs['y_nucl']
            if axles[1] > 100:
                axles = (30, 30)
            return [*axles, x, y, angle]
        except:
            return [0] * 5

    def size(self, img, orig, **kwargs):
        return [img.sum()]
    
    def color(self, img, orig, **kwargs):
        cell_pixels = orig[img]
        return [*cell_pixels.mean(axis=0), *cell_pixels.std(axis=0)]

    @property
    def feature_names(self):
        names = []
        for f in self.features:
            names += self.feature_dict[f][1]
        return names
    
    def compute_M(self, data):
        cols = np.arange(data.size)
        raveled = data.ravel()
        cols = cols[raveled != 0]
        raveled = raveled[raveled != 0]
        return csr_matrix((cols, (raveled, cols)),
                          shape=(data.max()+1, data.size))

    def get_indices_sparse(self, data):
        M = self.compute_M(data)
        return [np.unravel_index(row.data, data.shape) for row in M[1:]]

    def compute(self):
        self.computed_features = []
        img = self.seg
        for i in tqdm.tqdm(range(1, img.max())):
            if self.fast:
                sub = self.indices[i] 
            else:
                coo_id = self.coo.data == i
                sub = np.stack((self.coo.row[coo_id], self.coo.col[coo_id]))
            
            upright = np.max(sub, axis=1)
            downleft = np.min(sub, axis=1)
            cut = img[downleft[0]:(upright[0]+1), downleft[1]:(upright[1]+1)]
            orig_cut = self.orig[downleft[0]:(upright[0]+1), downleft[1]:(upright[1]+1)]
            tmp_img = cut == i
            tmp = []
            for f in self.features:
                tmp += self.feature_dict[f][0](tmp_img, orig_cut, x_nucl=downleft[0], y_nucl=downleft[1])
            self.computed_features.append(tmp)
        return self

    def df(self):
        if self.computed_features is None:
            self.compute()
        df = pd.DataFrame(self.computed_features, columns=self.feature_names)
        return df