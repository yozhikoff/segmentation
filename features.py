import os
import numpy as np
from tqdm import tqdm_notebook as tqdm
from PIL import Image
import cv2 as cv
import scipy.ndimage
import pandas as pd
from tqdm import tqdm as tqdm
Image.MAX_IMAGE_PIXELS = None

def get_x_and_y(name):
    x, y = os.path.splitext(name)[0].split('_')[-2:]
    return int(x), int(y)

class NucleiFeatures():

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
                             'ellips': (
                             self.ellips, ['first_axis', 'second_axis', 'ellipse_x', 'ellipse_y', 'ellipse_angle']),
                             'color': (
                             self.color, ['Blue_mean', 'Red_mean', 'Green_mean', 'Blue_std', 'Red_std', 'Green_std'])}
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
            print(f'{self.tif_folder}/{filename}.tif', f'{self.png_folder}/{filename}/image/{filename}.png')
            img = cv.imread(f'{self.tif_folder}/{filename}.tif', -1)
            orig = cv.imread(f'{self.png_folder}/{filename}/images/{filename}.png', -1)

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

    def df(self):
        if self.computed_features is None:
            self.compute()
        df = pd.DataFrame(self.computed_features, columns=self.feature_names)
        return df
