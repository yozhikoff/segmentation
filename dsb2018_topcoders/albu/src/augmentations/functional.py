import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np
import math
from scipy.ndimage.filters import gaussian_filter
from functools import wraps
import torch
import torchvision.transforms.functional as F


def vflip(img):
    return cv2.flip(img, 0)


def hflip(img):
    return cv2.flip(img, 1)


def random_flip(img, code):
    return cv2.flip(img, code)


def transpose(img):
    return img.transpose(1, 0, 2) if len(img.shape) > 2 else img.transpose(1, 0)


def rot90(img, factor):
    img = np.rot90(img, factor)
    return np.ascontiguousarray(img)


def rotate(img, angle):
    height, width = img.shape[0:2]
    mat = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
    img = cv2.warpAffine(img, mat, (width, height),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return img


def shift_scale_rotate(img, angle, scale, dx, dy):
    height, width = img.shape[:2]

    cc = math.cos(angle/180*math.pi) * scale
    ss = math.sin(angle/180*math.pi) * scale
    rotate_matrix = np.array([[cc, -ss], [ss, cc]])

    box0 = np.array([[0, 0], [width, 0],  [width, height], [0, height], ])
    box1 = box0 - np.array([width/2, height/2])
    box1 = np.dot(box1, rotate_matrix.T) + np.array([width/2+dx*width, height/2+dy*height])

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0, box1)
    img = cv2.warpPerspective(img, mat, (width, height),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)

    return img


def center_crop(img, height, width):
    h, w, c = img.shape
    dy = (h-height)//2
    dx = (w-width)//2
    y1 = dy
    y2 = y1 + height
    x1 = dx
    x2 = x1 + width
    img = img[y1:y2, x1:x2, :]
    return img


def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)


def clipped(func):
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        dtype, maxval = img.dtype, np.max(img)
        return clip(func(img, *args, **kwargs), dtype, maxval)
    return wrapped_function


def shift_hsv(img, hue_shift, sat_shift, val_shift):
    dtype = img.dtype
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int32)
    h, s, v = cv2.split(img)
    h = cv2.add(h, hue_shift)
    h = np.where(h < 0, 255 - h, h)
    h = np.where(h > 255, h - 255, h)
    h = h.astype(dtype)
    s = clip(cv2.add(s, sat_shift), dtype, 255 if dtype == np.uint8 else 1.)
    v = clip(cv2.add(v, val_shift), dtype, 255 if dtype == np.uint8 else 1.)
    img = cv2.merge((h, s, v)).astype(dtype)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img

@clipped
def shift_rgb(img, r_shift, g_shift, b_shift):
    img[...,0] = img[...,0] + r_shift
    img[...,1] = img[...,1] + g_shift
    img[...,2] = img[...,2] + b_shift
    return img

def clahe(img, clipLimit=2.0, tileGridSize=(8,8)):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_LAB2RGB)
    return img_output

def blur(img, ksize):
    return cv2.blur(img, (ksize, ksize))

def median_blur(img, ksize):
    return cv2.medianBlur(img, ksize)

def motion_blur(img, ksize):
    kernel = np.zeros((ksize, ksize))
    xs, ys = np.random.randint(0, kernel.shape[1]), np.random.randint(0, kernel.shape[0])
    xe, ye = np.random.randint(0, kernel.shape[1]), np.random.randint(0, kernel.shape[0])
    cv2.line(kernel, (xs, ys), (xe, ye), 1, thickness=1)
    return cv2.filter2D(img, -1, kernel / np.sum(kernel))

def random_polosa(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if np.mean(gray) < 100:
        empty = np.zeros(img.shape[:2], dtype=np.uint8)
        xs, ys = np.random.randint(0, empty.shape[1]), np.random.randint(0, empty.shape[0])
        xe, ye = np.random.randint(0, empty.shape[1]), np.random.randint(0, empty.shape[0])
        factor = np.random.randint(1, 10) / 3.
        cv2.line(empty, (xs, ys), (xe, ye), np.max(gray) / factor, thickness=np.random.randint(10, 100))
        empty = cv2.blur(empty, (5, 5))
        empty = empty | gray
        return cv2.cvtColor(empty, cv2.COLOR_GRAY2RGB)
    return img

def distort1(img, k=0, dx=0, dy=0):
    """"
    ## unconverntional augmnet ################################################################################3
    ## https://stackoverflow.com/questions/6199636/formulas-for-barrel-pincushion-distortion

    ## https://stackoverflow.com/questions/10364201/image-transformation-in-opencv
    ## https://stackoverflow.com/questions/2477774/correcting-fisheye-distortion-programmatically
    ## http://www.coldvision.io/2017/03/02/advanced-lane-finding-using-opencv/

    ## barrel\pincushion distortion
    """
    height, width = img.shape[:2]
    #  map_x, map_y =
    # cv2.initUndistortRectifyMap(intrinsics, dist_coeffs, None, None, (width,height),cv2.CV_32FC1)
    # https://stackoverflow.com/questions/6199636/formulas-for-barrel-pincushion-distortion
    # https://stackoverflow.com/questions/10364201/image-transformation-in-opencv
    k = k * 0.00001
    dx = dx * width
    dy = dy * height
    x, y = np.mgrid[0:width:1, 0:height:1]
    x = x.astype(np.float32) - width/2 - dx
    y = y.astype(np.float32) - height/2 - dy
    theta = np.arctan2(y, x)
    d = (x*x + y*y)**0.5
    r = d*(1+k*d*d)
    map_x = r*np.cos(theta) + width/2 + dx
    map_y = r*np.sin(theta) + height/2 + dy

    img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return img


def distort2(img, num_steps=10, xsteps=[], ysteps=[]):
    """
    #http://pythology.blogspot.sg/2014/03/interpolation-on-regular-distorted-grid.html
    ## grid distortion
    """
    height, width = img.shape[:2]

    x_step = width // num_steps
    xx = np.zeros(width, np.float32)
    prev = 0
    for idx, x in enumerate(range(0, width, x_step)):
        start = x
        end = x + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step*xsteps[idx]

        xx[start:end] = np.linspace(prev, cur, end-start)
        prev = cur

    y_step = height // num_steps
    yy = np.zeros(height, np.float32)
    prev = 0
    for idx, y in enumerate(range(0, height, y_step)):
        start = y
        end = y + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step*ysteps[idx]

        yy[start:end] = np.linspace(prev, cur, end-start)
        prev = cur

    map_x, map_y = np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    img = cv2.remap(img, map_x, map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101)
    return img

def elastic_transform_fast(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(1234)

    shape = image.shape
    shape_size = shape[:2]


    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    alpha = float(alpha)
    sigma = float(sigma)
    alpha_affine = float(alpha_affine)

    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)

    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = np.float32(gaussian_filter((random_state.rand(*shape_size) * 2 - 1), sigma) * alpha)
    dy = np.float32(gaussian_filter((random_state.rand(*shape_size) * 2 - 1), sigma) * alpha)

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    mapx = np.float32(x + dx)
    mapy = np.float32(y + dy)

    return cv2.remap(image, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def remap_color(img, bg, center, max):
    def get_lut(img, bg, center, max):
        ma = np.max(img)
        # me = np.mean(img)
        # th = np.mean([ma, me]) * 1.5
        th = ma / 2
        gap = 10
        channels = [[], [], []]
        range2 = ma - int(th)
        for i in range(3):
            channels[i].append(np.linspace(bg[i] - gap, center[i] - gap, int(th)).astype(np.uint8))
            channels[i].append(np.linspace(center[i] - gap, max[i] + gap, range2).astype(np.uint8))
            channels[i].append([max[i] + gap] * (256 - sum(map(len, channels[i]))))
            channels[i] = np.hstack(channels[i])
        return np.dstack(channels)

    # img = adjust_gamma(img, 5.)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if np.mean(gray) > 100:
        return img
    lut = get_lut(img, bg, center, max)
    res = cv2.LUT(img, lut).astype(np.uint8)
    return res

def invert(img):
    return 255 - img

def channel_shuffle(img):
    ch_arr = [0, 1, 2]
    np.random.shuffle(ch_arr)
    img = img[..., ch_arr]
    return img

@clipped
def gauss_noise(image, var):
    row, col, ch = image.shape
    mean = var
    # var = 30
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    gauss = (gauss - np.min(gauss)).astype(np.uint8)
    return image.astype(np.int32) + gauss

def salt_pepper_noise(image):
    #todo
    s_vs_p = 0.5
    amount = 0.004
    noisy = image
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    noisy[coords] = 255

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    noisy[coords] = 0
    return noisy

def poisson_noise(image):
    #todo
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    return noisy

def speckle_noise(image):
    #todo
    row, col, ch = image.shape
    gauss = np.random.randn(row,col,ch)
    gauss = gauss.reshape(row,col,ch)
    noisy = image + image * gauss
    return noisy

@clipped
def random_brightness(img, alpha):
    return alpha * img

@clipped
def random_contrast(img, alpha):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
    return alpha * img + gray

def to_three_channel_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    invgray = 255 - gray
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    if np.mean(invgray) < np.mean(gray):
        invgray, gray = gray, invgray
    res = [invgray, gray, clahe.apply(invgray)]
    return cv2.merge(res)

def to_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if np.mean(gray) > 127:
        gray = 255 - gray
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def add_channel(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(21, 21))
    lab = clahe.apply(lab[:, :, 0])
    if lab.mean() > 127:
        lab = 255 - lab
    return np.dstack((img, lab))


def fix_mask(msk, sigmoid=False):
    if not sigmoid:
        msk[..., 2] = (msk[..., 2] > 127)
        msk[..., 1] = (msk[..., 1] > 127) * (msk[..., 2] == 0)
        msk[..., 0] = (msk[..., 1] == 0) * (msk[..., 2] == 0)
    else:
        msk = (msk > 127)
    return msk.astype(np.uint8) * 255


def img_to_tensor(im, normalize=None):
    tensor = torch.from_numpy(np.moveaxis(im / (255. if im.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
    if normalize is not None:
        return F.normalize(tensor, **normalize)
    return tensor


def mask_to_tensor(mask, num_classes, sigmoid):
    mask = fix_mask(mask, sigmoid)
    if num_classes > 1:
        if not sigmoid:
            #softmax
            long_mask = np.zeros((mask.shape[:2]), dtype=np.int64)
            if len(mask.shape) == 3:
                for c in range(mask.shape[2]):
                    long_mask[mask[...,c] > 0] = c
            else:
                long_mask[mask > 127] = 1
                long_mask[mask == 0] = 0
            mask = long_mask
        else:
            mask = np.moveaxis(mask / (255. if mask.dtype == np.uint8 else 1), -1, 0).astype(np.float32)
    else:
        mask = np.expand_dims(mask / (255. if mask.dtype == np.uint8 else 1), 0).astype(np.float32)
    return torch.from_numpy(mask)
