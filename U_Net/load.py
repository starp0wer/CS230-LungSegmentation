import numpy as np
from skimage import transform, io, img_as_float, exposure
import cv2
import os.path

"""
Data was preprocessed in the following ways:
    - resize to im_shape;
    - equalize histogram (skimage.exposure.equalize_hist);
    - normalize by data set mean and std.
Resulting shape should be (n_samples, img_width, img_height, 1).
It may be more convenient to store preprocessed data for faster loading.
Dataframe should contain paths to images and masks as two columns (relative to `path`).
"""


def loadDataGeneral(df, im_shape):
    """Function for loading arbitrary data in standard formats"""
    X, y = [], []
    imgID = []
    for i, item in df.iterrows():
        img = cv2.imread(item[0])
        img = img_as_float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[2] == 3 else np.squeeze(img))
        mask = cv2.imread(item[1])
        mask = img_as_float(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if mask.shape[2] == 3 else np.squeeze(mask))
        img = transform.resize(img, im_shape)
        img = exposure.equalize_hist(img)
        img = np.expand_dims(img, -1)
        mask = transform.resize(mask, im_shape)
        mask = np.expand_dims(mask, -1)
        imageID = os.path.basename(item[0])
        X.append(img)
        y.append(mask)
        imgID.append(imageID)
    X = np.array(X)
    y = np.array(y)
    imageID = np.array(imageID)
    X -= X.mean()
    X /= X.std()

    print ('### Dataset loaded')
    # print ('\t{}'.format(path))
    print ('\t{}\t{}'.format(X.shape, y.shape))
    print ('\tX:{:.1f}-{:.1f}\ty:{:.1f}-{:.1f}\n'.format(X.min(), X.max(), y.min(), y.max()))
    print ('\tX.mean = {}, X.std = {}'.format(X.mean(), X.std()))
    return X, y, imgID
