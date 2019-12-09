from load import loadDataGeneral
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from skimage import morphology, color, io, exposure
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import colorsys
import datetime
import os

def IoU(y_true, y_pred):
    """Returns Intersection over Union score for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return (intersection + 1) * 1. / (union + 1)

def Dice(y_true, y_pred):
    """Returns Dice Similarity Coefficient for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    return (2. * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.)

def masked(img, gt, mask, alpha=1):
    """Returns image with GT lung field outlined with red, predicted lung field
    filled with blue."""
    rows, cols = img.shape
    color_mask = np.zeros((rows, cols, 3))
    m = morphology.dilation(gt, morphology.disk(3))
    # boundary = morphology.dilation(gt, morphology.disk(3)) - gt
    boundary = (m.astype(np.float32) - gt.astype(np.float32)).astype(np.bool)
    color_mask[mask == 1] = [0, 0, 1]
    color_mask[boundary == 1] = [1, 0, 0]
    img_color = np.dstack((img, img, img))

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    return img_masked



def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img

if __name__ == '__main__':

    # Path to csv-file. File should contain X-ray filenames as first column,
    # mask filenames as second column.
    # csv_path = '/path/to/JSRT/idx.csv'
    # # Path to the folder with images. Images will be read from path + path_from_csv
    # path = csv_path[:csv_path.rfind('/')] + '/'

    # df = pd.read_csv(csv_path)
    # csv_path = '/SFS/project/ry/cheyiyun/cs230/img_mask.csv'
    csv_path = '/SFS/project/ry/cheyiyun/cs230/test.csv'
    df = pd.read_csv(csv_path)
    # Shuffle rows in dataframe. Random state is set for reproducibility.
    # df = df.sample(frac=1, random_state=23)
    # n_val = int(len(df)*0.9)
    # df_test = df[n_val:]
    df_test = df
    
    # Load test data
    im_shape = (256, 256)
    X, y, imageID = loadDataGeneral(df_test, im_shape)

    n_test = X.shape[0]
    inp_shape = X[0].shape

    # Load model
    model_name = 'model.020.hdf5'
    UNet = load_model(model_name)

    # For inference standard keras ImageGenerator is used.
    test_gen = ImageDataGenerator(rescale=1.)

    ious = np.zeros(n_test)
    dices = np.zeros(n_test)

    i = 0
    dif = []
    
    for xx, yy in test_gen.flow(X, y, batch_size=1, shuffle=False):
        img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0,1))
        pred = UNet.predict(xx)[..., 0].reshape(inp_shape[:2])
        mask = yy[..., 0].reshape(inp_shape[:2])

        # Binarize masks
        gt = mask > 0.5
        pr = pred > 0.5
        print(f"gt's shape: {gt.shape}")
        # Remove regions smaller than 2% of the image
        pr = remove_small_regions(pr, 0.02 * np.prod(im_shape))
        if not os.path.isdir("results/"):
            os.mkdir("results/")
        io.imsave('results/{}'.format(df.iloc[i][0].split('/')[-1]), masked(img, gt, pr, 0.5))
        ious[i] = IoU(gt, pr)
        dices[i] = Dice(gt, pr)
        dif1 = np.subtract(gt.flatten(), pr.flatten(), dtype=np.float)
        dif.append(np.sum(dif1))
        print (df.iloc[i][0], ious[i], dices[i])

        i += 1
        if i == n_test:
            break
    dif = np.asarray(dif)
    imageID = np.asarray(imageID)
    print ('Mean IoU:', ious.mean())
    print ('Mean Dice:', dices.mean())
    df = pd.DataFrame({"imageID": imageID, "ious" : ious, "dif(gt-pred)": dif})
    des_dir = "{:%Y%m%dT%H%M%S}.csv".format(datetime.datetime.now())
    df.to_csv(des_dir,index=False)