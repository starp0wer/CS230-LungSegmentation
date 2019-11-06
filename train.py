from generator import ImageDataGenerator
from load import loadDataGeneral
from model import build_UNet2D_4L

import pandas as pd
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from callback import TrainValTensorBoard
from time import time


if __name__ == '__main__':

    # Path to csv-file. File should contain X-ray filenames as first column,
    # mask filenames as second column.
    csv_path = '??.csv' #[path to your csv file]
    df = pd.read_csv(csv_path)
    # Shuffle rows in dataframe. Random state is set for reproducibility.
    df = df.sample(frac=1, random_state=23)
    n_train = int(len(df)*0.8)
    n_val = int(len(df)*0.9)
    df_train = df[:n_train]
    df_val = df[n_train:n_val]

    # Load training and validation data
    im_shape = (256, 256)
    X_train, y_train = loadDataGeneral(df_train, im_shape = im_shape)
    X_val, y_val = loadDataGeneral(df_val, im_shape = im_shape)

    # Build model
    inp_shape = X_train[0].shape
    UNet = build_UNet2D_4L(inp_shape)
    UNet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Load pre-trained model by Microsoft
    # UNet.load_weights("/SFS/project/ry/cheyiyun/cs230/trained_model.hdf5", by_name=False)

    # Visualize model
    plot_model(UNet, 'model.png', show_shapes=True)

    ##########################################################################################
    model_file_format = 'model.{epoch:03d}.hdf5'
    print(model_file_format)
    checkpointer = ModelCheckpoint(model_file_format, period=10)

    train_gen = ImageDataGenerator(rotation_range=10,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   rescale=1.,
                                   zoom_range=0.2,
                                   fill_mode='nearest',
                                   cval=0)

    test_gen = ImageDataGenerator(rescale=1.)
    tensorboard = TrainValTensorBoard(log_dir="logs/{}".format(time()), write_graph=False)
    lr_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.00001, cooldown=0, min_lr=0)
    batch_size = 8
    UNet.fit_generator(train_gen.flow(X_train, y_train, batch_size),
                       steps_per_epoch=(X_train.shape[0] + batch_size - 1) // batch_size,
                       epochs=50,
                       callbacks=[checkpointer, tensorboard, lr_decay],
                       validation_data=test_gen.flow(X_val, y_val),
                       validation_steps=(X_val.shape[0] + batch_size - 1) // batch_size)