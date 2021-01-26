import pandas as pd
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, concatenate
from keras.optimizers import Adam
import keras
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
import numpy as np
import argparse
import os
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.externals import joblib
from sklearn.metrics import r2_score

def rmse(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true)))


def tic():
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        runningTime = time.time() - startTime_for_tictoc
        toc_ = "Elapsed time is " + str(runningTime) + " seconds."
        print(toc_)
        return runningTime
    else:
        toc_ = "Toc: start time not set"
        print(toc_)
        return toc_

def scale(arr, std, mean):
    arr -= mean
    arr /= (std + 1e-7)
    return arr


def rescale(arr, std, mean):
    arr = arr * std
    arr = arr + mean

    return arr


def compress_image(prev_image, n):
    height = prev_image.shape[0] // n
    width = prev_image.shape[1] // n
    new_image = np.zeros((height, width), dtype="uint8")
    for i in range(0, height):
        for j in range(0, width):
            new_image[i, j] = prev_image[n * i, n * j]
    return new_image


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


## TRAIN
CSV_TRAIN_FILE_PATH = "./data/aby_full_0125_train_(107982, 5).csv"
CSV_TEST_FILE_PATH = "./data/aby_full_0125_test_(11999, 5).csv"
IMAGE_FILE_PATH = "./data/imaged_h_Xdata_(119981,160,160).npz"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Select model type.", default="cnn")
    parser.add_argument("-l", "--loss_function", help="Select loss functions.. (rmse,diff_rmse,diff_ce)", default='rmse')

    parser.add_argument("-tr", "--n_train", help="Set train set number", type=int, default=1000)

    parser.add_argument("-te", "--n_test", help="Set test set", type=int, default=100)

    parser.add_argument("-lr", "--learning_rate", help="Set learning_rate 1e-5", type=float, default=1e-3)
    parser.add_argument("-e", "--epochs", help="Set epochs", default=50)
    parser.add_argument("-b", "--batch_size", help="Set batch size", default=128)
    parser.add_argument("-n", "--is_normalized_output", help="Set is Normalized", action='store_true')
    parser.add_argument("-d", "--data_type", help="Select data type.. (train, valid, test)", default='train')
    # arg for testing parameters
    parser.add_argument("-u", "--unit_test", help="flag for testing source code", action='store_true')
    parser.add_argument("-oe", "--is_ordinal_encoding", help="flag for ordinal encoding 'a'", action='store_true')
    parser.add_argument("-oh", "--is_onehot_encoding", help="flag for onehot encoding 'a'", action='store_true')
    parser.add_argument("-g", "--is_global_average_pooling", help="is global average pooling", action='store_true')


    args = parser.parse_args()
    model_name = args.model
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    loss_functions = args.loss_function

    # TEST
    # args.unit_test = True

    print('CSV Data Loading....')
    csv_train_data = pd.read_csv(CSV_TRAIN_FILE_PATH)
    csv_test_data = pd.read_csv(CSV_TEST_FILE_PATH)
    print('CSV Data Loading finished (shape : train/test : {}/{})'.format(csv_train_data.shape, csv_test_data.shape))

    print('image data loading...')
    image_data = np.load(IMAGE_FILE_PATH)
    print('image data loaded')

    x_train_val_image = image_data['xtrain']
    y_train = image_data['ytrain']

    x_test_image = image_data['xtest']
    y_test = image_data['ytest']

    # of train, test set
    n_train = image_data['ytrain'].shape[0]
    n_test = image_data['ytest'].shape[0]

    # csv data train/test split
    # train_data = csv_data[:n_train].to_numpy()
    # test_data = csv_data[n_train:].to_numpy()
    train_data = csv_train_data.to_numpy()
    test_data = csv_test_data.to_numpy()

    # split train/valid image
    ##TODO
    # x_train_val_image = image_data['xtrain']
    y_train = image_data['ytrain']

    x_test_image = image_data['xtest']
    y_test = image_data['ytest']


    if args.unit_test:
        print('unit_test start')
        n_train = 1000
        n_test = 100
    else:
        print('Training Start. (Train/Test)=({}/{})'.format(args.n_train, args.n_test))
        n_train = args.n_train
        n_test = args.n_test

    # resample train
    x_train_val_image = x_train_val_image[:n_train]
    y_train = y_train[:n_train]
    train_data = train_data[:n_train, :]

    # resample test
    x_test_image = x_test_image[:n_test]
    y_test = y_test[:n_test]
    test_data = test_data[:n_test]

    # Binarize Image
    # binarized = 1.0 * (img > threshold)
    # x_train_val_image = 1.0 * (x_train_val_image > 0)
    # x_test_image = 1.0 * (x_test_image > 0)

    # preprocess image
    x_train_val_image = x_train_val_image / 116.0
    x_test_image = x_test_image / 116.0

    # get input columns
    print('x_train data before', train_data.shape)
    x_train = train_data[:, :1]
    x_test = test_data[:, :1]
    intput_column_size = x_test.shape[1]
    print('x_train data after', x_test.shape)

    output_activation = 'sigmoid'
    # y1, y2, y3, y4 normalize
    if args.is_normalized_output:
        TRANSMITTANCE_MEAN = 0.25
        TRANSMITTANCE_STD = 0.06
        output_activation = 'linear'
        print('normalize output y1 ~ y4, activation : {}'.format(output_activation))
        y_train = (y_train - TRANSMITTANCE_MEAN) / TRANSMITTANCE_STD
        y_test = (y_test - TRANSMITTANCE_MEAN) / TRANSMITTANCE_STD

    if args.is_ordinal_encoding:
        print('ordinal encoding for A start! ')
        ordinal_encoder = OrdinalEncoder()

        ordinal_encoder.fit(x_test)
        x_train = ordinal_encoder.transform(x_train)
        x_test = ordinal_encoder.transform(x_test)
        print('label successfully ordinal encoded shape : {}'.format(x_test.shape))
        intput_column_size = x_test.shape[1]

        if args.is_onehot_encoding:
            onehot_encoder = OneHotEncoder()
            onehot_encoder.fit(x_test)
            x_train = onehot_encoder.transform(x_train)
            x_test = onehot_encoder.transform(x_test)
            intput_column_size = x_test.shape[1]
            print('label successfully onehot encoded shape : {}'.format(x_test.shape))
            print('one hot encoding for A end! ')

    img_rows, img_cols, channels = 160, 160, 1


    print('Image data reshaping start')
    x_train_val_image = x_train_val_image.reshape(x_train_val_image.shape[0], img_rows, img_cols, channels)

    x_test_image = x_test_image.reshape(x_test_image.shape[0], img_rows, img_cols, channels)
    # x_validation_image = x_validation_image.reshape(x_validation_image.shape[0], img_rows, img_cols, channels)
    input_shape = (img_rows, img_cols, channels)

    print('Image data reshaping finished')
    # for DEBUG
    # print('x shape:', x_train.shape)
    # print('y shape:', y_train.shape)
    # print(x_train.shape[0], 'train samples')

    # img_model = create_model(model_name, input_shape, 'rmse')
    struct_input = Input(shape=input_shape, name="design_image")
    # input_2 = Input(shape=(2,), name="ab")
    wave_input = Input(shape=(intput_column_size,), name="wavelength")

    r = Conv2D(16, kernel_size=(3, 3), padding='same', use_bias=False, activation='relu')(struct_input)
    r = MaxPooling2D(pool_size=(2, 2))(r)
    r = Dropout(0.25)(r)
    r = Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False, activation='relu')(r)
    r = MaxPooling2D(pool_size=(2, 2))(r)
    r = Dropout(0.25)(r)
    r = Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False, activation='relu')(r)
    r = MaxPooling2D(pool_size=(2, 2))(r)
    r = Dropout(0.25)(r)
    r = Conv2D(64, kernel_size=(3, 3), padding='same', use_bias=False, activation='relu')(r)
    r = MaxPooling2D(pool_size=(2, 2))(r)
    r = Dropout(0.25)(r)
    if args.is_global_average_pooling:
        feature_output = GlobalAveragePooling2D()(r)
    else:
        feature_output = Flatten()(r)
    # add dense after flatten
    # feature_output = Dense(1024, activation='relu')(feature_output)
    concat = concatenate([feature_output, wave_input])
    tr = Dense(2048, activation='relu')(
        concat)
    tr = Dropout(0.4)(tr)
    tr = Dense(1024, activation='relu')(tr)
    tr = Dense(256, activation='relu')(tr)
    tr = Dense(64, activation='relu')(tr)
    tr = Dense(16, activation='relu')(tr)
    tr_output = Dense(4, activation=output_activation)(tr)
    model = Model([struct_input, wave_input], tr_output)
    # model.compile(loss=loss_function, optimizer=Adam(lr=learn_rate), metrics=['mae', r2])

    #
    # conv = Sequential()
    #
    # conv.add(Conv2D(16, kernel_size=(3, 3), padding='same', use_bias=False, activation='relu', input_shape=input_shape))
    # conv.add(MaxPooling2D(pool_size=(2, 2)))
    # # conv.add(Dropout(0.25))
    #
    # conv.add(Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False, activation='relu'))
    # conv.add(MaxPooling2D(pool_size=(2, 2)))
    # # conv.add(Dropout(0.25))
    #
    # conv.add(Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False, activation='relu'))
    # conv.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # conv.add(Conv2D(64, kernel_size=(3, 3), padding='same', use_bias=False, activation='relu'))
    # conv.add(MaxPooling2D(pool_size=(2, 2)))
    # conv.add(Flatten())
    # # conv.add(Dense(128, activation='relu'))
    # # conv.add(Dropout(0.4))
    #
    # concat = concatenate([conv, fc_output])
    # fc_model = Sequential()
    # fc_model.add(Dense(2048, input_shape=(1,), activation='relu'))
    # fc_model.add(Dropout(0.4))
    # fc_model.add(Dense(1024, activation='relu'))
    # fc_model.add(Dense(256, activation='relu'))
    # fc_model.add(Dense(64, activation='relu'))
    # fc_model.add(Dense(16, activation='relu'))
    # print('conv summary')
    # conv.summary()
    # print('fc_model summary')
    # fc_model.summary()
    # # exit()
    # conv_output = conv(input_1)
    # fc_output = fc_model(input_2)
    # print('conv_output shape : {}, fc_output shape: {}'.format(conv_output.shape, fc_output.shape))
    #
    # # concat = Concatenate()([conv_output, fc_output])
    # print('concat output shape: {}'.format(concat.shape))
    # concat = Concatenate()([conv_output, fc_output])
    # dense_layer = Dense(512, activation='relu')(concat)
    # dense_layer = Dropout(0.5)(dense_layer)

    # dense_layer = Dense(128, activation='relu')(concat)
    # output = Dense(4, activation='sigmoid')(dense_layer)
    #
    # model = Model([input_1, input_2], outputs=[output])
    # model.add(Dense(4, activation='sigmoid'))
    model.summary()

    model.compile(loss=rmse, optimizer=Adam(lr=args.learning_rate), metrics=[rmse])
    # model.compile(loss='mse', optimizer=Adam(lr=args.learning_rate), metrics=['mse', rmse])
    # model.compile(loss=loss_functions, optimizer=Adam(lr=args.learning_rate), metrics=['mse', rmse])

    # add callback
    model_export_path_folder = 'models_blue/{}_tr-te_{}-{}_{}_{}_lr{}'.format(model_name, n_train, n_test, batch_size, epochs, args.learning_rate)
    if not os.path.exists(model_export_path_folder):
        os.makedirs(model_export_path_folder)

    model_export_path_template = '{}/{}'.format(model_export_path_folder, loss_functions) + '-ep{epoch:02d}-val{val_loss:.4f}.hdf5'
    model_file_path = model_export_path_template
    print('model_file_path : ', model_file_path)
    # serialize weights to HDF5

    mc = keras.callbacks.ModelCheckpoint(model_file_path,
                                         monitor='val_loss',
                                         mode='auto',
                                         verbose=2,
                                         save_best_only=True)

    # add reduce_lr, earlystopping
    stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, verbose=2)  # 8

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        factor=0.1, #0.1
        patience=2, #2
        verbose=2,
        min_lr=args.learning_rate * 0.001)

    print('Training Start : Lr={}'.format(args.learning_rate))

    if model_name.startswith('cnn') or model_name.startswith('nn'):
        tic()
        history = model.fit([x_train_val_image, x_train], y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=0.2,
                            shuffle=True,
                            callbacks=[reduce_lr, mc, stopping])
        # history = model.fit([x_train_image, x_train_col], y_train,
        #                     batch_size=batch_size,
        #                     epochs=epochs,
        #                     # pass validtation for monitoring
        #                     # validation loss and metrics
        #                     validation_data=([x_validation_image, x_validation_col], y_validation),
        #                     callbacks=[reduce_lr, stopping])
        toc()
        score = model.evaluate([x_train_val_image, x_train], y_train, verbose=0)
        y_train_pred = model.predict([x_train_val_image, x_train])
        print('Train loss:', score[0])
        print('Train accuracy:', score[1])
        print('Train R-squared', r2_score(y_train, y_train_pred))
        print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

        test_score = model.evaluate([x_test_image, x_test], y_test, verbose=0)
        y_test_pred = model.predict([x_test_image, x_test])
        print('Test loss:', test_score[0])
        print('Test accuracy:', test_score[1])
        print('Test R-squared', r2_score(y_test, y_test_pred))
        print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

        # serialize model to JSON
        # model_json = model.to_json()
        # model_export_path_folder = 'models_blue/{}_{}_{}_lr{}'.format(model_name, batch_size, epochs, args.learning_rate)
        # if not os.path.exists(model_export_path_folder):
        #     os.makedirs(model_export_path_folder)
        #
        # model_export_path_template = '{}/{}_1.{}'
        # model_export_path = model_export_path_template.format(model_export_path_folder, loss_functions, 'json')
        # with open(model_export_path, "w") as json_file:
        #     json_file.write(model_json)
        #
        # # serialize weights to HDF5
        # model.save_weights(
        #     model_export_path_template.format(model_export_path_folder, loss_functions, 'h5'))
        # print("Saved model to disk")

        # Loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Model - Loss')
        plt.legend(['Training', 'Validation'], loc='upper right')
        train_progress_figure_path_folder = 'result_samsung/train_progress'
        if not os.path.exists(train_progress_figure_path_folder):
            os.makedirs(train_progress_figure_path_folder)
        plt.savefig('{}/{}_{}.png'.format(train_progress_figure_path_folder, model_name, loss_functions))
    else:
        regr = model.fit(x_train, y_train)

        model_export_path_folder = 'models_blue/{}_{}_{}_lr{}'.format(model_name, batch_size, epochs, args.learning_rate)
        if not os.path.exists(model_export_path_folder):
            os.makedirs(model_export_path_folder)

        model_export_path_template = '{}/{}_1.joblib'
        model_export_path = model_export_path_template.format(model_export_path_folder, loss_functions)
        joblib.dump(model, model_export_path)
        print("Saved model to disk")

