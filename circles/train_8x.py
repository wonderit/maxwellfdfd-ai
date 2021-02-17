from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, concatenate
from keras.optimizers import Adam, SGD
import keras
import tensorflow_addons as tfa
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


class CustomLoss:
    def __init__(self, _loss_function):
        super(CustomLoss, self).__init__()
        self.loss_function_array = _loss_function.split(',')

    def custom_loss(self, y_true, y_pred):
        loss = 0

        if 'mse' in self.loss_function_array:
            loss = loss + K.mean(K.square(y_pred - y_true))

        if 'rmse' in self.loss_function_array:
            loss = loss + K.sqrt(K.mean(K.square(y_pred - y_true)))

        return loss


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


# CPU TEST
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
TRAIN_1_DATA_PATH = "./data/wv_h580_train1_350000-160-160.npz"
TRAIN_2_DATA_PATH = "./data/wv_h580_train2_350000-160-160.npz"
TEST_DATA_PATH = "./data/wv_h580_test_69969-160-160.npz"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Select model type.", default="cnn")
    parser.add_argument("-l", "--loss_function", help="Select loss functions.. (rmse,diff_rmse,diff_ce)",
                        default='rmse')

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
    parser.add_argument("-mh", "--is_multiple_height", help="is height adjustable", action='store_true')
    parser.add_argument("-da", "--data_augmentation", help="add augmentation", action='store_true')

    parser.add_argument("-o", "--optimizer", help="Select optimizer.. (sgd, adam, adamw)", default='adam')

    parser.add_argument("-pl", "--patience_lr", help="Set patience of lr", type=int, default=2)
    parser.add_argument("-pes", "--patience_early_stopping", help="Set patience of es", type=int, default=8)

    args = parser.parse_args()
    model_name = args.model
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    loss_functions = args.loss_function

    # TEST
    # args.unit_test = True

    print('Train, Valid, Test Data Loading....')

    # args.unit_test = True
    # args.optimizer = 'adamw'
    # epochs = 4

    if args.unit_test:

        TRAIN_DATA_PATH = "./data/wv_image_val_0128_(9711,160,160).npz"
        train_data = np.load(TRAIN_1_DATA_PATH)
        validation_data = train_data
        test_data = train_data

        x_train_image = train_data['image']
        x_train = train_data['wv']
        y_train = train_data['y']

        x_validation_image = validation_data['image']
        x_validation = validation_data['wv']
        y_validation = validation_data['y']

        x_test_image = test_data['image']
        x_test = test_data['wv']
        y_test = test_data['y']
    else:
        print('use 7x (train set * 7)')
        train_1_data = np.load(TRAIN_1_DATA_PATH)

        train_2_data = np.load(TRAIN_2_DATA_PATH)

        test_data = np.load(TEST_DATA_PATH)
        print('keys : ', train_1_data.files)

        print('keys 2: ', train_2_data.files)

        print('keys test: ', test_data.files)

        x_train_image = train_1_data['xtrain']
        print('preprocess1 start')
        x_train_image = x_train_image.astype(np.uint8)
        x_train_image = np.true_divide(x_train_image, 116).astype(np.uint8)
        print('preprocess1 end')

        x_train = train_1_data['wvtrain']
        y_train = train_1_data['ytrain']

        print('preprocess2 start')
        x_train_image_2 = train_2_data['xtrain']
        x_train_image_2 = x_train_image_2.astype(np.uint8)
        x_train_image_2 = np.true_divide(x_train_image_2, 116).astype(np.uint8)
        print('preprocess2 end')

        x_train_image = np.append(x_train_image, x_train_image_2, axis=0)
        x_train = np.append(x_train, train_2_data['wvtrain'], axis=0)
        y_train = np.append(y_train, train_2_data['ytrain'], axis=0)

        x_validation_image = train_1_data['xtest']

        print('preprocess3 start')
        x_validation_image = x_validation_image.astype(np.uint8)
        x_validation_image = np.true_divide(x_validation_image, 116).astype(np.uint8)
        print('preprocess3 end')

        x_validation = train_1_data['wvtest']
        y_validation = train_1_data['ytest']

        print('preprocess4 start')
        x_validation_image_2 = train_2_data['xtest']
        x_validation_image_2 = x_validation_image_2.astype(np.uint8)
        x_validation_image_2 = np.true_divide(x_validation_image_2, 116).astype(np.uint8)
        print('preprocess4 end')

        x_validation_image = np.append(x_validation_image, x_validation_image_2, axis=0)
        x_validation = np.append(x_validation, train_2_data['wvtest'], axis=0)
        y_validation = np.append(y_validation, train_2_data['ytest'], axis=0)

        x_test_image = test_data['image']
        print('preprocess5 start')
        x_test_image = x_test_image.astype(np.uint8)
        x_test_image = np.true_divide(x_test_image, 116).astype(np.uint8)
        print('preprocess5 end')
        x_test = test_data['wv']
        y_test = test_data['y']

        print('freeing memory start')
        del train_1_data, train_2_data, test_data, x_train_image_2, x_validation_image_2
        print('freeing memory end')

    print('Train, Valid, Test Data Loading finished (shape : train/valid/test > {}/{}/{})'.format(len(y_train),
                                                                                                  len(y_validation),
                                                                                                  len(y_test)))

    # of train, test set
    n_train = len(y_train)
    n_validation = len(y_validation)
    n_test = len(y_test)

    # Binarize Image
    # binarized = 1.0 * (img > threshold)
    # x_train_val_image = 1.0 * (x_train_val_image > 0)
    # x_test_image = 1.0 * (x_test_image > 0)

    # preprocess image
    if not args.unit_test:
        print('preprocess start')
        # x_train_image = np.true_divide(x_train_image, 116.0)
        # x_validation_image = np.true_divide(x_validation_image, 116.0)
        # x_test_image = np.true_divide(x_test_image, 116.0)
        print('preprocess end')
        # x_train_image = x_train_image / 116.0
        # x_validation_image = x_validation_image / 116.0
        # x_test_image = x_test_image / 116.0

    # resize input column
    x_train = x_train.reshape(-1, 1)
    x_validation = x_validation.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)

    # get input columns
    # x_train = train_data[:, :1]
    # x_test = test_data[:, :1]
    # intput_column_size = x_test.shape[1]
    input_column_size = x_train.shape[1]
    print('x_test data after', x_test.shape)

    # if not args.is_multiple_height:
    #     print('Input Image has same Height')
    #     train_indices = []
    #     test_indices = []
    #     for i in range(x_train_val_image.shape[0]):
    #         if x_train_val_image[i][0][0] == 1.0:
    #             train_indices.append(True)
    #         else:
    #             train_indices.append(False)
    #     for j in range(x_test_image.shape[0]):
    #         if x_test_image[j][0][0] == 1.0:
    #             test_indices.append(True)
    #         else:
    #             test_indices.append(False)
    #
    #     x_train_val_image = x_train_val_image[train_indices, :, :]
    #     y_train = y_train[train_indices]
    #     x_test_image = x_test_image[test_indices, :, :]
    #     y_test = y_test[test_indices]
    #
    #     x_train = x_train[train_indices, :]
    #     x_test = x_test[test_indices, :]

    if args.unit_test:
        print('unit_test start')
        n_train = 1000
        n_test = 100
    else:
        print('Training Start. (Train/Test)=({}/{})'.format(args.n_train, args.n_test))
        n_train = args.n_train
        n_test = args.n_test

    if n_train > 0:
        # resample x image
        x_train_image = x_train_image[:n_train]
        x_train = x_train[:n_train, :]
        y_train = y_train[:n_train]

    if n_test > 0:
        x_validation_image = x_validation_image[:n_test]
        x_test_image = x_test_image[:n_test]
        y_validation = y_validation[:n_test]
        y_test = y_test[:n_test]
        x_validation = x_validation[:n_test, :]
        x_test = x_test[:n_test, :]

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

        ordinal_encoder.fit(x_train)
        x_train = ordinal_encoder.transform(x_train)
        x_validation = ordinal_encoder.transform(x_validation)
        x_test = ordinal_encoder.transform(x_test)
        print('label successfully ordinal encoded shape : {}'.format(x_test.shape))
        input_column_size = x_test.shape[1]

        if args.is_onehot_encoding:
            onehot_encoder = OneHotEncoder()
            onehot_encoder.fit(x_train)
            x_train = onehot_encoder.transform(x_train)
            x_validation = onehot_encoder.transform(x_validation)
            x_test = onehot_encoder.transform(x_test)
            input_column_size = x_test.shape[1]
            print('label successfully onehot encoded shape : {}'.format(x_test.shape))
            print('one hot encoding for A end! ')

    img_rows, img_cols, channels = x_train_image.shape[1], x_train_image.shape[2], 1

    print('Image data reshaping start')
    x_train_image = x_train_image.reshape(-1, img_rows, img_cols, channels)
    x_validation_image = x_validation_image.reshape(-1, img_rows, img_cols, channels)
    x_test_image = x_test_image.reshape(-1, img_rows, img_cols, channels)

    input_shape = (img_rows, img_cols, channels)
    print('Image data reshaping finished')

    struct_input = Input(shape=input_shape, name="design_image")
    wave_input = Input(shape=(input_column_size,), name="wavelength")

    r = Conv2D(16, kernel_size=(3, 3), padding='same', use_bias=False, activation='relu')(struct_input)
    r = MaxPooling2D(pool_size=(2, 2))(r)
    # r = Dropout(0.25)(r)
    r = Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False, activation='relu')(r)
    r = MaxPooling2D(pool_size=(2, 2))(r)
    # r = Dropout(0.25)(r)
    r = Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False, activation='relu')(r)
    r = MaxPooling2D(pool_size=(2, 2))(r)
    # r = Dropout(0.25)(r)
    r = Conv2D(64, kernel_size=(3, 3), padding='same', use_bias=False, activation='relu')(r)
    r = MaxPooling2D(pool_size=(2, 2))(r)
    # r = Dropout(0.25)(r)
    if args.is_global_average_pooling:
        feature_output = GlobalAveragePooling2D()(r)
    else:
        feature_output = Flatten()(r)
    # add dense after flatten
    # feature_output = Dense(1024, activation='relu')(feature_output)
    concat = concatenate([feature_output, wave_input])
    tr = Dense(2048, activation='relu')(concat)
    tr = Dropout(0.4)(tr)
    tr = Dense(1024, activation='relu')(tr)
    tr = Dense(256, activation='relu')(tr)
    tr = Dense(64, activation='relu')(tr)
    tr = Dense(16, activation='relu')(tr)
    tr_output = Dense(4, activation=output_activation)(tr)
    model = Model([struct_input, wave_input], tr_output)
    # model.compile(loss=loss_function, optimizer=Adam(lr=learn_rate), metrics=['mae', r2])

    model.summary()

    custom_loss = CustomLoss(loss_functions)

    # Optimizer
    optimizer = Adam(lr=args.learning_rate)
    if args.optimizer == 'sgd':
        optimizer = SGD(lr=args.learning_rate)
    elif args.optimizer == 'adamw':
        optimizer = tfa.optimizers.AdamW(learning_rate=args.learning_rate, weight_decay=1e-4)
    else:
        optimizer = Adam(lr=args.learning_rate)
    model.compile(loss=custom_loss.custom_loss, optimizer=optimizer, metrics=['accuracy'])
    # model.compile(loss='mse', optimizer=Adam(lr=args.learning_rate), metrics=['mse', rmse])
    # model.compile(loss=loss_functions, optimizer=Adam(lr=args.learning_rate), metrics=['mse', rmse])

    # add callback
    model_export_path_folder = 'models_blue/{}_tr-te_{}-{}_{}_{}_lr{}'.format(model_name, n_train, n_test, batch_size,
                                                                              epochs, args.learning_rate)
    if not os.path.exists(model_export_path_folder):
        os.makedirs(model_export_path_folder)

    model_export_path_template = '{}/{}'.format(model_export_path_folder,
                                                loss_functions) + '-ep{epoch:02d}-val{val_loss:.4f}.hdf5'
    model_file_path = model_export_path_template
    print('model_file_path : ', model_file_path)
    # serialize weights to HDF5

    mc = keras.callbacks.ModelCheckpoint(model_file_path,
                                         monitor='val_loss',
                                         mode='auto',
                                         verbose=2,
                                         save_best_only=True)

    # add reduce_lr, earlystopping
    stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience_early_stopping,
                                             verbose=2)  # 8, 50
    #
    # stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, verbose=2)  # 8
    patience_lr_factor = 0.1
    if args.patience_lr == 20:
        patience_lr_factor = 0.5

    print('patience lr : {}, lr factor : {}'.format(args.patience_lr, patience_lr_factor))
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        factor=patience_lr_factor,  # 0.1, 0.5
        patience=args.patience_lr,  # 2, 20
        verbose=2,
    )
    #
    #
    # reduce_lr = keras.callbacks.ReduceLROnPlateau(
    #     factor=0.1, #0.1
    #     patience=2, #2
    #     verbose=2,
    #     min_lr=args.learning_rate * 0.001
    # )

    callbacks = [reduce_lr, mc, stopping]

    print('Training Start : Lr={}'.format(args.learning_rate))

    if model_name.startswith('cnn') or model_name.startswith('nn'):
        tic()
        # history = model.fit([x_train_val_image, x_train], y_train,
        #                     batch_size=batch_size,
        #                     epochs=epochs,
        #                     validation_split=0.1,
        #                     shuffle=True,
        #                     callbacks=[reduce_lr, mc, stopping])
        history = model.fit([x_train_image, x_train],
                            y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            # validation loss and metrics
                            validation_data=([x_validation_image, x_validation], y_validation),
                            shuffle=True,
                            callbacks=callbacks)

        toc()
        score = model.evaluate([x_train_image, x_train], y_train, verbose=0)
        y_train_pred = model.predict([x_train_image, x_train])
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

        model_export_path_folder = 'models_blue/{}_{}_{}_lr{}'.format(model_name, batch_size, epochs,
                                                                      args.learning_rate)
        if not os.path.exists(model_export_path_folder):
            os.makedirs(model_export_path_folder)

        model_export_path_template = '{}/{}_1.joblib'
        model_export_path = model_export_path_template.format(model_export_path_folder, loss_functions)
        joblib.dump(model, model_export_path)
        print("Saved model to disk")
