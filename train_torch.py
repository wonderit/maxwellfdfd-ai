import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.optimizers import Adam, SGD
import keras
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image
import numpy as np
import argparse
import os

torch.manual_seed(1)

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


def create_model(model_type, model_input_shape, loss_function, premodel, optim=Adam(), l2=0):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), padding='same', input_shape=model_input_shape,
                     use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(l2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False,
                     kernel_regularizer=tf.keras.regularizers.l2(l2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False,
                     kernel_regularizer=tf.keras.regularizers.l2(l2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False,
                     kernel_regularizer=tf.keras.regularizers.l2(l2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(l2),
                    activity_regularizer=tf.keras.regularizers.l2(l2)))
    model.add(Dropout(0.4))
    model.add(Dense(24, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l2),
                    activity_regularizer=tf.keras.regularizers.l2(l2)))
    model.compile(loss=CustomLoss(loss_function, _prev_model=premodel), optimizer=optim, metrics=['accuracy'])

    return model

# Check GPU
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
DATAPATH_TRAIN = os.path.join('data', 'train')
DATASETS_TRAIN = [
    'binary_501',
    'binary_502',
    'binary_503',
    'binary_504',
    'binary_505',
    'binary_506',
    'binary_507',
    'binary_508',
    'binary_509',
    'binary_510',
    'binary_511',
    'binary_512',
    'binary_1001',
    'binary_1002',
    'binary_1003',
    'binary_rl_fix_501',
    'binary_rl_fix_502',
    'binary_rl_fix_503',
    'binary_rl_fix_504',
    'binary_rl_fix_505',
    'binary_rl_fix_506',
    'binary_rl_fix_507',
    'binary_rl_fix_508',
    'binary_rl_fix_509',
    'binary_rl_fix_510',
    'binary_rl_fix_511',
    'binary_rl_fix_512',
    'binary_rl_fix_513',
    'binary_rl_fix_514',
    'binary_rl_fix_515',
    'binary_rl_fix_516',
    'binary_rl_fix_517',
    'binary_rl_fix_518',
    'binary_rl_fix_519',
    'binary_rl_fix_520',
    'binary_rl_fix_1001',
    'binary_rl_fix_1002',
    'binary_rl_fix_1003',
    'binary_rl_fix_1004',
    'binary_rl_fix_1005',
    'binary_rl_fix_1006',
    'binary_rl_fix_1007',
    'binary_rl_fix_1008',
]

## VALIDATION
DATAPATH_VALID = './data/valid'
DATASETS_VALID = [
    'binary_1004',
    'binary_test_1001',
    'binary_test_1002',
    'binary_rl_fix_1009',
    'binary_rl_fix_1010',
    'binary_rl_fix_1011',
    'binary_rl_fix_1012',
    'binary_rl_fix_1013',
    'binary_rl_fix_test_1001',
]

## TEST
DATAPATH_TEST = './data/test'
DATASETS_TEST = [
    'binary_new_test_501',
    'binary_new_test_1501',
    'binary_rl_fix_1014',
    'binary_rl_fix_1015',
    'binary_rl_fix_test_1002',
    'binary_rl_fix_test_1003',
    'binary_rl_fix_test_1004',
    'binary_rl_fix_test_1005',
    'binary_test_1101',
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Select model type.", default="cnn")
    parser.add_argument("-s", "--shape", help="Select input image shape. (rectangle or square?)", default='rect')
    parser.add_argument("-l", "--loss_function", help="Select loss functions.. (rmse,diff_rmse,diff_ce)",
                        default='rmse')
    parser.add_argument("-lr", "--learning_rate", help="Set learning_rate", type=float, default=0.001)
    parser.add_argument("-e", "--max_epoch", help="Set max epoch", type=int, default=50)
    parser.add_argument("-b", "--batch_size", help="Set batch size", type=int, default=128)

    # arg for AL
    parser.add_argument("-it", "--iteration", help="Set iteration for training", type=int, default=1)
    parser.add_argument("-n", "--num_models", help="Set number of models for evaluation", type=int, default=3)
    parser.add_argument("-a", "--is_active_learning", help="Set is Active Learning", action='store_true')
    parser.add_argument("-ar", "--is_active_random", help="Set is Active random result", action='store_true')
    parser.add_argument("-r", "--labeled_ratio", help="Set R", type=float, default=0.2)
    parser.add_argument("-t", "--top_ratio", help="Set T", type=float, default=0.1)

    # arg for testing parameters
    parser.add_argument("-u", "--unit_test", help="flag for testing source code", action='store_true')
    parser.add_argument("-d", "--debug", help="flag for debugging", action='store_true')

    # arg for rpo lossfunction
    parser.add_argument("-dl", "--is_different_losses", action='store_true')
    parser.add_argument("-dm", "--is_different_models", action='store_true')

    # arg for weight decay scheduling
    parser.add_argument("-ws", "--weight_schedule_factor", type=float, default=0)
    parser.add_argument("-wd", "--weight_decay_factor", type=float, default=0)
    parser.add_argument("-rm", "--remember_model", action='store_true')
    parser.add_argument("-tor", "--teacher_outlier_rejection", action='store_true')

    parser.add_argument("-o", "--optimizer", help="Select optimizer.. (sgd, adam, adamw)", default='adam')

    args = parser.parse_args()

    # TEST
    args.unit_test = True
    args.debug = True
    args.teacher_outlier_rejection = True
    args.max_epoch = 1
    args.is_active_learning = True

    model_name = args.model
    batch_size = int(args.batch_size)
    epochs = int(args.max_epoch)
    loss_functions = args.loss_function
    input_shape_type = args.shape

    img_rows, img_cols, channels = 100, 200, 1

    if args.unit_test:
        DATASETS_TRAIN = [
            'binary_501',
        ]
        DATASETS_VALID = [
            'binary_1004',
        ]

    x_train = []
    y_train = []

    print('Training model args : batch_size={}, max_epoch={}, lr={}, loss_function={}, al={}, iter={}, R={}, T={}'
          .format(args.batch_size, args.max_epoch, args.learning_rate, args.loss_function, args.is_active_learning,
                  args.iteration, args.labeled_ratio, args.top_ratio))

    print('Data Loading... Train dataset Start.')

    # load Train dataset
    for data_train in DATASETS_TRAIN:
        dataframe = pd.read_csv(os.path.join(DATAPATH_TRAIN, '{}.csv'.format(data_train)), delim_whitespace=False,
                                header=None)
        dataset = dataframe.values

        # split into input (X) and output (Y) variables
        fileNames = dataset[:, 0]
        y_train.extend(dataset[:, 1:25])
        for idx, file in enumerate(fileNames):
            try:
                image = Image.open(os.path.join(DATAPATH_TRAIN, data_train, '{}.tiff'.format(int(file))))
                image = np.array(image, dtype=np.uint8)
            except (TypeError, FileNotFoundError) as te:
                image = Image.open(os.path.join(DATAPATH_TRAIN, data_train, '{}.tiff'.format(idx + 1)))
                try:
                    image = np.array(image, dtype=np.uint8)
                except:
                    continue

            x_train.append(image)

    print('Data Loading... Train dataset Finished.')
    print('Data Loading... Validation dataset Start.')

    x_validation = []
    y_validation = []
    for data_valid in DATASETS_VALID:
        dataframe = pd.read_csv(os.path.join(DATAPATH_VALID, '{}.csv'.format(data_valid)), delim_whitespace=False,
                                header=None)
        dataset = dataframe.values

        # split into input (X) and output (Y) variables
        fileNames = dataset[:, 0]
        y_validation.extend(dataset[:, 1:25])
        for idx, file in enumerate(fileNames):

            try:
                image = Image.open(os.path.join(DATAPATH_VALID, data_valid, '{}.tiff'.format(int(file))))
                image = np.array(image, dtype=np.uint8)
            except (TypeError, FileNotFoundError) as te:
                image = Image.open(os.path.join(DATAPATH_VALID, data_valid, '{}.tiff'.format(idx + 1)))
                try:
                    image = np.array(image, dtype=np.uint8)
                except:
                    continue

            x_validation.append(image)
    print('Data Loading... Validation dataset Finished.')
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y_train = np.true_divide(y_train, 2767.1)

    x_validation = np.array(x_validation)
    y_validation = np.array(y_validation)
    y_validation = np.true_divide(y_validation, 2767.1)

    # Set total dataset : D_, labeled dataset : L_, unlabeled dataset : U_
    D_x = x_train
    D_y = y_train
    L_x = D_x
    L_y = D_y

    if args.is_active_learning:
        n_row = int(x_train.shape[0])

        shuffled_indices = np.random.permutation(n_row)
        labeled_set_size = int(n_row * args.labeled_ratio)

        if args.is_active_random:
            labeled_set_size = labeled_set_size * 2

        L_indices = shuffled_indices[:labeled_set_size]
        U_indices = shuffled_indices[labeled_set_size:]

        L_x = x_train[L_indices]
        L_y = y_train[L_indices]
        U_x = x_train[U_indices]
        U_y = y_train[U_indices]

    # for DEBUG
    if args.debug:
        print('x shape:', L_x.shape)
        print('y shape:', L_y.shape)
        print(L_x.shape[0], 'train samples')

    # Set RPO LOSS
    if args.is_different_losses:
        rpo_losses = [
            'rmse', 'rmse,diff_rmse', 'rmse'
        ]
    else:
        rpo_losses = [
            'rmse', 'rmse', 'rmse'
        ]

    # Set RPO Models
    if args.is_different_models:
        rpo_models = [
            'cnn', 'cnn', 'nn'
        ]
    else:
        rpo_models = [
            'cnn', 'cnn', 'cnn', 'cnn', 'cnn'
        ]

    # add reduce_lr, earlystopping
    stopping = keras.callbacks.EarlyStopping(monitor='val_loss', verbose=2, patience=8)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        factor=0.1,
        patience=2,
        verbose=2,
        min_lr=args.learning_rate * 0.001)

    ITERATION = args.iteration

    # Set model, result folder
    model_folder_path = 'models_al'
    result_folder_path = 'result_al'
    model_folder_name = '{}_bs{}_e{}_lr{}'.format(
        model_name, batch_size, epochs, args.learning_rate
    )
    if args.is_active_learning:
        model_folder_name = '{}_al_from_l0_r{}_t{}_bs{}_e{}_lr{}'.format(
            model_name, args.labeled_ratio, args.top_ratio, batch_size, epochs, args.learning_rate
        )
        if args.is_different_losses:
            model_folder_name = '{}_al_from_l0_w_diff_losses_r{}_t{}_bs{}_e{}_lr{}'.format(
                model_name, args.labeled_ratio, args.top_ratio, batch_size, epochs, args.learning_rate
            )
        if args.is_different_models:
            model_folder_name = '{}_al_from_l0_w_diff_models_r{}_t{}_bs{}_e{}_lr{}'.format(
                model_name, args.labeled_ratio, args.top_ratio, batch_size, epochs, args.learning_rate
            )
        ITERATION = ITERATION + 1

    model_export_path_folder = '{}/{}'.format(model_folder_path, model_folder_name)

    if not os.path.exists(model_export_path_folder):
        os.makedirs(model_export_path_folder)

    result_train_progress_path_template = '{}/train_progress/{}'

    prev_model = None
    for i in range(ITERATION):

        print('Training Iteration : {}'.format(i + 1))

        # model_list = []
        X_pr = []

        if args.debug:
            print('L_x, L_y shape:', L_x.shape, L_y.shape)
            print(L_x.shape[0], 'Labeled samples')
            print(U_x.shape[0], 'Unlabeled samples')

        num_models = args.num_models
        is_different_losses = args.is_different_losses

        if i == (ITERATION - 1):
            num_models = 7
            is_different_losses = False

        for m in range(num_models):

            model_export_path_template = '{}/{}_{}_it{}_m{}.{}'
            tor_x_pr = None
            if args.teacher_outlier_rejection and len(X_pr) > 0:
                tor_x_pr = X_pr[-1]

            # custom_loss = CustomLoss(loss_functions, tor_x_pr)

            if args.is_different_models:
                if i == (ITERATION - 1):
                    model_name = 'cnn'
                else:
                    model_name = rpo_models[m]
                model_export_path_template = '{}/{}_{}_it{}_m{}_{}.{}'
                model_export_path = model_export_path_template.format(model_export_path_folder,
                                                                      loss_functions,
                                                                      input_shape_type,
                                                                      i,
                                                                      m,
                                                                      model_name,
                                                                      'h5')
            else:
                model_export_path = model_export_path_template.format(model_export_path_folder,
                                                                      loss_functions,
                                                                      input_shape_type,
                                                                      i,
                                                                      m,
                                                                      'h5')

            input_L_x = L_x
            valid_x = x_validation
            if model_name.startswith('cnn'):
                input_L_x = L_x.reshape(L_x.shape[0], img_rows, img_cols, channels)
                valid_x = x_validation.reshape(x_validation.shape[0], img_rows, img_cols, channels)
                input_shape = (img_rows, img_cols, channels)
            else:
                input_L_x = L_x.reshape(L_x.shape[0], img_rows * img_cols * channels)
                valid_x = x_validation.reshape(x_validation.shape[0], img_rows * img_cols * channels)
                input_shape = channels * img_rows * img_cols

            tic()

            # scheduling weight decay
            wd = 0
            if args.weight_decay_factor > 0:
                wd = args.weight_decay_factor - args.weight_schedule_factor * i
                print('Weight Decay Scheduling activated : lambda={}'.format(wd))

            mc = keras.callbacks.ModelCheckpoint(model_export_path, monitor='val_loss', mode='min',
                                                 save_best_only=True)
            callbacks = [mc, reduce_lr, stopping]
            # Optimizer
            optimizer = Adam(lr=args.learning_rate)
            if args.optimizer == 'sgd':
                optimizer = SGD(lr=args.learning_rate)
            elif args.optimizer == 'adamw':
                optimizer = tfa.optimizers.AdamW(learning_rate=args.learning_rate, weight_decay=wd)
                callbacks = [mc, stopping]
            else:
                optimizer = Adam(lr=args.learning_rate)

            inputs = keras.Input(shape=(100, 200, 1), name="digits")
            x = Conv2D(16, kernel_size=(3, 3), padding='same', use_bias=False, activation='relu')(inputs)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False, activation='relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False, activation='relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False, activation='relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Flatten()(x)
            x = Dense(1024, activation="relu", name="dense_1")(x)
            x = Dropout(0.4)(x)
            outputs = Dense(24, activation="sigmoid", name="predictions")(x)
            model = keras.Model(inputs=inputs, outputs=outputs)

            temp_x = tf.Variable(0, trainable=False)


            def new_loss(noise):
                def loss(y_true, y_pred):
                    if temp_x is not None:
                        print(temp_x.eval())
                    test = noise
                    exit()
                    return K.mean(K.square(y_pred - y_true) - K.square(y_true - noise))

                return loss


            model.compile(loss='mse', optimizer=Adam(lr=0.001))

            # Initialize weights
            if args.remember_model and prev_model is not None:
                print('Initializing model with previous model 0')
                model.set_weights(prev_model.get_weights())

            temp_x = input_L_x
            history = model.fit(input_L_x, L_y,
                                batch_size=batch_size,
                                epochs=epochs,
                                # pass validation for monitoring
                                # validation loss and metrics
                                validation_data=(valid_x, y_validation),
                                callbacks=callbacks)
            toc()
            score = model.evaluate(input_L_x, L_y, verbose=0)
            print('score', score)
            print('Model : {}, Train loss: {}'.format(model_name, score))

            # Loss
            plt.clf()
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Model - Loss')
            plt.legend(['Training', 'Validation'], loc='upper right')
            train_progress_figure_path_folder = result_train_progress_path_template.format(
                result_folder_path, model_export_path_folder
            )

            if not os.path.exists(train_progress_figure_path_folder):
                os.makedirs(train_progress_figure_path_folder)
            plt.savefig(
                '{}/{}_{}_it{}_m{}.png'.format(train_progress_figure_path_folder, model_name, loss_functions, i, m))

            if args.remember_model and m == 0:
                print('ITERATION : {}, prev model updated'.format(i))
                prev_model = model

            if not args.is_active_random and args.is_active_learning:
                resized_U_x = U_x
                resized_U_x = U_x.reshape(U_x.shape[0], img_rows, img_cols, channels)
                predict_from_model = model.predict(resized_U_x)
                X_pr.append(predict_from_model)

        # shuffle Labeled data
        shuffle_index = np.random.permutation(len(L_x))
        L_x = L_x[shuffle_index]
        L_y = L_y[shuffle_index]
