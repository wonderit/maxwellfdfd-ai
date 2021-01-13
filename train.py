import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.optimizers import Adam
import keras
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
from keras import losses
from PIL import Image
import numpy as np
import argparse
import os
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


class CustomLoss:
    def __init__(self, _loss_function):
        super(CustomLoss, self).__init__()
        self.loss_function_array = _loss_function.split(',')

    def tf_diff_axis_1(self, a):
        return a[:, 1:] - a[:, :-1]

    def tf_minmax_axis_1(self, a):
        b = self.tf_diff_axis_1(a)
        sign = K.sign(b)
        abs_sign = tf.abs(self.tf_diff_axis_1(sign))
        mask_array = K.greater(abs_sign, 0)

        result = tf.where(mask_array, a[:, 1:-1], tf.zeros_like(a[:, 1:-1]))

        return result

    def custom_loss(self, y_true, y_pred):
        loss = 0
        y_true_diff = self.tf_diff_axis_1(y_true)
        y_pred_diff = self.tf_diff_axis_1(y_pred)
        threshold_value = 0
        y_true_diff_binary = K.cast(K.greater(y_true_diff, threshold_value), K.floatx())
        y_pred_diff_binary = K.cast(K.greater(y_pred_diff, threshold_value), K.floatx())
        y_true_minmax = self.tf_minmax_axis_1(y_true)
        y_pred_minmax = self.tf_minmax_axis_1(y_pred)

        if 'mse' in self.loss_function_array:
            loss = loss + K.mean(K.square(y_pred - y_true))

        if 'diff_mse' in self.loss_function_array:
            loss = loss + K.mean(K.square(y_pred_diff - y_true_diff))

        if 'rmse' in self.loss_function_array:
            loss = loss + K.sqrt(K.mean(K.square(y_pred - y_true)))

        if 'diff_rmse' in self.loss_function_array:
            loss = loss + K.sqrt(K.mean(K.square(y_pred_diff - y_true_diff)))

        if 'diff_ce' in self.loss_function_array:
            loss = loss + losses.binary_crossentropy(y_true_diff, y_pred_diff)

        if 'diff_bce' in self.loss_function_array:
            loss = loss + losses.binary_crossentropy(y_true_diff_binary, y_pred_diff_binary)

        if 'diff_rmse_minmax' in self.loss_function_array:
            loss = loss + K.sqrt(K.mean(K.square(y_pred_minmax - y_true_minmax)))

        if 'diff_poly' in self.loss_function_array:
            x = np.arange(24)
            loss = loss + np.sum(
                (np.polyval(np.polyfit(x, y_pred, 3)) - np.polyval(np.polyfit(x, y_true, 3))) ** 2
            )

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


def create_model(model_type, model_input_shape, loss_function):
    if model_type.startswith('cnn'):
        model = Sequential()
        model.add(Conv2D(16, kernel_size=(3, 3), padding='same', input_shape=model_input_shape, use_bias=False))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(24, activation='sigmoid'))
        model.compile(loss=loss_function, optimizer=Adam(lr=args.learning_rate), metrics=['accuracy'])
    elif model_type.startswith('rf'):
        regr = RandomForestRegressor(n_estimators=100, max_depth=30, random_state=2)
        return regr
    elif model_type.startswith('svm'):
        regr = SVR(kernel='rbf', C=1e3, gamma=0.1)
        return regr
    elif model_type.startswith('lasso'):
        regr = Lasso()
        return regr
    elif model_type.startswith('lr'):
        regr = LinearRegression()
        return regr
    elif model_type.startswith('ridge'):
        regr = Ridge()
        return regr
    elif model_type.startswith('mlp'):
        regr = MLPRegressor(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(20, 10), random_state=1)
        return regr
    elif model_type.startswith('knn'):
        regr = KNeighborsRegressor()
        return regr
    elif model_type.startswith('elasticnet'):
        regr = ElasticNet(random_state=0)
        return regr
    elif model_type.startswith('extratree'):
        regr = ExtraTreesRegressor(n_estimators=10,
                                   max_features=32,  # Out of 20000
                                   random_state=0)
        return regr
    elif model_type.startswith('dt'):
        regr = DecisionTreeRegressor(max_depth=5)
        return regr
    elif model_type.startswith('gbr'):
        regr = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, max_depth=5))
        return regr
    elif model_type.startswith('ada'):
        regr = MultiOutputRegressor(AdaBoostRegressor(n_estimators=300))
        return regr
    else:
        model = Sequential()
        model.add(Dense(512, activation='relu', input_dim=model_input_shape))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(24, activation='sigmoid'))
        model.compile(loss=loss_function, optimizer=Adam(lr=args.learning_rate), metrics=['accuracy'])

    return model


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
    parser.add_argument("-e", "--max_epoch", help="Set max epoch", type=int, default=100)
    parser.add_argument("-b", "--batch_size", help="Set batch size", type=int, default=128)

    # arg for AL
    parser.add_argument("-it", "--iteration", help="Set iteration for training", type=int,  default=1)
    parser.add_argument("-n", "--num_models", help="Set number of models for evaluation", type=int, default=3)
    parser.add_argument("-a", "--is_active_learning", help="Set is Active Learning", action='store_true')
    parser.add_argument("-ar", "--is_active_random", help="Set is Active random result", action='store_true')
    parser.add_argument("-r", "--labeled_ratio", help="Set R", type=float, default=0.2)
    parser.add_argument("-t", "--top_ratio", help="Set T", type=float, default=0.1)

    # arg for testing parameters
    parser.add_argument("-u", "--unit_test", help="flag for testing source code", action='store_true')
    parser.add_argument("-d", "--debug", help="flag for debugging", action='store_true')


    args = parser.parse_args()

    model_name = args.model
    batch_size = int(args.batch_size)
    epochs = int(args.max_epoch)
    loss_functions = args.loss_function
    input_shape_type = args.shape

    # DATASETS = DATASETS_TRAIN

    if input_shape_type.startswith('rect'):
        img_rows, img_cols, channels = 100, 200, 1
    else:
        img_rows, img_cols, channels = 200, 200, 1

    if model_name.startswith('cnn') is False and model_name.startswith('nn') is False:
        img_rows = img_rows // 10
        img_cols = img_cols // 10

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
        dataframe = pd.read_csv(os.path.join(DATAPATH_TRAIN, '{}.csv'.format(data_train)), delim_whitespace=False, header=None)
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

            if model_name.startswith('cnn') is False and model_name.startswith('nn') is False:
                image = compress_image(image, 10)

            if model_name.startswith('cnn_small'):
                image = compress_image(image, 5)

            if input_shape_type.startswith('rect'):
                x_train.append(image)
            else:
                v_flipped_image = np.flip(image, 0)
                square_image = np.vstack([image, v_flipped_image])
                x_train.append(square_image)

    print('Data Loading... Train dataset Finished.')
    print('Data Loading... Validation dataset Start.')

    x_validation = []
    y_validation = []
    for data_valid in DATASETS_VALID:
        dataframe = pd.read_csv(os.path.join(DATAPATH_VALID, '{}.csv'.format(data_valid)), delim_whitespace=False, header=None)
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

            if model_name.startswith('cnn') is False and model_name.startswith('nn') is False:
                image = compress_image(image, 10)

            if model_name.startswith('cnn_small'):
                image = compress_image(image, 5)

            if input_shape_type.startswith('rect'):
                x_validation.append(image)
            else:
                v_flipped_image = np.flip(image, 0)
                square_image = np.vstack([image, v_flipped_image])
                x_validation.append(square_image)
    print('Data Loading... Validation dataset Finished.')
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y_train = np.true_divide(y_train, 2767.1)

    x_validation = np.array(x_validation)
    y_validation = np.array(y_validation)
    y_validation = np.true_divide(y_validation, 2767.1)

    if model_name.startswith('cnn'):
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
            y_train = y_train.reshape(y_train.shape[0], channels, img_rows, img_cols)

            x_validation = x_validation.reshape(x_validation.shape[0], channels, img_rows, img_cols)
            y_validaton = y_validaton.reshape(y_validaton.shape[0], channels, img_rows, img_cols)
            input_shape = (channels, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
            x_validation = x_validation.reshape(x_validation.shape[0], img_rows, img_cols, channels)
            input_shape = (img_rows, img_cols, channels)
    else:
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], channels * img_rows * img_cols)
            y_train = y_train.reshape(y_train.shape[0], channels * img_rows * img_cols)
            x_validation = x_validation.reshape(x_validation.shape[0], channels * img_rows * img_cols)
            y_validaton = y_validaton.reshape(y_validaton.shape[0], channels * img_rows * img_cols)
            input_shape = channels * img_rows * img_cols
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows * img_cols * channels)
            x_validation = x_validation.reshape(x_validation.shape[0], img_rows * img_cols * channels)
            input_shape = channels * img_rows * img_cols

    D_x = x_train
    D_y = y_train
    L_x, L_y, U_x, U_y = None, None, None, None

    if args.is_active_learning:
        # import random
        n_row = int(x_train.shape[0])

        shuffled_indices = np.random.permutation(n_row)
        labeled_set_size = int(n_row*args.labeled_ratio)

        if args.is_active_random:
            labeled_set_size = labeled_set_size * 2

        # random_row = random.sample(list(range(n_row)), random_n_row)
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

    custom_loss = CustomLoss(loss_functions)

    # add reduce_lr, earlystopping
    stopping = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=8)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        factor=0.1,
        patience=2,
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
        ITERATION = ITERATION+1

    model_export_path_folder = '{}/{}'.format(model_folder_path, model_folder_name)

    if not os.path.exists(model_export_path_folder):
        os.makedirs(model_export_path_folder)

    model_export_path_template = '{}/{}_{}_it{}_m{}.{}'
    result_train_progress_path_template = '{}/train_progress/{}'

    for i in range(ITERATION):

        print('Training Iteration : {}'.format(i+1))

        # model_list = []
        X_pr = []

        if args.debug:
            print('L_x, L_y shape:', L_x.shape, L_y.shape)
            print(L_x.shape[0], 'Labeled samples')
            print(U_x.shape[0], 'Unlabeled samples')

        for m in range(args.num_models):
            model = create_model(model_name, input_shape, custom_loss.custom_loss)

            model_export_path = model_export_path_template.format(model_export_path_folder,
                                                                  loss_functions,
                                                                  input_shape_type,
                                                                  i,
                                                                  m,
                                                                  'h5')

            mc = keras.callbacks.ModelCheckpoint(model_export_path, monitor='val_loss', mode='min', save_best_only=True)

            if model_name.startswith('cnn') or model_name.startswith('nn'):
                tic()
                history = model.fit(L_x, L_y,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    # pass validation for monitoring
                                    # validation loss and metrics
                                    validation_data=(x_validation, y_validation),
                                    callbacks=[mc, reduce_lr, stopping])
                toc()
                score = model.evaluate(L_x, L_y, verbose=0)
                print('Train loss:', score[0])
                print('Train accuracy:', score[1])
                print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

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
                plt.savefig('{}/{}_{}_it{}_m{}.png'.format(train_progress_figure_path_folder, model_name, loss_functions, i, m))
            else:
                regr = model.fit(x_train, y_train)


                model_export_path_folder = 'models_al/{}_{}_{}'.format(model_name, batch_size, epochs)
                if not os.path.exists(model_export_path_folder):
                    os.makedirs(model_export_path_folder)

                model_export_path_template = '{}/{}_{}_{}.joblib'
                model_export_path = model_export_path_template.format(model_export_path_folder, loss_functions,
                                                                      input_shape_type, (i+1))
                joblib.dump(model, model_export_path)
                print("Saved model to disk")

            predict_from_model = model.predict(U_x)
            X_pr.append(predict_from_model)

        X_pr = np.array(X_pr)

        # Ascending order Sorted
        rpo_array = np.max(X_pr, axis=0) - np.min(X_pr, axis=0)
        rpo_array_sum = np.sum(rpo_array, axis=1)
        rpo_array_arg_sort = np.argsort(rpo_array_sum)
        rpo_array_sort = np.sort(rpo_array_sum)

        T_indices = int(len(D_x) * args.labeled_ratio * args.top_ratio)
        U_length = len(rpo_array_arg_sort) - T_indices
        print('t', T_indices, 'U_length', U_length)
        U_indices = rpo_array_arg_sort[:U_length]
        L_indices = rpo_array_arg_sort[U_length:]

        L_x = np.append(L_x, U_x[L_indices], axis=0)
        L_y = np.append(L_y, U_y[L_indices], axis=0)

        # shuffle Labeled data
        shuffle_index = np.random.permutation(len(L_x))
        L_x = L_x[shuffle_index]
        L_y = L_y[shuffle_index]

        U_x = U_x[U_indices]
        U_y = U_y[U_indices]


