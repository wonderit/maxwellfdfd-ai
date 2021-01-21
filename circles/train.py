import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, Activation, Concatenate
from keras.optimizers import Adam
import keras
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
import numpy as np
import argparse
import os
from sklearn.externals import joblib
from sklearn.metrics import r2_score

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

#
# def create_model(model_type, model_input_shape, loss_function):
#     if model_type.startswith('cnn'):
#         model = Sequential()
#         model.add(Conv2D(16, kernel_size=(3, 3), padding='same', input_shape=model_input_shape, use_bias=False))
#         model.add(Activation('relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False))
#         model.add(Activation('relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False))
#         model.add(Activation('relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False))
#         model.add(Activation('relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Flatten())
#         model.add(Dense(1024, activation='relu'))
#         model.add(Dropout(0.4))
#         model.add(Dense(24, activation='sigmoid'))
#         model.compile(loss=loss_function, optimizer=Adam(lr=args.learning_rate), metrics=['accuracy'])
#     elif model_type.startswith('rf'):
#         regr = RandomForestRegressor(n_estimators=100, max_depth=30, random_state=2)
#         return regr
#     elif model_type.startswith('svm'):
#         regr = SVR(kernel='rbf', C=1e3, gamma=0.1)
#         return regr
#     elif model_type.startswith('lasso'):
#         regr = Lasso()
#         return regr
#     elif model_type.startswith('lr'):
#         regr = LinearRegression()
#         return regr
#     elif model_type.startswith('ridge'):
#         regr = Ridge()
#         return regr
#     elif model_type.startswith('mlp'):
#         regr = MLPRegressor(solver='lbfgs', alpha=1e-5,
#                             hidden_layer_sizes=(20, 10), random_state=1)
#         return regr
#     elif model_type.startswith('knn'):
#         regr = KNeighborsRegressor()
#         return regr
#     elif model_type.startswith('elasticnet'):
#         regr = ElasticNet(random_state=0)
#         return regr
#     elif model_type.startswith('extratree'):
#         regr = ExtraTreesRegressor(n_estimators=10,
#                                    max_features=32,  # Out of 20000
#                                    random_state=0)
#         return regr
#     elif model_type.startswith('dt'):
#         regr = DecisionTreeRegressor(max_depth=5)
#         return regr
#     elif model_type.startswith('gbr'):
#         regr = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, max_depth=5))
#         return regr
#     elif model_type.startswith('ada'):
#         regr = MultiOutputRegressor(AdaBoostRegressor(n_estimators=300))
#         return regr
#     else:
#         model = Sequential()
#         model.add(Dense(512, activation='relu', input_dim=model_input_shape))
#         model.add(Dense(512, activation='relu'))
#         model.add(Dense(24, activation='sigmoid'))
#         model.compile(loss=loss_function, optimizer='adam', metrics=['accuracy'])
#
#     return model


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
CSV_FILE_PATH = "../../data/aby_data.csv"
IMAGE_FILE_PATH = "../../data/image_data.npz"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Select model type.", default="cnn")
    parser.add_argument("-l", "--loss_function", help="Select loss functions.. (rmse,diff_rmse,diff_ce)",
                        default='mse')
    parser.add_argument("-lr", "--learning_rate", help="Set learning_rate", default=0.001)
    parser.add_argument("-e", "--epochs", help="Set epochs", default=50)
    parser.add_argument("-b", "--batch_size", help="Set batch size", default=128)
    parser.add_argument("-n", "--is_normalized", help="Set is Normalized", action='store_true')
    parser.add_argument("-d", "--data_type", help="Select data type.. (train, valid, test)",
                        default='train')
    # arg for testing parameters
    parser.add_argument("-u", "--unit_test", help="flag for testing source code", action='store_true')

    args = parser.parse_args()
    model_name = args.model
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    loss_functions = args.loss_function

    print('CSV Data Loading....')
    csv_data = pd.read_csv(CSV_FILE_PATH)
    print('CSV Data Loading finished (shape : {})'.format(csv_data.shape))

    print('image data loading...')
    image_data = np.load(IMAGE_FILE_PATH)
    print('image data loaded')

    # of train, test set
    n_train = image_data['ytrain'].shape[0]
    n_test = image_data['ytest'].shape[0]

    # csv data train/test split
    train_data = csv_data[:n_train].to_numpy()
    test_data = csv_data[n_train:].to_numpy()

    # split train/valid image
    x_train_val_image = image_data['xtrain']
    ytrain = image_data['ytrain']

    x_test_image = image_data['xtest']
    y_test = image_data['ytest']

    # Binarize Image
    # binarized = 1.0 * (img > threshold)
    x_train_val_image = 1.0 * (x_train_val_image > 0)
    x_test_image = 1.0 * (x_test_image > 0)

    # get input columns
    x_train = train_data[:, :2]
    x_test = test_data[:, :2]

    shuffled_indices = np.random.permutation(n_train)
    train_size = int(n_train * 0.75)
    train_idx, valid_idx = shuffled_indices[:train_size], shuffled_indices[train_size:]

    print('Data split.')
    # Train : Validation : Test = 3:1:1
    x_train_col = x_train[train_idx]
    x_train_image = x_train_val_image[train_idx]
    y_train = ytrain[train_idx]

    x_validation_col = x_train[valid_idx]
    x_validation_image = x_train_val_image[valid_idx]
    y_validation = ytrain[valid_idx]

    if args.unit_test:
        print('unit_test start')
        x_train_col = x_train_col[:1000]
        x_train_image = x_train_image[:1000]
        y_train = y_train[:1000]

        x_validation_col = x_validation_col[:1000]
        x_validation_image = x_validation_image[:1000]
        y_validation = y_validation[:1000]

    print('Data Split Finished.')
    print(K.image_data_format())

    img_rows, img_cols, channels = 160, 160, 1


    print('Image data reshaping start')
    x_train_image = x_train_image.reshape(x_train_image.shape[0], img_rows, img_cols, channels)

    x_test_image = x_test_image.reshape(x_test_image.shape[0], img_rows, img_cols, channels)
    x_validation_image = x_validation_image.reshape(x_validation_image.shape[0], img_rows, img_cols, channels)
    input_shape = (img_rows, img_cols, channels)

    print('Image data reshaping finished')
    # for DEBUG
    # print('x shape:', x_train.shape)
    # print('y shape:', y_train.shape)
    # print(x_train.shape[0], 'train samples')

    # img_model = create_model(model_name, input_shape, 'rmse')
    input_1 = Input(shape=input_shape, name="design_image")
    input_2 = Input(shape=(2,), name="ab")

    conv = Sequential()

    conv.add(Conv2D(16, kernel_size=(3, 3), padding='same', input_shape=input_shape, use_bias=False))
    conv.add(Activation('relu'))
    conv.add(MaxPooling2D(pool_size=(2, 2)))
    conv.add(Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False))
    conv.add(Activation('relu'))
    conv.add(MaxPooling2D(pool_size=(2, 2)))
    conv.add(Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False))
    conv.add(Activation('relu'))
    conv.add(MaxPooling2D(pool_size=(2, 2)))
    conv.add(Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False))
    conv.add(Activation('relu'))
    conv.add(MaxPooling2D(pool_size=(2, 2)))
    conv.add(Flatten())
    conv.add(Dense(1024, activation='relu'))
    conv.add(Dropout(0.4))

    fc_model = Sequential()
    fc_model.add(Dense(2, input_shape=(2,), activation='relu'))

    conv_output = conv(input_1)
    fc_output = fc_model(input_2)

    concat = Concatenate()([conv_output, fc_output])

    dense_layer = Dense(128, activation='relu')(concat)

    output = Dense(4, activation='sigmoid')(dense_layer)

    model = Model([input_1, input_2], output)
    # model.add(Dense(4, activation='sigmoid'))

    model.compile(loss=loss_functions, optimizer=Adam(lr=args.learning_rate), metrics=['accuracy'])

    # add reduce_lr, earlystopping
    stopping = keras.callbacks.EarlyStopping(patience=8)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        factor=0.1,
        patience=2,
        min_lr=args.learning_rate * 0.001)

    print('Training Start : Lr={}'.format(args.learning_rate))

    if model_name.startswith('cnn') or model_name.startswith('nn'):
        tic()
        history = model.fit([x_train_image, x_train_col], y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            # pass validtation for monitoring
                            # validation loss and metrics
                            validation_data=([x_validation_image, x_validation_col], y_validation),
                            callbacks=[reduce_lr, stopping])
        toc()
        score = model.evaluate([x_train_image, x_train_col], y_train, verbose=0)
        y_train_pred = model.predict([x_train_image, x_train_col])
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
        model_json = model.to_json()
        model_export_path_folder = 'models_blue/{}_{}_{}'.format(model_name, batch_size, epochs)
        if not os.path.exists(model_export_path_folder):
            os.makedirs(model_export_path_folder)

        model_export_path_template = '{}/{}_1.{}'
        model_export_path = model_export_path_template.format(model_export_path_folder, loss_functions, 'json')
        with open(model_export_path, "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights(
            model_export_path_template.format(model_export_path_folder, loss_functions, 'h5'))
        print("Saved model to disk")

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

        model_export_path_folder = 'models_blue/{}_{}_{}'.format(model_name, batch_size, epochs)
        if not os.path.exists(model_export_path_folder):
            os.makedirs(model_export_path_folder)

        model_export_path_template = '{}/{}_1.joblib'
        model_export_path = model_export_path_template.format(model_export_path_folder, loss_functions)
        joblib.dump(model, model_export_path)
        print("Saved model to disk")
