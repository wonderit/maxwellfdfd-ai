
from math import sqrt
import pandas as pd
from keras import backend as K
from keras.models import model_from_json
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

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


def rescale(arr, std, mean):
    arr = arr * std
    arr = arr + mean

    return arr

model_name_details = [
    'cnn_128_50/mse_1',
    'cnn_128_50/rmse_1'
]

model_folder_path = 'models_blue'

model_name = './'

x_test = []
x_test_compressed = []
y_test = []
y_test_compressed = []

print('Data Loading....')

## TRAIN
CSV_FILE_PATH = "../../data/aby_data.csv"
IMAGE_FILE_PATH = "../../data/image_data.npz"

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
# train_data = csv_data[:n_train].to_numpy()
test_data = csv_data[n_train:].to_numpy()

# split train/valid image
# x_train_val_image = image_data['xtrain']
# ytrain = image_data['ytrain']

x_test_image = image_data['xtest']
y_test = image_data['ytest']

# Binarize Image
# binarized = 1.0 * (img > threshold)
# x_train_val_image = 1.0 * (x_train_val_image > 0)
x_test_image = 1.0 * (x_test_image > 0)

# get input columns
# x_train = train_data[:, :2]
x_test = test_data[:, :2]

# shuffled_indices = np.random.permutation(n_train)
# train_size = int(n_train * 0.75)
# train_idx, valid_idx = shuffled_indices[:train_size], shuffled_indices[train_size:]

img_rows, img_cols, channels = 160, 160, 1

print('Image data reshaping start')
x_test_image = x_test_image.reshape(x_test_image.shape[0], img_rows, img_cols, channels)
input_shape = (img_rows, img_cols, channels)

print('Image data reshaping finished')
# for DEBUG
# print('x shape:', x_train.shape)
# print('y shape:', y_train.shape)
# print(x_train.shape[0], 'train samples')

# img_model = create_model(model_name, input_shape, 'rmse')


# model = Model([input_1, input_2], output)


print('Data Loading... Finished.')

result = dict()
result['real'] = x_test

MODEL_JSON_PATH = ''
MODEL_H5_PATH = ''
myeongjo = 'NanumMyeongjo'

result_runningTime = dict()
result_r2 = dict()
result_rmse = dict()
rmse_for_boxplot = dict()


result_list = []
for i, model_name_detail in enumerate(model_name_details):
    print(model_name_detail)
    parsed_model_name = model_name_detail.split('/')[0]
    runningTime = 0
    if model_name_detail.startswith('cnn') or model_name_detail.startswith('nn'):
        parsed_model_name = model_name_detail.split('/')[0] + '_' + model_name_detail.split('/')[1]
        MODEL_JSON_PATH = '{}/{}.json'.format(model_folder_path, model_name_detail)
        MODEL_H5_PATH = '{}/{}.h5'.format(model_folder_path, model_name_detail)
        # load json and create model
        json_file = open(MODEL_JSON_PATH, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(MODEL_H5_PATH)
        print("Loaded model from disk")

        if model_name_detail.startswith('cnn'):
            tic()
            y_predict = loaded_model.predict([x_test_image, x_test])
        else:
            x_test_nn = x_test.reshape(x_test.shape[0], img_rows * img_cols * channels)
            tic()
            y_predict = loaded_model.predict(x_test_nn)
        runningTime = toc()

    else:
        MODEL_PATH = '{}/{}/{}.joblib'.format(model_folder_path, model_name, model_name_detail)
        loaded_model = joblib.load(MODEL_PATH)
        tic()
        y_predict = loaded_model.predict(x_test_compressed)
        runningTime = toc()
        # corr = np.corrcoef(y_test_compressed, y_predict)[0, 1]
        # rmse = root_mean_squared_error(y_test_compressed, y_predict)

    # corr = np.corrcoef(y_test, y_predict)[0, 1]
    r2 = r2_score(y_test, y_predict)

    meanSquaredError = mean_squared_error(y_test, y_predict)
    rmse = sqrt(meanSquaredError)

    rmse_all = []
    count = 0

    message = 'r2:{0:.4f}, RMSE:{1:.4f}'.format(r2, rmse)

    rmse_for_boxplot[parsed_model_name] = rmse_all

    result_r2[parsed_model_name] = r2
    result_rmse[parsed_model_name] = rmse
    result_runningTime[parsed_model_name] = runningTime

    # x_margin = -0.05
    x_margin = 0
    plt.text(x_margin, 1, 'R² = %0.4f' % r2)
    plt.text(x_margin, 0.95, 'RMSE = %0.4f' % rmse)
    plt.xlabel('Predictions')
    plt.ylabel('Actual')
    plt.savefig("{}/scatter_alpha/{}_all.png".format('result', parsed_model_name))
    plt.clf()

print('running time:', result_runningTime)
print('rmse: ', result_rmse)
print('r2: ', result_r2)
