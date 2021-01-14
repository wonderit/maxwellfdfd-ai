from math import sqrt
from PIL import Image
import pandas as pd
from keras import backend as K
from keras import losses
from keras.layers import Average
from keras.models import Model, load_model
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error, r2_score
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import argparse

import os
import glob
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def normalized_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square((y_pred - y_true) / 2600 / (y_true / 2600)), axis=-1))

def ensemble(models, model_input):
    outputs = [model(model_input) for model in models]
    y = Average()(outputs)
    model = Model(model_input, y, name='ensemble')
    return model

def compress_image(prev_image):
    height = prev_image.shape[0] // 10
    width = prev_image.shape[1] // 10
    new_image = np.zeros((height, width), dtype="uint8")
    for i in range(0, height):
        for j in range(0, width):
            new_image[i, j] = prev_image[10*i, 10*j]
    return new_image

def ensembleModels(models, model_input):
    # collect outputs of models in a list
    yModels = [model(model_input) for model in models]
    # averaging outputs
    yAvg = Average(yModels)
    # build model from same input and avg output
    modelEns = Model(inputs=model_input, outputs=yAvg, name='ensemble')

    return modelEns

def tf_diff(a):
    return a[1:] - a[:-1]


def tf_diff_axis_1(a):
    return a[:, 1:] - a[:, :-1]

def image_trim(image, x=8, y=8):
    print(image.shape)
    images = []
    width = image.shape[1] // x
    height = image.shape[0] // y
    print(width, height)
    for i in range(0, x):
        for j in range(0, y):
            trimmed_image = image[j*height:j*height+height, i*width:i*width + width]
            resized_image = cv2.resize(trimmed_image, None, fx=5, fy=5, interpolation=cv2.INTER_AREA)
            cv2.imwrite('./data_test/image_from_gan/' + str(j) + '_' + str(i) + '.tiff', resized_image)
            resized_image //= 255
            images.append(resized_image)
    return images

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



# PARAMETERS
MODEL_SHAPE_TYPE = 'rect'

## TEST
DATAPATH = os.path.join('data', 'test')
DATASETS = [
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

model_folder_path = 'models_al'
is_mean_std = False

if MODEL_SHAPE_TYPE == 'rect':
    img_rows, img_cols, channels = 100, 200, 1
else:
    img_rows, img_cols, channels = 200, 200, 1

img_rows_compressed = img_rows // 10
img_cols_compressed = img_cols // 10
lowest_RMSE = 999
lowest_RMSE_id = 0

lowest_RMSE_DIFF_RMSE = 999
lowset_RMSE_DIFF_RMSE_ID = 0

lowset_local_RMSE = 999
lowset_local_RMSE_id = 0

lowest_POLY_RMSE = 999
lowest_POLY_RMSE_ID = 0

model_name = './'

x_test = []
x_test_compressed = []
y_test = []
y_test_compressed = []

print('Data Loading....')

# load dataset
for i, data in enumerate(DATASETS):

    dataframe = pd.read_csv('{}/{}.csv'.format(DATAPATH, data), delim_whitespace=False, header=None)
    dataset = dataframe.values
    # split into input (X) and output (Y) variables
    fileNames = dataset[:, 0]
    y_test.extend(dataset[:, 1:25])
    for idx, file in enumerate(fileNames):

        try:
            image = Image.open(os.path.join(DATAPATH, data, '{}.tiff'.format(int(file))))
            image = np.array(image, dtype=np.uint8)
        except (TypeError, FileNotFoundError) as te:

            image = Image.open(os.path.join(DATAPATH, data, '{}.tiff'.format(idx + 1)))
            # image = cv2.imread('{}/{}/{}.tiff'.format(DATAPATH, data, idx + 1), 0)

            image = np.array(image, dtype=np.uint8)
        # image //= 255
        # print(image)

        compressed_image = compress_image(image)

        if MODEL_SHAPE_TYPE.startswith('rect'):
            x_test.append(image)
            x_test_compressed.append(compressed_image)
        else:
            v_flipped_image = np.flip(image, 0)
            square_image = np.vstack([image, v_flipped_image])
            x_test.append(square_image)

            v_flipped_image_compressed = np.flip(compressed_image, 0)
            square_image_compressed = np.vstack([compressed_image, v_flipped_image_compressed])
            x_test_compressed.append(square_image_compressed)


print('Data Loading... Finished.')

x_test = np.array(x_test)
x_test_compressed = np.array(x_test_compressed)
y_test = np.array(y_test)
y_test = np.true_divide(y_test, 2767.1)

if K.image_data_format() == 'channels_first':
    x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
    y_test = y_test.reshape(y_test.shape[0], channels, img_rows, img_cols)
    x_test_compressed = x_test_compressed.reshape(x_test.shape[0], channels * img_rows_compressed * img_cols_compressed)
    # y_test_compressed = y_test.reshape(y_test.shape[0], channels * img_rows_compressed * img_cols_compressed)
    input_shape = (channels, img_rows, img_cols)
    input_shape_compressed = channels*img_rows_compressed*img_cols_compressed
else:
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
    x_test_compressed = x_test_compressed.reshape(x_test_compressed.shape[0], channels*img_rows_compressed*img_cols_compressed)
    # y_test_compressed = y_test.reshape(y_test.shape[0], channels*img_rows_compressed*img_cols_compressed)
    input_shape = (img_rows, img_cols, channels)
    input_shape_compressed = channels * img_rows_compressed * img_cols_compressed

result = dict()
result['real'] = x_test
x_axis = range(400, 1600, 50)
# fig, ax = plt.subplots(1, 1, figsize=(14, 7))
# ax.plot(x_axis, y_test, label='real', color='black')
MODEL_JSON_PATH = ''
MODEL_H5_PATH = ''
myeongjo = 'NanumMyeongjo'

mask_array = np.ones_like(y_test, np.bool)

for j in range(len(y_test)):
    peaks_positive, _ = find_peaks(y_test[j], height=0)
    peaks_negative, _ = find_peaks(1 - y_test[j], height=0)
    mask = np.ones(len(y_test[j]), np.bool)
    mask[peaks_positive] = 0
    mask[peaks_negative] = 0
    mask_array[j][mask] = 0

result_runningTime = dict()
result_r2 = dict()
result_r2_local_minmax = dict()
result_rmse = dict()
result_rmse2 = dict()
result_diff_rmse = dict()
result_rmse_add_diff_rmse = dict()
result_poly = dict()
rmse_for_boxplot = dict()
rmse_local_for_boxplot = dict()


result_list = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Select model type.", default="cnn")
    parser.add_argument("-s", "--shape", help="Select input image shape. (rectangle or square?)", default='rect')
    parser.add_argument("-l", "--loss_function", help="Select loss functions.. (rmse,diff_rmse,diff_ce)",
                        default='rmse')
    parser.add_argument("-lr", "--learning_rate", help="Set learning_rate", type=float, default=0.001)
    parser.add_argument("-e", "--max_epoch", help="Set max epoch", type=int, default=100)
    parser.add_argument("-b", "--batch_size", help="Set batch size", type=int, default=128)
    parser.add_argument("-n", "--is_normalized", help="Set is Normalized", action='store_true')

    # arg for AL
    parser.add_argument("-it", "--iteration", help="Set iteration for training", type=int,  default=1)
    parser.add_argument("-a", "--is_active_learning", help="Set is Active Learning", action='store_true')
    parser.add_argument("-r", "--labeled_ratio", help="Set R", type=float, default=0.2)
    parser.add_argument("-t", "--top_ratio", help="Set T", type=float, default=0.1)

    # arg for testing parameters
    parser.add_argument("-u", "--unit_test", help="flag for testing source code", action='store_true')
    parser.add_argument("-d", "--debug", help="flag for debugging", action='store_true')


    args = parser.parse_args()

    # Set model, result folder
    model_folder_path = 'models_al'

    model_folder_name = '{}_bs{}_e{}_lr{}'.format(
        args.model, args.batch_size, args.max_epoch, args.learning_rate
    )

    if args.is_active_learning:
        model_folder_name = '{}_al_from_l0_r{}_t{}_bs{}_e{}_lr{}'.format(
            args.model, args.labeled_ratio, args.top_ratio, args.batch_size, args.max_epoch, args.learning_rate
        )
        if args.is_different_losses:
            model_folder_name = '{}_al_from_l0_w_diff_losses_r{}_t{}_bs{}_e{}_lr{}'.format(
                args.model, args.labeled_ratio, args.top_ratio, args.batch_size, args.max_epoch, args.learning_rate
            )

    result_folder_path = 'result_al'
    model_export_path_folder = '{}/scatter_alpha/{}'.format(result_folder_path, model_folder_name)

    if not os.path.exists(model_export_path_folder):
        os.makedirs(model_export_path_folder)

    folder_path_template = '{}/{}/*.h5'
    search_template = folder_path_template.format(model_folder_path, model_folder_name)
    print(search_template)
    files = glob.glob(search_template)
    print('model file paths', files)

    for i, model_file_path in enumerate(files):
        model_file_folder, model_file_name = os.path.split(model_file_path)
        _, parsed_model_name = os.path.split(model_file_folder)

        runningTime = 0
        if parsed_model_name.startswith('cnn') or parsed_model_name.startswith('nn'):
            # parsed_model_name = '{}_{}'.format(parsed_model_name, model_file_name)
            # parsed_model_name = model_file_name

            loss_function = CustomLoss(args.loss_function)
            loaded_model = load_model(model_file_path, compile=False)
            # loaded_model.compile(loss=loss_function)
            print("Loaded model({}) from disk".format(parsed_model_name))

            if parsed_model_name.startswith('cnn'):
                tic()
                y_predict = loaded_model.predict(x_test)
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

        if is_mean_std == True:
            MEAN = 0.5052
            STD = 0.2104
            y_predict = rescale(y_predict, MEAN, STD)


        # corr = np.corrcoef(y_test, y_predict)[0, 1]
        r2 = r2_score(y_test, y_predict)

        meanSquaredError = mean_squared_error(y_test, y_predict)
        rmse = sqrt(meanSquaredError)

        rmse_all = []
        count = 0

        message = 'r2:{0:.4f}, RMSE:{1:.4f}'.format(r2, rmse)

        parsed_model_name = model_file_name
        rmse_for_boxplot[parsed_model_name] = rmse_all
        y_test_for_local_minmax = y_test[mask_array]
        y_predict_for_local_minmax = y_predict[mask_array]

        y_test_for_local_minmax_inverse = y_test[~mask_array]
        y_predict_for_local_minmax_inverse = y_predict[~mask_array]

        rmse2 = sqrt(mean_squared_error(y_test_for_local_minmax, y_predict_for_local_minmax))
        r2_local_minmax = r2_score(y_test_for_local_minmax, y_predict_for_local_minmax)
        result_rmse2[parsed_model_name] = rmse2
        result_r2[parsed_model_name] = r2
        result_r2_local_minmax[parsed_model_name] = r2_local_minmax
        result_rmse[parsed_model_name] = rmse
        result_runningTime[parsed_model_name] = runningTime

        y_test_diff = tf_diff_axis_1(y_test)
        y_predict_diff = tf_diff_axis_1(y_predict)
        mse_diff = mean_squared_error(y_test_diff, y_predict_diff)
        rmse_diff = sqrt(mse_diff)
        result_diff_rmse[parsed_model_name] = rmse_diff

        result_rmse_add_diff_rmse[parsed_model_name] = rmse_diff + rmse

        plt.scatter(y_predict_for_local_minmax_inverse, y_test_for_local_minmax_inverse, s=3, alpha=0.3, label='all', marker='+')
        plt.scatter(y_predict_for_local_minmax, y_test_for_local_minmax, s=2, alpha=0.3, label='local_minmax', marker='.')
        # x_margin = -0.05
        x_margin = 0
        plt.text(x_margin, 1, 'R² = %0.4f' % r2)
        plt.text(x_margin, 0.95, 'RMSE = %0.4f' % rmse)
        plt.text(x_margin, 0.9, 'local minmax R² = %0.4f' % r2_local_minmax)
        plt.text(x_margin, 0.85, 'local minmax RMSE = %0.4f' % rmse2)
        plt.xlabel('Predictions')
        plt.ylabel('Actual')
        plt.savefig("{}/{}.png".format(model_export_path_folder, parsed_model_name))
        plt.clf()

    print('running time:', result_runningTime)
    print('rmse: ', result_rmse)
    print('rmse local minmax: ', result_rmse2)
    print('r2: ', result_r2)
    print('r2-local: ', result_r2_local_minmax)

    import csv
    with open('{}/r2.csv'.format(model_export_path_folder), 'w') as csvfile:
        for key in result_r2.keys():
            csvfile.write("%s,%s\n" % (key, result_r2[key]))

    with open('{}/rmse.csv'.format(model_export_path_folder), 'w') as csvfile:
        for key in result_rmse.keys():
            csvfile.write("%s,%s\n" % (key, result_rmse[key]))

    with open('{}/rmse-local.csv'.format(model_export_path_folder), 'w') as csvfile:
        for key in result_rmse2.keys():
            csvfile.write("%s,%s\n" % (key, result_rmse2[key]))

    with open('{}/r2-locaal.csv'.format(model_export_path_folder), 'w') as csvfile:
        for key in result_r2_local_minmax.keys():
            csvfile.write("%s,%s\n" % (key, result_r2_local_minmax[key]))