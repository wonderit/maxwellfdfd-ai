import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from PIL import Image
import os
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from pytorchtools import EarlyStopping
import random

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Torch is running on Device : {}'.format(device))
#
# # Hyper parameters
# num_epochs = 10
# num_classes = 24
# batch_size = 128
# learning_rate = 0.001

# Set deterministic random seed
random_seed = 999
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


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


class MaxwellFDFDDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).float()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        return x, y

    def __len__(self):
        return len(self.data)


def scatter_plot(y_true, y_pred, message, result_path, iter_number, model_number):
    result = np.column_stack((y_true,y_pred))

    if not os.path.exists('{}/{}'.format(result_path, 'csv')):
        os.makedirs('{}/{}'.format(result_path, 'csv'))

    if not os.path.exists('{}/{}'.format(result_path, 'scatter')):
        os.makedirs('{}/{}'.format(result_path, 'scatter'))

    pd.DataFrame(result).to_csv("{}/csv/{}_{}.csv".format(result_path, iter_number, model_number), index=False)

    plt.clf()
    plt.scatter(y_pred, y_true, s=3)
    plt.suptitle(message)
    plt.xlabel('Predictions')
    plt.ylabel('Actual')
    plt.savefig("{}/scatter/{}_{}.png".format(result_path, iter_number, model_number))
    # plt.show()


print('Converting to TorchDataset...')

def mse_loss(input, target):
    return ((input - target) ** 2).sum() / input.data.nelement()

def sqrt_loss(input, target):
    return ((input-target) ** 0.5).sum() / input.data.nelement()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Select model type.", default="cnn")
    parser.add_argument("-s", "--shape", help="Select input image shape. (rectangle or square?)", default="rect")
    parser.add_argument("-l", "--loss_function", help="Select loss functions.. (rmse,diff_rmse,diff_ce)", default="rmse")
    parser.add_argument("-lr", "--learning_rate", help="Set learning_rate", type=float, default=0.001)
    parser.add_argument("-e", "--max_epoch", help="Set max epoch", type=int, default=10)
    parser.add_argument("-b", "--batch_size", help="Set batch size", type=int, default=128)

    # arg for testing parameters
    parser.add_argument("-u", "--unit_test", help="flag for testing source code", action='store_true')
    parser.add_argument("-d", "--debug", help="flag for debugging", action='store_true')

    # arg for rpo lossfunction
    parser.add_argument("-dl", "--is_different_losses", action='store_true')
    parser.add_argument("-dm", "--is_different_models", action='store_true')

    parser.add_argument("-o", "--optimizer", help="Select optimizer.. (sgd, adam, adamw)", default='adam')
    # arg for AL
    parser.add_argument("-it", "--iteration", help="Set iteration for AL", type=int, default=1)
    parser.add_argument("-n", "--num_models", help="Set number of models for active regressors", type=int, default=3)
    parser.add_argument("-a", "--is_active_learning", help="Set is AL", action='store_true')
    parser.add_argument("-ar", "--is_active_random", help="Set is AL random set", action='store_true')
    parser.add_argument("-r", "--labeled_ratio", help="Set R", type=float, default=0.2)
    parser.add_argument("-t", "--top_ratio", help="Set T", type=float, default=0.1)
    parser.add_argument("-ll", "--loss_lambda", help="set loss lambda", type=float, default=0.5)

    # arg for KD
    parser.add_argument("-rm", "--remember_model", action='store_true')
    parser.add_argument("-rms", "--remember_models", action='store_true')
    parser.add_argument("-w", "--weight", action='store_true')
    parser.add_argument("-tor", "--teacher_outlier_rejection", action='store_true')
    parser.add_argument("-tbr", "--teacher_bounded_regression", action='store_true')
    parser.add_argument("-tbra", "--tbr_addition", action='store_true')
    parser.add_argument("-z", "--z_score", type=float, default=2.0)
    parser.add_argument("-pl", "--pseudo_label", action='store_true')
    # arg for rpo type
    parser.add_argument("-rt", "--rpo_type", help="Select rpo type.. (max_diff, min_diff, random)", default='max_diff')
    parser.add_argument("-rts", "--rpo_type_schedule", help="rpo type scheduling", action='store_true')

    # arg for uncertainty attention
    parser.add_argument("-ua", "--uncertainty_attention", help="flag for uncertainty attention of gradients", action='store_true')
    args = parser.parse_args()

    # TEST
    # args.unit_test = True
    # args.debug = True
    # args.max_epoch = 1
    # args.is_active_learning = True

    # Hyper parameters
    # num_epochs = 10
    num_classes = 24
    # batch_size = 128
    # learning_rate = 0.001

    model_name = args.model
    batch_size = int(args.batch_size)
    num_epochs = int(args.max_epoch)
    loss_functions = args.loss_function
    learning_rate = args.learning_rate
    num_models = args.num_models

    img_rows, img_cols, channels = 100, 200, 1

    if args.unit_test:
        DATASETS_TRAIN = [
            'binary_501',
        ]
        DATASETS_VALID = [
            'binary_1004',
        ]
        args.debug = True
        args.max_epoch = 1
        args.iteration = 2

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

    print('Data Loading... Test dataset Start.')

    x_test = []
    y_test = []
    for data_test in DATASETS_TEST:
        dataframe = pd.read_csv(os.path.join(DATAPATH_TEST, '{}.csv'.format(data_test)), delim_whitespace=False,
                                header=None)
        dataset = dataframe.values

        # split into input (X) and output (Y) variables
        fileNames = dataset[:, 0]
        y_test.extend(dataset[:, 1:25])
        for idx, file in enumerate(fileNames):

            try:
                image = Image.open(os.path.join(DATAPATH_TEST, data_test, '{}.tiff'.format(int(file))))
                image = np.array(image, dtype=np.uint8)
            except (TypeError, FileNotFoundError) as te:
                image = Image.open(os.path.join(DATAPATH_TEST, data_test, '{}.tiff'.format(idx + 1)))
                try:
                    image = np.array(image, dtype=np.uint8)
                except:
                    continue

            x_test.append(image)
    print('Data Loading... Test dataset Finished.')


    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y_train = np.true_divide(y_train, 2767.1)

    x_validation = np.array(x_validation)
    y_validation = np.array(y_validation)
    y_validation = np.true_divide(y_validation, 2767.1)

    x_test = np.array(x_test)
    y_test = np.array(y_test)
    y_test = np.true_divide(y_test, 2767.1)

    # reshape dataset
    x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
    x_validation = x_validation.reshape(x_validation.shape[0], channels, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)

    # Dataset for AL Start
    ITERATION = args.iteration
    if args.is_active_learning:
        # import random
        n_row = int(x_train.shape[0])
        print('Labeled Dataset row : {}'.format(n_row))

        shuffled_indices = np.random.permutation(n_row)
        labeled_set_size = int(n_row * args.labeled_ratio)

        if args.is_active_random:
            labeled_set_size = labeled_set_size * 2

        # random_row = random.sample(list(range(n_row)), random_n_row)
        L_indices = shuffled_indices[:labeled_set_size]
        U_indices = shuffled_indices[labeled_set_size:]

        L_x = x_train[L_indices]
        L_y = y_train[L_indices]

        U_x = x_train[U_indices]
        U_y = y_train[U_indices]
        ITERATION = ITERATION + 1

        if args.pseudo_label:
            PL_x = L_x
            PL_y = L_y


    # Convolutional neural network (two convolutional layers)
    class ConvNet(nn.Module):
        def __init__(self, num_classes=24):
            super(ConvNet, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.layer3 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.layer4 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.fc1 = nn.Linear(4608, 1024)
            self.fc2 = nn.Linear(1024, num_classes)
            self.dropout = nn.Dropout(p=0.4)

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc1(out)
            out = self.dropout(out)
            out = self.fc2(out)
            out = torch.sigmoid(out)
            return out

        # create loss log folder
    al_type = 'al'
    if args.is_active_random:
        al_type = al_type + '_random'

    if args.pseudo_label:
        al_type = al_type + '_pl'

    if args.remember_model:
        al_type = al_type + '_rm'

    if args.remember_models:
        al_type = al_type + '_rms'

    if args.weight:
        al_type = al_type + '_weight'

    if args.teacher_outlier_rejection:
        al_type = al_type + '_tor_z{}_lambda{}'.format(args.z_score, args.loss_lambda)

    if args.teacher_bounded_regression:
        tbr_type = 'upper_bound'
        if args.tbr_addition:
            tbr_type = 'addition'
        al_type = al_type + '_tbr_{}_lambda{}'.format(tbr_type, args.loss_lambda)

    if args.uncertainty_attention:
        al_type = al_type + '_ua'

    log_folder = 'torch/{}_{}_{}_n{}_b{}_e{}_lr{}_it{}_R{}'.format(
        al_type, args.loss_function, args.rpo_type, args.num_models, batch_size, num_epochs, learning_rate, args.iteration, args.labeled_ratio
    )

    torch_loss_folder = '{}/train_progress'.format(log_folder)
    torch_model_folder = '{}/model'.format(log_folder)

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    if not os.path.exists(torch_loss_folder):
        os.makedirs(torch_loss_folder)

    if not os.path.exists(torch_model_folder):
        os.makedirs(torch_model_folder)

    prev_model = None
    prev_model_path = 'prev_model.pth'
    prev_models_path = 'prev_{}th_model.pth'
    prev_models = []
    for iter_i in range(ITERATION):
        print('Training Iteration : {}, Labeled dataset size : {}'.format(iter_i + 1, L_x.shape[0]))
        X_pr = []

        if args.debug:
            print('L_x, L_y shape:', L_x.shape, L_y.shape)
            print(L_x.shape[0], 'Labeled samples')
            print(U_x.shape[0], 'Unlabeled samples')
        if iter_i == (ITERATION - 1):
            num_models = 7
            is_different_losses = False

        #set data
        train_set = MaxwellFDFDDataset(L_x, L_y, transform=False)

        valid_set = MaxwellFDFDDataset(x_validation, y_validation, transform=False)

        test_set = MaxwellFDFDDataset(x_test, y_test, transform=False)

        # Data loader
        train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        valid_loader = torch.utils.data.DataLoader(dataset=valid_set,
                                                   batch_size=batch_size,
                                                   shuffle=False)

        test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                  batch_size=batch_size,
                                                  shuffle=False)

        if iter_i > 0 and args.pseudo_label:
            train_set = MaxwellFDFDDataset(PL_x, PL_y, transform=False)

            train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                       batch_size=batch_size,
                                                       shuffle=True)

        # Train model
        total_step = len(train_loader)

        # active regressor
        for m in range(num_models):
            print('Training models ({}/{}), Labeled data size: {}'.format(m + 1, num_models, total_step * batch_size))
            # train, val loss
            val_loss_array = []
            train_loss_array = []

            model = ConvNet(num_classes).to(device)


            # Initialize weights
            if args.remember_model and iter_i > 0:
                print('Get teacher model for tor loss')
                prev_model = ConvNet(num_classes).to(device)
                prev_model.load_state_dict(torch.load(prev_model_path))
                prev_model.eval()

                if args.weight:
                    print('Initializing model with previous model 0')
                    model.load_state_dict(torch.load(prev_model_path))
                model.train()

            # Loss and optimizer
            # criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            # Lr scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1,
                                                                   min_lr=learning_rate * 0.001, verbose=True)

            # Early Stopping
            early_stopping = EarlyStopping(patience=8, verbose=True)

            for epoch in range(num_epochs):
                model.train()
                train_loss = 0
                count = 0
                for i, (images, labels) in enumerate(train_loader):
                    images = images.to(device)
                    labels = labels.to(device)

                    # Backward and optimize
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(images)

                    if args.loss_function == 'rmse':
                        loss = torch.sqrt(mse_loss(outputs, labels))
                    elif args.loss_function == 'smoothl1':
                        loss = F.smooth_l1_loss(outputs, labels)
                    elif args.loss_function == 'l1':
                        loss = F.l1_loss(outputs, labels)
                    else:
                        loss = mse_loss(outputs, labels)

                    if args.teacher_outlier_rejection and iter_i > 0:
                        outputs_prev = prev_model(images)
                        mse_output_prev = (outputs_prev - labels) ** 2
                        z_flag_1 = ((mse_output_prev - mse_output_prev.mean()) / mse_output_prev.std()) > args.z_score
                        z_flag_0 = ((mse_output_prev - mse_output_prev.mean()) / mse_output_prev.std()) <= args.z_score
                        loss = loss + args.loss_lambda * (z_flag_1 * torch.sqrt(torch.abs(outputs-outputs_prev) + 1e-7) + z_flag_0 * (outputs-labels)**2).sum() / outputs.data.nelement()


                    if args.teacher_bounded_regression and iter_i > 0:
                        outputs_prev = prev_model(images)
                        mse_output_prev = (outputs_prev - labels) ** 2
                        mse_output = (outputs - labels) ** 2
                        flag = (mse_output - mse_output_prev) > 0
                        if args.tbr_addition:
                            loss = loss + args.loss_lambda * (((outputs-labels)**2).sum() / outputs.data.nelement())
                        else:
                            loss = loss + args.loss_lambda * (flag * (outputs - labels) ** 2).sum() / outputs.data.nelement()

                    loss.backward()

                    optimizer.step()

                    train_loss += loss.item()
                    count += 1

                    if (i + 1) % 10 == 0:
                        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                              .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

                train_loss_array.append(train_loss / count)


                # Validate the model
                # Test the model
                model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
                with torch.no_grad():
                    total = 0
                    pred_array = []
                    labels_array = []
                    for images, labels in valid_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = model(images)

                        pred_array.extend(outputs.cpu().detach().numpy().reshape(-1))
                        labels_array.extend(labels.cpu().detach().numpy().reshape(-1))

                        total += labels.size(0)

                    pred_array = np.array(pred_array)
                    labels_array = np.array(labels_array)

                    pred_array = pred_array.reshape(-1)
                    labels_array = labels_array.reshape(-1)
                    if np.any(np.isnan(pred_array)):
                        print('INPUT CONTAINS NAN ERROR!!!', )
                    # val_loss = torch.sqrt(mse_loss(outputs, labels))
                    val_loss = np.sqrt(mean_squared_error(labels_array, pred_array))
                    r2 = r2_score(y_true=labels_array, y_pred=pred_array, multioutput='uniform_average')
                    val_loss_array.append(val_loss)

                    print('Validation Accuracy of the model on the {} validation images, loss: {:.4f}, R^2 : {:.4f} '.format(total, val_loss, r2))

                    early_stopping(val_loss, model)

                    scheduler.step(val_loss)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            # Test the model
            model.eval()

            test_r2 = 0
            test_rmse = 0
            with torch.no_grad():
                total = 0
                pred_array = []
                labels_array = []
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)

                    pred_array.extend(outputs.cpu().numpy().reshape(-1))
                    labels_array.extend(labels.cpu().numpy().reshape(-1))

                    total += labels.size(0)

                pred_array = np.array(pred_array)
                labels_array = np.array(labels_array)

                pred_array = pred_array.reshape(-1)
                labels_array = labels_array.reshape(-1)
                print('labels array shape: {}, pred array shape: {}'.format(labels_array.shape, pred_array.shape))
                test_rmse = np.sqrt(mean_squared_error(labels_array, pred_array))
                # test_rmse = torch.sqrt(mse_loss(outputs, labels))
                test_r2 = r2_score(y_true=labels_array, y_pred=pred_array, multioutput='uniform_average')
                print('Test Accuracy of the model on the {} test images, loss: {:.4f}, R^2 : {:.4f} '.format(total, test_rmse, test_r2))
                scatter_plot(y_true=labels_array, y_pred=pred_array,
                             message='RMSE: {:.4f}, R^2: {:4f}'.format(test_rmse, test_r2),
                             result_path=log_folder,
                             iter_number=iter_i,
                             model_number=m)

            # Save the model checkpoint
            model_file_name = '{}/model_it{}_m{}-{:.4f}-{:.4f}-ep{}-lr{}.ckpt'.format(torch_model_folder, iter_i, m,
                                                                                      test_rmse, test_r2, num_epochs, learning_rate)
            torch.save(model.state_dict(), model_file_name)

            # remove m == 0
            if args.remember_model:
                print('ITERATION : {}, prev model updated'.format(iter_i))
                torch.save(model.state_dict(), prev_model_path)
                # prev_model = model

            if args.remember_models:
                print('ITERATION : {}, prev {}th model updated'.format(iter_i, m))
                torch.save(model.state_dict(), prev_models_path.format(m))

            # Save learning curve
            plt.clf()
            plt.plot(train_loss_array)
            plt.plot(val_loss_array)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Model - Loss')
            plt.legend(['Training', 'Validation'], loc='upper right')
            log_curve_file_name = '{}/log-curve-it{}-m{}-{:.4f}-{:.4f}-ep{}-lr{}.png'.format(torch_loss_folder,
                                                                                             iter_i, m,
                                                                                             test_rmse, test_r2, num_epochs,
                                                                                     learning_rate)
            plt.savefig(log_curve_file_name)

            # AL start
            if not args.is_active_random and args.is_active_learning:
                active_set = MaxwellFDFDDataset(U_x, U_y, transform=False)
                # Data loader
                active_loader = torch.utils.data.DataLoader(dataset=active_set,
                                                            batch_size=batch_size,
                                                            shuffle=False)
                model.eval()
                with torch.no_grad():
                    x_pr_active = []
                    for (active_images, active_labels) in active_loader:
                        torch_U_x_image = active_images.to(device)
                        predict_from_model = model(torch_U_x_image)

                        np_pred = predict_from_model.cpu().data.numpy()
                        x_pr_active.extend(np_pred)
                    x_pr_active = np.array(x_pr_active)
                    X_pr.append(x_pr_active)

        if not args.is_active_random:
            X_pr = np.array(X_pr)

            # Ascending order Sorted
            rpo_array = np.max(X_pr, axis=0) - np.min(X_pr, axis=0)
            rpo_array_sum = np.sum(rpo_array, axis=1)

            if args.rpo_type == 'max_diff' or args.rpo_type == 'mid_diff' or args.rpo_type == 'max_random_diff':
                rpo_array_arg_sort = np.argsort(rpo_array_sum)
            elif args.rpo_type == 'random':
                rpo_array_arg_sort = np.random.permutation(len(rpo_array_sum))
            else:
                rpo_array_arg_sort = np.argsort(-rpo_array_sum)

            # add labeled to L_iter
            T_indices = int(len(x_train) * args.labeled_ratio * args.top_ratio)
            U_length = len(rpo_array_arg_sort) - T_indices
            if args.rpo_type == 'mid_diff':
                start_idx = int(U_length / 2)
                U_indices = rpo_array_arg_sort[:start_idx]
                U_indices = np.append(U_indices, rpo_array_arg_sort[start_idx+T_indices:], axis=0)

                L_indices = rpo_array_arg_sort[start_idx:start_idx+T_indices]
            elif args.rpo_type == 'max_random_diff':
                T_length_half = int(T_indices / 2)
                U_length = len(rpo_array_arg_sort) - T_length_half
                U_indices = rpo_array_arg_sort[:U_length]
                L_indices = rpo_array_arg_sort[U_length:]

                # start random sampling for T/2
                random_u_indices = np.random.permutation(len(U_indices))
                U_length = len(random_u_indices) - T_length_half
                U_indices = random_u_indices[:U_length]
                L_indices = np.append(L_indices, random_u_indices[U_length:], axis=0)
            else:
                U_indices = rpo_array_arg_sort[:U_length]
                L_indices = rpo_array_arg_sort[U_length:]

            L_x = np.append(L_x, U_x[L_indices], axis=0)
            L_y = np.append(L_y, U_y[L_indices], axis=0)

            U_x = U_x[U_indices]
            U_y = U_y[U_indices]

            # if pseudo label
            if args.pseudo_label:
                X_pr_avg = np.average(X_pr, axis=0)
                X_pr_avg_U = X_pr_avg[U_indices]
                PL_x = np.append(L_x, U_x, axis=0)
                PL_y = np.append(L_y, X_pr_avg_U, axis = 0)
                # shuffle Pseudo Labeled data
                shuffle_index = np.random.permutation(len(PL_x))
                PL_x = PL_x[shuffle_index]
                PL_y = PL_y[shuffle_index]

        # shuffle Labeled data
        shuffle_index = np.random.permutation(len(L_x))
        L_x = L_x[shuffle_index]
        L_y = L_y[shuffle_index]