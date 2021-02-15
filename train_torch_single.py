import torch
import torch.nn as nn
import numpy as np
import argparse
from PIL import Image
import os
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

from torch.utils.data import Dataset
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
# # Hyper parameters
# num_epochs = 10
# num_classes = 24
# batch_size = 128
# learning_rate = 0.001


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



# MNIST dataset
# train_dataset = torchvision.datasets.MNIST(root='../../data/',
#                                            train=True,
#                                            transform=transforms.ToTensor(),
#                                            download=True)
#
# test_dataset = torchvision.datasets.MNIST(root='../../data/',
#                                           train=False,
#                                           transform=transforms.ToTensor())


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

print('Converting to TorchDataset...')\

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Select model type.", default="cnn")
    parser.add_argument("-s", "--shape", help="Select input image shape. (rectangle or square?)", default='rect')
    parser.add_argument("-l", "--loss_function", help="Select loss functions.. (rmse,diff_rmse,diff_ce)",
                        default='rmse')
    parser.add_argument("-lr", "--learning_rate", help="Set learning_rate", type=float, default=0.001)
    parser.add_argument("-e", "--max_epoch", help="Set max epoch", type=int, default=10)
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
    # args.unit_test = True
    args.debug = True
    args.teacher_outlier_rejection = True
    # args.max_epoch = 1
    args.is_active_learning = True

    # Hyper parameters
    # num_epochs = 10
    num_classes = 24
    # batch_size = 128
    # learning_rate = 0.001

    model_name = args.model
    batch_size = int(args.batch_size)
    num_epochs = int(args.max_epoch)
    loss_functions = args.loss_function
    input_shape_type = args.shape
    learning_rate = args.learning_rate

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

    input_shape = (channels, img_rows, img_cols)

    train_set = MaxwellFDFDDataset(x_train, y_train, transform=False)

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

    # Convolutional neural network (two convolutional layers)
    class ConvNet(nn.Module):
        def __init__(self, num_classes=24):
            super(ConvNet, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(16),
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

            self.fc = nn.Linear(4608, num_classes)

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
            return out


    model = ConvNet(num_classes).to(device)

    # Loss and optimizer
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)

    # train, val loss
    val_loss_array = []
    train_loss_array = []

    # create loss log folder
    torch_loss_folder = 'torch/train_progress'
    torch_model_folder = 'torch/model'

    if not os.path.exists(torch_loss_folder):
        os.makedirs(torch_loss_folder)

    if not os.path.exists(torch_model_folder):
        os.makedirs(torch_model_folder)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = torch.sqrt(criterion(outputs, labels))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = loss.item()
            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, train_loss))

        train_loss_array.append(train_loss)


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

                pred_array.extend(outputs.cpu().numpy().reshape(-1))
                labels_array.extend(labels.cpu().numpy().reshape(-1))
                total += labels.size(0)

            pred_array = np.array(pred_array)
            labels_array = np.array(labels_array)

            pred_array = pred_array.reshape(-1)
            labels_array = labels_array.reshape(-1)
            rmse = np.sqrt(mean_squared_error(labels_array, pred_array))
            r2 = r2_score(y_true=labels_array, y_pred=pred_array)
            val_loss_array.append(rmse)
            print('Validation Accuracy of the model on the {} validation images, loss: {}, R^2 : {} '.format(total, rmse, r2))

    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

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
        test_r2 = r2_score(y_true=labels_array, y_pred=pred_array)
        print('Test Accuracy of the model on the {} test images, loss: {}, R^2 : {} '.format(total, test_rmse, test_r2))

    # Save the model checkpoint
    model_file_name = '{}/model-{}-{}-ep{}-lr{}.ckpt'.format(torch_model_folder, test_rmse, test_r2, num_epochs, learning_rate)
    torch.save(model.state_dict(), model_file_name)


    # Save learning curve
    plt.clf()
    plt.plot(train_loss_array)
    plt.plot(val_loss_array)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model - Loss')
    plt.legend(['Training', 'Validation'], loc='upper right')
    log_curve_file_name = '{}/log-curve-{}-{}-ep{}-lr{}.png'.format(torch_loss_folder, test_rmse, test_r2, num_epochs,
                                                                             learning_rate)
    plt.savefig(log_curve_file_name)