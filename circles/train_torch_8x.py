import torch
import torch.nn as nn
import numpy as np
import argparse
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.externals import joblib
import os
from sklearn.metrics import r2_score, mean_squared_error

from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from pytorchtools import EarlyStopping

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
# # Hyper parameters
# num_epochs = 10
# num_classes = 24
# batch_size = 128
# learning_rate = 0.001


## TRAIN
TRAIN_1_DATA_PATH = "./data/wv_h580_train1_350000-160-160.npz"
TRAIN_2_DATA_PATH = "./data/wv_h580_train2_350000-160-160.npz"
TEST_DATA_PATH = "./data/wv_h580_test_69969-160-160.npz"


class MaxwellFDFDDataset(Dataset):
    def __init__(self, data_image, data_wv, target, transform=None):

        data_image = np.true_divide(data_image, 116).astype(np.uint8)
        self.data_image = torch.from_numpy(data_image).float()
        self.data_wv = torch.from_numpy(data_wv).float()
        self.target = torch.from_numpy(target).float()
        self.transform = transform

    def __getitem__(self, index):
        x_image = self.data[index]
        x_wv = self.data_wv[index]
        y = self.target[index]

        return x_image, x_wv, y

    def __len__(self):
        return len(self.data)

print('Converting to TorchDataset...')\

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Select model type.", default="cnn")
    parser.add_argument("-l", "--loss_function", help="Select loss functions.. (rmse,diff_rmse,diff_ce)",
                        default='rmse')
    parser.add_argument("-lr", "--learning_rate", help="Set learning_rate", type=float, default=0.001)
    parser.add_argument("-e", "--max_epoch", help="Set max epoch", type=int, default=10)
    parser.add_argument("-b", "--batch_size", help="Set batch size", type=int, default=128)

    #
    parser.add_argument("-tr", "--n_train", help="Set train set number", type=int, default=1000)
    parser.add_argument("-te", "--n_test", help="Set test set", type=int, default=100)
    parser.add_argument("-oe", "--is_ordinal_encoding", help="flag for ordinal encoding 'a'", action='store_true')
    parser.add_argument("-oh", "--is_onehot_encoding", help="flag for onehot encoding 'a'", action='store_true')

    # arg for testing parameters
    parser.add_argument("-u", "--unit_test", help="flag for testing source code", action='store_true')
    parser.add_argument("-d", "--debug", help="flag for debugging", action='store_true')

    # arg for rpo lossfunction
    parser.add_argument("-rm", "--remember_model", action='store_true')

    parser.add_argument("-o", "--optimizer", help="Select optimizer.. (sgd, adam, adamw)", default='adam')

    args = parser.parse_args()

    # TEST
    # args.unit_test = True
    args.debug = True
    # args.max_epoch = 1

    # Hyper parameters
    # num_epochs = 10
    num_classes = 4
    # batch_size = 128
    # learning_rate = 0.001

    model_name = args.model
    batch_size = int(args.batch_size)
    num_epochs = int(args.max_epoch)
    loss_functions = args.loss_function
    learning_rate = args.learning_rate

    img_rows, img_cols, channels = 160, 160, 1



    print('Data Loading... dataset Start.')
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
        print('preprocess1 end')

        x_train = train_1_data['wvtrain']
        y_train = train_1_data['ytrain']

        print('preprocess2 start')
        x_train_image_2 = train_2_data['xtrain']
        x_train_image_2 = x_train_image_2.astype(np.uint8)
        # x_train_image_2 = np.true_divide(x_train_image_2, 116).astype(np.uint8)
        print('preprocess2 end')

        x_train_image = np.append(x_train_image, x_train_image_2, axis=0)
        x_train = np.append(x_train, train_2_data['wvtrain'], axis=0)
        y_train = np.append(y_train, train_2_data['ytrain'], axis=0)

        x_validation_image = train_1_data['xtest']

        print('preprocess3 start')
        x_validation_image = x_validation_image.astype(np.uint8)
        # x_validation_image = np.true_divide(x_validation_image, 116).astype(np.uint8)
        print('preprocess3 end')

        x_validation = train_1_data['wvtest']
        y_validation = train_1_data['ytest']

        print('preprocess4 start')
        x_validation_image_2 = train_2_data['xtest']
        x_validation_image_2 = x_validation_image_2.astype(np.uint8)
        # x_validation_image_2 = np.true_divide(x_validation_image_2, 116).astype(np.uint8)
        print('preprocess4 end')

        x_validation_image = np.append(x_validation_image, x_validation_image_2, axis=0)
        x_validation = np.append(x_validation, train_2_data['wvtest'], axis=0)
        y_validation = np.append(y_validation, train_2_data['ytest'], axis=0)

        x_test_image = test_data['image']
        print('preprocess5 start')
        x_test_image = x_test_image.astype(np.uint8)
        # x_test_image = np.true_divide(x_test_image, 116).astype(np.uint8)
        print('preprocess5 end')
        x_test = test_data['wv']
        y_test = test_data['y']

        print('freeing memory start')
        del train_1_data, train_2_data, test_data, x_train_image_2, x_validation_image_2
        print('freeing memory end')

    print('Train, Valid, Test Data Loading finished (shape : train/valid/test > {}/{}/{})'.format(len(y_train),
                                                                                                  len(y_validation),
                                                                                                  len(y_test)))

    print('Training model args : batch_size={}, max_epoch={}, lr={}, loss_function={}'
          .format(args.batch_size, args.max_epoch, args.learning_rate, args.loss_function))


    # of train, test set
    n_train = len(y_train)
    n_validation = len(y_validation)
    n_test = len(y_test)

    # resize input column
    x_train = x_train.reshape(-1, 1)
    x_validation = x_validation.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)

    # get input columns
    input_column_size = x_train.shape[1]
    print('x_test data after', x_test.shape)

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

    train_set = MaxwellFDFDDataset(x_train_image, x_train, y_train, transform=False)

    valid_set = MaxwellFDFDDataset(x_validation_image, x_validation, y_validation, transform=False)

    test_set = MaxwellFDFDDataset(x_test_image, x_test, y_test, transform=False)

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
            return out


    model = ConvNet(num_classes).to(device)

    # Loss and optimizer
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Lr scheduler
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch, verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

    # Early Stopping
    early_stopping = EarlyStopping(patience=10, verbose = True)

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
        count = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Backward and optimize
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = torch.sqrt(criterion(outputs, labels))

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

                pred_array.extend(outputs.cpu().numpy().reshape(-1))
                labels_array.extend(labels.cpu().numpy().reshape(-1))
                total += labels.size(0)

            pred_array = np.array(pred_array)
            labels_array = np.array(labels_array)

            pred_array = pred_array.reshape(-1)
            labels_array = labels_array.reshape(-1)
            val_loss = np.sqrt(mean_squared_error(labels_array, pred_array))
            r2 = r2_score(y_true=labels_array, y_pred=pred_array)
            val_loss_array.append(val_loss)

            print('Validation Accuracy of the model on the {} validation images, loss: {:.4f}, R^2 : {:.4f} '.format(total, val_loss, r2))

            early_stopping(val_loss, model)

            scheduler.step(val_loss)


        if early_stopping.early_stop:
            print("Early stopping")
            break

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
        print('Test Accuracy of the model on the {} test images, loss: {:.4f}, R^2 : {:.4f} '.format(total, test_rmse, test_r2))

    # Save the model checkpoint
    model_file_name = '{}/model-{:.4f}-{:.4f}-ep{}-lr{}.ckpt'.format(torch_model_folder, test_rmse, test_r2, num_epochs, learning_rate)
    torch.save(model.state_dict(), model_file_name)


    # Save learning curve
    plt.clf()
    plt.plot(train_loss_array)
    plt.plot(val_loss_array)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model - Loss')
    plt.legend(['Training', 'Validation'], loc='upper right')
    log_curve_file_name = '{}/log-curve-{:.4f}-{:.4f}-ep{}-lr{}.png'.format(torch_loss_folder, test_rmse, test_r2, num_epochs,
                                                                             learning_rate)
    plt.savefig(log_curve_file_name)