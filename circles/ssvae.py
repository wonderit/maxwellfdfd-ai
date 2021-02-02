import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn, optim
import torch.utils.data as tud
from sklearn.metrics import mean_squared_error, r2_score
print(torch.__version__)

## TRAIN
TRAIN_DATA_PATH = "./data/wv_image_train_0128_(77684,160,160).npz"
VALIDATION_DATA_PATH = "./data/wv_image_val_0128_(9711,160,160).npz"
TEST_DATA_PATH = "./data/wv_image_test_0128_(9711,160,160).npz"

# BATCH_SIZE
DATA_SIZE = 8000

class TwoLayerConvNet(nn.Module):
    """
    A 2-layer convolutional NN with dropout and batch-normalization
    Dimension progression:
        (if raw resolution = 128). 128*128*3 -> 128*128*10 -> 64*64*10 -> 64*64*20 -> 16*16*20 -> 64 -> 10
    """

    def __init__(self, image_reso, filter_size, dropout_rate):
        super(TwoLayerConvNet, self).__init__()

        self.best_dev_accuracy = 0

        assert filter_size % 2 == 1, "filter_size = %r but it has to be an odd number" % filter_size

        # 3 input channels, 10 output channels, 5 x 5 filter size
        # self.conv1_drop = nn.Dropout2d(dropout_rate)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=filter_size, stride=1, padding=(filter_size - 1) // 2)
        # self.bn1 = nn.BatchNorm2d(num_features=10)

        # reduce spatial dimension by 2 times

        # 10 input channels, 20 output channels, 5 x 5 filter size
        # self.conv2_drop = nn.Dropout2d(dropout_rate)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=filter_size, stride=1, padding=(filter_size - 1) // 2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=filter_size, stride=1, padding=(filter_size - 1) // 2)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=filter_size, stride=1, padding=(filter_size - 1) // 2)
        # self.bn2 = nn.BatchNorm2d(num_features=20)

        # reduce spatial dimension by 4 times

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # self.fc1 = nn.Linear(20 * (image_reso // 8) * (image_reso // 8) + 1, 64)
        # self.fc2 = nn.Linear(64, 4)
        self.fc1 = nn.Linear(6401, 2048)
        self.fc1_drop = nn.Dropout(0.4)
        self.fc2 = nn.Linear(2048, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x_image, x_wv):
        # 3 x 96 x 96 -> 10 x 96 x 96
        # print('x_image shape : ', x_image.shape)
        # x = self.conv1_drop(x_image)
        x = self.conv1(x_image)
        # x = self.bn1(x)
        x = F.relu(x)

        # 10 x 96 x 96 -> 10 x 48 x 48
        x = self.pool1(x)

        # 10 x 48 x 48 -> 20 x 48 x 48
        # x = self.conv2_drop(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool4(x)

        # resize the 2d representation to a vector
        # x = x.view(-1, 20 * 12 * 12)
        # print('x before', x.shape)
        x = x.view(x.shape[0], -1)

        # print('x after', x.shape)
        #
        # print('xv shape', x_wv.shape)
        x_wv = x_wv.view(x_wv.shape[0], 1).float()

        # add x_wv
        # print(x.type(), x_wv.type())
        x = torch.cat((x, x_wv), dim=1)

        # 1 x (20 * 12 * 12) -> 1 x 64
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc1(x))

        # 1 x 64 -> 1 x 10
        x = torch.sigmoid(self.fc3(x))

        return x


class BaselineConvNet(object):
    """
    Packaged version of the baseline CNN model, including train and test function
    Parameters:
    filter_size: filter size for ConvNet
    dropout_rate: drop out rate for Conv layer
    image_reso: 64, 128, or 256. The size of the input image
    lr: learning rate for the optimizerfloat, learning rate (default: 0.001)
    batch_size: int, batch size (default: 128)
    cuda: bool, whether to use GPU if available (default: True)
    """

    def __init__(self, image_reso=160, path="baseline.pth", filter_size=3, dropout_rate=.2,
                 lr=1.0e-3, batch_size=10, cuda=True):

        self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

        self.model = TwoLayerConvNet(image_reso, filter_size, dropout_rate)
        self.model.to(self.device)
        self.path = path
        self.image_reso = image_reso
        self.filter_size = filter_size
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.trainset_loader = None
        self.testset_loader = None

        self.initialize()

    def train(self, epoch, log_interval=10):
        self.model.train()
        iteration = 0
        best_dev_accuracy = 0
        for ep in range(epoch):
            for batch_idx, ([data_image, data_wv], target) in enumerate(self.trainset_loader):
                data_image = data_image.view(-1, 1, data_image.shape[1], data_image.shape[2])
                # print('di, dw', data_image.shape, data_wv.shape)
                data_image, data_wv, target = data_image.to(self.device), data_wv.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data_image, data_wv)

                batch_size = output.shape[0]
                # loss = ((output - target) ** 2).sum() / batch_size
                #
                loss = nn.MSELoss(reduction='mean')(output, target.float())

                # print('loss type: ', loss.type(), output.type(), target.type())
                # loss = F.nll_loss(output, target)
                loss.backward()
                self.optimizer.step()
                if iteration % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        ep, batch_idx * len(data_image), len(self.trainset_loader.dataset),
                            100. * batch_idx / len(self.trainset_loader), loss.item()))
                iteration += 1
            dev_accuracy = self.dev()
            if dev_accuracy > best_dev_accuracy:
                best_dev_accuracy = dev_accuracy
                self.model.best_dev_accuracy = best_dev_accuracy
                torch.save(self.model, self.path)

    # dev set evaluation
    def dev(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for [data_image, data_wv], target in self.testset_loader:
                data_image = data_image.view(-1, 1, data_image.shape[1], data_image.shape[2])
                data_image, data_wv, target = data_image.to(self.device), data_wv.to(self.device), target.to(self.device)
                # data, target = data.to(self.device), target.to(self.device)
                output = self.model(data_image, data_wv)

                # target = target.view(target.shape[0], -1)

                test_loss += ((output - target) ** 2).sum()
                # test_loss += nn.MSELoss(reduction='mean')(output, target)
                # test_loss += F.mse_loss(output, target).item()
                # test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
                # pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                # correct += pred.eq(target.view_as(pred)).sum().item()
                print(target.shape, output.shape)
                r2 = r2_score(target.data.numpy(), output.data.numpy())
                print(target.data.numpy())

                print('output', output.data.numpy())
                import matplotlib.pyplot as plt
                # plot and show learning process
                plt.cla()

                plt.scatter(output.data.numpy(), target.data.numpy())
                # plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=2)
                plt.text(0.5, 0, 'Loss=%.4f' % test_loss, fontdict={'size': 10, 'color': 'red'})
                plt.pause(0.1)


        test_loss /= len(self.testset_loader.dataset)
        print('\nDev set: Average loss: {:.4f}, r^2: {:.0f}%\n'.format(
            test_loss, r2))
            # 100. * correct / len(self.testset_loader.dataset)))
        return r2

    # test set evaluation
    def test(self, testset_loader, path, return_confusion_matrix=False):

        if not torch.cuda.is_available():
            self.model = torch.load(path, map_location='cpu')
        else:
            self.model = torch.load(path)
        self.model.eval()
        correct = 0
        with torch.no_grad():
            confusion_matrix = torch.zeros(10, 10)
            for data, target in testset_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

                for t, p in zip(target.view(-1), pred.view(-1)):  # make confusion matrix
                    confusion_matrix[t.long(), p.long()] += 1

                correct += pred.eq(target.view_as(pred)).sum().item()

        print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(testset_loader.dataset),
            100. * correct / len(testset_loader.dataset)))

        if return_confusion_matrix:
            return (confusion_matrix, confusion_matrix.diag() / confusion_matrix.sum(1))

    def fit(self, train_loader, test_loader):

        self.trainset_loader = train_loader
        self.testset_loader = test_loader

        return

    def initialize(self):
        """
        Model Initialization
        """
        nn.init.xavier_uniform_(self.model.conv1.weight)
        nn.init.xavier_uniform_(self.model.conv2.weight)
        nn.init.xavier_uniform_(self.model.fc1.weight)
        nn.init.xavier_uniform_(self.model.fc2.weight)

        return
class MetaPrismDataset(Dataset):
    """ Diabetes dataset."""

    # Initialize your data, download, etc.
    def __init__(self, data_path):
        self.data_path = data_path
        npz_data = np.load(self.data_path)
        print('datapath : ', self.data_path)
        self.len = npz_data['image'].shape[0]
        self.x_image_data = torch.from_numpy(npz_data['image'])
        self.x_wv_data = torch.from_numpy(npz_data['wv'])
        self.y_data = torch.from_numpy(npz_data['y'])

    def __getitem__(self, index):
        return [self.x_image_data[index], self.x_wv_data[index]], self.y_data[index]

    def __len__(self):
        return self.len


dataset = MetaPrismDataset(data_path=TRAIN_DATA_PATH)
train_loader = DataLoader(dataset=dataset,
                          batch_size=8000,
                          shuffle=False,
                          num_workers=0)

[data_images, data_wvs], labels = next(iter(train_loader))

# keep 1500 as labeled data
np.random.seed(5)
labeled_index = np.random.choice(DATA_SIZE,int(DATA_SIZE*0.2), replace=False)
unlabeled_index = np.setdiff1d(list(range(DATA_SIZE)), labeled_index)
labels = labels.numpy()
np.put(labels, list(unlabeled_index), 10)

#make 0.3 of the labeled data dev set, dev set is made sure to have balanced labels
np.random.seed(5)
dev_index = labeled_index[np.random.choice(int(DATA_SIZE*0.2),int(DATA_SIZE*0.2*0.2), replace = False)]

train_index = np.setdiff1d(list(range(DATA_SIZE)), dev_index)

#prepare dataloader for pytorch
class TorchInputData(tud.Dataset):
    """
    A simple inheretance of torch.DataSet to enable using our customized DogBreed dataset in torch
    """
    def __init__(self, X_image, X_wv, Y, transform=None):
        """
        X: a list of numpy images
        Y: a list of labels coded using 0-9
        """
        self.X_image = X_image
        self.X_wv = X_wv
        self.Y = Y

    def __len__(self):
        return len(self.X_wv)

    def __getitem__(self, idx):
        x = [self.X_image[idx], self.X_wv[idx]]
        y = self.Y[idx]

        return x, y


images_train = [data_images[i] / 116.0 for i in train_index]
wv_train = [data_wvs[i] for i in train_index]
# print('images_train, wv_train shape', images_train[0].shape, wv_train[0].shape)
trainset = TorchInputData(images_train, wv_train, labels[train_index])
train_loader = tud.DataLoader(trainset, batch_size=50, shuffle=True)
images_dev = [data_images[i] / 116.0 for i in dev_index]
wv_dev = [data_wvs[i] for i in dev_index]
devset = TorchInputData(images_dev, wv_dev, labels[dev_index])
dev_loader = tud.DataLoader(devset, batch_size=50, shuffle=True)
baseline = BaselineConvNet(160, path="baseline.pth", lr=1e-2)
print(baseline.model)

baseline.fit(train_loader, dev_loader)
baseline.train(10)

#best dev set accuracy
print(baseline.model.best_dev_accuracy)