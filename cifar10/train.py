
import numpy as np
import argparse
import os
import pandas as pd
from scipy.stats import entropy
import random
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from networks import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'

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

def mse_loss(input, target):
    return ((input - target) ** 2).sum() / input.data.nelement()

def sqrt_loss(input, target):
    return ((input-target) ** 0.5).sum() / input.data.nelement()


def softmax(x):
    max_x = np.max(x)
    exp_x = np.exp(x - max_x)
    sum_exp_x = np.exp(exp_x)
    return exp_x / sum_exp_x

    # f_x = np.exp(x) / np.sum(np.exp(x))
    # return f_x

# def lr_decay(step):
#     epoch = step // (args.sample_number // batch_size)
#     # print(f'step:{step}, epoch:{epoch}, num_samples:{num_samples}, batch size:{batch_size}')
#     if epoch < 150:
#         return 1.0
#     elif epoch >= 150 and epoch < 250:
#         return 0.1
#     else:
#         return 0.01

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--loss_function", help="Select loss functions.. (rmse,diff_rmse,diff_ce)", default="ce")
    parser.add_argument("-lr", "--learning_rate", help="Set learning_rate", type=float, default=0.001)
    parser.add_argument("-e", "--max_epoch", help="Set max epoch", type=int, default=100)
    parser.add_argument("-b", "--batch_size", help="Set batch size", type=int, default=128)

    # arg for testing parameters
    parser.add_argument("-u", "--unit_test", help="flag for testing source code", action='store_true')
    parser.add_argument("-d", "--debug", help="flag for debugging", action='store_true')

    parser.add_argument("-o", "--optimizer", help="Select optimizer.. (sgd, adam, adamw)", default='adam')
    parser.add_argument("-bn", "--is_batch_norm", help="Set is_batch_norm", action='store_true')
    # arg for AL
    parser.add_argument("-it", "--iteration", help="Set iteration for AL", type=int, default=1)
    parser.add_argument("-n", "--num_models", help="Set number of models for active regressors", type=int, default=1)
    parser.add_argument("-a", "--is_active_learning", help="Set is AL", action='store_true')
    parser.add_argument("-ar", "--is_active_random", help="Set is AL random set", action='store_true')
    parser.add_argument("-k", "--sample_number", help="Set K", type=int, default=500)
    parser.add_argument("-ll", "--loss_lambda", help="set loss lambda", type=float, default=0.5)
    parser.add_argument("-rtl", "--rpo_type_lambda", help="max random data ratio", type=float, default=0.5)

    # arg for KD
    parser.add_argument("-rm", "--remember_model", action='store_true')
    parser.add_argument("-w", "--weight", action='store_true')
    parser.add_argument("-pl", "--pseudo_label", action='store_true')
    # arg for rpo type
    parser.add_argument("-rt", "--rpo_type", help="Select rpo type.. (max_diff, min_diff, random)", default='max_ce')
    parser.add_argument("-mt", "--model_type", help="Select model type.. (resnet, densenet)", default='resnet')

    # arg for uncertainty attention
    parser.add_argument("-ua", "--uncertainty_attention", help="flag for uncertainty attention of gradients", action='store_true')

    parser.add_argument("-sb", "--sigmoid_beta", help="beta of sigmoid", type=float, default=1.0)
    parser.add_argument("-uaa", "--uncertainty_attention_activation", help="flag for uncertainty attention of gradients",
                        default='sigmoid')
    parser.add_argument("-ut", "--uncertainty_attention_type", default='lambda_residual')
    parser.add_argument("-ual", "--uncertainty_attention_lambda", type=float, default=0.1)
    parser.add_argument("-uag", "--uncertainty_attention_grad", action='store_true')

    # arg for wd
    parser.add_argument("-wd", "--weight_decay", type=float, default=1e-4)
    parser.add_argument("-wds", "--weight_decay_schedule", action='store_true')

    # arg for gpu
    parser.add_argument("-g", "--gpu", help="set gpu num", type=int, default=0)
    parser.add_argument("-sn", "--server_num", help="set server_num", type=int, default=0)
    parser.add_argument("-rs", "--random_seed", help="set server_num", type=int, default=999)

    args = parser.parse_args()

    GPU_NUM = args.gpu
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')

    # Additional Infos
    if device.type == 'cuda':
        torch.cuda.set_device(device)  # change allocation of current GPU
        print('Current cuda device ', torch.cuda.current_device())  # check
        print(torch.cuda.get_device_name(GPU_NUM))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(GPU_NUM) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(GPU_NUM) / 1024 ** 3, 1), 'GB')

    # Set deterministic random seed
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    # TEST
    # args.unit_test = True

    # Hyper parameters
    num_classes = 10

    args.is_active_learning = True
    batch_size = int(args.batch_size)
    num_epochs = int(args.max_epoch)
    loss_functions = args.loss_function
    learning_rate = args.learning_rate
    num_models = args.num_models
    uncertainty_attention_grad = False
    if args.uncertainty_attention_grad:
        uncertainty_attention_grad = True



    if args.unit_test:
        args.debug = True
        args.max_epoch = 1
        args.iteration = 2
        args.is_active_learning = True
        args.uncertainty_attention = True
        args.loss_function = 'l1'
        args.sample_number = 50


    print('Training model args : batch_size={}, max_epoch={}, lr={}, loss_function={}, al={}, iter={}, K={}'
          .format(args.batch_size, args.max_epoch, args.learning_rate, args.loss_function, args.is_active_learning,
                  args.iteration, args.sample_number))

    print('Data Loading... Train dataset Start.')

    # Image preprocessing modules
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    #
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    train_dataset = datasets.CIFAR10(root=f'./data/',
                                     train=True,
                                     download=False,
                                     transform=transform)
    test_dataset = datasets.CIFAR10(root='data/',
                                    train=False,
                                    download=False,
                                    transform=transform)

    # Dataset for AL Start
    ITERATION = args.iteration
    if args.is_active_learning:
        # import random
        n_row = len(train_dataset)
        print('Labeled Dataset row : {}'.format(n_row))

        shuffled_indices = np.random.permutation(n_row)
        labeled_set_size = args.sample_number

        if args.is_active_random:
            labeled_set_size = labeled_set_size * args.iteration

        # random_row = random.sample(list(range(n_row)), random_n_row)
        L_indices = shuffled_indices[:labeled_set_size]
        U_indices = shuffled_indices[labeled_set_size:]

        labeled_set = torch.utils.data.Subset(train_dataset, L_indices)
        unlabeled_set = torch.utils.data.Subset(train_dataset, U_indices)


    # create loss log folder
    al_type = f'al_g{args.gpu}_s{args.server_num}_rs{random_seed}'
    if args.is_active_random:
        al_type = al_type + '_random'

    if args.pseudo_label:
        al_type = al_type + '_pl'

    if args.remember_model:
        al_type = al_type + '_rm'
    if args.is_batch_norm:
        al_type = al_type + '_bn'
    else:
        al_type = al_type + '_nobn'

    if args.weight:
        al_type = al_type + '_weight'

    if args.weight_decay_schedule:
        al_type = al_type + '_wds'

    if args.uncertainty_attention:
        if args.uncertainty_attention_activation == 'sigmoid':
            al_type = al_type + f'_ua{args.uncertainty_attention_type}_sb{args.sigmoid_beta}'
            if args.uncertainty_attention_type == 'lambda_residual':
                al_type = al_type + f'_ual{args.uncertainty_attention_lambda}'
        elif args.uncertainty_attention_activation == 'identity':
            al_type = al_type + f'_ua{args.uncertainty_attention_type}_ual{args.uncertainty_attention_lambda}'
        else:
            al_type = al_type + '_ua{}_{}'.format(args.uncertainty_attention_type, args.uncertainty_attention_activation)

    log_folder = 'torch/{}_{}_{}_{}_{}{}_wd{}_b{}_e{}_lr{}_it{}_K{}'.format(
        al_type, args.loss_function, args.optimizer, args.model_type, args.rpo_type, args.rpo_type_lambda, args.weight_decay, batch_size, num_epochs, learning_rate, args.iteration, args.sample_number
    )

    torch_loss_folder = '{}/train_progress'.format(log_folder)
    torch_ua_log_folder = '{}/ua'.format(log_folder)
    torch_model_result_text_folder = '{}/txt'.format(log_folder)

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    if not os.path.exists(torch_loss_folder):
        os.makedirs(torch_loss_folder)

    if not os.path.exists(torch_ua_log_folder):
        os.makedirs(torch_ua_log_folder)

    if not os.path.exists(torch_model_result_text_folder):
        os.makedirs(torch_model_result_text_folder)

    prev_model = None
    prev_model_path = 'prev_model_gpu{}_server{}.pth'.format(args.gpu, args.server_num)
    prev_models_path = 'prev_{}th_model_gpu{}_server{}.pth'
    acc = []
    prev_models = []
    uas = []
    uas_uaa = []
    uncertainty_attention = None
    if args.pseudo_label:
        pseudo_labeled_set = None
    for iter_i in range(ITERATION):
        print('Training Iteration : {}, Labeled dataset size : {}'.format(iter_i + 1, len(labeled_set)))
        X_pr = []

        if args.debug:
            print('labeled set', len(labeled_set))

        # Data loader
        train_loader = torch.utils.data.DataLoader(dataset=labeled_set,
                                                   batch_size=batch_size,
                                                   shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)

        if iter_i > 0 and args.pseudo_label:
            train_loader = torch.utils.data.DataLoader(dataset=pseudo_labeled_set,
                                                       batch_size=batch_size,
                                                       shuffle=False)

        # Train model
        total_step = len(train_loader)


        # active regressor
        for m in range(num_models):
            print('Training models ({}/{}), Labeled data size: {}'.format(m + 1, num_models, (iter_i+1) * args.sample_number))
            # train, val loss
            val_loss_array = []
            train_loss_array = []

            # model = ConvNet(num_classes).to(device)
            if args.model_type=='resnet':
                model = ResNet18().to(device)
            else:
                model = DenseNet121().to(device)
            # if device.type == 'cuda':
            #     model = torch.nn.DataParallel(model)

            # Initialize weights
            if args.remember_model and iter_i > 0:
                print('Get teacher model for tor loss')
                # prev_model = ConvNet(num_classes).to(device)
                if args.model_type == 'resnet':
                    prev_model = ResNet18().to(device)
                else:
                    prev_model = DenseNet121().to(device)
                prev_model.load_state_dict(torch.load(prev_model_path))
                prev_model.eval()

                if args.weight:
                    print('Initializing model with previous model 0')
                    model.load_state_dict(torch.load(prev_model_path))
                model.train()

            # Loss and optimizer
            # criterion = nn.MSELoss()
            # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            weight_decay = args.weight_decay

            # N1 / N2 * lambda = K / 2K
            if args.weight_decay_schedule:
                weight_decay = weight_decay * (0.5 ** iter_i)

            print(f'weight decay : {weight_decay}, iter_i:{iter_i}')

            if args.optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            elif args.optimizer == 'adamw':
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)

            # Lr scheduler

            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.5)
            # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160])
            # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_decay, last_epoch=-1)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1,
            #                                                        min_lr=learning_rate * 0.001, verbose=True)

            # Early Stopping
            # early_stopping = EarlyStopping(patience=8, verbose=True)
            # early_stopped_epoch = 0
            criterion = nn.CrossEntropyLoss()
            for epoch in range(num_epochs):
                model.train()
                train_loss = 0
                count = 0
                for i, (images, labels) in enumerate(train_loader):
                    images = images.to(device)
                    labels = labels.to(device)

                    # Forward pass
                    outputs = model(images)

                    if args.uncertainty_attention and uncertainty_attention is not None:
                       # uncertainty_attention_resize = np.array(num_classes * [uncertainty_attention]).T
                        ua_end = batch_size * i + batch_size
                        ua_start = batch_size * i
                        if ua_end < len(uncertainty_attention):
                            batch_ua = uncertainty_attention[ua_start:ua_end]
                        else:
                            batch_ua = uncertainty_attention[ua_start:]
                        batch_ua_torch = torch.from_numpy(batch_ua).to(device)
                        batch_ua_torch.requires_grad = uncertainty_attention_grad

                        if args.uncertainty_attention_type == 'multiply':
                            loss = (torch.abs(outputs - labels) * batch_ua_torch).sum() / outputs.data.nelement()
                        elif args.uncertainty_attention_type == 'residual':
                            log_softmax = torch.nn.LogSoftmax(dim=1)
                            x_log = log_softmax(outputs)
                            #loss = (-x_log[range(labels.shape[0]), labels] * (1. + args.uncertainty_attention_lambda * batch_ua_torch)).sum() / outputs.data.nelement()
                            loss = (-x_log[range(labels.shape[0]), labels]* (1. + batch_ua_torch)).mean()
                            #loss = (torch.abs(outputs - labels) * (1. + batch_ua_torch)).sum() / outputs.data.nelement()
                        elif args.uncertainty_attention_type == 'add':
                            loss = (torch.abs(
                                outputs - labels) + args.uncertainty_attention_lambda * batch_ua_torch).sum() / outputs.data.nelement()
                        elif args.uncertainty_attention_type == 'lambda_residual':
                            log_softmax = torch.nn.LogSoftmax(dim=1)
                            x_log = log_softmax(outputs)
                            loss = (-x_log[range(labels.shape[0]), labels] * (1. + args.uncertainty_attention_lambda * batch_ua_torch)).mean()
                        elif args.uncertainty_attention_type == 'lambda_residual_minus':
                            log_softmax = torch.nn.LogSoftmax(dim=1)
                            x_log = log_softmax(outputs)
                            loss = (-x_log[range(labels.shape[0]), labels] * (
                                        1. - args.uncertainty_attention_lambda * batch_ua_torch)).mean()
                        else:
                            loss = nn.CrossEntropyLoss()(outputs, labels)
                    else:
                        log_softmax = torch.nn.LogSoftmax(dim=1)
                        x_log = log_softmax(outputs)
                        loss = (-x_log[range(labels.shape[0]), labels]).mean()
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()

                    optimizer.step()
                    scheduler.step()

                    train_loss += loss.item()
                    count += 1

                    if (i + 1) % 10 == 0:
                        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')

                train_loss_array.append(train_loss / count)

                # # Test the model
                # model.eval()
                # test_running_loss = 0
                # count = 0
                # with torch.no_grad():
                #     for images, labels in test_loader:
                #         images = images.to(device)
                #         labels = labels.to(device)
                #         outputs = model(images)
                #         test_loss = nn.CrossEntropyLoss()(outputs, labels)
                #     test_running_loss += test_loss.item()
                #     count += 1
                # val_loss_array.append(test_running_loss / count)


            # Test the model
            model.eval()
            acc = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for (images, labels) in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                acc = 100 * correct / total
                print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))


            # Save the model result text
            model_file_result = f'{torch_model_result_text_folder}/model_it{iter_i}_m{m}_acc{acc}_lr{learning_rate}.txt'

            with open(model_file_result, "w") as f:
                f.write(f'{model_file_result}')

            # remove m == 0
            if args.remember_model:
                print(f'ITERATION : {iter_i}, prev model updated')
                torch.save(model.state_dict(), prev_model_path)
                # prev_model = model

            if args.uncertainty_attention:
                print(f'ITERATION : {iter_i}, prev {m}th model updated')
                torch.save(model.state_dict(), prev_models_path.format(m, args.gpu, args.server_num))

            # Save learning curve
            plt.clf()
            plt.plot(train_loss_array)
            plt.plot(val_loss_array)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Model - Loss')
            plt.legend(['Training', 'Test'], loc='upper right')
            log_curve_file_name = f'{torch_loss_folder}/log-curve-it{iter_i}-m{m}-acc{acc}-lr{learning_rate}.png'
            plt.savefig(log_curve_file_name)

            # AL start
            if not args.is_active_random and args.is_active_learning:
                # Data loader
                active_loader = torch.utils.data.DataLoader(dataset=unlabeled_set,
                                                            batch_size=batch_size,
                                                            shuffle=False)
                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    x_pr_active = []
                    for active_images, active_labels in active_loader:
                        torch_U_x_image = active_images.to(device)
                        # labels = active_labels.to(device)
                        predict_from_model = model(torch_U_x_image)
                        np_pred = predict_from_model.cpu().data.numpy()
                        x_pr_active.extend(np_pred)

                    x_pr_active = np.array(x_pr_active)
                    X_pr.extend(x_pr_active)

        if not args.is_active_random:
            X_pr = np.array(X_pr)
            rpo_array = []
            # Ascending order Sorted
            for i in range(X_pr.shape[0]):
                softmax_pr = softmax(X_pr[i])
                rpo_array.append(entropy(softmax_pr))
            if args.rpo_type == 'random':
                rpo_array_arg_sort = np.random.permutation(len(rpo_array))
            else:
                rpo_array_arg_sort = np.argsort(rpo_array)

            # add labeled to L_iter
            T_indices = args.sample_number
            U_length = len(rpo_array_arg_sort) - T_indices
            if args.rpo_type == 'mid_diff':
                start_idx = int(U_length / 2)
                U_indices = rpo_array_arg_sort[:start_idx]
                U_indices = np.append(U_indices, rpo_array_arg_sort[start_idx+T_indices:], axis=0)

                L_indices = rpo_array_arg_sort[start_idx:start_idx+T_indices]
            elif args.rpo_type == 'max_random_diff':
                max_length = int(T_indices * args.rpo_type_lambda)
                U_length = len(rpo_array_arg_sort) - max_length
                U_indices = rpo_array_arg_sort[:U_length]
                L_indices = rpo_array_arg_sort[U_length:]

                # start random sampling for T/2
                random_u_indices = np.random.permutation(len(U_indices))
                random_length = int(T_indices * (1 - args.rpo_type_lambda))
                U_length = len(random_u_indices) - random_length
                U_indices = random_u_indices[:U_length]
                L_indices = np.append(L_indices, random_u_indices[U_length:], axis=0)
            else:
                U_indices = rpo_array_arg_sort[:U_length]
                L_indices = rpo_array_arg_sort[U_length:]

            active_labeled_set = torch.utils.data.Subset(unlabeled_set, L_indices)
            active_unlabeled_set = torch.utils.data.Subset(unlabeled_set, U_indices)
            labeled_set = torch.utils.data.ConcatDataset([labeled_set, active_labeled_set])
            unlabeled_set = active_unlabeled_set

        # shuffle Labeled data
        shuffle_index = np.random.permutation(len(labeled_set))
        labeled_set = torch.utils.data.Subset(labeled_set, shuffle_index)

        # Pseudo-labeled set

        # add uncertainty attention
        if args.uncertainty_attention:
            print('load model and calculate uncertainty for attention model')
            X_pr_L = []
            for ua_i in range(num_models):
                # prev_model = ConvNet(num_classes).to(device)
                if args.model_type == 'resnet':
                    prev_model = ResNet18().to(device)
                else:
                    prev_model = DenseNet121().to(device)
                prev_model.load_state_dict(torch.load(prev_models_path.format(ua_i, args.gpu, args.server_num)))
                prev_model.eval()
                # if args.pseudo_label:
                #     ua_set = MaxwellFDFDDataset(PL_x, PL_y, transform=False)
                # else:
                #     ua_set = MaxwellFDFDDataset(L_x, L_y, transform=False)
                # Data loader
                ua_loader = torch.utils.data.DataLoader(dataset=labeled_set,
                                                            batch_size=batch_size,
                                                            shuffle=False)
                prev_model.eval()
                with torch.no_grad():
                    X_pr_L_ua = []
                    for (active_images, active_labels) in ua_loader:
                        torch_L_x_image = active_images.to(device)
                        predict_from_model = prev_model(torch_L_x_image)

                        np_pred = predict_from_model.cpu().data.numpy()
                        X_pr_L_ua.extend(np_pred)
                    X_pr_L_ua = np.array(X_pr_L_ua)
                    X_pr_L.extend(X_pr_L_ua)
            X_pr_L = np.array(X_pr_L)
            rpo_array_l = []
            # Ascending order Sorted
            for i in range(X_pr_L.shape[0]):
                softmax_pr_l = softmax(X_pr_L[i])
                rpo_array_l.append(entropy(softmax_pr_l))

            rpo_ua_array_average = np.array(rpo_array_l)

            if args.uncertainty_attention_activation == 'sigmoid':
                uncertainty_attention = 1/(1 + np.exp(-args.sigmoid_beta * rpo_ua_array_average))
            elif args.uncertainty_attention_activation == 'std_sigmoid':
                std_ua = (rpo_ua_array_average - np.mean(rpo_ua_array_average)) / np.std(rpo_ua_array_average)
                uncertainty_attention = 1/(1 + np.exp(-args.sigmoid_beta * std_ua))
            elif args.uncertainty_attention_activation == 'minmax':
                minmax_ua = (rpo_ua_array_average - np.min(rpo_ua_array_average)) / (
                        np.max(rpo_ua_array_average) - np.min(rpo_ua_array_average)
                )
                uncertainty_attention = minmax_ua
            elif args.uncertainty_attention_activation == 'minmax_tanh':
                minmax_ua = (rpo_ua_array_average - np.min(rpo_ua_array_average)) / (
                        np.max(rpo_ua_array_average) - np.min(rpo_ua_array_average)
                )
                uncertainty_attention = np.tanh(minmax_ua)
            elif args.uncertainty_attention_activation == 'std_tanh':
                std_ua = (rpo_ua_array_average - np.mean(rpo_ua_array_average)) / np.std(rpo_ua_array_average)
                uncertainty_attention = np.tanh(std_ua)
            elif args.uncertainty_attention_activation == 'tanh':
                uncertainty_attention = np.tanh(rpo_ua_array_average)
            elif args.uncertainty_attention_activation == 'softplus':
                uncertainty_attention = np.log1p(np.exp(rpo_ua_array_average))
            else:
                uncertainty_attention = rpo_ua_array_average

            # boxplot logging
            uas.append(rpo_ua_array_average)
            uas_uaa.append(uncertainty_attention)
            ua_prev_plot_path = '{}/ua_boxplot_it{}.png'.format(torch_ua_log_folder, iter_i)
            ua_after_activation_plot_path = '{}/ua_{}_boxplot_it{}.png'.format(torch_ua_log_folder, args.uncertainty_attention_activation, iter_i)

            green_diamond = dict(markerfacecolor='r', marker='s')
            plt.close()
            plt.boxplot(uas, flierprops=green_diamond)
            plt.title("box plot ua")
            plt.savefig(ua_prev_plot_path, dpi=300)

            plt.close()
            plt.boxplot(uas_uaa, flierprops=green_diamond)
            plt.title("box plot ua activation: {}".format(args.uncertainty_attention_activation))
            plt.savefig(ua_after_activation_plot_path, dpi=300)
