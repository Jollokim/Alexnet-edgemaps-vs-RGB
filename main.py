import math
import os.path

import torch.optim
import model
import argparse
import random

import numpy as np

from torch.utils.data import DataLoader
from timm import create_model
from torchsummary import summary

from datasets import AlexDataset
from engine import train_one_epoch, test_accuracy


# function for defining all the commandline parameters
def get_args_parser():
    parser = argparse.ArgumentParser('Main', add_help=False)

    # Model mode:
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'pass'], required=True,
                        help='train or test a model')
    # Model settings
    parser.add_argument('--name', type=str, help='Name of folder and weights file')
    parser.add_argument('--dataset', type=str, choices=['NI', 'DD'], required=True,
                        help='Choose what dataset you want to choose')
    parser.add_argument('--pretrained_weights', type=str, help='the path to pretrained weights file')

    # Dataset folder paths
    parser.add_argument('--train_folder', type=str, help='The train root folder')
    parser.add_argument('--valid_folder', type=str, help='The valid root folder')
    parser.add_argument('--test_folder', type=str, help='The test root folder')

    # Dataloader settings
    parser.add_argument('--batch_size', type=int, default=32, help='number of samples per iteration in the epoch')
    parser.add_argument('--num_workers', default=10, type=int)

    # optimizer settings
    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')

    # trainng related parameters
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train for')

    return parser


def main(args):
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    # setting the device to do stuff on
    print('Training on GPU:', torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Creates a model of 8 class head with NI dataset and 2 class head with DD dataset\
    if args.dataset == 'NI':
        model = create_model('Alexnet_NI').to(device)

    else:
        model = create_model('Alexnet_DD').to(device)

    # Prints summary of the model
    print('Model summary:')
    summary(model, (3, 227, 227))

    # creates necessary datasets and dataloader based on if we are training or testing a model.
    if args.mode == 'train':
        dataset_train = AlexDataset(args.train_folder)
        dataset_valid = AlexDataset(args.valid_folder)
        dataset_test = AlexDataset(args.test_folder)

        # the dataloader automatically divides the dataset into batches, on each epoch the batches are reshuffled
        dataloader_train = DataLoader(dataset_train, args.batch_size, shuffle=True, num_workers=args.num_workers)

        # creates dir for saving pretrained weights
        if not os.path.exists(args.name):
            os.mkdir(args.name)
            os.mkdir(f'{args.name}/log')

            with open(f'{args.name}/log/log.csv', 'a') as f:
                f.write('epoch,acc,loss\n')
        else:
            raise Exception('This model name is already in use!')

    elif args.mode == 'test':
        dataset_test = AlexDataset(args.test_folder)

        # Loads pretrained weights, the loading method is different depending on if a GPU is available
        model.load_state_dict(torch.load(args.pretrained_weights)) \
            if torch.cuda.is_available() \
            else model.load_state_dict(torch.load(args.pretrained_weights, map_location=torch.device('cpu')))

        # When mode is test the model is only tested and program ends
        print(f'Calculating accuracy, please wait...')
        print(f'Accuracy:\n{test_accuracy(model, dataset_test, device)}')
        quit()

    # creates adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # variables for holding best validation score and the epoch it was achieved
    best_valid_acc = 0
    best_valid_epoch = 0

    # training loop
    for epoch in range(args.epochs):
        # training one epoch
        loss = train_one_epoch(model, dataloader_train, optimizer, (epoch + 1), device)
        print(
            f'\nEPOCH {epoch + 1} COMPLETED! loss over epoch: {loss} learning rate: {optimizer.param_groups[0]["lr"]}')

        # getting validation accuracy
        acc = test_accuracy(model, dataset_valid, device)
        print(f'Validation accuracy: {acc:3f}\n')

        # logging validation accuracy and loss
        with open(f'{args.name}/log/log.csv', 'a') as f:
            f.write(f'{epoch + 1},{acc},{loss}\n')

        # saving the model and some statistics when ever we beat the best validation accuracy
        if acc > best_valid_acc:
            # removes previous best weights
            if os.path.exists(f'{args.name}/epoch{best_valid_epoch}.pt'):
                os.remove(f'{args.name}/epoch{best_valid_epoch}.pt')

            best_valid_acc = acc
            best_test_acc = test_accuracy(model, dataset_test, device)
            best_valid_epoch = epoch + 1

            print('NEW BEST! best_valid:', best_valid_acc, 'best_test_acc:', best_test_acc, 'best_valid_epoch:',
                  best_valid_epoch)

            torch.save(model.state_dict(), f'{args.name}/epoch{best_valid_epoch}.pt')

    print('Final test accuracy of model:', best_test_acc, 'Best epoch:', best_valid_epoch)
    print()
    print(args)


# program start
if __name__ == '__main__':
    # creates commandline parser
    arg_parser = argparse.ArgumentParser('Alexnet study with different edge detector applied to images',
                                         parents=[get_args_parser()])
    args = arg_parser.parse_args()

    # passes the commandline argument to the main function
    main(args)
