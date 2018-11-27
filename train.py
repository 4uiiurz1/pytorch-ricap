import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils import *
from wide_resnet import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--dataset', default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='dataset name')
    parser.add_argument('--ricap', default=False, type=str2bool,
                        help='use RICAP')
    parser.add_argument('--beta', default=0.3, type=float)
    parser.add_argument('--depth', default=28, type=int)
    parser.add_argument('--width', default=10, type=int)

    args = parser.parse_args()

    return args


def train(args, train_loader, model, criterion, optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    scores = AverageMeter()

    model.train()

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        # from original paper's appendix
        if args.ricap:
            I_x, I_y = input.size()[2:]

            w = int(np.round(I_x * np.random.beta(args.beta, args.beta)))
            h = int(np.round(I_y * np.random.beta(args.beta, args.beta)))
            w_ = [w, I_x - w, w, I_x - w]
            h_ = [h, h, I_y - h, I_y - h]

            cropped_images = {}
            c_ = {}
            W_ = {}
            for k in range(4):
                idx = torch.randperm(input.size(0))
                x_k = np.random.randint(0, I_x - w_[k] + 1)
                y_k = np.random.randint(0, I_y - h_[k] + 1)
                cropped_images[k] = input[idx][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
                c_[k] = target[idx].cuda()
                W_[k] = w_[k] * h_[k] / (I_x * I_y)

            patched_images = torch.cat(
                (torch.cat((cropped_images[0], cropped_images[1]), 2),
                torch.cat((cropped_images[2], cropped_images[3]), 2)),
            3)
            patched_images = patched_images.cuda()

            output = model(patched_images)
            loss = sum([W_[k] * criterion(output, c_[k]) for k in range(4)])

            acc1 = sum([W_[k] * accuracy(output, c_[k])[0] for k in range(4)])

        else:
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            loss = criterion(output, target)

            acc1, _ = accuracy(output, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        scores.update(acc1.item(), input.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('acc', scores.avg),
    ])

    return log


def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    scores = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            scores.update(acc1.item(), input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('acc', scores.avg),
    ])

    return log


def main():
    args = parse_args()

    if args.name is None:
        if args.ricap:
            args.name = '%s_WideResNet_%sx%s_wRICAP' %(args.dataset, args.depth, args.width)
        else:
            args.name = '%s_WideResNet_%sx%s_woRICAP' %(args.dataset, args.depth, args.width)

    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    # data loading code
    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        train_set = datasets.CIFAR10(
            root='~/data',
            train=True,
            download=True,
            transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=128,
            shuffle=True,
            num_workers=8)

        test_set = datasets.CIFAR10(
            root='~/data',
            train=False,
            download=True,
            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=128,
            shuffle=False,
            num_workers=8)

    elif args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        train_set = datasets.CIFAR100(
            root='~/data',
            train=True,
            download=True,
            transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=128,
            shuffle=True,
            num_workers=8)

        test_set = datasets.CIFAR100(
            root='~/data',
            train=False,
            download=True,
            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=128,
            shuffle=False,
            num_workers=8)

    # create model
    model = WideResNet(args.depth, args.width)
    model = model.cuda()

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1,
            momentum=0.9, weight_decay=1e-4)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'acc', 'val_loss', 'val_acc'
    ])

    best_acc = 0
    for epoch in range(200):
        print('Epoch [%d/%d]' %(epoch, 200))

        # train for one epoch
        train_log = train(args, train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        val_log = validate(args, test_loader, model, criterion)

        lr = adjust_learning_rate(optimizer, epoch)

        print('loss %.4f - acc %.4f - val_loss %.4f - val_acc %.4f'
            %(train_log['loss'], train_log['acc'], val_log['loss'], val_log['acc']))

        tmp = pd.Series([
            epoch,
            lr,
            train_log['loss'],
            train_log['acc'],
            val_log['loss'],
            val_log['acc'],
        ], index=['epoch', 'lr', 'loss', 'acc', 'val_loss', 'val_acc'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv' %args.name, index=False)

        if val_log['acc'] > best_acc:
            torch.save(model.state_dict(), 'models/%s/model.pth' %args.name)
            best_acc = val_log['acc']
            print("=> saved best model")

    print("best val_acc: %f" %best_acc)


if __name__ == '__main__':
    main()
