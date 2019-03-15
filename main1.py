'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
from models1 import *
# import models
from utils import progress_bar
from utils1 import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

# parser.add_argument('--stages', type=str, metavar='STAGE DEPTH',
#                      help='per layer depth')
# parser.add_argument('--bottleneck', default=4, type=int, metavar='B',
#                      help='bottleneck (default: 4)')
# parser.add_argument('--group-1x1', type=int, metavar='G', default=4,
#                      help='1x1 group convolution (default: 4)')
# parser.add_argument('--group-3x3', type=int, metavar='G', default=4,
#                      help='3x3 group convolution (default: 4)')
# parser.add_argument('--condense-factor', type=int, metavar='C', default=4,
#                      help='condense factor (default: 4)')
# parser.add_argument('--growth', type=str, metavar='GROWTH RATE',
#                      help='per layer growth')
# parser.add_argument('--reduction', default=0.5, type=float, metavar='R',
#                      help='transition reduction (default: 0.5)')
# parser.add_argument('--dropout-rate', default=0, type=float,
#                      help='drop out (default: 0)')
# parser.add_argument('--group-lasso-lambda', default=0., type=float, metavar='LASSO',
#                      help='group lasso loss weight (default: 0)')
#
# parser.add_argument('--evaluate', action='store_true',
#                      help='evaluate model on validation set (default: false)')
# parser.add_argument('--convert-from', default=None, type=str, metavar='PATH',
#                      help='path to saved checkpoint (default: none)')
# parser.add_argument('--evaluate-from', default=None, type=str, metavar='PATH',
#                      help='path to saved checkpoint (default: none)')
# parser.add_argument('--model', default='condensenet', type=str, metavar='M',
#                      help='model to train the dataset')
#
args = parser.parse_args()
# args.num_classes = 10
# args.stages = '14-14-14'
# args.stages = list(map(int, str(args.stages).split('-')))
# args.growth = '8-16-32'
# args.growth = list(map(int, str(args.growth).split('-')))
# args.condense_factor = args.group_1x1


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1536, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=8)
# normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
#
# train_dataset = torchvision.datasets.CIFAR100(
#             root='./data',
#             train=True,
#             download=True,
#             transform=transforms.Compose([
#                 transforms.RandomCrop(32, padding=4),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 normalize,
#             ]))
# trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
#
# test_dataset = torchvision.datasets.CIFAR100(
#             root='./data',
#             train=False,
#             download=True,
#             transform=transforms.Compose([
#                 transforms.ToTensor(),
#                 normalize,
#             ]))
# testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG32()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = squeezenet()
# net = getattr(models, args.model)(args)
# net=PNASNetB()
# print(net)
# print(net)
net = net.to(device)
print(net)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    adjust_learning_rate(optimizer, epoch)
    # for param_group in optimizer.param_groups:
    #     print(param_group['lr'] )
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        b=str(best_acc)
        torch.save(state, './checkpoint/'+b)
        best_acc = acc

def adjust_learning_rate(optimizer, epoch):

    if epoch < 50:
            lr = args.lr
    elif epoch < 75:
            lr = args.lr * 0.5
    else:
            lr = args.lr * 0.25

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

#for epoch in range(start_epoch, start_epoch+100):
#    train(epoch)
#    test(epoch)
number = compute_num_parameters(net)/1000000
print(number)
flots = compute_num_flops(net)/(1000000)
print(flots)
