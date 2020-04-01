'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

import pandas as pd
import numpy as np
import time
import datetime
from pathlib import Path
import json
import glob

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

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

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = net.to(device)
net = ResNet18()
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

# Training
def train(epoch, batch_size):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
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
    return train_loss / (batch_idx + 1), 100. * correct / total, correct, total


def test(epoch, batch_size):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
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
    return test_loss / (batch_idx + 1), 100. * correct / total

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
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


optimizers_param_json_path = glob.glob("best_configs/*.json")

for param_json_path in optimizers_param_json_path:
    net = ResNet18()
    optimizer_params = params_dict.copy()
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    net = net.to(device)

    with open(param_json_path,"r") as f:
        params_dict = json.load(f)
    batch_size = int(params_dict['batch_size'])
    del params_dict['batch_size']
    del params_dict['algorithm']
    print(params_dict,batch_size)
    params_dict['params'] = net.parameters()

    if "Adam" in param_json_path:
        optimizer = optim.SGD(**params_dict)
    elif "RMSprop" in param_json_path:
        optimizer = optim.RMSprop(**params_dict)
    elif "SGD" in param_json_path:
        optimizer = optim.SGD(**params_dict)
    criterion = nn.CrossEntropyLoss()
    log_df = pd.DataFrame(columns=['epoch_number','train-test','time','loss','accuracy'])
    # Train:1
    # Test: 0
    start_time = time.time()
    for epoch in range(start_epoch, start_epoch+3):
        start_time = time.time()
        # try:
        train_loss, train_accuracy, _, _ = train(epoch, batch_size)
        # except:
        #     print("Error on train on " + optimizer_name)
        #     break
        iteration_train_time = time.time() - start_time

        start_time = time.time()
        # try:
        test_loss, test_accuracy = test(epoch, batch_size)
        # except:
        #     print("Error on test on " + optimizer_name)
        #     break
        iteration_test_time = time.time() - start_time

        buf_dict_train = {'epoch_number': epoch,
                          'train-test': 1,
                          'time': iteration_train_time,
                          'loss': train_loss,
                          'accuracy': train_accuracy}
        buf_dict_test = {'epoch_number': epoch,
                         'train-test': 0,
                         'time': iteration_test_time,
                         'loss': test_loss,
                         'accuracy': test_accuracy}
        # loc[nir_caviar_forward_model_EW.shape[0]] = list(buf_dict.values())
        log_df.loc[log_df.shape[0]] = list(buf_dict_train.values())
        log_df.loc[log_df.shape[0]] = list(buf_dict_test.values())

    optimizer_params_buf = optimizer_params.copy()
    optimizer_params_buf = removekey(optimizer_params_buf, 'params')
    parameters = tuple(optimizer_params_buf.values())
    string_parameters = "%1.2f_"*len(parameters)%parameters + str(batch_size) + '_'
    optimizer_name = param_json_path.split('/')[-1].split('.')[0]
    optimizer_params_buf['algorithm'] = optimizer_name
    optimizer_params['batch_size'] = batch_size
    now = datetime.datetime.now()
    dir_name = ("outputs/" + "ResNet18" + "/" + optimizer_name + "/"
                + string_parameters + now.strftime("_%d_%H_%m_%S"))
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    with open(dir_name + '/parameters.json', 'w') as f:
        json.dump(optimizer_params_buf, f)
    log_df['train-test'] = pd.to_numeric(log_df['train-test'], downcast='unsigned')
    log_df.to_csv(dir_name + "/log.csv")
    net = net.to('cpu')
    del net