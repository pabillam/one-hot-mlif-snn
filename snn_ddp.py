from __future__ import print_function
from enum import Enum
import builtins
import csv
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import datasets, transforms, models
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from functools import reduce
from collections import OrderedDict
import numpy as np
import datetime
import time
import pdb
from models import *
import sys
import os
import shutil
import argparse

parser = argparse.ArgumentParser(description='SNN training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
parser.add_argument('--batch_size', default=64, type=int, help='minibatch size of all GPUs on the current node')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--epochs', default=30, type=int, help='number of training epochs')
parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, help='initial learning_rate')
parser.add_argument('-thr_lr', default=0.0, type=float, help='learning rate for thresholds')
parser.add_argument('--lr_interval', default='0.60 0.80 0.90', type=str,
                    help='intervals at which to reduce lr, expressed as %%age of total epochs')
parser.add_argument('--warm_up', action='store_true', help='perform initial warm up learning rate')
parser.add_argument('--warm_up_epochs', default=5, type=int, help='number of warm up epochs')
parser.add_argument('--warm_up_lr', default=1e-6, type=float, help='initial warm up learning rate')
parser.add_argument('--lr_reduce', default=10, type=int, help='reduction factor for learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay parameter for the optimizer')
parser.add_argument('--optimizer', default='SGD', type=str, help='optimizer for SNN backpropagation',
                    choices=['SGD', 'Adam'])
parser.add_argument('--amsgrad', default=True, type=bool, help='amsgrad parameter for Adam optimizer')
parser.add_argument('--betas', default='0.9,0.999', type=str, help='betas for Adam optimizer')
parser.add_argument('--momentum', default=0.95, type=float, help='momentum parameter for the SGD optimizer')
parser.add_argument('--dropout', default=0.5, type=float, help='dropout percentage for conv layers')
parser.add_argument('--timesteps', default=20, type=int, help='simulation timesteps')
parser.add_argument('--leak', default=1.0, type=float, help='membrane leak')
parser.add_argument('--scaling_factor', default=0.3, type=float,
                    help='scaling factor for thresholds at reduced timesteps')
parser.add_argument('--default_threshold', default=1.0, type=float,
                    help='initial threshold to train SNN from scratch')
parser.add_argument('--activation', default='Linear', type=str, help='SNN activation function',
                    choices=['Linear'])
parser.add_argument('--kernel_size', default=3, type=int, help='filter size for the conv layers')
parser.add_argument('--pretrained_ann', default='', type=str, help='pretrained ANN model')
parser.add_argument('--pretrained_snn', default='', type=str, help='pretrained SNN for inference')
parser.add_argument('-a', '--architecture', default='VGG16', type=str, help='network architecture',
                    choices=['VGG16', 'RESNET20'])
parser.add_argument('--bias', action='store_true', help='enable bias for conv layers in network')
parser.add_argument('-multi_channel', default=0, type=int, help="multi channel SNN", choices=[0, 1])
parser.add_argument('-num_channels', default=1, type=int, help='number of in out channels for all layers')
parser.add_argument('-hardware_opt', default=0, type=int, help="neuron model variation",
                    choices=[1, 2])  # 1 --> one-hot
parser.add_argument('-reset_to_zero', default=0, type=int, help="reset_to_zero", choices=[0, 1])
parser.add_argument('-input_encoding', default=1, type=int, help="direct vs multi-channel rate",
                    choices=[0, 1])  # 1 --> direct
parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset name',
                    choices=['CIFAR10', 'CIFAR100', 'IMAGENET'])
parser.add_argument('--dataset_dir', metavar='path', default='./data', help='dataset path')
parser.add_argument('-s', '--seed', default=0, type=int, help='seed for random number')
parser.add_argument('--dont_save', action='store_true', help='don\'t save training model during testing')
parser.add_argument('--world_size', default=-1, type=int, help='Number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='Node rank for distributed training')
parser.add_argument('--dist_url', default='env://', type=str, help='URL used to setup distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str, help='Distributed backend')
parser.add_argument('--gpu', default=None, type=int, help='GPU ID to use')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multiprocessing distributed training to launch'
                         'N processes per node, which has N GPUs. This is the'
                         'fastest way to use PyTorch for either single node or'
                         'multi node data parallel training')

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def main():
    args = parser.parse_args()

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely'
                      'disable data parallelism')
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    args.ngpus_per_node = ngpus_per_node
    print("Number of GPUs per node = {}".format(ngpus_per_node))

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.set_start_method("spawn")
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    # Suppress printing if not master

    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ.get("SLURM_LOCALID"))
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = int(os.environ.get('SLURM_NODEID')) * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # Start from the same point in different nodes
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset = args.dataset
    dataset_dir = args.dataset_dir
    batch_size = args.batch_size
    architecture = args.architecture
    multi_channel = args.multi_channel
    num_channels = args.num_channels
    hardware_opt = args.hardware_opt
    reset_to_zero = args.reset_to_zero
    input_encoding = args.input_encoding
    learning_rate = args.learning_rate
    pretrained_ann = args.pretrained_ann
    pretrained_snn = args.pretrained_snn
    epochs = args.epochs
    timesteps = args.timesteps
    leak = args.leak
    scaling_factor = args.scaling_factor
    default_threshold = args.default_threshold
    activation = args.activation
    optimizer = args.optimizer
    weight_decay = args.weight_decay
    momentum = args.momentum
    amsgrad = args.amsgrad
    beta1 = float(args.betas.split(',')[0])
    beta2 = float(args.betas.split(',')[1])
    dropout = args.dropout
    kernel_size = args.kernel_size
    start_epoch = 1
    max_accuracy = 0.0

    values = args.lr_interval.split()
    lr_interval = []
    for value in values:
        lr_interval.append(int(float(value) * args.epochs))
    args.lr_interval = lr_interval

    log_file = './logs/snn/'
    if args.gpu == 0:
        try:
            os.mkdir(log_file)
        except OSError:
            pass
    if multi_channel == 1:
        identifier = 'snn_' + architecture.lower() + '_' + dataset.lower() + '_MC_T_' + str(
            timesteps) + '_H_' + str(hardware_opt) + '_C_' + str(num_channels) + '_Z_' + str(
            reset_to_zero) + '_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    else:
        identifier = 'snn_' + architecture.lower() + '_' + dataset.lower() + '_SC_T_' + str(
            timesteps) + '_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    print('\n Run on time: {}'.format(datetime.datetime.now()))
    print(f'Process ID: {os.getpid()}')

    print('\n Arguments: ')
    for arg in vars(args):
        if arg == 'lr_interval':
            print('\t {:20} : {}'.format(arg, lr_interval))
        elif arg == 'pretrained_ann':
            print('\t {:20} : {}'.format(arg, pretrained_ann))
        else:
            print('\t {:20} : {}'.format(arg, getattr(args, arg)))

    # Loading Dataset
    if dataset == 'CIFAR10':
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        labels = 10
    elif dataset == 'CIFAR100':
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        labels = 100
    elif dataset == 'IMAGENET':
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        labels = 1000
    
    if dataset in ['CIFAR10', 'CIFAR100']:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    if dataset == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                         transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True,
                                        transform=transform_test)

    elif dataset == 'CIFAR100':
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True,
                                          transform=transform_train)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True,
                                         transform=transform_test)
    elif dataset == 'IMAGENET':
        traindir = os.path.join(dataset_dir, 'train')
        valdir = os.path.join(dataset_dir, 'val')
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        test_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False, drop_last=True)

    else:
        train_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True, sampler=test_sampler)

    # Create model
    print("Creating model '{}'".format(args.architecture))
    if architecture[0:3].lower() == 'vgg':
        if multi_channel == 1:
            model = VGG_SNN_MC(vgg_name=architecture, labels=labels, timesteps=timesteps, leak=leak,
                               default_threshold=default_threshold, dropout=dropout, dataset=dataset,
                               hardware_opt=hardware_opt, input_encoding=input_encoding, reset_to_zero=reset_to_zero,
                               num_channels=num_channels, kernel_size=kernel_size, bias=args.bias)
        else:
            model = VGG_SNN(vgg_name=architecture, activation=activation, labels=labels, timesteps=timesteps,
                            leak=leak, default_threshold=default_threshold, dropout=dropout, kernel_size=kernel_size,
                            dataset=dataset)

    elif architecture[0:3].lower() == 'res':
        if multi_channel == 1:
            model = RESNET_SNN_MC(resnet_name=architecture, labels=labels, timesteps=timesteps, leak=leak,
                                  default_threshold=default_threshold, dropout=dropout, dataset=dataset,
                                  hardware_opt=hardware_opt, input_encoding=input_encoding, reset_to_zero=reset_to_zero,
                                  num_channels=num_channels, bias=args.bias)
        else:
            model = RESNET_SNN(resnet_name=architecture, activation=activation, labels=labels, timesteps=timesteps,
                               leak=leak, default_threshold=default_threshold, dropout=dropout, dataset=dataset)
    print('\n {}'.format(model))

    # Load from pretrained checkpoint, before DistributedDataParallel constructor
    if pretrained_ann:
        state = torch.load(pretrained_ann, map_location='cpu')
        # Create new OrderedDict that does not contain 'module.'
        new_state_dict = OrderedDict()
        for k, v in state['state_dict'].items():
            name = k[7:]  # remove 'module.'
            new_state_dict[name] = v
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        print('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))
        if not args.bias:
            print('\n Info: Accuracy of loaded ANN model: {}'.format(state['accuracy']))

    # ANN-SNN conversion:
    if pretrained_ann:
        # If thresholds present in loaded ANN file
        if 'thresholds' in state.keys():
            thresholds = state['thresholds']
            print('\n Info: Thresholds loaded from trained ANN: {}'.format(thresholds))
            model.threshold_update(scaling_factor=scaling_factor, thresholds=thresholds[:])
        else:
            thresholds = find_threshold(train_dataset, args, model, batch_size=512, timesteps=100,
                                        architecture=architecture)
            model.threshold_update(scaling_factor=scaling_factor, thresholds=thresholds[:])

            # Save the thresholds in the ANN file
            temp = {}
            for key, value in state.items():
                temp[key] = value
            temp['thresholds'] = thresholds
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.world_size == 0):
                torch.save(temp, pretrained_ann)

    # Define loss and optimizer
    threshold_params = dict()
    other_params = dict()
    for name, param in model.named_parameters():
        if 'threshold' in name:
            threshold_params[name] = param
        else:
            other_params[name] = param

    if optimizer == 'Adam':
        if args.warm_up:
            optimizer = optim.Adam(model.parameters(), lr=args.warm_up_lr, amsgrad=amsgrad,
                                   weight_decay=weight_decay,
                                   betas=(beta1, beta2))
        else:
            if args.thr_lr == 0.0:
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=amsgrad, weight_decay=weight_decay,
                                       betas=(beta1, beta2))
            else:
                optimizer = optim.Adam([{"params": threshold_params.values(), "lr": args.thr_lr},
                                        {"params": other_params.values()}], lr=learning_rate, amsgrad=amsgrad,
                                       weight_decay=weight_decay,
                                       betas=(beta1, beta2))
    elif optimizer == 'SGD':
        if args.warm_up:
            optimizer = optim.SGD(model.parameters(), lr=args.warm_up_lr, momentum=momentum,
                                  weight_decay=weight_decay,
                                  nesterov=False)
        else:
            if args.thr_lr == 0.0:
                optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,
                                      weight_decay=weight_decay,
                                      nesterov=False)
            else:
                optimizer = optim.SGD([{"params": threshold_params.values(), "lr": args.thr_lr},
                                       {"params": other_params.values()}], lr=learning_rate, momentum=momentum,
                                      weight_decay=weight_decay,
                                      nesterov=False)
    print('{}'.format(optimizer))

    if pretrained_snn:
        state = torch.load(pretrained_snn, map_location='cpu')
        # Create new OrderedDict that does not contain 'module.'
        # new_state_dict = OrderedDict()
        # for k, v in state['state_dict'].items():
        #     name = k[7:]  # remove 'module.'
        #     new_state_dict[name] = v
        missing_keys, unexpected_keys = model.load_state_dict(state['state_dict'], strict=False)
        print('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))
        epoch = state['epoch']
        start_epoch = epoch + 1
        max_accuracy = state['accuracy']
        optimizer.load_state_dict(state['optimizer'])
        print('\n Loaded from resume epoch: {}, accuracy: {:.4f}'.format(epoch, max_accuracy))
        thresholds = state['thresholds']
        print('\n Info: Thresholds loaded from trained SNN: {}'.format(thresholds))
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate 

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor should
        # always set the single device scope, otherwise, DistributedDataParallel will use
        # all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            optimizer_to(optimizer, 'cuda:' + str(args.gpu))
            # When using a single GPU per process and per DistributedDataParallel,
            # we need to divide the batch size ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            # model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all available GPUs if
            # device_ids are not set
            print("Setting DDP")
            # model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
            model = nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        optimizer_to(optimizer, 'cuda:' + str(args.gpu))
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    model.module.network_update(timesteps=timesteps, leak=leak)

    for epoch in range(start_epoch, epochs):
        start_time = datetime.datetime.now()
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train(epoch, train_loader, optimizer, model, args)
        max_accuracy = test(epoch, test_loader, model, args, max_accuracy, start_time, identifier, optimizer)

    print('Highest accuracy: {:.4f}'.format(max_accuracy))


def find_threshold(train_dataset, args, model, batch_size=512, timesteps=2500, architecture='VGG16'):
    loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    model = nn.DataParallel(model.cuda()).cuda()
    model.module.network_update(timesteps=timesteps, leak=1.0)
    thresholds = []

    def find(layer):
        max_act = 0

        print('\n Finding threshold for layer {}'.format(layer))
        for batch_idx, (data, target) in enumerate(loader):
            if args.gpu is not None:
                data, target = data.cuda(args.gpu, non_blocking=True), target.cuda(args.gpu, non_blocking=True)
            elif torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                model.eval()
                output = model(data, find_max_mem=True, max_mem_layer=layer)
                if torch.max(output).item() > max_act:
                    max_act = torch.max(output).item()

                if batch_idx == 0:
                    thresholds.append(max_act)
                    print(' {}'.format(thresholds))
                    model.module.threshold_update(scaling_factor=1.0, thresholds=thresholds[:])
                    break

    if architecture.lower().startswith('vgg'):
        for l in model.module.features.named_children():
            if isinstance(l[1], nn.Conv2d):
                find(int(l[0]))

        for c in model.module.classifier.named_children():
            if isinstance(c[1], nn.Linear):
                if int(c[0]) == len(model.module.classifier) - 1:
                    break
                else:
                    find(int(l[0]) + int(c[0]) + 1)

    if architecture.lower().startswith('res'):
        for l in model.module.pre_process.named_children():
            if isinstance(l[1], nn.Conv2d):
                find(int(l[0]))

        pos = len(model.module.pre_process)

        for i in range(1, 5):
            layer = model.module.layers[i]
            for index in range(len(layer)):
                for l in range(len(layer[index].residual)):
                    if isinstance(layer[index].residual[l], nn.Conv2d):
                        pos = pos + 1

        for c in model.module.classifier.named_children():
            if isinstance(c[1], nn.Linear):
                if int(c[0]) == len(model.module.classifier) - 1:
                    break
                else:
                    find(int(c[0]) + pos)

    model.module.to('cpu')
    print('ANN thresholds: {}'.format(thresholds))
    return thresholds


def train(epoch, loader, optimizer, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    if args.warm_up:
        if epoch <= (args.warm_up_epochs + 1):
            if args.thr_lr == 0.0: 
                for param_group in optimizer.param_groups:
                    param_group['lr'] = (args.learning_rate - args.warm_up_lr) * (epoch - 1) / args.warm_up_epochs + args.warm_up_lr
            else:
                optimizer.param_groups[1]['lr'] = (args.learning_rate - args.warm_up_lr) * (epoch - 1) / args.warm_up_epochs + args.warm_up_lr
        else:
            if args.thr_lr == 0.0:
                if epoch in args.lr_interval:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] / args.lr_reduce
                        # args.learning_rate = param_group['lr']
            else:
                if epoch in args.lr_interval:
                    optimizer.param_groups[1]['lr'] = optimizer.param_groups[1]['lr'] / args.lr_reduce
                    # args.learning_rate = param_group['lr']
    else:
        if args.thr_lr == 0.0: 
            if epoch in args.lr_interval:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] / args.lr_reduce
                    # args.learning_rate = param_group['lr']
        else:
            # if epoch in args.lr_interval:
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = param_group['lr'] / args.lr_reduce
            if epoch in args.lr_interval:
                optimizer.param_groups[1]['lr'] = optimizer.param_groups[1]['lr'] / args.lr_reduce
                # args.learning_rate = param_group['lr']
    
    if args.thr_lr == 0.0:
        print('Current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
    else:
        print('Current lr {:.5e}, thr_lr {:.5e}'.format(optimizer.param_groups[1]['lr'], optimizer.param_groups[0]['lr']))

    model.train()
    end = time.time()
    for batch_idx, (data, target) in enumerate(loader, 0):
        data_time.update(time.time() - end)

        if args.gpu is not None:
            data, target = data.cuda(args.gpu, non_blocking=True), target.cuda(args.gpu, non_blocking=True)
        elif torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        mean_output = output.mean(1)
        loss = F.cross_entropy(mean_output, target)
        loss.backward()
        
        optimizer.step()
        pred = mean_output.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()
        
        for key, value in model.module.leak.items():
            model.module.leak[key].data.clamp_(max=1.0)

        losses.update(loss.item(), data.size(0))
        top1.update(correct.item() / data.size(0), data.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            progress.display(batch_idx)

    print('TRAIN: * Epoch {0} Acc@1 {1:.4f} Loss {2:.4f}'.format(epoch, top1.avg, losses.avg))


def test(epoch, loader, model, args, max_accuracy, start_time, identifier, optimizer):
    def run_test(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            total_loss = 0

            for batch_idx, (data, target) in enumerate(loader, 0):
                batch_idx = base_progress + batch_idx
                if args.gpu is not None:
                    data, target = data.cuda(args.gpu, non_blocking=True), target.cuda(args.gpu, non_blocking=True)
                elif torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()

                output = model(data)
                mean_output = output.mean(1)
                loss = F.cross_entropy(mean_output, target)
                    
                total_loss += loss.item()
                pred = mean_output.max(1, keepdim=True)[1]
                correct = pred.eq(target.data.view_as(pred)).cpu().sum()

                losses.update(loss.item(), data.size(0))
                top1.update(correct.item() / data.size(0), data.size(0))
                batch_time.update(time.time() - end)
                end = time.time()

                if batch_idx % args.print_freq == 0:
                    progress.display(batch_idx)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(loader) + (args.distributed and (len(loader.sampler) * args.world_size < len(loader.dataset))),
        [batch_time, losses, top1],
        prefix="Test: [{}]".format(epoch))
    model.eval()
    run_test(loader)

    if args.distributed:
        top1.all_reduce()

    if args.distributed and (len(loader.sampler) * args.world_size < len(loader.dataset)):
        aux_test_dataset = Subset(test_loader.dataset,
                                  range(len(test_loader.sampler) * args.world_size, len(test_loader.dataset)))
        aux_test_loader = torch.utils.data.DataLoader(aux_test_dataset, batch_size=args.batch_size, shuffle=False,
                                                      num_workers=args.workers, pin_memory=True)
        run_test(aux_test_loader, base_progress=len(loader))

    if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0):
        temp1 = []
        temp2 = []
        for key, value in sorted(model.module.threshold.items(), key=lambda x: (int(x[0][1:]), (x[1]))):
            temp1 = temp1 + [value.item()]
        for key, value in sorted(model.module.leak.items(), key=lambda x: (int(x[0][1:]), (x[1]))):
            temp2 = temp2 + [value.item()]
        print('Thresholds: {}, leak: {}'.format(temp1, temp2))

        if top1.avg > max_accuracy:
            max_accuracy = top1.avg
            print('Saving model check point')
            state = {
                'accuracy': max_accuracy,
                'epoch': epoch,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'thresholds': temp1,
                'timesteps': args.timesteps,
                'leak': temp2,
                'activation': args.activation
            }
            try:
                os.mkdir('./trained_models/snn/')
            except OSError:
                pass

            filename = './trained_models/snn/' + identifier + '.pth'
            if not args.dont_save:
                torch.save(state, filename)

    print(
        'TEST: * Acc@1 {0:.4f} Loss {1:.4f}, Best Acc@1: {2:.4f}, Time: {3}'.format(top1.avg, losses.avg, max_accuracy,
                                                                                    datetime.timedelta(seconds=(
                                                                                            datetime.datetime.now() - start_time).seconds)))

    return max_accuracy


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.summary_type = summary_type
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()
