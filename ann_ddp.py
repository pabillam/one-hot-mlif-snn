import builtins
import argparse
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import datasets, transforms, models
from matplotlib import pyplot as plt
import pdb
import sys
import datetime
import time
import os
import numpy as np
from models import *
from augment import *

parser = argparse.ArgumentParser(description='Train ANN',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
parser.add_argument('--batch_size', default=64, type=int, help='minibatch size of all GPUs on the current node')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--epochs', default=300, type=int, help='number of training epochs')
parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help='initial learning_rate')
parser.add_argument('--lr_interval', default='0.45 0.70 0.90', type=str,
                    help='intervals at which to reduce lr, expressed as %%age of total epochs')
parser.add_argument('--lr_reduce', default=10, type=int, help='reduction factor for learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay for Adam/SGD optimizers')
parser.add_argument('--optimizer', default='SGD', type=str, help='optimizer for SNN backpropagation',
                    choices=['SGD', 'Adam'])
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD optimizer')
parser.add_argument('--dropout', default=0.2, type=float, help='dropout percentage for conv layers')
parser.add_argument('--kernel_size', default=3, type=int, help='filter size for the conv layers')
parser.add_argument('--pretrained_ann', default='', type=str, help='pretrained model to initialize ANN')
parser.add_argument('-a', '--architecture', default='VGG16', type=str, help='network architecture',
                    choices=['VGG16', 'RESNET20'])
parser.add_argument('--bn', action='store_true', help='enable batch normalization layers in network')
parser.add_argument('--bias', action='store_true', help='enable bias for conv layers in network')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset name',
                    choices=['CIFAR10', 'CIFAR100', 'IMAGENET'])
parser.add_argument('--dataset_dir', metavar='path', default='./data', help='dataset path')
parser.add_argument('-s', '--seed', default=0, type=int, help='seed for random number')
parser.add_argument('--test_only', action='store_true', help='perform only inference')
parser.add_argument('--dont_save', action='store_true', help='don\'t save training model during testing')
parser.add_argument('--quant_act', action='store_true', help='quantize activations to match SNN resolution')
parser.add_argument('--act_bits', default=1, type=int, help='number of bits to quantize activations')
parser.add_argument('--quant_scale_max', default=0.8, type=float, help='scaling factor to apply to max')
parser.add_argument('--quant_num_batches', default=10, type=int, help='number of batches to go over during quantization analysis')
parser.add_argument('--quant_id', default='log', type=str, help='quantization scheme identifier') # TODO
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
    learning_rate = args.learning_rate
    pretrained_ann = args.pretrained_ann
    epochs = args.epochs
    optimizer = args.optimizer
    momentum = args.momentum
    weight_decay = args.weight_decay
    dropout = args.dropout
    kernel_size = args.kernel_size

    values = args.lr_interval.split()
    lr_interval = []
    for value in values:
        lr_interval.append(int(float(value) * args.epochs))
    args.lr_interval = lr_interval

    if args.quant_act:
        identifier = 'ann_' + architecture.lower() + '_' + dataset.lower() + '_AB_' + str(
            args.act_bits) + '_' + args.quant_id + '_' + datetime.datetime.now().strftime(
            "%Y_%m_%d_%H_%M_%S")
    else:
        identifier = 'ann_' + architecture.lower() + '_' + dataset.lower() + '_' + datetime.datetime.now().strftime(
            "%Y_%m_%d_%H_%M_%S")

    print('\n Run on time: {}'.format(datetime.datetime.now()))
    print(f'Process ID: {os.getpid()}')

    print('\n Arguments:')
    for arg in vars(args):
        if arg == 'lr_interval':
            print('\t {:20} : {}'.format(arg, lr_interval))
        else:
            print('\t {:20} : {}'.format(arg, getattr(args, arg)))

    # Loading Dataset
    if dataset == 'CIFAR100':
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        labels = 100
    elif dataset == 'CIFAR10':
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        labels = 10
    elif dataset == 'IMAGENET':
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        labels = 1000

    if dataset == 'CIFAR10' or dataset == 'CIFAR100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    if dataset == 'CIFAR100':
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True,
                                          transform=transform_train)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True,
                                         transform=transform_test)
    elif dataset == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                         transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True,
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
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                   shuffle=False, num_workers=args.workers, pin_memory=True,
                                                   sampler=train_sampler)
    else:
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

    # Create model
    print("Creating model '{}'".format(args.architecture))
    if architecture[0:3].lower() == 'vgg':
        model = VGG(vgg_name=architecture, labels=labels, dataset=dataset, kernel_size=kernel_size, dropout=dropout,
                    bn=args.bn,
                    bias=args.bias, quant_act=args.quant_act, act_bits=args.act_bits,
                    quant_scale_max=args.quant_scale_max, quant_id=args.quant_id)
        if args.quant_act:
            base_model = VGG(vgg_name=architecture, labels=labels, dataset=dataset, kernel_size=kernel_size,
                             dropout=dropout,
                             bn=args.bn,
                             bias=args.bias, quant_act=False, act_bits=args.act_bits,
                             quant_scale_max=args.quant_scale_max, quant_id=args.quant_id)
    elif architecture[0:3].lower() == 'res':
        if architecture.lower() == 'resnet20':
            model = ResNet20(labels=labels, dropout=dropout, dataset=dataset, bn=args.bn, bias=args.bias,
                             quant_act=args.quant_act, act_bits=args.act_bits, quant_scale_max=args.quant_scale_max, quant_id=args.quant_id)
        if args.quant_act:
            if architecture.lower() == 'resnet20':
                base_model = ResNet20(labels=labels, dropout=dropout, dataset=dataset, bn=args.bn, bias=args.bias,
                                      quant_act=False, act_bits=args.act_bits, quant_scale_max=args.quant_scale_max, quant_id=args.quant_id)

    if args.quant_act:
        if args.pretrained_ann:
            state = torch.load(args.pretrained_ann, map_location='cpu')
            cur_dict = base_model.state_dict()
            for key in state['state_dict'].keys():
                if key in cur_dict:
                    if state['state_dict'][key].shape == cur_dict[key].shape and torch.is_floating_point(
                            state['state_dict'][key].data):
                        cur_dict[key] = nn.Parameter(state['state_dict'][key].data)
                        print('Success: Loaded {} from {}'.format(key, pretrained_ann))
                    elif not torch.is_floating_point(state['state_dict'][key].data):
                        cur_dict[key] = state['state_dict'][key].data
                        print('Success: Loaded {} from {}'.format(key, pretrained_ann))
                    else:
                        print('Error: Size mismatch, size of loaded model {}, size of current model {}'.format(
                            state['state_dict'][key].shape, base_model.state_dict()[key].shape))
                else:
                    print('Error: Loaded weight {} not present in current model'.format(key))
            base_model.load_state_dict(cur_dict)
            transfer_models(base_model, model, args)
    else:
        if args.pretrained_ann:
            state = torch.load(args.pretrained_ann, map_location='cpu')
            cur_dict = model.state_dict()
            for key in state['state_dict'].keys():
                if key in cur_dict:
                    if state['state_dict'][key].shape == cur_dict[key].shape and torch.is_floating_point(
                            state['state_dict'][key].data):
                        cur_dict[key] = nn.Parameter(state['state_dict'][key].data)
                        print('Success: Loaded {} from {}'.format(key, pretrained_ann))
                    elif not torch.is_floating_point(state['state_dict'][key].data):
                        cur_dict[key] = state['state_dict'][key].data
                        print('Success: Loaded {} from {}'.format(key, pretrained_ann))
                    else:
                        print('Error: Size mismatch, size of loaded model {}, size of current model {}'.format(
                            state['state_dict'][key].shape, model.state_dict()[key].shape))
                else:
                    print('Error: Loaded weight {} not present in current model'.format(key))
            model.load_state_dict(cur_dict)
    print('{}'.format(model))

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor should
        # always set the single device scope, otherwise, DistributedDataParallel will use
        # all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per DistributedDataParallel,
            # we need to divide the batch size ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all available GPUs if
            # device_ids are not set
            print("Setting DDP")
            model = nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # Define loss and optimizer
    if optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True, weight_decay=weight_decay)

    print('{}'.format(optimizer))

    max_accuracy = 0
    for epoch in range(1, epochs):
        start_time = datetime.datetime.now()
        if args.distributed:
            train_sampler.set_epoch(epoch)
        print('Current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        if args.quant_act:
            for n, m in model.named_modules():
                if isinstance(m, QuantizeActivation):
                    m.mode = 1  # Analysis
                    if args.gpu is not None or torch.cuda.is_available():
                        m.minimum = torch.tensor([0.0]).cuda()
                        m.maximum = torch.tensor([0.0]).cuda()
            find_max_min(train_dataset, args, model, num_batches=args.quant_num_batches)
            for n, m in model.named_modules():
                if isinstance(m, QuantizeActivation):
                    m.mode = 0  # Quantize
                    print("Epoch {}, Layer {}: Maximum = {}, Minimum = {}".format(epoch, n, m.maximum.item(), m.minimum.item()))
        train(epoch, train_loader, optimizer, model, args)
        max_accuracy = test(epoch, test_loader, model, args, max_accuracy, start_time, identifier, optimizer)

    print('Highest accuracy: {:.4f}'.format(max_accuracy))


def transfer_models(model, quantized_model, args):
    if args.architecture[0:3].lower() == 'vgg':
        co = 0
        for l in range(len(model.features)):
            if isinstance(model.features[l], nn.Conv2d):
                conv = model.features[l]
                quantized_model.features[co].weight = nn.Parameter(conv.weight)
                if conv.bias is not None:
                    quantized_model.features[co].bias = nn.Parameter(conv.bias)
                if args.bn:
                    co += 1
                else:
                    co += 4
            elif isinstance(model.features[l], nn.BatchNorm2d):
                bn = model.features[l]
                quantized_model.features[co].weight = nn.Parameter(bn.weight)
                quantized_model.features[co].bias = nn.Parameter(bn.bias)
                quantized_model.features[co].running_mean = bn.running_mean
                quantized_model.features[co].running_var = bn.running_var
                quantized_model.features[co].num_batches_tracked = bn.num_batches_tracked
                co += 4
        co = 0
        for l in range(len(model.classifier)):
            if isinstance(model.classifier[l], nn.Linear):
                quantized_model.classifier[co].weight = nn.Parameter(model.classifier[l].weight)
                co += 4
    elif args.architecture[0:3].lower() == 'res':
        co = 0
        for l in range(len(model.pre_process)):
            if isinstance(model.pre_process[l], nn.Conv2d):
                conv = model.pre_process[l]
                quantized_model.pre_process[co].weight = nn.Parameter(conv.weight)
                if conv.bias is not None:
                    quantized_model.pre_process[co].bias = nn.Parameter(conv.bias)
                if args.bn:
                    co += 1
                else:
                    co += 4
            elif isinstance(model.pre_process[l], nn.BatchNorm2d):
                bn = model.pre_process[l]
                quantized_model.pre_process[co].weight = nn.Parameter(bn.weight)
                quantized_model.pre_process[co].bias = nn.Parameter(bn.bias)
                quantized_model.pre_process[co].running_mean = bn.running_mean
                quantized_model.pre_process[co].running_var = bn.running_var
                quantized_model.pre_process[co].num_batches_tracked = bn.num_batches_tracked
                co += 4
        for i in range(1, 5):
            if i == 1:
                layer = model.layer1
                other_layer = quantized_model.layer1
            elif i == 2:
                layer = model.layer2
                other_layer = quantized_model.layer2
            elif i == 3:
                layer = model.layer3
                other_layer = quantized_model.layer3
            elif i == 4:
                layer = model.layer4
                other_layer = quantized_model.layer4
            for index in range(len(layer)):
                co = 0
                for l in range(len(layer[index].residual)):
                    if isinstance(layer[index].residual[l], nn.Conv2d):
                        conv = layer[index].residual[l]
                        other_layer[index].residual[co].weight = nn.Parameter(conv.weight)
                        if conv.bias is not None:
                            other_layer[index].residual[co].bias = nn.Parameter(conv.bias)
                        if args.bn:
                            co += 1
                        else:
                            co += 4
                    elif isinstance(layer[index].residual[l], nn.BatchNorm2d):
                        bn = layer[index].residual[l]
                        other_layer[index].residual[co].weight = nn.Parameter(bn.weight)
                        other_layer[index].residual[co].bias = nn.Parameter(bn.bias)
                        other_layer[index].residual[co].running_mean = bn.running_mean
                        other_layer[index].residual[co].running_var = bn.running_var
                        other_layer[index].residual[co].num_batches_tracked = bn.num_batches_tracked
                        co += 4
                co = 0
                for l in range(len(layer[index].identity)):
                    if isinstance(layer[index].identity[l], nn.Conv2d):
                        conv = layer[index].identity[l]
                        other_layer[index].identity[co].weight = nn.Parameter(conv.weight)
                        if conv.bias is not None:
                            other_layer[index].identity[co].bias = nn.Parameter(conv.bias)
                        if args.bn:
                            co += 1
                        else:
                            co += 4
                    elif isinstance(layer[index].identity[l], nn.BatchNorm2d):
                        bn = layer[index].identity[l]
                        other_layer[index].identity[co].weight = nn.Parameter(bn.weight)
                        other_layer[index].identity[co].bias = nn.Parameter(bn.bias)
                        other_layer[index].identity[co].running_mean = bn.running_mean
                        other_layer[index].identity[co].running_var = bn.running_var
                        other_layer[index].identity[co].num_batches_tracked = bn.num_batches_tracked
                        co += 4
        for l in range(len(model.classifier)):
            if isinstance(model.classifier[l], nn.Linear):
                quantized_model.classifier[l].weight = nn.Parameter(model.classifier[l].weight)


def find_max_min(dataset, args, model, num_batches):
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size,
                                         shuffle=True,
                                         num_workers=args.workers, pin_memory=True)
    with torch.no_grad():
        model.eval()
        for batch_idx, (data, target) in enumerate(loader, 0):
            if args.gpu is not None:
                data, target = data.cuda(args.gpu, non_blocking=True), target.cuda(args.gpu, non_blocking=True)
            elif torch.cuda.is_available():
                data, target = data.cuda(), target.cuda() 
            model(data)
            if batch_idx == num_batches:
                break


def train(epoch, loader, optimizer, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    if epoch in args.lr_interval:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / args.lr_reduce
            args.learning_rate = param_group['lr']

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
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()

        losses.update(loss.item(), data.size(0))
        top1.update(correct.item() / data.size(0), data.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            progress.display(batch_idx)

    print('TRAIN: * Epoch {0} Acc@1 {1:.4f} Loss {2:.4f}'.format(epoch, top1.avg, losses.avg))


def test(epoch, loader, model, args, max_accuracy, start_time, identifier, optimizer):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(loader),
        [batch_time, losses, top1],
        prefix="Test: [{}]".format(epoch))

    with torch.no_grad():
        model.eval()
        end = time.time()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(loader, 0):
            if args.gpu is not None:
                data, target = data.cuda(args.gpu, non_blocking=True), target.cuda(args.gpu, non_blocking=True)
            elif torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = F.cross_entropy(output, target)
            total_loss += loss.item()
            pred = output.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()

            losses.update(loss.item(), data.size(0))
            top1.update(correct.item() / data.size(0), data.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.print_freq == 0:
                progress.display(batch_idx)

    if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0):
        if top1.avg > max_accuracy:
            max_accuracy = top1.avg
            print('Saving model check point')
            state = {
                'accuracy': max_accuracy,
                'epoch': epoch,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            try:
                os.mkdir('./trained_models/ann/')
            except OSError:
                pass

            filename = './trained_models/ann/' + identifier + '.pth'
            if not args.dont_save:
                torch.save(state, filename)

    print(
        'TEST: * Acc@1 {0:.4f} Loss {1:.4f}, Best Acc@1: {2:.4f}, Time: {3}'.format(top1.avg, losses.avg, max_accuracy,
                                                                                    datetime.timedelta(seconds=(
                                                                                            datetime.datetime.now() - start_time).seconds)))

    return max_accuracy


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
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

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()
