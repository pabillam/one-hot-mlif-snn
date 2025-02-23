from __future__ import print_function
import csv
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms, models, utils
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from functools import reduce
import numpy as np
import datetime
import time
import pdb
from models import *
import sys
import os
import shutil
import argparse

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


def find_threshold(batch_size=512, timesteps=2500, architecture='VGG16'):
    if args.gpu:
        loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=False,
                                             generator=torch.Generator(device='cuda'))
    else:
        loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=False)
    model.module.network_update(timesteps=timesteps, leak=1.0)
    pos = 0
    thresholds = []

    def find(layer):
        max_act = 0

        print('\n Finding threshold for layer {}'.format(layer))
        for batch_idx, (data, target) in enumerate(loader):

            if torch.cuda.is_available() and args.gpu:
                data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                model.eval()
                output = model(data, find_max_mem=True, max_mem_layer=layer)
                if output > max_act:
                    max_act = output.item()

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
                        find(pos)
                        pos = pos + 1

        for c in model.module.classifier.named_children():
            if isinstance(c[1], nn.Linear):
                if int(c[0]) == len(model.module.classifier) - 1:
                    break
                else:
                    find(int(c[0]) + pos)

    print('\n ANN thresholds: {}'.format(thresholds))
    return thresholds


def train(epoch):
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')

    if args.thr_lr == 0.0: 
        if epoch in args.lr_interval:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / args.lr_reduce
    else:
        if epoch in args.lr_interval:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / args.lr_reduce
        # if epoch in args.lr_interval:
        #     optimizer.param_groups[1]['lr'] = optimizer.param_groups[1]['lr'] / args.lr_reduce
    
    if args.thr_lr == 0.0:
        print('Current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
    else:
        print('Current lr {:.5e}, thr_lr {:.5e}'.format(optimizer.param_groups[1]['lr'], optimizer.param_groups[0]['lr']))

    model.train()
    local_time = datetime.datetime.now()

    for batch_idx, (data, target) in enumerate(train_loader):

        if torch.cuda.is_available() and args.gpu:
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

        if (batch_idx + 1) % train_acc_batches == 0:
            temp1 = []
            temp2 = []
            for key, value in sorted(model.module.threshold.items(), key=lambda x: (int(x[0][1:]), (x[1]))):
                temp1 = temp1 + [round(value.item(), 5)]
            for key, value in sorted(model.module.leak.items(), key=lambda x: (int(x[0][1:]), (x[1]))):
                temp2 = temp2 + [round(value.item(), 5)]
            print(
                '\n\nEpoch: {}, batch: {}, train_loss: {:.4f}, train_acc: {:.4f}, threshold: {}, leak: {}, timesteps: {}, time: {}'
                    .format(epoch,
                            batch_idx + 1,
                            losses.avg,
                            top1.avg,
                            temp1,
                            temp2,
                            model.module.timesteps,
                            datetime.timedelta(seconds=(datetime.datetime.now() - local_time).seconds)
                            )
            )
            local_time = datetime.datetime.now()

    print('\nEpoch: {}, train_loss: {:.4f}, train_acc: {:.4f}'.format(epoch, losses.avg, top1.avg))

def test(epoch):
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')

    if args.test_only:
        temp1 = []
        temp2 = []
        for key, value in sorted(model.module.threshold.items(), key=lambda x: (int(x[0][1:]), (x[1]))):
            temp1 = temp1 + [round(value.item(), 2)]
        for key, value in sorted(model.module.leak.items(), key=lambda x: (int(x[0][1:]), (x[1]))):
            temp2 = temp2 + [round(value.item(), 2)]
        print('\n Thresholds: {}, leak: {}'.format(temp1, temp2))

    with torch.no_grad():
        model.eval()
        global max_accuracy
        global test_snn_ops

        for batch_idx, (data, target) in enumerate(test_loader):

            if torch.cuda.is_available() and args.gpu:
                data, target = data.cuda(), target.cuda()

            output = model(data)
            if args.print_ops:
                test_snn_ops = reduce(lambda d1, d2: {k: d1.get(k, 0) + d2.get(k, 0) for k in set(d1) | set(d2)},
                                      [model.module.ops, test_snn_ops])
            
            mean_output = output.mean(1)
            loss = F.cross_entropy(mean_output, target)

            pred = mean_output.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()

            losses.update(loss.item(), data.size(0))
            top1.update(correct.item() / data.size(0), data.size(0))

            if test_acc_every_batch:
                print('\n Images {}/{} Accuracy: {}/{}({:.4f})'.format(
                    test_loader.batch_size * (batch_idx + 1),
                    len(test_loader.dataset),
                    correct.item(),
                    data.size(0),
                    top1.avg
                )
                )

        temp1 = []
        temp2 = []
        for key, value in sorted(model.module.threshold.items(), key=lambda x: (int(x[0][1:]), (x[1]))):
            temp1 = temp1 + [value.item()]
        for key, value in sorted(model.module.leak.items(), key=lambda x: (int(x[0][1:]), (x[1]))):
            temp2 = temp2 + [value.item()]

        if epoch > 5 and top1.avg < 0.15:
            print('\n Quitting as the training is not progressing')
            exit(0)

        if top1.avg > max_accuracy:
            max_accuracy = top1.avg

            state = {
                'accuracy': max_accuracy,
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'thresholds': temp1,
                'timesteps': timesteps,
                'leak': temp2,
                'activation': activation
            }
            try:
                os.mkdir('./trained_models/snn/')
            except OSError:
                pass
            filename = './trained_models/snn/' + identifier + '.pth'
            if not args.dont_save:
                torch.save(state, filename)

        print(' test_loss: {:.4f}, test_acc: {:.4f}, best: {:.4f} time: {}'.format(
            losses.avg,
            top1.avg,
            max_accuracy,
            datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)
        )
        )


def compute_ops(model, dataset, num_samples, architecture, filename):
    if dataset.lower().startswith('cifar'):
        h_in, w_in = 32, 32
    elif dataset.lower().startswith('image'):
        h_in, w_in = 224, 224

    fields = ['Layer', 'Type', '# MAC', '# Spikes', '# Neurons', 'Spike Rate', 'SNN OP (Paper)']
    rows = []

    ann_ops = []
    snn_ops = []
    snn_spikes = []

    global test_snn_ops
    print("Num Samples: {}".format(num_samples))
    if architecture == "res":
        # Layer 0 is ANN
        l = 0
        c_in = model.module.pre_process[l].in_channels
        k = model.module.pre_process[l].kernel_size[0]
        h_out = int(
            (h_in - k + 2 * model.module.pre_process[l].padding[0]) / (model.module.pre_process[l].stride[0])) + 1
        w_out = int(
            (w_in - k + 2 * model.module.pre_process[l].padding[0]) / (model.module.pre_process[l].stride[0])) + 1
        c_out = model.module.pre_process[l].out_channels
        mac = k * k * c_in * h_out * w_out * c_out
        print('h_in {}, w_in {}, c_in {}, h_out {}, w_out {}, c_out {}'.format(h_in, w_in, c_in, h_out, w_out, c_out))
        num_neurons = h_in * w_in * c_in
        spike_rate = 0
        snn_op = 0
        snn_spike = 0
        snn_spikes.append(snn_spike)
        snn_ops.append(snn_op)
        ann_ops.append(mac)
        h_in = h_out
        w_in = w_out
        rows.append(['{}'.format(l), 'Conv2d', '{}'.format(mac), '{}'.format(snn_spike),
                     '{}'.format(num_neurons), '{}'.format(spike_rate), '{}'.format(snn_op)])
        print('Layer {} Conv2d, MAC: {}'.format(l, mac))
        print('Layer {} Conv2d, # Spikes: {}'.format(l, snn_spike))
        print('Layer {} Conv2d, Num Neurons: {}'.format(l, num_neurons))
        print('Layer {} Conv2d, Spike Rate: {}'.format(l, spike_rate))
        print('Layer {} Conv2d, # OP SNN: {}'.format(l, snn_op))
        print('\n')
        for l in range(1, len(model.module.pre_process)):
            if isinstance(model.module.pre_process[l], nn.Conv2d):
                c_in = model.module.pre_process[l].in_channels
                k = model.module.pre_process[l].kernel_size[0]
                h_out = int((h_in - k + 2 * model.module.pre_process[l].padding[0]) / (
                    model.module.pre_process[l].stride[0])) + 1
                w_out = int((w_in - k + 2 * model.module.pre_process[l].padding[0]) / (
                    model.module.pre_process[l].stride[0])) + 1
                c_out = model.module.pre_process[l].out_channels
                mac = k * k * c_in * h_out * w_out * c_out
                num_neurons = h_in * w_in * c_in
                spike_rate = test_snn_ops[l] / num_samples / num_neurons
                snn_op = spike_rate * mac
                snn_spike = test_snn_ops[l] / num_samples
                snn_spikes.append(snn_spike)
                snn_ops.append(snn_op)
                ann_ops.append(mac)
                h_in = h_out
                w_in = w_out
                rows.append(['{}'.format(l), 'Conv2d', '{}'.format(mac), '{}'.format(snn_spike),
                             '{}'.format(num_neurons), '{}'.format(spike_rate), '{}'.format(snn_op)])
                print('Layer {} Conv2d, MAC: {}'.format(l, mac))
                print('Layer {} Conv2d, # Spikes: {}'.format(l, snn_spike))
                print('Layer {} Conv2d, Num Neurons: {}'.format(l, num_neurons))
                print('Layer {} Conv2d, Spike Rate: {}'.format(l, spike_rate))
                print('Layer {} Conv2d, # OP SNN: {}'.format(l, snn_op))
                print('\n')
            elif isinstance(model.module.pre_process[l], nn.AvgPool2d):
                h_in = h_in // model.module.pre_process[l].kernel_size
                w_in = w_in // model.module.pre_process[l].kernel_size

        pos = len(model.module.pre_process)
        for i in range(1, 5):
            layer = model.module.layers[i]
            for index in range(len(layer)):
                first_pos = pos
                first_h_in = h_in
                first_w_in = w_in
                for l in range(len(layer[index].residual)):
                    if isinstance(layer[index].residual[l], nn.Conv2d):
                        c_in = layer[index].residual[l].in_channels
                        k = layer[index].residual[l].kernel_size[0]
                        h_out = int((h_in - k + 2 * layer[index].residual[l].padding[0]) / (
                            layer[index].residual[l].stride[0])) + 1
                        w_out = int((w_in - k + 2 * layer[index].residual[l].padding[0]) / (
                            layer[index].residual[l].stride[0])) + 1
                        c_out = layer[index].residual[l].out_channels
                        mac = k * k * c_in * h_out * w_out * c_out
                        num_neurons = h_in * w_in * c_in
                        spike_rate = test_snn_ops[pos] / num_samples / num_neurons
                        snn_op = spike_rate * mac
                        snn_spike = test_snn_ops[pos] / num_samples
                        snn_spikes.append(snn_spike)
                        snn_ops.append(snn_op)
                        ann_ops.append(mac)
                        h_in = h_out
                        w_in = w_out
                        rows.append(['{}'.format(pos), 'Conv2d', '{}'.format(mac), '{}'.format(snn_spike),
                                     '{}'.format(num_neurons), '{}'.format(spike_rate), '{}'.format(snn_op)])
                        print('Layer {} Conv2d, MAC: {}'.format(pos, mac))
                        print('Layer {} Conv2d, # Spikes: {}'.format(pos, snn_spike))
                        print('Layer {} Conv2d, Spike Rate: {}'.format(pos, spike_rate))
                        print('Layer {} Conv2d, Num Neurons: {}'.format(pos, num_neurons))
                        print('Layer {} Conv2d, # OP SNN: {}'.format(pos, snn_op))
                        print('\n')
                        pos = pos + 1
                for l in range(len(layer[index].identity)):
                    if isinstance(layer[index].identity[l], nn.Conv2d):
                        c_in = layer[index].identity[l].in_channels
                        c_out = layer[index].identity[l].out_channels
                        k = layer[index].identity[l].kernel_size[0]
                        h_out = int((first_h_in - k + 2 * layer[index].residual[l].padding[0]) / (
                            layer[index].residual[l].stride[0])) + 1
                        w_out = int((first_w_in - k + 2 * layer[index].residual[l].padding[0]) / (
                            layer[index].residual[l].stride[0])) + 1
                        mac = k * k * h_out * w_out * c_out * c_in
                        num_neurons = first_h_in * first_w_in * c_in
                        spike_rate = test_snn_ops[first_pos] / num_samples / num_neurons
                        snn_op = spike_rate * mac
                        snn_spike = test_snn_ops[first_pos] / num_samples
                        snn_spikes.append(snn_spike)
                        snn_ops.append(snn_op)
                        ann_ops.append(mac)
                        rows.append(['{}'.format(first_pos + l), 'Identity', '{}'.format(mac), '{}'.format(snn_spike),
                                     '{}'.format(num_neurons), '{}'.format(spike_rate), '{}'.format(snn_op)])
                        print('Layer {} Identity, MAC: {}'.format(first_pos + l, mac))
                        print('Layer {} Identity, # Spikes: {}'.format(first_pos + l, snn_spike))
                        print('Layer {} Identity, Spike Rate: {}'.format(first_pos + l, spike_rate))
                        print('Layer {} Identity, Num Neurons: {}'.format(first_pos + l, num_neurons))
                        print('Layer {} Identity, # OP SNN: {}'.format(first_pos + l, snn_op))
                        print('\n')
        for l in range(len(model.module.classifier)):
            if isinstance(model.module.classifier[l], nn.Linear):
                mac = model.module.classifier[l].in_features * model.module.classifier[l].out_features
                num_neurons = model.module.classifier[l].in_features
                spike_rate = test_snn_ops[pos + l] / num_samples / num_neurons
                snn_op = spike_rate * mac
                snn_spike = test_snn_ops[pos + l] / num_samples
                snn_ops.append(snn_op)
                ann_ops.append(mac)
                snn_spikes.append(snn_spike)
                rows.append(['{}'.format(pos + l), 'Linear', '{}'.format(mac), '{}'.format(snn_spike),
                             '{}'.format(num_neurons), '{}'.format(spike_rate), '{}'.format(snn_op)])
                print('Layer {} Linear, MAC: {}'.format(pos + l, mac))
                print('Layer {} Linear, # Spikes: {}'.format(pos + l, snn_spike))
                print('Layer {} Linear, Spike Rate: {}'.format(pos + l, spike_rate))
                print('Layer {} Linear, Num Neurons: {}'.format(pos + l, num_neurons))
                print('Layer {} Linear, # OP SNN: {}'.format(pos + l, snn_op))
                print('\n')
    elif architecture == "vgg":
        # Layer 0 is ANN
        l = 0
        c_in = model.module.features[l].in_channels
        k = model.module.features[l].kernel_size[0]
        h_out = int((h_in - k + 2 * model.module.features[l].padding[0]) / (model.module.features[l].stride[0])) + 1
        w_out = int((w_in - k + 2 * model.module.features[l].padding[0]) / (model.module.features[l].stride[0])) + 1
        c_out = model.module.features[l].out_channels
        mac = k * k * c_in * h_out * w_out * c_out
        num_neurons = h_in * w_in * c_in
        spike_rate = 0
        snn_op = 0
        snn_spike = 0
        snn_spikes.append(snn_spike)
        snn_ops.append(snn_op)
        ann_ops.append(mac)
        h_in = h_out
        w_in = w_out
        rows.append(['{}'.format(l), 'Conv2d', '{}'.format(mac), '{}'.format(snn_spike),
                     '{}'.format(num_neurons), '{}'.format(spike_rate), '{}'.format(snn_op)])
        print('Layer {} Conv2d, MAC: {}'.format(l, mac))
        print('Layer {} Conv2d, # Spikes: {}'.format(l, snn_spike))
        print('Layer {} Conv2d, Num Neurons: {}'.format(l, num_neurons))
        print('Layer {} Conv2d, Spike Rate: {}'.format(l, spike_rate))
        print('Layer {} Conv2d, # OP SNN: {}'.format(l, snn_op))
        print('\n')
        for l in range(1, len(model.module.features)):
            if isinstance(model.module.features[l], nn.Conv2d):
                c_in = model.module.features[l].in_channels
                k = model.module.features[l].kernel_size[0]
                h_out = int(
                    (h_in - k + 2 * model.module.features[l].padding[0]) / (model.module.features[l].stride[0])) + 1
                w_out = int(
                    (w_in - k + 2 * model.module.features[l].padding[0]) / (model.module.features[l].stride[0])) + 1
                c_out = model.module.features[l].out_channels
                mac = k * k * c_in * h_out * w_out * c_out
                num_neurons = h_in * w_in * c_in
                spike_rate = test_snn_ops[l] / num_samples / num_neurons
                snn_op = spike_rate * mac
                snn_spike = test_snn_ops[l] / num_samples
                snn_spikes.append(snn_spike)
                snn_ops.append(snn_op)
                ann_ops.append(mac)
                h_in = h_out
                w_in = w_out
                rows.append(['{}'.format(l), 'Conv2d', '{}'.format(mac), '{}'.format(snn_spike),
                             '{}'.format(num_neurons), '{}'.format(spike_rate), '{}'.format(snn_op)])
                print('Layer {} Conv2d, MAC: {}'.format(l, mac))
                print('Layer {} Conv2d, # Spikes: {}'.format(l, snn_spike))
                print('Layer {} Conv2d, Spike Rate: {}'.format(l, spike_rate))
                print('Layer {} Conv2d, Num Neurons: {}'.format(l, num_neurons))
                print('Layer {} Conv2d, # OP SNN: {}'.format(l, snn_op))
            elif isinstance(model.module.features[l], nn.AvgPool2d):
                h_in = h_in // model.module.features[l].kernel_size
                w_in = w_in // model.module.features[l].kernel_size

        prev = len(model.module.features)
        for l in range(len(model.module.classifier)):
            if isinstance(model.module.classifier[l], nn.Linear):
                mac = model.module.classifier[l].in_features * model.module.classifier[l].out_features
                num_neurons = model.module.classifier[l].in_features
                spike_rate = test_snn_ops[prev + l] / num_samples / num_neurons
                snn_op = spike_rate * mac
                snn_spike = test_snn_ops[prev + l] / num_samples
                snn_spikes.append(snn_spike)
                snn_ops.append(snn_op)
                ann_ops.append(mac)
                rows.append(['{}'.format(l), 'Linear', '{}'.format(mac), '{}'.format(snn_spike),
                             '{}'.format(num_neurons), '{}'.format(spike_rate), '{}'.format(snn_op)])
                print('Layer {} Linear, MAC: {}'.format(prev + l, mac))
                print('Layer {} Linear, # Spikes: {}'.format(prev + l, snn_spike))
                print('Layer {} Linear, Spike Rate: {}'.format(prev + l, spike_rate))
                print('Layer {} Linear, Num Neurons: {}'.format(prev + l, num_neurons))
                print('Layer {} Linear, # OP SNN: {}'.format(prev + l, snn_op))

    with open(filename, 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(rows)

    print('Total MACs: {:e}'.format(sum(ann_ops)))
    print('Total SNN OPs: {:e}'.format(sum(snn_ops)))
    print('Total SNN Spikes: {:e}'.format(sum(snn_spikes)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SNN training')
    parser.add_argument('--gpu', default=1, type=int, help='use gpu', choices=[0, 1])
    parser.add_argument('-s', '--seed', default=0, type=int, help='seed for random number')
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset name',
                        choices=['CIFAR10', 'CIFAR100', 'IMAGENET'])
    parser.add_argument('--dataset_dir', metavar='path', default='./data', help='dataset path')
    parser.add_argument('--batch_size', default=64, type=int, help='minibatch size')
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
    parser.add_argument('-print_ops', default=0, type=int, help="print # snn ops", choices=[0, 1])
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, help='initial learning_rate')
    parser.add_argument('-thr_lr', default=0.0, type=float, help='learning rate for thresholds')
    parser.add_argument('--pretrained_ann', default='', type=str, help='pretrained ANN model')
    parser.add_argument('--pretrained_snn', default='', type=str, help='pretrained SNN for inference')
    parser.add_argument('--test_only', action='store_true', help='perform only inference')
    parser.add_argument('--log', action='store_true', help='to print the output on terminal or to log file')
    parser.add_argument('--epochs', default=30, type=int, help='number of training epochs')
    parser.add_argument('--lr_interval', default='0.60 0.80 0.90', type=str,
                        help='intervals at which to reduce lr, expressed as %%age of total epochs')
    parser.add_argument('--lr_reduce', default=10, type=int, help='reduction factor for learning rate')
    parser.add_argument('--timesteps', default=20, type=int, help='simulation timesteps')
    parser.add_argument('--leak', default=1.0, type=float, help='membrane leak')
    parser.add_argument('--scaling_factor', default=0.3, type=float,
                        help='scaling factor for thresholds at reduced timesteps')
    parser.add_argument('--default_threshold', default=1.0, type=float,
                        help='initial threshold to train SNN from scratch')
    parser.add_argument('--activation', default='Linear', type=str, help='SNN activation function',
                        choices=['Linear'])
    parser.add_argument('--optimizer', default='SGD', type=str, help='optimizer for SNN backpropagation',
                        choices=['SGD', 'Adam'])
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay parameter for the optimizer')
    parser.add_argument('--momentum', default=0.95, type=float, help='momentum parameter for the SGD optimizer')
    parser.add_argument('--amsgrad', default=True, type=bool, help='amsgrad parameter for Adam optimizer')
    parser.add_argument('--betas', default='0.9,0.999', type=str, help='betas for Adam optimizer')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout percentage for conv layers')
    parser.add_argument('--kernel_size', default=3, type=int, help='filter size for the conv layers')
    parser.add_argument('--test_acc_every_batch', action='store_true', help='print acc of every batch during inference')
    parser.add_argument('--train_acc_batches', default=1000, type=int,
                        help='print training progress after this many batches')
    parser.add_argument('--devices', default='0', type=str, help='list of gpu device(s)')
    parser.add_argument('--resume', default='', type=str, help='resume training from this state')
    parser.add_argument('--dont_save', action='store_true', help='don\'t save training model during testing')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

    # Seed random number
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
    print_ops = args.print_ops
    learning_rate = args.learning_rate
    thr_lr = args.thr_lr
    pretrained_ann = args.pretrained_ann
    pretrained_snn = args.pretrained_snn
    epochs = args.epochs
    lr_reduce = args.lr_reduce
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
    test_acc_every_batch = args.test_acc_every_batch
    train_acc_batches = args.train_acc_batches
    resume = args.resume
    start_epoch = 1
    max_accuracy = 0.0
    test_snn_ops = {}

    values = args.lr_interval.split()
    lr_interval = []
    for value in values:
        lr_interval.append(int(float(value) * args.epochs))
    args.lr_interval = lr_interval

    log_file = './logs/snn/'
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
    log_file += identifier + '.log'

    if args.log:
        f = open(log_file, 'w', buffering=1)
        sys.stdout = f

    if not pretrained_ann:
        ann_file = './trained_models/ann/ann_' + architecture.lower() + '_' + dataset.lower() + '.pth'
        if os.path.exists(ann_file):
            val = input('\n Do you want to use the pretrained ANN {}? Y or N: '.format(ann_file))
            if val.lower() == 'y' or val.lower() == 'yes':
                pretrained_ann = ann_file

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

    # Training settings
    if torch.cuda.is_available() and args.gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if dataset == 'CIFAR10':
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    elif dataset == 'CIFAR100':
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    elif dataset == 'IMAGENET':
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    if dataset in ['CIFAR10', 'CIFAR100']:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    if dataset == 'CIFAR10':
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        labels = 10

    elif dataset == 'CIFAR100':
        trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        labels = 100

    elif dataset == 'IMAGENET':
        traindir = os.path.join(dataset_dir, 'train')
        valdir = os.path.join(dataset_dir, 'val')
        trainset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        testset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        labels = 1000

    if args.gpu:
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False,
                                  generator=torch.Generator(device='cuda'))
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                                 generator=torch.Generator(device='cuda'))
    else:
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

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
                               leak=leak, default_threshold=default_threshold, dropout=dropout,
                               dataset=dataset)

    # Please comment this line if you find key mismatch error and uncomment the DataParallel after the if block
    # if dataset != 'IMAGENET':
    #     model = nn.DataParallel(model)

    if pretrained_ann:
        state = torch.load(pretrained_ann, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(state['state_dict'], strict=False)
        print('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))
        if not args.bias:
            print('\n Info: Accuracy of loaded ANN model: {}'.format(state['accuracy']))

        # If thresholds present in loaded ANN file
        if 'thresholds' in state.keys():
            thresholds = state['thresholds']
            print('\n Info: Thresholds loaded from trained ANN: {}'.format(thresholds))
            model.module.threshold_update(scaling_factor=scaling_factor, thresholds=thresholds[:])
        else:
            thresholds = find_threshold(batch_size=512, timesteps=100, architecture=architecture)
            model.module.threshold_update(scaling_factor=scaling_factor, thresholds=thresholds[:])

            # Save the thresholds in the ANN file
            temp = {}
            for key, value in state.items():
                temp[key] = value
            temp['thresholds'] = thresholds
            torch.save(temp, pretrained_ann)

    elif pretrained_snn:
        state = torch.load(pretrained_snn, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(state['state_dict'], strict=False)
        print('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))
        print('\n Info: Accuracy of loaded ANN model: {}'.format(state['accuracy']))

    # if dataset == 'IMAGENET':
    model = nn.DataParallel(model)

    print('\n {}'.format(model))

    if torch.cuda.is_available() and args.gpu:
        model.cuda()

    threshold_params = dict()
    other_params = dict()
    for name, param in model.named_parameters():
        if 'threshold' in name:
            threshold_params[name] = param
        else:
            other_params[name] = param 

    if optimizer == 'Adam':
        if args.thr_lr == 0.0:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=amsgrad, weight_decay=weight_decay,
                                   betas=(beta1, beta2))
        else:
            optimizer = optim.Adam([{"params": threshold_params.values(), "lr": args.thr_lr},
                                    {"params": other_params.values()}], lr=learning_rate, amsgrad=amsgrad,
                                   weight_decay=weight_decay,
                                   betas=(beta1, beta2))
    elif optimizer == 'SGD':
        if args.thr_lr == 0.0:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,
                                  weight_decay=weight_decay,
                                  nesterov=False)
        else:
            optimizer = optim.SGD([{"params": threshold_params.values(), "lr": args.thr_lr},
                                   {"params": other_params.values()}], lr=learning_rate, momentum=momentum,
                                  weight_decay=weight_decay,
                                  nesterov=False)

    print('\n {}'.format(optimizer))

    # find_threshold() alters the timesteps and leak, restoring it here
    model.module.network_update(timesteps=timesteps, leak=leak)

    if resume:
        print('\n Resuming from checkpoint {}'.format(resume))
        state = torch.load(resume, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(state['state_dict'], strict=False)
        print('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))
        print('\n Info: Accuracy of loaded ANN model: {}'.format(state['accuracy']))

        epoch = state['epoch']
        start_epoch = epoch + 1
        max_accuracy = state['accuracy']
        optimizer.load_state_dict(state['optimizer'])
        for param_group in optimizer.param_groups:
            learning_rate = param_group['lr']

        print('\n Loaded from resume epoch: {}, accuracy: {:.4f} lr: {:.1e}'.format(epoch, max_accuracy, learning_rate))

    torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, epochs):
        start_time = datetime.datetime.now()
        if not args.test_only:
            train(epoch)
        test(epoch)
        if print_ops:
            print("\n")
            if dataset == 'IMAGENET':
                samps = 50000
            else:
                samps = len(testset.data)
            compute_ops(model, dataset, samps, architecture[0:3].lower(),
                        './ops/snn/' + identifier + '.csv')
            print("\n")

    print('\n Highest accuracy: {:.4f}'.format(max_accuracy))
