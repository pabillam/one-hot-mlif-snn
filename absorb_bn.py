import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from matplotlib import pyplot as plt
import pdb
import sys
import datetime
import os
from models import *
import numpy as np
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fuse batch normalization parameters in the layerwise weights for ANN-SNN conversion',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--seed', default=0, type=int, help='seed for random number')
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset name',
                        choices=['CIFAR10', 'CIFAR100', 'IMAGENET'])
    parser.add_argument('-a', '--architecture', default='VGG16', type=str, help='network architecture',
                        choices=['VGG16', 'RESNET20'])
    parser.add_argument('--pretrained_ann', default='', type=str, help='pretrained model to initialize ANN')
    parser.add_argument('--dropout', default=0.2, type=float, help='dropout percentage for conv layers')
    parser.add_argument('--kernel_size', default=3, type=int, help='filter size for the conv layers')
    parser.add_argument('--bn', action='store_true', help='enable batch normalization layers in network')
    parser.add_argument('--bias', action='store_true', help='enable bias for conv layers in network')

    args = parser.parse_args()

    # Seed random number
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset = args.dataset
    architecture = args.architecture
    pretrained_ann = args.pretrained_ann
    dropout = args.dropout
    kernel_size = args.kernel_size

    # Loading Dataset
    if dataset == 'CIFAR100':
        labels = 100
    elif dataset == 'CIFAR10':
        labels = 10
    elif dataset == 'IMAGENET':
        labels= 1000

    if architecture[0:3].lower() == 'vgg':
        model = VGG(vgg_name=architecture, labels=labels, dataset=dataset, kernel_size=kernel_size, dropout=dropout, bn=False, bias=True)
        model_bn = VGG(vgg_name=architecture, labels=labels, dataset=dataset, kernel_size=kernel_size, dropout=dropout, bn=args.bn, bias=args.bias)
    elif architecture[0:3].lower() == 'res':
        if architecture.lower() == 'resnet20':
            model = ResNet20(labels=labels, dropout=dropout, dataset=dataset, bn=False, bias=True)
            model_bn = ResNet20(labels=labels, dropout=dropout, dataset=dataset, bn=args.bn, bias=args.bias)

    # Load model with batch norm and bias
    if pretrained_ann:
        state = torch.load(pretrained_ann, map_location='cpu')
        missing_keys, unexpected_keys = model_bn.load_state_dict(state['state_dict'], strict=False)
        print('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))

    model = nn.DataParallel(model)
    model_bn = nn.DataParallel(model_bn)
    
    if architecture[0:3].lower() == 'vgg':
        co = 0
        for l in range(len(model_bn.module.features)):
            if isinstance(model_bn.module.features[l], nn.Conv2d):
                # Here beta and gamma are used oppositely, beta in the actual formula is gamma here and vice-versa
                conv = model_bn.module.features[l]
                bn = model_bn.module.features[l + 1]
                w = conv.weight
                mean = bn.running_mean
                var_sqrt = torch.sqrt(bn.running_var + bn.eps)
                beta = bn.weight
                gamma = bn.bias
                if conv.bias is not None:
                    b = conv.bias
                else:
                    b = mean.new_zeros(mean.shape)
                w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
                b = (b - mean) / var_sqrt * beta + gamma
                model.module.features[co].weight = nn.Parameter(w)
                model.module.features[co].bias = nn.Parameter(b)
                co += 3
        for l in range(len(model.module.classifier)):
            if isinstance(model.module.classifier[l], nn.Linear):
                model.module.classifier[l].weight = nn.Parameter(model_bn.module.classifier[l].weight)
    elif architecture[0:3].lower() == 'res':
        co = 0
        for l in range(len(model_bn.module.pre_process)):
            if isinstance(model_bn.module.pre_process[l], nn.Conv2d):
                conv = model_bn.module.pre_process[l]
                bn = model_bn.module.pre_process[l + 1]
                w = conv.weight
                mean = bn.running_mean
                var_sqrt = torch.sqrt(bn.running_var + bn.eps)
                beta = bn.weight
                gamma = bn.bias
                if conv.bias is not None:
                    b = conv.bias
                else:
                    b = mean.new_zeros(mean.shape)
                w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
                b = (b - mean) / var_sqrt * beta + gamma
                model.module.pre_process[co].weight = nn.Parameter(w)
                model.module.pre_process[co].bias = nn.Parameter(b)
                co += 3
        for i in range(1, 5):
            if i == 1:
                layer = model_bn.module.layer1
                other_layer = model.module.layer1
            elif i == 2:
                layer = model_bn.module.layer2
                other_layer = model.module.layer2
            elif i == 3:
                layer = model_bn.module.layer3
                other_layer = model.module.layer3
            elif i == 4:
                layer = model_bn.module.layer4
                other_layer = model.module.layer4
            for index in range(len(layer)):
                co = 0
                for l in range(len(layer[index].residual)):
                    if isinstance(layer[index].residual[l], nn.Conv2d):
                        conv = layer[index].residual[l]
                        bn = layer[index].residual[l + 1]
                        w = conv.weight
                        mean = bn.running_mean
                        var_sqrt = torch.sqrt(bn.running_var + bn.eps)
                        beta = bn.weight
                        gamma = bn.bias
                        if conv.bias is not None:
                            b = conv.bias
                        else:
                            b = mean.new_zeros(mean.shape)
                        w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
                        b = (b - mean) / var_sqrt * beta + gamma
                        other_layer[index].residual[co].weight = nn.Parameter(w)
                        other_layer[index].residual[co].bias = nn.Parameter(b)
                        co += 3
                co = 0
                for l in range(len(layer[index].identity)):
                    if isinstance(layer[index].identity[l], nn.Conv2d):
                        conv = layer[index].identity[l]
                        bn = layer[index].identity[l + 1]
                        w = conv.weight
                        mean = bn.running_mean
                        var_sqrt = torch.sqrt(bn.running_var + bn.eps)
                        beta = bn.weight
                        gamma = bn.bias
                        if conv.bias is not None:
                            b = conv.bias
                        else:
                            b = mean.new_zeros(mean.shape)
                        w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
                        b = (b - mean) / var_sqrt * beta + gamma
                        other_layer[index].identity[co].weight = nn.Parameter(w)
                        other_layer[index].identity[co].bias = nn.Parameter(b)
                        co += 3
        for l in range(len(model.module.classifier)):
            if isinstance(model.module.classifier[l], nn.Linear):
                model.module.classifier[l].weight = nn.Parameter(model_bn.module.classifier[l].weight)

    state = {'state_dict': model.state_dict()}

    res = re.search(r"((_[0-9]+)+)", pretrained_ann)
    identifier = 'ann_' + architecture.lower() + '_' + dataset.lower() + '_' + 'bn_fused' + res.group()
    identifier_clean = 'ann_' + architecture.lower() + '_' + dataset.lower() + '_' + 'bn_fused_clean' + res.group()
    filename = './trained_models/ann/' + identifier + '.pth'
    filename_clean = './trained_models/ann/' + identifier_clean + '.pth'

    torch.save(state, filename)
    torch.save(state, filename_clean)
    print('Finished BN Fusion')


