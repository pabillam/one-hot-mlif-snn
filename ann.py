import argparse
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from matplotlib import pyplot as plt
import pdb
import sys
import datetime
import time
import sys
import os
import numpy as np
from models import *

qann_batch_sparsity = {}
qann_sparsity = {}

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

def hook(name):
    def dump_act(module, input, output):
        if len(output) > 0:
            qann_batch_sparsity[name] = np.count_nonzero(input[0].detach().cpu().numpy())
    return dump_act

def train(epoch, loader):
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')

    if epoch in lr_interval:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / lr_reduce
            learning_rate = param_group['lr']

    model.train()
    for batch_idx, (data, target) in enumerate(loader):

        if torch.cuda.is_available() and args.gpu:
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

    print('\n Epoch: {}, train_loss: {:.4f}, train_acc: {:.4f}'.format(
            epoch,
            losses.avg,
            top1.avg
        )
    )

def test(loader):
    global qann_batch_sparsity
    global qann_sparsity

    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')

    if args.test_only:
        for n, m in model.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.register_forward_hook(hook(n))

    with torch.no_grad():
        model.eval()
        total_loss = 0
        correct = 0
        global max_accuracy, start_time

        for batch_idx, (data, target) in enumerate(loader):

            if torch.cuda.is_available() and args.gpu:
                data, target = data.cuda(), target.cuda()

            output = model(data)
            if batch_idx == 0:
                qann_sparsity = qann_batch_sparsity
            else:
                for i in qann_sparsity.keys():
                    qann_sparsity[i] = qann_sparsity[i] + qann_batch_sparsity[i]
            qann_batch_sparsity = {}

            loss = F.cross_entropy(output, target)
            total_loss += loss.item()
            pred = output.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()
            losses.update(loss.item(), data.size(0))
            top1.update(correct.item() / data.size(0), data.size(0))

        if epoch > 30 and top1.avg < 0.15:
            print('Quitting as the training is not progressing')
            exit(0)

        if top1.avg > max_accuracy:
            max_accuracy = top1.avg
            state = {
                'accuracy': max_accuracy,
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            try:
                os.mkdir('./trained_models/ann/')
            except OSError:
                pass

            filename = './trained_models/ann/' + identifier + '.pth'
            if not args.dont_save:
                torch.save(state, filename)

        print(' test_loss: {:.4f}, test_acc: {:.4f}, best: {:.4f}, time: {}'.format(
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

    fields = ['Layer', 'Type', '# MAC', 'QANN OP']
    rows = []

    ann_ops = []
    qann_ops = []

    global qann_sparsity
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
        qann_op = 0
        qann_ops.append(qann_op)
        ann_ops.append(mac)
        h_in = h_out
        w_in = w_out
        rows.append(['{}'.format(l), 'Conv2d', '{}'.format(mac), '{}'.format(qann_op)])
        print('Layer {} Conv2d, MAC: {}'.format(l, mac))
        print('Layer {} Conv2d, # OP QANN: {}'.format(l, qann_op))
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
                qann_op = mac * qann_sparsity['module.pre_process.{}'.format(l)] / num_samples / num_neurons
                qann_ops.append(qann_op)
                ann_ops.append(mac)
                h_in = h_out
                w_in = w_out
                rows.append(['{}'.format(l), 'Conv2d', '{}'.format(mac), '{}'.format(qann_op)])
                print('Layer {} Conv2d, MAC: {}'.format(l, mac))
                print('Layer {} Conv2d, # OP QANN: {}'.format(l, qann_op))
                print('\n')
            elif isinstance(model.module.pre_process[l], nn.AvgPool2d):
                h_in = h_in // model.module.pre_process[l].kernel_size
                w_in = w_in // model.module.pre_process[l].kernel_size

        pos = len(model.module.pre_process)
        for i in range(1, 5):
            if i == 1:
                layer = model.module.layer1
            elif i == 2:
                layer = model.module.layer2
            elif i == 3:
                layer = model.module.layer3
            elif i == 4:
                layer = model.module.layer4
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
                        qann_op = mac * qann_sparsity['module.layer{}.{}.residual.{}'.format(i, index, l)] / num_samples / num_neurons
                        qann_ops.append(qann_op)
                        ann_ops.append(mac)
                        h_in = h_out
                        w_in = w_out
                        rows.append(['{}'.format(pos), 'Conv2d', '{}'.format(mac), '{}'.format(qann_op)])
                        print('Layer {} Conv2d, MAC: {}'.format(pos, mac))
                        print('Layer {} Conv2d, # OP QANN: {}'.format(pos, qann_op))
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
                        qann_op = mac * qann_sparsity['module.layer{}.{}.identity.{}'.format(i, index, l)] / num_samples / num_neurons
                        qann_ops.append(qann_op)
                        ann_ops.append(mac)
                        rows.append(['{}'.format(pos), 'Conv2d', '{}'.format(mac), '{}'.format(qann_op)])
                        print('Layer {} Conv2d, MAC: {}'.format(first_pos + l, mac))
                        print('Layer {} Conv2d, # OP QANN: {}'.format(first_pos + l, qann_op))
                        print('\n')
        for l in range(len(model.module.classifier)):
            if isinstance(model.module.classifier[l], nn.Linear):
                mac = model.module.classifier[l].in_features * model.module.classifier[l].out_features
                num_neurons = model.module.classifier[l].in_features
                qann_op = mac * qann_sparsity['module.classifier.{}'.format(l)] / num_samples / num_neurons
                qann_ops.append(qann_op)
                ann_ops.append(mac)
                rows.append(['{}'.format(pos + l), 'Linear', '{}'.format(mac), '{}'.format(qann_op)])
                print('Layer {} Linear, MAC: {}'.format(pos + l, mac))
                print('Layer {} Linear, # OP QANN: {}'.format(pos + l, qann_op))
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
        qann_op = 0 
        qann_ops.append(qann_op)
        ann_ops.append(mac)
        h_in = h_out
        w_in = w_out
        rows.append(['{}'.format(l), 'Conv2d', '{}'.format(mac), '{}'.format(qann_op)])
        print('Layer {} Conv2d, MAC: {}'.format(l, mac))
        print('Layer {} Conv2d, # OP QANN: {}'.format(l, qann_op))
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
                qann_op = mac * qann_sparsity['module.features.{}'.format(l)] / num_samples / num_neurons
                qann_ops.append(qann_op)
                ann_ops.append(mac)
                h_in = h_out
                w_in = w_out
                rows.append(['{}'.format(l), 'Conv2d', '{}'.format(mac), '{}'.format(qann_op)])
                print('Layer {} Conv2d, MAC: {}'.format(l, mac))
                print('Layer {} Conv2d, # OP QANN: {}'.format(l, qann_op))
                print('Layer {} Conv2d, Density: {}'.format(l, qann_sparsity['module.features.{}'.format(l)] / num_samples / num_neurons))
            elif isinstance(model.module.features[l], nn.AvgPool2d):
                h_in = h_in // model.module.features[l].kernel_size
                w_in = w_in // model.module.features[l].kernel_size

        prev = len(model.module.features)
        for l in range(len(model.module.classifier)):
            if isinstance(model.module.classifier[l], nn.Linear):
                mac = model.module.classifier[l].in_features * model.module.classifier[l].out_features
                num_neurons = model.module.classifier[l].in_features
                qann_op = mac * qann_sparsity['module.classifier.{}'.format(l)] / num_samples / num_neurons
                qann_ops.append(qann_op)
                ann_ops.append(mac)
                rows.append(['{}'.format(l), 'Linear', '{}'.format(mac), '{}'.format(qann_op)])
                print('Layer {} Linear, MAC: {}'.format(prev + l, mac))
                print('Layer {} Linear, # OP QANN: {}'.format(prev + l, qann_op))

    with open(filename, 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(rows)

    print('Total MACs: {:e}'.format(sum(ann_ops)))
    print('Total Quantized ANN OPs: {:e}'.format(sum(qann_ops)))

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ANN',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu', default=True, type=bool, help='use gpu')
    parser.add_argument('--log', action='store_true', help='to print the output on terminal or to log file')
    parser.add_argument('-s', '--seed', default=0, type=int, help='seed for random number')
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset name',
                        choices=['CIFAR10', 'CIFAR100', 'IMAGENET'])
    parser.add_argument('--dataset_dir', metavar='path', default='./data/', help='dataset path')
    parser.add_argument('--batch_size', default=64, type=int, help='minibatch size')
    parser.add_argument('-a', '--architecture', default='VGG16', type=str, help='network architecture',
                        choices=['VGG16', 'RESNET20'])
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help='initial learning_rate')
    parser.add_argument('--pretrained_ann', default='', type=str, help='pretrained model to initialize ANN')
    parser.add_argument('--test_only', action='store_true', help='perform only inference')
    parser.add_argument('--epochs', default=300, type=int, help='number of training epochs')
    parser.add_argument('--lr_interval', default='0.45 0.70 0.90', type=str,
                        help='intervals at which to reduce lr, expressed as %%age of total epochs')
    parser.add_argument('--lr_reduce', default=10, type=int, help='reduction factor for learning rate')
    parser.add_argument('--optimizer', default='SGD', type=str, help='optimizer for SNN backpropagation',
                        choices=['SGD', 'Adam'])
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD optimizer')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay for Adam/SGD optimizers')
    parser.add_argument('--dropout', default=0.2, type=float, help='dropout percentage for conv layers')
    parser.add_argument('--kernel_size', default=3, type=int, help='filter size for the conv layers')
    parser.add_argument('--dont_save', action='store_true', help='don\'t save training model during testing')
    parser.add_argument('--quant_act', action='store_true', help='quantize activations to match SNN resolution')
    parser.add_argument('--act_bits', default=1, type=int, help='number of bits to quantize activations')
    parser.add_argument('--quant_scale_max', default=0.8, type=float, help='scaling factor to apply to max')
    parser.add_argument('--quant_num_batches', default=10,
                        help='number of batches to go over during quantization analysis')
    parser.add_argument('--quant_id', default='log', type=str, help='quantization scheme identifier') # TODO
    parser.add_argument('--devices', default='0', type=str, help='list of gpu device(s)')
    parser.add_argument('--bn', action='store_true', help='enable batch normalization layers in network')
    parser.add_argument('--bias', action='store_true', help='enable bias for conv layers in network')
    parser.add_argument('-print_ops', default=0, type=int, help="print # qann ops", choices=[0, 1])

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
    learning_rate = args.learning_rate
    pretrained_ann = args.pretrained_ann
    epochs = args.epochs
    lr_reduce = args.lr_reduce
    optimizer = args.optimizer
    momentum = args.momentum
    weight_decay = args.weight_decay
    dropout = args.dropout
    kernel_size = args.kernel_size

    values = args.lr_interval.split()
    lr_interval = []
    for value in values:
        lr_interval.append(int(float(value) * args.epochs))

    log_file = './logs/ann/'
    try:
        os.mkdir(log_file)
    except OSError:
        pass

    if args.quant_act:
        identifier = 'ann_' + architecture.lower() + '_' + dataset.lower() + '_AB_' + str(
            args.act_bits) + '_' + args.quant_id + '_' + datetime.datetime.now().strftime(
            "%Y_%m_%d_%H_%M_%S")
    else:
        identifier = 'ann_' + architecture.lower() + '_' + dataset.lower() + '_' + datetime.datetime.now().strftime(
            "%Y_%m_%d_%H_%M_%S")

    log_file += identifier + '.log'

    if args.log:
        f = open(log_file, 'w', buffering=1)
        sys.stdout = f

    print('\n Run on time: {}'.format(datetime.datetime.now()))
    print(f'Process ID: {os.getpid()}')

    print('\n Arguments:')
    for arg in vars(args):
        if arg == 'lr_interval':
            print('\t {:20} : {}'.format(arg, lr_interval))
        else:
            print('\t {:20} : {}'.format(arg, getattr(args, arg)))

    # Training settings
    if torch.cuda.is_available() and args.gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Loading Dataset
    if dataset == 'CIFAR100':
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        labels = 100
    elif dataset == 'IMAGENET':
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        labels = 1000
    elif dataset == 'CIFAR10':
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        labels = 10

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

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

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
                             quant_act=args.quant_act, act_bits=args.act_bits, quant_scale_max=args.quant_scale_max,
                             quant_id=args.quant_id)

        if args.quant_act:
            if architecture.lower() == 'resnet20':
                base_model = ResNet20(labels=labels, dropout=dropout, dataset=dataset, bn=args.bn, bias=args.bias,
                                      quant_act=False, act_bits=args.act_bits, quant_scale_max=args.quant_scale_max,
                                      quant_id=args.quant_id)

    # model = nn.DataParallel(model)
    if args.quant_act:
        if args.pretrained_ann and not args.test_only:
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
        if args.test_only:
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
    model = nn.DataParallel(model)

    if torch.cuda.is_available() and args.gpu:
        print('Running on GPU')
        model.cuda()

    if optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True, weight_decay=weight_decay)

    print('{}'.format(optimizer))

    max_accuracy = 0
    for epoch in range(1, epochs):
        start_time = datetime.datetime.now()
        if args.quant_act:
            for n, m in model.named_modules():
                if isinstance(m, QuantizeActivation):
                    m.mode = 0  # Quantize
        if not args.test_only:
            train(epoch, train_loader)
        test(test_loader)
        if args.print_ops:
            print("\n")
            if dataset == 'IMAGENET':
                samps = 50000
            else:
                samps = len(test_dataset.data)
            compute_ops(model, dataset, samps, architecture[0:3].lower(),
                        './ops/ann/' + identifier + '.csv')
            print("\n")

    print('Highest accuracy: {:.4f}'.format(max_accuracy))
