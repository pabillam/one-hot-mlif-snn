import argparse
import shutil
import os
import time
import torch
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from models.VGG_models import *
import data_loaders
from functions import TET_loss, seed_all

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', help='number of data loading workers (default: 10)')
parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=1000, type=int, help='seed for initializing training. ')
parser.add_argument('--multi_channel', default=1, type=int, help="multi channel SNN", choices=[0, 1])
parser.add_argument('--num_channels', default=2, type=int, help='number of in out channels for all layers')
parser.add_argument('--T_prime', default=5, type=int, help='number of timesteps to reduce the data first, T_F')

filename = 'VGGSNN_woAP_T3_C2'
args = parser.parse_args()

def compute_ops(model, dataset, architecture):
    ann_ops = []
    snn_ops = []

    if dataset.lower().startswith('dvs-cifar10') and architecture.lower().startswith('vggsnn_woap'):
        num_samples = 1000
        h_in = 48
        w_in = 48
        c_ins = [2, 64, 128, 256, 256, 512, 512, 512]
        c_outs = [64, 128, 256, 256, 512, 512, 512, 512]
        ks = [3, 3, 3, 3, 3, 3, 3, 3]
        paddings = [1, 1, 1, 1, 1, 1, 1, 1]
        strides = [1, 2, 1, 2, 1, 2, 1, 2]
        for i in range(len(c_ins)):
            c_in = c_ins[i]
            k = ks[i]
            c_out = c_outs[i]
            h_out = int((h_in - k + 2 * paddings[i]) / (strides[i])) + 1
            w_out = int((w_in - k + 2 * paddings[i]) / (strides[i])) + 1
            mac = k * k * c_in * h_out * w_out * c_out
            num_neurons = h_in * w_in * c_in
            spike_rate = model.module.features[i].ops.item() / num_samples / num_neurons
            snn_op = spike_rate * mac
            snn_ops.append(snn_op)
            ann_ops.append(mac)
            h_in = h_out
            w_in = w_out

        mac = (512*9) * 10
        num_neurons = (512*9)
        spike_rate = model.module.classifier_ops.item() / num_samples / num_neurons
        snn_op = spike_rate * mac
        snn_ops.append(snn_op)
        ann_ops.append(mac)

        print('Total MACs: {:e}'.format(sum(ann_ops)))
        print('Total SNN OPs: {:e}'.format(sum(snn_ops)))

@torch.no_grad()
def test(model, test_loader, device):
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        # inputs = inputs.unsqueeze_(1).repeat(1, args.time, 1, 1, 1)
        outputs = model(inputs)
        mean_out = outputs.mean(1)
        _, predicted = mean_out.cpu().max(1)
        ## _, predicted = outputs.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
        if batch_idx % 100 == 0:
            acc = 100. * float(correct) / float(total)
            print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)
    final_acc = 100 * correct / total
    return final_acc


if __name__ == '__main__':
    seed_all(args.seed)

    train_dataset, val_dataset = data_loaders.build_dvscifar('./data/dvs-cifar10/', channels=args.num_channels, multi_channel=args.multi_channel, T_prime=args.T_prime)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)

    model = VGGSNNwoAP(args.num_channels, args.multi_channel, classes=10)

    state_dict = torch.load('trained_models/CIFAR10-DVS/VGGSNN_woAP/{}.pth'.format(filename), map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)

    parallel_model = torch.nn.DataParallel(model)
    parallel_model.to(device)

    facc = test(parallel_model, test_loader, device)
    compute_ops(parallel_model, 'dvs-cifar10', 'vggsnn_woap')
    print('Test Accuracy of the model: %.3f' % facc)
