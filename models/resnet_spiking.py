import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
import random
import matplotlib.pyplot as plt
from collections import OrderedDict
import copy

cfg = {
    'resnet20': [2, 2, 2, 2]
}

out_channel_cfg = {
    'resnet20': {0: 2, 3: 2, 6: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2,
                 17: 2, 18: 2, 19: 2, 20: 2, 21: 2, 22: 2, 23: 2, 24: 2, 25: 2}
}

in_channels = 1
in_channel_cfg = {
    'resnet20': {x: out_channel_cfg['resnet20'][y] for (x, y) in zip([3, 6] + list(range(9, 26)), [0, 3, 6] + list(range(9, 25)))}
}

for d in in_channel_cfg:
    in_channel_cfg[d].update({0: in_channels})

class LinearSpike(torch.autograd.Function):
    gamma = 0.3  # Controls the dampening of the piecewise-linear surrogate gradient

    @staticmethod
    def forward(ctx, input, last_spike):
        ctx.save_for_backward(input)
        if torch.cuda.is_available():
            out = torch.zeros_like(input).cuda()
        else:
            out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = LinearSpike.gamma * F.threshold(1.0 - torch.abs(input), 0, 0)
        return grad * grad_input, None

class LinearSpikeMC(torch.autograd.Function):
    gamma = 0.3  # Controls the dampening of the piecewise-linear surrogate gradient

    @staticmethod
    def forward(ctx, mem_thr, hardware_opt, threshold):
        variables = torch.tensor([hardware_opt, threshold])
        ctx.save_for_backward(mem_thr, variables)
        if hardware_opt != 2:
            if torch.cuda.is_available():
                out = torch.zeros_like(mem_thr[::2]).cuda()
            else:
                out = torch.zeros_like(mem_thr[::2])
            cond1 = mem_thr[::2]
            cond2 = mem_thr[1::2]
            out[torch.logical_and(cond1 > 0, cond2 >= 0)] = 1.0
        else:
            if torch.cuda.is_available():
                out = torch.zeros_like(mem_thr).cuda()
            else:
                out = torch.zeros_like(mem_thr)
            out[mem_thr > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        mem_thr, variables = ctx.saved_tensors
        hardware_opt = variables[0].item()
        threshold = variables[1].item()
        grad_input = grad_output.clone()
        if hardware_opt != 2:
            grad = torch.zeros_like(mem_thr)
            for x in range(mem_thr.shape[0] // 2 - 1):
                grad[2 * x] = LinearSpikeMC.gamma * F.threshold(1.0 - torch.abs(mem_thr[2 * x]), 0, 0) * grad_input[x]
                grad[2 * x + 1] = - LinearSpikeMC.gamma * F.threshold(1.0 - torch.abs(mem_thr[2 * x + 1]), 0, 0) * grad_input[x]
            grad[2 * (x + 1)] = LinearSpikeMC.gamma * F.threshold(1.0 - torch.abs(mem_thr[2 * (x + 1)]), 0, 0) * grad_input[x + 1]
        else:
            grad = LinearSpikeMC.gamma * F.threshold(1.0 - torch.abs(mem_thr), 0, 0) * grad_input

        return grad, None, None

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, dropout):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.identity = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.identity = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, dic):
        out_prev = dic['out_prev']
        pos = dic['pos']
        act_func = dic['act_func']
        mem = dic['mem']
        spike = dic['spike']
        mask = dic['mask']
        threshold = dic['threshold']
        t = dic['t']
        leak = dic['leak']
        ops = dic['ops']
        inp = out_prev.clone()

        # conv1
        ops[pos] += torch.sum(inp).item()
        delta_mem = self.residual[0](inp)
        mem[pos] = getattr(leak, 'l' + str(pos)) * mem[pos] + delta_mem
        mem_thr = (mem[pos] / getattr(threshold, 't' + str(pos))) - 1.0
        rst = getattr(threshold, 't' + str(pos)) * (mem_thr > 0).float()
        mem[pos] = mem[pos] - rst

        # relu1
        out = act_func(mem_thr, (t - 1 - spike[pos]))
        spike[pos] = spike[pos].masked_fill(out.bool(), t - 1)
        out_prev = out.clone()

        # dropout1
        out_prev = out_prev * mask[pos]

        # conv2+identity
        ops[pos + 1] += torch.sum(out_prev).item()
        delta_mem = self.residual[3](out_prev) + self.identity(inp)
        mem[pos + 1] = getattr(leak, 'l' + str(pos + 1)) * mem[pos + 1] + delta_mem
        mem_thr = (mem[pos + 1] / getattr(threshold, 't' + str(pos + 1))) - 1.0
        rst = getattr(threshold, 't' + str(pos + 1)) * (mem_thr > 0).float()
        mem[pos + 1] = mem[pos + 1] - rst

        # relu2
        out = act_func(mem_thr, (t - 1 - spike[pos + 1]))
        spike[pos + 1] = spike[pos + 1].masked_fill(out.bool(), t - 1)
        out_prev = out.clone()

        return out_prev

class RESNET_SNN(nn.Module):

    def __init__(self, resnet_name, activation='Linear', labels=10, timesteps=75, leak=1.0, default_threshold=1.0, dropout=0.2, dataset='CIFAR10'):
        super().__init__()
        self.resnet_name = resnet_name.lower()
        if activation == 'Linear':
            self.act_func = LinearSpike.apply
        self.labels = labels
        self.timesteps = timesteps
        self.dropout = dropout
        self.dataset = dataset
        self.mem = {}
        self.mask = {}
        self.spike = {}
        self.ops = {}

        self.pre_process = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        block = BasicBlock
        self.in_planes = 64

        self.layer1 = self._make_layer(block, 64, cfg[self.resnet_name][0], stride=1, dropout=self.dropout)
        self.layer2 = self._make_layer(block, 128, cfg[self.resnet_name][1], stride=2, dropout=self.dropout)
        self.layer3 = self._make_layer(block, 256, cfg[self.resnet_name][2], stride=2, dropout=self.dropout)
        self.layer4 = self._make_layer(block, 512, cfg[self.resnet_name][3], stride=2, dropout=self.dropout)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, labels, bias=False)
        )

        self.layers = {1: self.layer1, 2: self.layer2, 3: self.layer3, 4: self.layer4}

        self._initialize_weights2()

        threshold = {}
        lk = {}
        for l in range(len(self.pre_process)):
            if isinstance(self.pre_process[l], nn.Conv2d):
                # self.register_buffer('threshold[l]', torch.tensor(default_threshold, requires_grad=True))
                threshold['t' + str(l)] = nn.Parameter(torch.tensor(default_threshold))
                lk['l' + str(l)] = nn.Parameter(torch.tensor(leak))

        pos = len(self.pre_process)

        for i in range(1, 5):

            layer = self.layers[i]
            for index in range(len(layer)):
                for l in range(len(layer[index].residual)):
                    if isinstance(layer[index].residual[l], nn.Conv2d):
                        threshold['t' + str(pos)] = nn.Parameter(torch.tensor(default_threshold))
                        lk['l' + str(pos)] = nn.Parameter(torch.tensor(leak))
                        pos = pos + 1

        for l in range(len(self.classifier) - 1):
            if isinstance(self.classifier[l], nn.Linear):
                threshold['t' + str(pos + l)] = nn.Parameter(torch.tensor(default_threshold))
                lk['l' + str(pos + l)] = nn.Parameter(torch.tensor(leak))

        self.threshold = nn.ParameterDict(threshold)
        self.leak = nn.ParameterDict(lk)
        self.scaling_factor = 1.0
        self.batch_size = 0
        self.width = 0
        self.height = 0

    def _initialize_weights2(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def threshold_update(self, scaling_factor=1.0, thresholds=[]):
        self.scaling_factor = scaling_factor

        for pos in range(len(self.pre_process)):
            if isinstance(self.pre_process[pos], nn.Conv2d):
                if thresholds:
                    self.threshold.update(
                        {'t' + str(pos): nn.Parameter(torch.tensor(thresholds.pop(0) * self.scaling_factor))})

        pos = len(self.pre_process)
        for i in range(1, 5):
            layer = self.layers[i]
            for index in range(len(layer)):
                for l in range(len(layer[index].residual)):
                    if isinstance(layer[index].residual[l], nn.Conv2d):
                        # self.threshold[pos].data = torch.tensor(thresholds.pop(0)*self.scaling_factor)
                        pos = pos + 1

        for l in range(len(self.classifier)):
            if isinstance(self.classifier[l], nn.Linear):
                if thresholds:
                    self.threshold.update(
                        {'t' + str(pos + l): nn.Parameter(torch.tensor(thresholds.pop(0) * self.scaling_factor))})

    def _make_layer(self, block, planes, num_blocks, stride, dropout):

        if num_blocks == 0:
            return nn.Sequential()
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def network_update(self, timesteps, leak):
        self.timesteps = timesteps

    def neuron_init(self, x):
        self.batch_size = x.size(0)
        self.width = x.size(2)
        self.height = x.size(3)
        self.mem = {}
        self.spike = {}
        self.mask = {}
        self.ops = {}

        # Pre processing layers
        for l in range(len(self.pre_process)):
            if isinstance(self.pre_process[l], nn.Conv2d):
                self.mem[l] = torch.zeros(self.batch_size, self.pre_process[l].out_channels, self.width, self.height)
                self.spike[l] = torch.ones(self.mem[l].shape) * (-1000)
                self.ops[l] = 0.0
            elif isinstance(self.pre_process[l], nn.Dropout):
                self.mask[l] = self.pre_process[l](torch.ones(self.mem[l - 2].shape))
            elif isinstance(self.pre_process[l], nn.AvgPool2d):
                self.width = self.width // self.pre_process[l].kernel_size
                self.height = self.height // self.pre_process[l].kernel_size

        pos = len(self.pre_process)
        for i in range(1, 5):
            layer = self.layers[i]
            self.width = self.width // layer[0].residual[0].stride[0]
            self.height = self.height // layer[0].residual[0].stride[0]
            for index in range(len(layer)):
                for l in range(len(layer[index].residual)):
                    if isinstance(layer[index].residual[l], nn.Conv2d):
                        self.mem[pos] = torch.zeros(self.batch_size, layer[index].residual[l].out_channels, self.width, self.height)
                        self.spike[pos] = torch.ones(self.mem[pos].shape) * (-1000)
                        self.ops[pos] = 0.0
                        pos = pos + 1
                    elif isinstance(layer[index].residual[l], nn.Dropout):
                        self.mask[pos - 1] = layer[index].residual[l](torch.ones(self.mem[pos - 1].shape))

        for l in range(len(self.classifier)):
            if isinstance(self.classifier[l], nn.Linear):
                self.mem[pos + l] = torch.zeros(self.batch_size, self.timesteps, self.classifier[l].out_features)
                self.spike[pos + l] = torch.ones(self.mem[pos + l].shape) * (-1000)
                self.ops[pos + l] = 0.0
            elif isinstance(self.classifier[l], nn.Dropout):
                self.mask[pos + l] = self.classifier[l](torch.ones(self.mem[pos + l - 2].shape))

    def percentile(self, t, q):
        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        result = t.view(-1).kthvalue(k).values.item()
        return result

    def forward(self, x, find_max_mem=False, max_mem_layer=0):
        self.neuron_init(x)
        max_mem = 0.0
        for t in range(self.timesteps):
            out_prev = x
            for l in range(len(self.pre_process)):
                if isinstance(self.pre_process[l], nn.Conv2d):
                    if find_max_mem and l == max_mem_layer:
                        cur = self.percentile(self.pre_process[l](out_prev).view(-1), 99.7)
                        if cur > max_mem:
                            max_mem = torch.tensor([cur])
                        break
                    self.ops[l] += torch.sum(out_prev).item()
                    delta_mem = self.pre_process[l](out_prev)
                    self.mem[l] = getattr(self.leak, 'l' + str(l)) * self.mem[l] + delta_mem
                    mem_thr = (self.mem[l] / getattr(self.threshold, 't' + str(l))) - 1.0
                    rst = getattr(self.threshold, 't' + str(l)) * (mem_thr > 0).float()
                    self.mem[l] = self.mem[l] - rst
                elif isinstance(self.pre_process[l], nn.ReLU):
                    out = self.act_func(mem_thr, (t - 1 - self.spike[l - 1]))
                    self.spike[l - 1] = self.spike[l - 1].masked_fill(out.bool(), t - 1)
                    out_prev = out.clone()
                elif isinstance(self.pre_process[l], nn.AvgPool2d):
                    out_prev = self.pre_process[l](out_prev)
                elif isinstance(self.pre_process[l], nn.Dropout):
                    out_prev = out_prev * self.mask[l]

            if find_max_mem and max_mem_layer < len(self.pre_process):
                continue

            pos = len(self.pre_process)

            for i in range(1, 5):
                layer = self.layers[i]
                for index in range(len(layer)):
                    out_prev = layer[index](
                        {'out_prev': out_prev.clone(), 'pos': pos, 'act_func': self.act_func, 'mem': self.mem,
                         'spike': self.spike, 'mask': self.mask, 'threshold': self.threshold, 't': t,
                         'leak': self.leak, 'ops': self.ops})
                    pos = pos + 2

            out_prev = out_prev.reshape(self.batch_size, -1)

            for l in range(len(self.classifier) - 1):
                if isinstance(self.classifier[l], nn.Linear):
                    if find_max_mem and (pos + l) == max_mem_layer:
                        if (self.classifier[l](out_prev)).max() > max_mem:
                            max_mem = (self.classifier[l](out_prev)).max()
                        break
                    mem_thr = (self.mem[pos + l] / getattr(self.threshold, 't' + str(pos + l))) - 1.0
                    out = self.act_func(mem_thr, (t - 1 - self.spike[pos + l]))
                    self.ops[pos + l] += torch.sum(out_prev).item()
                    rst = getattr(self.threshold, 't' + str(pos + l)) * (mem_thr > 0).float()
                    self.spike[pos + l] = self.spike[pos + l].masked_fill(out.bool(), t - 1)
                    self.mem[pos + l] = getattr(self.leak, 'l' + str(pos + l)) * self.mem[pos + l] + self.classifier[l](
                        out_prev) - rst
                    out_prev = out.clone()

                elif isinstance(self.classifier[l], nn.Dropout):
                    out_prev = out_prev * self.mask[pos + l]

            # Compute the final layer outputs
            if not find_max_mem:
                if len(self.classifier) > 1:
                    self.ops[pos + l + 1] += torch.sum(out_prev).item()
                    self.mem[pos + l + 1] = self.mem[pos + l + 1] + self.classifier[l + 1](out_prev)
                    self.mem[pos + l + 1][:, t, ...] = self.mem[pos + l + 1][:, t, ...] + self.classifier[l + 1](out_prev)
                else:
                    self.ops[pos] += torch.sum(out_prev).item()
                    self.mem[pos] = self.mem[pos] + self.classifier[0](out_prev)
                    self.mem[pos][:, t, ...] = self.mem[pos][:, t, ...] + self.classifier[0](out_prev)

        if find_max_mem:
            return max_mem

        if len(self.classifier) > 1:
            return self.mem[pos + l + 1]
        else:
            return self.mem[pos]

class BasicBlock_MC(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, dropout, bias=False):
        super().__init__()
        self.bias = bias

        if self.bias:
            self.residual = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
            )
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            )

        self.identity = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if self.bias:
                self.identity = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=True)
                )
            else:
                self.identity = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
                )

    def percentile(self, t, q):
        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        result = t.view(-1).kthvalue(k).values.item()
        return result

    def forward(self, dic):
        out_prev = dic['out_prev']
        pos = dic['pos']
        act_func = dic['act_func']
        mem = dic['mem']
        mask = dic['mask']
        threshold = dic['threshold']
        t = dic['t']
        leak = dic['leak']
        ops = dic['ops']
        resnet_name = dic['resnet_name']
        hardware_opt = dic['hardware_opt']
        reset_to_zero = dic['reset_to_zero']
        mem_id = dic['mem_id']
        find_max_mem = dic['find_max_mem']
        max_mem_layer = dic['max_mem_layer']
        inp = out_prev.clone()
        max_mem = 0.0

        # conv1
        merged_inp = torch.zeros_like(inp[0])
        for in_ch in range(in_channel_cfg[resnet_name][pos]):
            merged_inp += (2 ** in_ch) * inp[in_ch]
        
        if find_max_mem and pos == max_mem_layer:
            cur = self.percentile(self.residual[0](merged_inp).view(-1), 99.7) / 2 ** (out_channel_cfg[resnet_name][pos] - 1)
            if cur > max_mem:
                if torch.cuda.is_available():
                    max_mem = torch.tensor([cur]).cuda()
                else:
                    max_mem = torch.tensor([cur])
            return out_prev, max_mem
        
        ops[pos] += torch.sum(inp).item()
        delta_mem = self.residual[0](merged_inp)
        mem[pos] = getattr(leak, 'le' + str(pos)) * mem[pos] + delta_mem
        thresh = getattr(threshold, 'th' + str(pos))
        mem_thr = []
        rst = torch.zeros_like(mem[pos])
        if out_channel_cfg[resnet_name][pos] == 1:
            if hardware_opt != 2:
                mem_thr.append(mem[pos] / thresh - 1.0)
            mem_thr.append(mem[pos] / thresh - 1.0)
            rst += (mem_thr[0] > 0).float() * thresh
        else:
            for out_ch in range(out_channel_cfg[resnet_name][pos] - 1):
                if hardware_opt != 2:
                    mem_thr.append(mem[pos] / ((2 ** out_ch) * thresh) - 1.0)
                    mem_thr.append(1.0 - mem[pos] / (thresh * (2 ** (out_ch + 1))))
                    rst += (2 ** out_ch) * (torch.logical_and(mem_thr[2 * out_ch] > 0, mem_thr[2 * out_ch + 1] >= 0)).float() * thresh
                else:
                    mem_thr.append(mem[pos] / ((2 ** out_ch) * thresh) - 1.0)
                    rst += (2 ** out_ch) * (mem_thr[out_ch] > 0).float() * thresh
            mem_thr.append(mem[pos] / (thresh * (2 ** (out_ch + 1))) - 1.0)
            if hardware_opt != 2:
                mem_thr.append(mem[pos] / ((2 ** (out_ch + 1)) * thresh) - 1.0)
                rst += (2 ** (out_ch + 1)) * (mem_thr[2 * (out_ch + 1)] > 0).float() * thresh
            else:
                rst += (2 ** (out_ch + 1)) * (mem_thr[out_ch + 1] > 0).float() * thresh
        if reset_to_zero == 0:
            mem[pos] = mem[pos] - rst
        else:
            mem[pos] = 0

        # relu1
        out = act_func(torch.stack(mem_thr), hardware_opt, thresh)
        out_prev = out.clone()

        # dropout1
        out_prev = out_prev * mask[pos]

        # conv2+identity
        merged_out = torch.zeros_like(out_prev[0])
        for in_ch in range(in_channel_cfg[resnet_name][pos + 1]):
            merged_out += (2 ** in_ch) * out_prev[in_ch]
        ops[pos + 1] += torch.sum(out_prev).item()
        delta_mem = self.residual[3](merged_out) + self.identity(merged_inp)

        if find_max_mem and pos + 1 == max_mem_layer:
            cur = self.percentile(delta_mem.view(-1), 99.7) / 2 ** (out_channel_cfg[resnet_name][pos + 1] - 1)
            if cur > max_mem:
                if torch.cuda.is_available():
                    max_mem = torch.tensor([cur]).cuda()
                else:
                    max_mem = torch.tensor([cur])
            return out_prev, max_mem

        mem[pos + 1] = getattr(leak, 'le' + str(pos + 1)) * mem[pos + 1] + delta_mem
        thresh = getattr(threshold, 'th' + str(pos + 1))
        mem_thr = []
        rst = torch.zeros_like(mem[pos + 1])
        if out_channel_cfg[resnet_name][pos + 1] == 1:
            if hardware_opt != 2:
                mem_thr.append(mem[pos + 1] / thresh - 1.0)
            mem_thr.append(mem[pos + 1] / thresh - 1.0)
            rst += (mem_thr[0] > 0).float() * thresh
        else:
            for out_ch in range(out_channel_cfg[resnet_name][pos + 1] - 1):
                if hardware_opt != 2:
                    mem_thr.append(mem[pos + 1] / ((2 ** out_ch) * thresh) - 1.0)
                    mem_thr.append(1.0 - mem[pos + 1] / ((2 ** (out_ch + 1)) * thresh))
                    rst += (2 ** out_ch) * (torch.logical_and(mem_thr[2 * out_ch] > 0, mem_thr[2 * out_ch + 1] >= 0)).float() * thresh
                else:
                    mem_thr.append(mem[pos + 1] / (thresh * (2 ** out_ch)) - 1.0)
                    rst += (2 ** out_ch) * (mem_thr[out_ch] > 0).float() * thresh
            mem_thr.append(mem[pos + 1] / (thresh * (2 ** (out_ch + 1))) - 1.0)
            if hardware_opt != 2:
                mem_thr.append(mem[pos + 1] / (thresh * (2 ** (out_ch + 1))) - 1.0)
                rst += (2 ** (out_ch + 1)) * (mem_thr[2 * (out_ch + 1)] > 0).float() * thresh
            else:
                rst += (2 ** (out_ch + 1)) * (mem_thr[out_ch + 1] > 0).float() * thresh
        if reset_to_zero == 0:
            mem[pos + 1] = mem[pos + 1] - rst
        else:
            mem[pos + 1] = 0

        # relu2
        out = act_func(torch.stack(mem_thr), hardware_opt, thresh)
        out_prev = out.clone()

        return out_prev, max_mem

class RESNET_SNN_MC(nn.Module):

    def __init__(self, resnet_name, labels=10, timesteps=75, leak=1.0, default_threshold=1.0, dropout=0.2, hardware_opt=0, reset_to_zero=0, input_encoding=1, num_channels=1, dataset='CIFAR10', bias=False):
        super().__init__()

        self.bias = bias
        self.resnet_name = resnet_name.lower()
        self.act_func = LinearSpikeMC.apply
        self.hardware_opt = hardware_opt
        self.reset_to_zero = reset_to_zero
        self.input_encoding = input_encoding
        self.num_channels = num_channels
        self.labels = labels
        self.timesteps = timesteps
        self.dropout = dropout
        self.dataset = dataset
        self.mem = {}
        self.mem_id = {}
        self.mask = {}
        self.ops = {}

        if dataset == 'CIFAR10' or dataset == 'CIFAR100':
            if self.bias:
                self.pre_process = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Dropout(self.dropout),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Dropout(self.dropout),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.AvgPool2d(2)
                )
            else:
                self.pre_process = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Dropout(self.dropout),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Dropout(self.dropout),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.AvgPool2d(2)
                )

        block = BasicBlock_MC
        self.in_planes = 64

        self.layer1 = self._make_layer(block, 64, cfg[self.resnet_name][0], stride=1, dropout=self.dropout)
        self.layer2 = self._make_layer(block, 128, cfg[self.resnet_name][1], stride=2, dropout=self.dropout)
        self.layer3 = self._make_layer(block, 256, cfg[self.resnet_name][2], stride=2, dropout=self.dropout)
        self.layer4 = self._make_layer(block, 512, cfg[self.resnet_name][3], stride=2, dropout=self.dropout)

        if dataset == 'CIFAR10' or dataset == 'CIFAR100':
            self.classifier = nn.Sequential(nn.Linear(512 * 2 * 2, labels, bias=False))

        self.layers = {1: self.layer1, 2: self.layer2, 3: self.layer3, 4: self.layer4}

        self._initialize_weights2()

        threshold = {}
        lk = {}
        for l in range(len(self.pre_process)):
            if isinstance(self.pre_process[l], nn.Conv2d):
                threshold['th' + str(l)] = nn.Parameter(torch.tensor(default_threshold))
                lk['le' + str(l)] = nn.Parameter(torch.tensor(leak))

        pos = len(self.pre_process)

        for i in range(1, 5):

            layer = self.layers[i]
            for index in range(len(layer)):
                for l in range(len(layer[index].residual)):
                    if isinstance(layer[index].residual[l], nn.Conv2d):
                        threshold['th' + str(pos)] = nn.Parameter(torch.tensor(default_threshold))
                        lk['le' + str(pos)] = nn.Parameter(torch.tensor(leak))
                        pos = pos + 1

        for l in range(len(self.classifier) - 1):
            if isinstance(self.classifier[l], nn.Linear):
                threshold['th' + str(pos + l)] = nn.Parameter(torch.tensor(default_threshold))
                lk['le' + str(pos + l)] = nn.Parameter(torch.tensor(leak))

        self.threshold = nn.ParameterDict(threshold)
        self.leak = nn.ParameterDict(lk)
        self.scaling_factor = 1.0
        self.batch_size = 0
        self.width = 0
        self.height = 0
        for layer in out_channel_cfg[self.resnet_name]:
            out_channel_cfg[self.resnet_name][layer] = self.num_channels
        for layer in in_channel_cfg[self.resnet_name]:
            if layer != 0:
                in_channel_cfg[self.resnet_name][layer] = self.num_channels

    def _initialize_weights2(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def threshold_update(self, scaling_factor=1.0, thresholds=[]):
        self.scaling_factor = scaling_factor

        for pos in range(len(self.pre_process)):
            if isinstance(self.pre_process[pos], nn.Conv2d):
                if thresholds:
                    if torch.cuda.is_available():
                        self.threshold.update(
                            {'th' + str(pos): nn.Parameter(torch.tensor(thresholds.pop(0) * self.scaling_factor).cuda())})
                    else:
                        self.threshold.update(
                            {'th' + str(pos): nn.Parameter(
                                torch.tensor(thresholds.pop(0) * self.scaling_factor))})

        pos = len(self.pre_process)
        for i in range(1, 5):
            layer = self.layers[i]
            for index in range(len(layer)):
                first_pos = pos
                for l in range(len(layer[index].residual)):
                    if isinstance(layer[index].residual[l], nn.Conv2d):
                        if thresholds:
                            if torch.cuda.is_available():
                                self.threshold.update(
                                    {'th' + str(pos): nn.Parameter(
                                        torch.tensor(thresholds.pop(0) * self.scaling_factor).cuda())})
                            else:
                                self.threshold.update(
                                    {'th' + str(pos): nn.Parameter(
                                        torch.tensor(thresholds.pop(0) * self.scaling_factor))})
                        pos = pos + 1

        for l in range(len(self.classifier)):
            if isinstance(self.classifier[l], nn.Linear):
                if thresholds:
                    if torch.cuda.is_available():
                        self.threshold.update(
                            {'th' + str(pos + l): nn.Parameter(torch.tensor(thresholds.pop(0) * self.scaling_factor).cuda())})
                    else:
                        self.threshold.update(
                            {'th' + str(pos + l): nn.Parameter(
                                torch.tensor(thresholds.pop(0) * self.scaling_factor))})

    def _make_layer(self, block, planes, num_blocks, stride, dropout):
        if num_blocks == 0:
            return nn.Sequential()
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout, self.bias))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def network_update(self, timesteps, leak):
        self.timesteps = timesteps

    def convert_single_to_multi(self, raw_in, num_channels):
        raw_out = [torch.div(raw_in, 2 ** (num_channels - 1), rounding_mode='floor')]
        for j in range(num_channels, 1, -1):
            raw_out.append(torch.div(torch.remainder(raw_in, 2 ** (j - 1)), 2 ** (j - 2), rounding_mode='floor'))

        raw_out.reverse()
        return torch.stack(raw_out)

    def neuron_init(self, x):
        self.batch_size = x.size(0)
        self.width = x.size(2)
        self.height = x.size(3)
        self.mem = {}
        self.mem_id = {}
        self.mask = {}
        self.ops = {}

        # Pre processing layers
        for l in range(len(self.pre_process)):
            if isinstance(self.pre_process[l], nn.Conv2d):
                self.width = ((self.width + (2 * self.pre_process[l].padding[0]) - self.pre_process[l].kernel_size[0]) // self.pre_process[l].stride[0]) + 1
                self.height = ((self.height + (2 * self.pre_process[l].padding[0]) - self.pre_process[l].kernel_size[0]) // self.pre_process[l].stride[0]) + 1
                if torch.cuda.is_available():
                    self.mem[l] = torch.zeros(self.batch_size, self.pre_process[l].out_channels, self.width, self.height, device=torch.device('cuda'))
                else:
                    self.mem[l] = torch.zeros(self.batch_size, self.pre_process[l].out_channels, self.width, self.height)
                self.ops[l] = 0.0
            elif isinstance(self.pre_process[l], nn.Dropout):
                if torch.cuda.is_available():
                    self.mask[l] = torch.stack([self.pre_process[l](torch.ones(self.mem[l - 2].shape, device=torch.device('cuda'))) for _ in range(out_channel_cfg[self.resnet_name][l - 2])])
                else:
                    self.mask[l] = torch.stack([self.pre_process[l](torch.ones(self.mem[l - 2].shape)) for _ in range(out_channel_cfg[self.resnet_name][l - 2])])
            elif isinstance(self.pre_process[l], nn.AvgPool2d):
                self.width = ((self.width + (2 * self.pre_process[l].padding) - self.pre_process[l].kernel_size) // self.pre_process[l].stride) + 1
                self.height = ((self.height + (2 * self.pre_process[l].padding) - self.pre_process[l].kernel_size) // self.pre_process[l].stride) + 1

        pos = len(self.pre_process)
        for i in range(1, 5):
            layer = self.layers[i]
            self.width = ((self.width + (2 * layer[0].residual[0].padding[0]) - layer[0].residual[0].kernel_size[0]) // layer[0].residual[0].stride[0]) + 1
            self.height = ((self.height + (2 * layer[0].residual[0].padding[0]) - layer[0].residual[0].kernel_size[0]) // layer[0].residual[0].stride[0]) + 1
            for index in range(len(layer)):
                for l in range(len(layer[index].residual)):
                    if isinstance(layer[index].residual[l], nn.Conv2d):
                        if torch.cuda.is_available():
                            self.mem[pos] = torch.zeros(self.batch_size, layer[index].residual[l].out_channels, self.width, self.height, device=torch.device('cuda'))
                        else:
                            self.mem[pos] = torch.zeros(self.batch_size, layer[index].residual[l].out_channels, self.width, self.height)
                        self.ops[pos] = 0.0
                        pos = pos + 1
                    elif isinstance(layer[index].residual[l], nn.Dropout):
                        if torch.cuda.is_available():
                            self.mask[pos - 1] = torch.stack([layer[index].residual[l](torch.ones(self.mem[pos - 1].shape, device=torch.device('cuda'))) for _ in range(out_channel_cfg[self.resnet_name][pos - 1])])
                        else:
                            self.mask[pos - 1] = torch.stack([layer[index].residual[l](torch.ones(self.mem[pos - 1].shape)) for _ in range(out_channel_cfg[self.resnet_name][pos - 1])])

        for l in range(len(self.classifier)):
            if isinstance(self.classifier[l], nn.Linear):
                if l == len(self.classifier) - 1:
                    if torch.cuda.is_available():
                        self.mem[pos + l] = torch.zeros(self.batch_size, self.timesteps, self.classifier[l].out_features, device=torch.device('cuda'))
                    else:
                        self.mem[pos + l] = torch.zeros(self.batch_size, self.timesteps, self.classifier[l].out_features)
                else:
                    if torch.cuda.is_available():
                        self.mem[pos + l] = torch.zeros(self.batch_size, self.classifier[l].out_features, device=torch.device('cuda'))
                    else:
                        self.mem[pos + l] = torch.zeros(self.batch_size, self.classifier[l].out_features)
                self.ops[pos + l] = 0.0
            elif isinstance(self.classifier[l], nn.Dropout):
                if torch.cuda.is_available():
                    self.mask[pos + l] = torch.stack([self.classifier[l](torch.ones(self.mem[pos + l - 2].shape, device=torch.device('cuda'))) for _ in range(out_channel_cfg[self.resnet_name][pos + l - 2])])
                else:
                    self.mask[pos + l] = torch.stack([self.classifier[l](torch.ones(self.mem[pos + l - 2].shape)) for _ in range(out_channel_cfg[self.resnet_name][pos + l - 2])])

    def percentile(self, t, q):
        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        result = t.view(-1).kthvalue(k).values.item()
        return result

    def forward(self, x, find_max_mem=False, max_mem_layer=0):
        self.neuron_init(x)
        max_mem = 0.0

        if self.input_encoding == 0:
            # Convert single channel input to multi channel
            num_pre_synapse = x.shape[3] * x.shape[2] * x.shape[1] * x.shape[0]
            x_flat = torch.div(self.convert_single_to_multi(x * (2 ** 8), in_channels), 2 ** 8)
            x_flat = x_flat.reshape(-1)
            x_flat = x_flat.repeat(self.timesteps)
            product = x_flat.reshape(self.timesteps, in_channels, num_pre_synapse)

            # Rate code
            x_rate = []
            for k in range(0, in_channels):
                temp = []
                for i in range(0, num_pre_synapse):
                    r = torch.bernoulli(product[:, k, i])
                    temp.append(r)
                x_rate.append(torch.stack(temp))
            x_rate = torch.stack(x_rate)

        for t in range(self.timesteps):
            if self.input_encoding == 0:
                out_prev = x_rate[:, :, t].reshape(in_channels, x.shape[3], x.shape[2], x.shape[1], x.shape[0])
            else:
                out_prev = x[None, :]

            # Pre-process
            for l in range(len(self.pre_process)):
                if isinstance(self.pre_process[l], nn.Conv2d):
                    # Combine input spikes into one tensor
                    merged_out_prev = torch.zeros_like(out_prev[0])
                    for in_ch in range(in_channel_cfg[self.resnet_name][l]):
                        merged_out_prev += (2 ** in_ch) * out_prev[in_ch]
                    # Find max membrane potential
                    if find_max_mem and l == max_mem_layer:
                        cur = self.percentile(self.pre_process[l](merged_out_prev).view(-1), 99.7) / 2 ** (out_channel_cfg[self.resnet_name][l] - 1)
                        if cur > max_mem:
                            if torch.cuda.is_available():
                                max_mem = torch.tensor([cur]).cuda()
                            else:
                                max_mem = torch.tensor([cur])
                        break
                    self.ops[l] += torch.sum(out_prev).item()
                    # Multiply weights by input spike trains
                    delta_mem = self.pre_process[l](merged_out_prev)
                    # Update membrane potential leaked prior and weight x input
                    self.mem[l] = getattr(self.leak, 'le' + str(l)) * self.mem[l] + delta_mem
                    # Compute output spike trains via thresholding
                    thresh = getattr(self.threshold, 'th' + str(l))
                    mem_thr = []
                    rst = torch.zeros_like(self.mem[l])
                    if out_channel_cfg[self.resnet_name][l] == 1:
                        if self.hardware_opt != 2:
                            mem_thr.append(self.mem[l] / thresh - 1.0)
                        mem_thr.append(self.mem[l] / thresh - 1.0)
                        rst += (mem_thr[0] > 0).float() * thresh
                    else:
                        for out_ch in range(out_channel_cfg[self.resnet_name][l] - 1):
                            if self.hardware_opt != 2:
                                mem_thr.append(self.mem[l] / ((2 ** out_ch) * thresh) - 1.0)
                                mem_thr.append(1.0 - self.mem[l] / (thresh * (2 ** (out_ch + 1))))
                                rst += (2 ** out_ch) * (torch.logical_and(mem_thr[2 * out_ch] > 0, mem_thr[2 * out_ch + 1] >= 0)).float() * thresh
                            else:
                                mem_thr.append(self.mem[l] / ((2 ** out_ch) * thresh) - 1.0)
                                rst += (2 ** out_ch) * (mem_thr[out_ch] > 0).float() * thresh
                        mem_thr.append(self.mem[l] / ((2 ** (out_ch + 1)) * thresh) - 1.0)
                        if self.hardware_opt != 2:
                            mem_thr.append(self.mem[l] / ((2 ** (out_ch + 1)) * thresh) - 1.0)
                            rst += (2 ** (out_ch + 1)) * (mem_thr[2 * (out_ch + 1)] > 0).float() * thresh
                        else:
                            rst += (2 ** (out_ch + 1)) * (mem_thr[out_ch + 1] > 0).float() * thresh
                    # Reset membrane potential
                    if self.reset_to_zero == 0:
                        self.mem[l] = self.mem[l] - rst
                    else:
                        self.mem[l] = 0
                elif isinstance(self.pre_process[l], nn.ReLU):
                    # mem_thr is a list containing output spike train tensors for each output channel
                    out = self.act_func(torch.stack(mem_thr), self.hardware_opt, thresh)
                    out_prev = out.clone()
                elif isinstance(self.pre_process[l], nn.AvgPool2d):
                    temp = []
                    for i in range(out_prev.shape[0]):
                        temp.append(self.pre_process[l](out_prev[i]))
                    out_prev = torch.stack(temp)
                elif isinstance(self.pre_process[l], nn.Dropout):
                    out_prev = out_prev * self.mask[l]

            if find_max_mem and max_mem_layer < len(self.pre_process):
                continue

            pos = len(self.pre_process)

            for i in range(1, 5):
                layer = self.layers[i]
                for index in range(len(layer)):
                    out_prev, max_mem = layer[index](
                        {'out_prev': out_prev.clone(), 'pos': pos, 'act_func': self.act_func, 'mem': self.mem,
                         'mask': self.mask, 'threshold': self.threshold, 't': t,
                         'leak': self.leak, 'resnet_name': self.resnet_name, 'hardware_opt': self.hardware_opt,
                         'reset_to_zero': self.reset_to_zero, 'ops': self.ops, 'mem_id': self.mem_id, 'find_max_mem': find_max_mem,
                         'max_mem_layer': max_mem_layer})
                    pos = pos + 2
                    if find_max_mem and max_mem_layer < pos:
                        break
                if find_max_mem and max_mem_layer < pos:
                    break
            
            if find_max_mem and max_mem_layer < pos:
                continue

            for l in range(len(self.classifier) - 1):
                if isinstance(self.classifier[l], nn.Linear):
                    merged_out_prev = torch.zeros_like(out_prev[0])
                    for in_ch in range(in_channel_cfg[self.resnet_name][pos + l]):
                        merged_out_prev += (2 ** in_ch) * out_prev[in_ch]
                    merged_out_prev = merged_out_prev.reshape(self.batch_size, -1)

                    if find_max_mem and (pos + l) == max_mem_layer:
                        if (self.classifier[l](merged_out_prev)).max() > max_mem:
                            max_mem = (self.classifier[l](merged_out_prev)).max() / 2 ** (out_channel_cfg[self.resnet_name][pos + l] - 1)
                        break
                    thresh = getattr(self.threshold, 'th' + str(pos + l))
                    mem_thr = []
                    rst = torch.zeros_like(self.mem[pos + l])
                    if out_channel_cfg[self.resnet_name][pos + l] == 1:
                        if self.hardware_opt != 2:
                            mem_thr.append(self.mem[pos + l] / thresh - 1.0)
                        mem_thr.append(self.mem[pos + l] / thresh - 1.0)
                        rst += (mem_thr[0] > 0).float() * thresh
                    else:
                        for out_ch in range(out_channel_cfg[self.resnet_name][pos + l] - 1):
                            if self.hardware_opt != 2:
                                mem_thr.append(self.mem[pos + l] / ((2 ** out_ch) * thresh) - 1.0)
                                mem_thr.append(1.0 - self.mem[pos + l] / ((2 ** (out_ch + 1)) * thresh))
                                rst += (2 ** out_ch) * (torch.logical_and(mem_thr[2 * out_ch] > 0, mem_thr[2 * out_ch + 1] >= 0)).float() * thresh
                            else:
                                mem_thr.append(self.mem[pos + l] / ((2 ** out_ch) * thresh) - 1.0)
                                rst += (2 ** out_ch) * (mem_thr[out_ch] > 0).float() * thresh
                        mem_thr.append(self.mem[pos + l] / ((2 ** (out_ch + 1)) * thresh) - 1.0)
                        if self.hardware_opt != 2:
                            mem_thr.append(self.mem[pos + l] / ((2 ** (out_ch + 1)) * thresh) - 1.0)
                            rst += (2 ** (out_ch + 1)) * (mem_thr[2 * (out_ch + 1)] > 0).float() * thresh
                        else:
                            rst += (2 ** (out_ch + 1)) * (mem_thr[out_ch + 1] > 0).float() * thresh
                    out = self.act_func(torch.stack(mem_thr), self.hardware_opt, thresh)
                    self.ops[pos + l] += torch.sum(out_prev).item()
                    if self.reset_to_zero == 0:
                        self.mem[pos + l] = getattr(self.leak, 'le' + str(pos + l)) * self.mem[pos + l] + self.classifier[l](merged_out_prev) - rst
                    else:
                        self.mem[pos + l] = 0

                    out_prev = out.clone()
                elif isinstance(self.classifier[l], nn.Dropout):
                    out_prev = out_prev * self.mask[pos + l]

            # Compute the final layer outputs
            if not find_max_mem:
                merged_out_prev = torch.zeros_like(out_prev[0])
                if len(self.classifier) > 1:
                    for in_ch in range(in_channel_cfg[self.resnet_name][pos + l + 1]):
                        merged_out_prev += (2 ** in_ch) * out_prev[in_ch]
                    self.ops[pos + l + 1] += torch.sum(out_prev).item()
                    self.mem[pos + l + 1][:, t, ...] = self.classifier[l + 1](merged_out_prev)
                else:
                    for in_ch in range(in_channel_cfg[self.resnet_name][pos]):
                        merged_out_prev += (2 ** in_ch) * out_prev[in_ch]
                    merged_out_prev = merged_out_prev.reshape(self.batch_size, -1)
                    self.ops[pos] += torch.sum(out_prev).item()
                    self.mem[pos][:, t, ...] = self.classifier[0](merged_out_prev)

        if find_max_mem:
            return max_mem

        if len(self.classifier) > 1:
            return self.mem[pos + l + 1]
        else:
            return self.mem[pos]
