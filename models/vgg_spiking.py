from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
import random
from collections import OrderedDict
import copy

cfg = {
    'VGG16': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A', 512, 512, 512, 'A', 512, 512, 512]
}

out_channel_cfg = {
    'VGG16': {0: 2, 3: 2, 6: 2, 9: 2, 12: 2, 15: 2, 18: 2, 21: 2, 24: 2, 27: 2, 30: 2, 33: 2, 36: 2, 39: 2, 42: 2,
              45: 2}
}

in_channels = 1
in_channel_cfg = {
    'VGG16': {x: out_channel_cfg['VGG16'][y] for (x, y) in zip(list(range(3, 48, 3)), list(range(0, 45, 3)))}
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
                grad[2 * x + 1] = - LinearSpikeMC.gamma * F.threshold(1.0 - torch.abs(mem_thr[2 * x + 1]), 0, 0) * \
                                  grad_input[x]
            grad[2 * (x + 1)] = LinearSpikeMC.gamma * F.threshold(1.0 - torch.abs(mem_thr[2 * (x + 1)]), 0, 0) * \
                                grad_input[x + 1]
        else:
            grad = LinearSpikeMC.gamma * F.threshold(1.0 - torch.abs(mem_thr), 0, 0) * grad_input

        return grad, None, None


class VGG_SNN(nn.Module):

    def __init__(self, vgg_name, activation='Linear', labels=10, timesteps=100, leak=1.0, default_threshold=1.0,
                 dropout=0.2, kernel_size=3, dataset='CIFAR10'):
        super().__init__()

        self.vgg_name = vgg_name
        if activation == 'Linear':
            self.act_func = LinearSpike.apply
        self.labels = labels
        self.timesteps = timesteps
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.dataset = dataset
        self.mem = {}
        self.mask = {}
        self.spike = {}
        self.ops = {}

        self.features, self.classifier = self._make_layers(cfg[self.vgg_name])

        self._initialize_weights2()

        threshold = {}
        lk = {}
        for l in range(len(self.features)):
            if isinstance(self.features[l], nn.Conv2d):
                threshold['t' + str(l)] = nn.Parameter(torch.tensor(default_threshold))
                lk['l' + str(l)] = nn.Parameter(torch.tensor(leak))

        prev = len(self.features)
        for l in range(len(self.classifier) - 1):
            if isinstance(self.classifier[l], nn.Linear):
                threshold['t' + str(prev + l)] = nn.Parameter(torch.tensor(default_threshold))
                lk['l' + str(prev + l)] = nn.Parameter(torch.tensor(leak))

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

        # Initialize thresholds
        self.scaling_factor = scaling_factor

        for pos in range(len(self.features)):
            if isinstance(self.features[pos], nn.Conv2d):
                if thresholds:
                    self.threshold.update(
                        {'t' + str(pos): nn.Parameter(torch.tensor(thresholds.pop(0) * self.scaling_factor))})

        prev = len(self.features)

        for pos in range(len(self.classifier) - 1):
            if isinstance(self.classifier[pos], nn.Linear):
                if thresholds:
                    self.threshold.update(
                        {'t' + str(prev + pos): nn.Parameter(torch.tensor(thresholds.pop(0) * self.scaling_factor))})

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3

        for x in cfg:
            stride = 1

            if x == 'A':
                layers.pop()
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]

            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2,
                                     stride=stride, bias=False),
                           nn.ReLU(inplace=True)
                           ]
                layers += [nn.Dropout(self.dropout)]
                in_channels = x

        if self.dataset == 'IMAGENET':
            layers.pop()
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]

        features = nn.Sequential(*layers)

        layers = []
        if self.dataset == 'IMAGENET':
            layers += [nn.Linear(512 * 7 * 7, 4096, bias=False)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Dropout(self.dropout)]
            layers += [nn.Linear(4096, 4096, bias=False)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Dropout(self.dropout)]
            layers += [nn.Linear(4096, self.labels, bias=False)]
        elif self.dataset == 'CIFAR10' or self.dataset == 'CIFAR100':
            layers += [nn.Linear(512 * 2 * 2, 4096, bias=False)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Dropout(self.dropout)]
            layers += [nn.Linear(4096, 4096, bias=False)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Dropout(self.dropout)]
            layers += [nn.Linear(4096, self.labels, bias=False)]

        classifier = nn.Sequential(*layers)
        return features, classifier

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

        for l in range(len(self.features)):

            if isinstance(self.features[l], nn.Conv2d):
                self.mem[l] = torch.zeros(self.batch_size, self.features[l].out_channels, self.width, self.height)
                self.ops[l] = 0.0
            elif isinstance(self.features[l], nn.ReLU):
                if isinstance(self.features[l - 1], nn.Conv2d):
                    self.spike[l] = torch.ones(self.mem[l - 1].shape) * (-1000)
                elif isinstance(self.features[l - 1], nn.AvgPool2d):
                    self.spike[l] = torch.ones(self.batch_size, self.features[l - 2].out_channels, self.width,
                                               self.height) * (-1000)

            elif isinstance(self.features[l], nn.Dropout):
                if torch.cuda.is_available():
                    self.mask[l] = self.features[l](torch.ones(self.mem[l - 2].shape).cuda())
                else:
                    self.mask[l] = self.features[l](torch.ones(self.mem[l - 2].shape))

            elif isinstance(self.features[l], nn.AvgPool2d):
                self.width = self.width // self.features[l].kernel_size
                self.height = self.height // self.features[l].kernel_size

        prev = len(self.features)

        for l in range(len(self.classifier)):

            if isinstance(self.classifier[l], nn.Linear):
                self.mem[prev + l] = torch.zeros(self.batch_size, self.timesteps, self.classifier[l].out_features)
                self.ops[prev + l] = 0.0

            elif isinstance(self.classifier[l], nn.ReLU):
                self.spike[prev + l] = torch.ones(self.mem[prev + l - 1].shape) * (-1000)

            elif isinstance(self.classifier[l], nn.Dropout):
                if torch.cuda.is_available():
                    self.mask[prev + l] = self.classifier[l](torch.ones(self.mem[prev + l - 2].shape).cuda())
                else:
                    self.mask[prev + l] = self.classifier[l](torch.ones(self.mem[prev + l - 2].shape))

    def percentile(self, t, q):

        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        result = t.view(-1).kthvalue(k).values.item()
        return result

    def forward(self, x, find_max_mem=False, max_mem_layer=0):

        self.neuron_init(x)
        max_mem = 0.0
        ann = [0] * len(self.features)
        for t in range(self.timesteps):
            out_prev = x

            for l in range(len(self.features)):
                if isinstance(self.features[l], nn.Conv2d):

                    if find_max_mem and l == max_mem_layer:
                        cur = self.percentile(self.features[l](out_prev).view(-1), 99.7)
                        if cur > max_mem:
                            max_mem = torch.tensor([cur])
                        break
                    self.ops[l] += torch.sum(out_prev).item()
                    delta_mem = self.features[l](out_prev)
                    self.mem[l] = getattr(self.leak, 'l' + str(l)) * self.mem[l] + delta_mem
                    mem_thr = (self.mem[l] / getattr(self.threshold, 't' + str(l))) - 1.0
                    rst = getattr(self.threshold, 't' + str(l)) * (mem_thr > 0).float()
                    self.mem[l] = self.mem[l] - rst

                elif isinstance(self.features[l], nn.ReLU):
                    if ann[l]:
                        out_prev = self.features[l](out_prev)
                    else:
                        out = self.act_func(mem_thr, (t - 1 - self.spike[l]))
                        self.spike[l] = self.spike[l].masked_fill(out.bool(), t - 1)
                        out_prev = out.clone()

                elif isinstance(self.features[l], nn.AvgPool2d):
                    out_prev = self.features[l](out_prev)

                elif isinstance(self.features[l], nn.Dropout):
                    out_prev = out_prev * self.mask[l]

            if find_max_mem and max_mem_layer < len(self.features):
                continue

            out_prev = out_prev.reshape(self.batch_size, -1)
            prev = len(self.features)
            for l in range(len(self.classifier) - 1):

                if isinstance(self.classifier[l], nn.Linear):

                    if find_max_mem and (prev + l) == max_mem_layer:
                        cur = self.percentile(self.classifier[l](out_prev).view(-1), 99.7)
                        if cur > max_mem:
                            max_mem = torch.tensor([cur])
                        break
                    self.ops[prev + l] = torch.sum(out_prev).item()
                    delta_mem = self.classifier[l](out_prev)
                    self.mem[prev + l] = getattr(self.leak, 'l' + str(prev + l)) * self.mem[prev + l] + delta_mem
                    mem_thr = (self.mem[prev + l] / getattr(self.threshold, 't' + str(prev + l))) - 1.0
                    rst = getattr(self.threshold, 't' + str(prev + l)) * (mem_thr > 0).float()
                    self.mem[prev + l] = self.mem[prev + l] - rst

                elif isinstance(self.classifier[l], nn.ReLU):
                    out = self.act_func(mem_thr, (t - 1 - self.spike[prev + l]))
                    self.spike[prev + l] = self.spike[prev + l].masked_fill(out.bool(), t - 1)
                    out_prev = out.clone()

                elif isinstance(self.classifier[l], nn.Dropout):
                    out_prev = out_prev * self.mask[prev + l]

            # Compute the classification layer outputs
            if not find_max_mem:
                self.ops[prev + l + 1] = torch.sum(out_prev).item()
                self.mem[prev + l + 1][:, t, ...] = self.mem[prev + l + 1] + self.classifier[l + 1](out_prev)

        if find_max_mem:
            return max_mem

        return self.mem[prev + l + 1]

class VGG_SNN_MC(nn.Module):

    def __init__(self, vgg_name, labels=10, timesteps=100, leak=1.0, default_threshold=1.0,
                 dropout=0.2, hardware_opt=0, reset_to_zero=0, input_encoding=1, num_channels=1, kernel_size=3,
                 dataset='CIFAR10', bias=False):
        super().__init__()

        self.bias = bias
        self.vgg_name = vgg_name
        self.act_func = LinearSpikeMC.apply
        self.hardware_opt = {i: hardware_opt for i in out_channel_cfg[vgg_name].keys()}
        self.reset_to_zero = reset_to_zero
        self.input_encoding = input_encoding
        self.num_channels = num_channels
        self.labels = labels
        self.timesteps = timesteps
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.dataset = dataset
        self.mem = {}
        self.mask = {}
        self.ops = {}

        self.features, self.classifier = self._make_layers(cfg[self.vgg_name])

        self._initialize_weights2()

        threshold = {}
        lk = {}
        for l in range(len(self.features)):
            if isinstance(self.features[l], nn.Conv2d):
                threshold['t' + str(l)] = nn.Parameter(torch.tensor(default_threshold))
                lk['l' + str(l)] = nn.Parameter(torch.tensor(leak))

        prev = len(self.features)
        for l in range(len(self.classifier) - 1):
            if isinstance(self.classifier[l], nn.Linear):
                threshold['t' + str(prev + l)] = nn.Parameter(torch.tensor(default_threshold))
                lk['l' + str(prev + l)] = nn.Parameter(torch.tensor(leak))

        self.threshold = nn.ParameterDict(threshold)
        self.leak = nn.ParameterDict(lk)
        self.scaling_factor = 1.0
        self.batch_size = 0
        self.width = 0
        self.height = 0
        for layer in out_channel_cfg[self.vgg_name]:
            out_channel_cfg[self.vgg_name][layer] = self.num_channels
        for layer in in_channel_cfg[self.vgg_name]:
            if layer != 0:
                in_channel_cfg[self.vgg_name][layer] = self.num_channels

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

        # Initialize thresholds
        self.scaling_factor = scaling_factor

        for pos in range(len(self.features)):
            if isinstance(self.features[pos], nn.Conv2d):
                if thresholds:
                    if torch.cuda.is_available():
                        self.threshold.update(
                            {'t' + str(pos): nn.Parameter(
                                torch.tensor(thresholds.pop(0) * self.scaling_factor).cuda())})
                    else:
                        self.threshold.update(
                            {'t' + str(pos): nn.Parameter(
                                torch.tensor(thresholds.pop(0) * self.scaling_factor))})

        prev = len(self.features)

        for pos in range(len(self.classifier) - 1):
            if isinstance(self.classifier[pos], nn.Linear):
                if thresholds:
                    if torch.cuda.is_available():
                        self.threshold.update(
                            {'t' + str(prev + pos): nn.Parameter(
                                torch.tensor(thresholds.pop(0) * self.scaling_factor).cuda())})
                    else:
                        self.threshold.update(
                            {'t' + str(prev + pos): nn.Parameter(
                                torch.tensor(thresholds.pop(0) * self.scaling_factor))})

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3

        for x in cfg:
            stride = 1

            if x == 'A':
                layers.pop()
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]

            else:
                if self.bias:
                    layers += [
                        nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2,
                                  stride=stride, bias=True),
                        nn.ReLU(inplace=True)
                    ]
                    layers += [nn.Dropout(self.dropout)]
                    in_channels = x
                else:
                    layers += [
                        nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2,
                                  stride=stride, bias=False),
                        nn.ReLU(inplace=True)
                    ]
                    layers += [nn.Dropout(self.dropout)]
                    in_channels = x

        if self.dataset == 'IMAGENET':
            layers.pop()
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]

        features = nn.Sequential(*layers)

        layers = []
        if self.dataset == 'IMAGENET':
            layers += [nn.Linear(512 * 7 * 7, 4096, bias=False)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Dropout(self.dropout)]
            layers += [nn.Linear(4096, 4096, bias=False)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Dropout(self.dropout)]
            layers += [nn.Linear(4096, self.labels, bias=False)]
        elif self.dataset == 'CIFAR10' or self.dataset == 'CIFAR100':
            layers += [nn.Linear(512 * 2 * 2, 4096, bias=False)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Dropout(self.dropout)]
            layers += [nn.Linear(4096, 4096, bias=False)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Dropout(self.dropout)]
            layers += [nn.Linear(4096, self.labels, bias=False)]

        classifier = nn.Sequential(*layers)
        return features, classifier

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
        self.mask = {}
        self.ops = {}

        for l in range(len(self.features)):
            if isinstance(self.features[l], nn.Conv2d):
                if torch.cuda.is_available():
                    self.mem[l] = torch.zeros(self.batch_size, self.features[l].out_channels, self.width, self.height,
                                              device=torch.device('cuda'))
                else:
                    self.mem[l] = torch.zeros(self.batch_size, self.features[l].out_channels, self.width, self.height)
                self.ops[l] = 0.0
            elif isinstance(self.features[l], nn.Dropout):
                if torch.cuda.is_available():
                    self.mask[l] = torch.stack([self.features[l](torch.ones(self.mem[l - 2].shape).cuda()) for _ in
                                                range(out_channel_cfg[self.vgg_name][l - 2])])
                else:
                    self.mask[l] = torch.stack([self.features[l](torch.ones(self.mem[l - 2].shape)) for _ in
                                                range(out_channel_cfg[self.vgg_name][l - 2])])
            elif isinstance(self.features[l], nn.AvgPool2d):
                self.width = self.width // self.features[l].kernel_size
                self.height = self.height // self.features[l].kernel_size

        prev = len(self.features)

        for l in range(len(self.classifier)):

            if isinstance(self.classifier[l], nn.Linear):
                if l == len(self.classifier) - 1:
                    if torch.cuda.is_available():
                        self.mem[prev + l] = torch.zeros(self.batch_size, self.timesteps, self.classifier[l].out_features,
                                                         device=torch.device('cuda'))
                    else:
                        self.mem[prev + l] = torch.zeros(self.batch_size, self.timesteps, self.classifier[l].out_features)
                else:
                    if torch.cuda.is_available():
                        self.mem[prev + l] = torch.zeros(self.batch_size, self.classifier[l].out_features,
                                                         device=torch.device('cuda'))
                    else:
                        self.mem[prev + l] = torch.zeros(self.batch_size, self.classifier[l].out_features)
                self.ops[prev + l] = 0.0
            elif isinstance(self.classifier[l], nn.Dropout):
                if torch.cuda.is_available():
                    self.mask[prev + l] = torch.stack(
                        [self.classifier[l](torch.ones(self.mem[prev + l - 2].shape).cuda()) for _ in
                         range(out_channel_cfg[self.vgg_name][prev + l - 2])])
                else:
                    self.mask[prev + l] = torch.stack(
                        [self.classifier[l](torch.ones(self.mem[prev + l - 2].shape)) for _ in
                         range(out_channel_cfg[self.vgg_name][prev + l - 2])])

    def percentile(self, t, q):
        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        result = t.view(-1).kthvalue(k).values.item()
        return result

    def forward(self, x, reset=True, find_max_mem=False, max_mem_layer=0):
        if reset:
            self.neuron_init(x)
            max_mem = 0.0
        ann = [0] * len(self.features)

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

            for l in range(len(self.features)):
                if isinstance(self.features[l], nn.Conv2d):
                    # Combine input spikes into one tensor
                    merged_out_prev = torch.zeros_like(out_prev[0])
                    for in_ch in range(in_channel_cfg[self.vgg_name][l]):
                        merged_out_prev += (2 ** in_ch) * out_prev[in_ch]
                    if find_max_mem and l == max_mem_layer:
                        cur = self.percentile(self.features[l](merged_out_prev).view(-1), 99.7) / 2 ** (out_channel_cfg[self.vgg_name][l] - 1)
                        if cur > max_mem:
                            max_mem = torch.tensor([cur]).cuda()
                        break
                    self.ops[l] += torch.count_nonzero(merged_out_prev).item()
                    # Multiply weights by input spike trains
                    delta_mem = self.features[l](merged_out_prev)
                    # Update membrane potential leaked prior and weight x input
                    self.mem[l] = getattr(self.leak, 'l' + str(l)) * self.mem[l] + delta_mem
                    # Compute output spike trains via thresholding
                    thresh = getattr(self.threshold, 't' + str(l))
                    mem_thr = []
                    rst = torch.zeros_like(self.mem[l])
                    if out_channel_cfg[self.vgg_name][l] == 1:
                        if self.hardware_opt[l] != 2:
                            mem_thr.append(self.mem[l] / thresh - 1.0)
                        mem_thr.append(self.mem[l] / thresh - 1.0)
                        rst += (mem_thr[0] > 0).float() * thresh
                    else:
                        for out_ch in range(out_channel_cfg[self.vgg_name][l] - 1):
                            if self.hardware_opt[l] != 2:
                                mem_thr.append(self.mem[l] / ((2 ** out_ch) * thresh) - 1.0)
                                mem_thr.append(1.0 - self.mem[l] / (thresh * (2 ** (out_ch + 1))))
                                rst += (2 ** out_ch) * (torch.logical_and(mem_thr[2 * out_ch] > 0, mem_thr[
                                    2 * out_ch + 1] >= 0)).float() * thresh
                            else:
                                mem_thr.append(self.mem[l] / ((2 ** out_ch) * thresh) - 1.0)
                                rst += (2 ** out_ch) * (mem_thr[out_ch] > 0).float() * thresh
                        mem_thr.append(self.mem[l] / ((2 ** (out_ch + 1)) * thresh) - 1.0)
                        if self.hardware_opt[l] != 2:
                            mem_thr.append(self.mem[l] / ((2 ** (out_ch + 1)) * thresh) - 1.0)
                            rst += (2 ** (out_ch + 1)) * (mem_thr[2 * (out_ch + 1)] > 0).float() * thresh
                        else:
                            rst += (2 ** (out_ch + 1)) * (mem_thr[out_ch + 1] > 0).float() * thresh
                    # Reset membrane potential
                    if self.reset_to_zero == 0:
                        self.mem[l] = self.mem[l] - rst
                    else:
                        self.mem[l] = 0
                elif isinstance(self.features[l], nn.ReLU):
                    # mem_thr is a list containing output spike train tensors for each output channel
                    if ann[l]:
                        out_prev = self.features[l](out_prev)
                    else:
                        out = self.act_func(torch.stack(mem_thr), self.hardware_opt[l - 1], thresh)
                        out_prev = out.clone()
                elif isinstance(self.features[l], nn.AvgPool2d):
                    temp = []
                    for i in range(out_prev.shape[0]):
                        temp.append(self.features[l](out_prev[i]))
                    out_prev = torch.stack(temp)
                elif isinstance(self.features[l], nn.Dropout):
                    out_prev = out_prev * self.mask[l]

            if find_max_mem and max_mem_layer < len(self.features):
                continue

            prev = len(self.features)
            for l in range(len(self.classifier) - 1):
                if isinstance(self.classifier[l], nn.Linear):
                    merged_out_prev = torch.zeros_like(out_prev[0])
                    for in_ch in range(in_channel_cfg[self.vgg_name][prev + l]):
                        merged_out_prev += (2 ** in_ch) * out_prev[in_ch]
                    merged_out_prev = merged_out_prev.reshape(self.batch_size, -1)
                    if find_max_mem and (prev + l) == max_mem_layer:
                        cur = self.percentile(self.classifier[l](merged_out_prev).view(-1), 99.7) / 2 ** (out_channel_cfg[self.vgg_name][prev + l] - 1)
                        if cur > max_mem:
                            max_mem = torch.tensor([cur]).cuda()
                        break
                    self.ops[prev + l] = torch.count_nonzero(merged_out_prev).item()
                    delta_mem = self.classifier[l](merged_out_prev)
                    
                    self.mem[prev + l] = getattr(self.leak, 'l' + str(prev + l)) * self.mem[prev + l] + delta_mem
                    thresh = getattr(self.threshold, 't' + str(prev + l))
                    mem_thr = []
                    rst = torch.zeros_like(self.mem[prev + l])
                    if out_channel_cfg[self.vgg_name][prev + l] == 1:
                        if self.hardware_opt[prev + l] != 2:
                            mem_thr.append(self.mem[prev + l] / thresh - 1.0)
                        mem_thr.append(self.mem[prev + l] / thresh - 1.0)
                        rst += (mem_thr[0] > 0).float() * thresh
                    else:
                        for out_ch in range(out_channel_cfg[self.vgg_name][prev + l] - 1):
                            if self.hardware_opt[prev + l] != 2:
                                mem_thr.append(self.mem[prev + l] / ((2 ** out_ch) * thresh) - 1.0)
                                mem_thr.append(1.0 - self.mem[prev + l] / (thresh * (2 ** (out_ch + 1))))
                                rst += (2 ** out_ch) * (torch.logical_and(mem_thr[2 * out_ch] > 0, mem_thr[
                                    2 * out_ch + 1] >= 0)).float() * thresh
                            else:
                                mem_thr.append(self.mem[prev + l] / ((2 ** out_ch) * thresh) - 1.0)
                                rst += (2 ** out_ch) * (mem_thr[out_ch] > 0).float() * thresh
                        mem_thr.append(self.mem[prev + l] / ((2 ** (out_ch + 1)) * thresh) - 1.0)
                        if self.hardware_opt[prev + l] != 2:
                            mem_thr.append(self.mem[prev + l] / ((2 ** (out_ch + 1)) * thresh) - 1.0)
                            rst += (2 ** (out_ch + 1)) * (mem_thr[2 * (out_ch + 1)] > 0).float() * thresh
                        else:
                            rst += (2 ** (out_ch + 1)) * (mem_thr[out_ch + 1] > 0).float() * thresh
                    # Reset membrane potential
                    if self.reset_to_zero == 0:
                        self.mem[prev + l] = self.mem[prev + l] - rst
                    else:
                        self.mem[prev + l] = 0
                elif isinstance(self.classifier[l], nn.ReLU):
                    out = self.act_func(torch.stack(mem_thr), self.hardware_opt[prev + l - 1], thresh)
                    out_prev = out.clone()
                elif isinstance(self.classifier[l], nn.Dropout):
                    out_prev = out_prev * self.mask[prev + l]

            # Compute the classification layer outputs
            if not find_max_mem:
                merged_out_prev = torch.zeros_like(out_prev[0])
                for in_ch in range(in_channel_cfg[self.vgg_name][prev + l + 1]):
                    merged_out_prev += (2 ** in_ch) * out_prev[in_ch]
                self.ops[prev + l + 1] = torch.count_nonzero(merged_out_prev).item()
                self.mem[prev + l + 1][:, t, ...] = self.classifier[l + 1](merged_out_prev)

        if find_max_mem:
            return max_mem

        return self.mem[prev + l + 1]
