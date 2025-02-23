import torch
import torch.nn as nn
import torch.nn.functional as F
from models.quantize import *
import pdb
import math

cfg = {
    'VGG16': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A', 512, 512, 512, 'A', 512, 512, 512],
}

class VGG(nn.Module):
    def __init__(self, vgg_name='VGG16', labels=10, dataset='CIFAR10', kernel_size=3, dropout=0.2, bn=False, bias=False,
                 quant_act=False, act_bits=1, quant_scale_max=0.8, quant_id='log'):
        super(VGG, self).__init__()

        self.dataset = dataset
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.bn = bn
        self.bias = bias
        self.quant_act = quant_act
        self.quant_id = quant_id
        self.act_bits = act_bits
        self.quant_scale_max = quant_scale_max
        self.features = self._make_layers(cfg[vgg_name])

        if self.quant_act:
            if dataset == 'IMAGENET':
                self.classifier = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096, bias=False),
                    nn.ReLU(inplace=True),
                    QuantizeActivation(act_bits=self.act_bits, minimum=0.0, maximum=0.0, mode=0, quant_scale_max=self.quant_scale_max, quant_id=self.quant_id),
                    nn.Dropout(self.dropout),
                    nn.Linear(4096, 4096, bias=False),
                    nn.ReLU(inplace=True),
                    QuantizeActivation(act_bits=self.act_bits, minimum=0.0, maximum=0.0, mode=0, quant_scale_max=self.quant_scale_max, quant_id=self.quant_id),
                    nn.Dropout(self.dropout),
                    nn.Linear(4096, labels, bias=False)
                )
            elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
                self.classifier = nn.Sequential(
                    nn.Linear(512 * 2 * 2, 4096, bias=False),
                    nn.ReLU(inplace=True),
                    QuantizeActivation(act_bits=self.act_bits, minimum=0.0, maximum=0.0, mode=0, quant_scale_max=self.quant_scale_max, quant_id=self.quant_id),
                    nn.Dropout(0.5),
                    nn.Linear(4096, 4096, bias=False),
                    nn.ReLU(inplace=True),
                    QuantizeActivation(act_bits=self.act_bits, minimum=0.0, maximum=0.0, mode=0, quant_scale_max=self.quant_scale_max, quant_id=self.quant_id),
                    nn.Dropout(0.5),
                    nn.Linear(4096, labels, bias=False)
                )
        else:
            if dataset == 'IMAGENET':
                self.classifier = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Dropout(self.dropout),
                    nn.Linear(4096, 4096, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Dropout(self.dropout),
                    nn.Linear(4096, labels, bias=False)
                )
            elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
                self.classifier = nn.Sequential(
                    nn.Linear(512 * 2 * 2, 4096, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(4096, 4096, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(4096, labels, bias=False)
                )

        self._initialize_weights2()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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

    def _make_layers(self, cfg):
        layers = []

        in_channels = 3

        for x in cfg:
            stride = 1

            if x == 'A':
                layers.pop()
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                if self.quant_act:
                    if self.bn and self.bias:
                        layers += [
                            nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2,
                                      stride=stride, bias=True),
                            nn.BatchNorm2d(x),
                            nn.ReLU(inplace=True),
                            QuantizeActivation(act_bits=self.act_bits, minimum=0.0, maximum=0.0, mode=0, quant_scale_max=self.quant_scale_max, quant_id=self.quant_id)
                        ]
                    elif self.bn:
                        layers += [
                            nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2,
                                      stride=stride, bias=False),
                            nn.BatchNorm2d(x),
                            nn.ReLU(inplace=True),
                            QuantizeActivation(act_bits=self.act_bits, minimum=0.0, maximum=0.0, mode=0, quant_scale_max=self.quant_scale_max, quant_id=self.quant_id)
                        ]
                    elif self.bias:
                        layers += [
                            nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2,
                                      stride=stride, bias=True),
                            nn.ReLU(inplace=True),
                            QuantizeActivation(act_bits=self.act_bits, minimum=0.0, maximum=0.0, mode=0, quant_scale_max=self.quant_scale_max, quant_id=self.quant_id)
                        ]
                    else:
                        layers += [
                            nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2,
                                      stride=stride, bias=False),
                            nn.ReLU(inplace=True),
                            QuantizeActivation(act_bits=self.act_bits, minimum=0.0, maximum=0.0, mode=0, quant_scale_max=self.quant_scale_max, quant_id=self.quant_id)
                        ]
                else:
                    if self.bn and self.bias:
                        layers += [
                            nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2,
                                      stride=stride, bias=True),
                            nn.BatchNorm2d(x),
                            nn.ReLU(inplace=True)
                        ]
                    elif self.bn:
                        layers += [
                            nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2,
                                      stride=stride, bias=False),
                            nn.BatchNorm2d(x),
                            nn.ReLU(inplace=True)
                        ]
                    elif self.bias:
                        layers += [
                            nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2,
                                      stride=stride, bias=True),
                            nn.ReLU(inplace=True)
                        ]
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

        return nn.Sequential(*layers)
