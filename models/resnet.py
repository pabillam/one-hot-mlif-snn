import torch
import torch.nn as nn
import torch.nn.functional as F
from models.quantize import *
import pdb
import math


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, dropout, bn=False, bias=False, quant_act=False, act_bits=1, quant_scale_max=0.8, quant_id='log'):
        super().__init__()
        self.bn = bn
        self.bias = bias
        self.quant_act = quant_act
        self.act_bits = act_bits
        self.quant_scale_max = quant_scale_max
        self.quant_id = quant_id
        
        if self.quant_act:
            self.quantizer = nn.Sequential(
                QuantizeActivation(act_bits=self.act_bits, minimum=0.0, maximum=0.0, mode=0, quant_scale_max=self.quant_scale_max, quant_id=self.quant_id),
            )

        if self.quant_act:
            if self.bn and self.bias:
                self.residual = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True),
                    nn.BatchNorm2d(planes),
                    nn.ReLU(inplace=True),
                    QuantizeActivation(act_bits=self.act_bits, minimum=0.0, maximum=0.0, mode=0, quant_scale_max=self.quant_scale_max, quant_id=self.quant_id),
                    nn.Dropout(dropout),
                    nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(planes)
                )
            elif self.bn:
                self.residual = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(planes),
                    nn.ReLU(inplace=True),
                    QuantizeActivation(act_bits=self.act_bits, minimum=0.0, maximum=0.0, mode=0, quant_scale_max=self.quant_scale_max, quant_id=self.quant_id),
                    nn.Dropout(dropout),
                    nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(planes)
                )
            elif self.bias:
                self.residual = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    QuantizeActivation(act_bits=self.act_bits, minimum=0.0, maximum=0.0, mode=0, quant_scale_max=self.quant_scale_max, quant_id=self.quant_id),
                    nn.Dropout(dropout),
                    nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
                )
            else:
                self.residual = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    QuantizeActivation(act_bits=self.act_bits, minimum=0.0, maximum=0.0, mode=0, quant_scale_max=self.quant_scale_max, quant_id=self.quant_id),
                    nn.Dropout(dropout),
                    nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
                )
        else:
            if self.bn and self.bias:
                self.residual = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True),
                    nn.BatchNorm2d(planes),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(planes)
                )
            elif self.bn:
                self.residual = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(planes),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(planes)
                )
            elif self.bias:
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
            if self.bn and self.bias:
                self.identity = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=True),
                    nn.BatchNorm2d(self.expansion * planes)
                )
            elif self.bn:
                self.identity = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )
            elif self.bias:
                self.identity = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=True)
                )
            else:
                self.identity = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
                )

    def forward(self, x):
        out = self.residual(x) + self.identity(x)
        out = F.relu(out)
        if self.quant_act:
            out = self.quantizer(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block, num_blocks, labels=10, dropout=0.2, dataset='CIFAR10', bn=False, bias=False,
                 quant_act=False, act_bits=1, quant_scale_max=0.8, quant_id='log'):
        super(ResNet, self).__init__()
        self.bn = bn
        self.bias = bias
        self.quant_act = quant_act
        self.act_bits = act_bits
        self.quant_scale_max = quant_scale_max
        self.quant_id = quant_id
        self.in_planes = 64
        self.dataset = dataset

        self.dropout = dropout
        if dataset == 'CIFAR10' or dataset == 'CIFAR100':
            if self.quant_act:
                if self.bn and self.bias:
                    self.pre_process = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        QuantizeActivation(act_bits=self.act_bits, minimum=0.0, maximum=0.0, mode=0, quant_scale_max=self.quant_scale_max, quant_id=self.quant_id),
                        nn.Dropout(self.dropout),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        QuantizeActivation(act_bits=self.act_bits, minimum=0.0, maximum=0.0, mode=0, quant_scale_max=self.quant_scale_max, quant_id=self.quant_id),
                        nn.Dropout(self.dropout),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        QuantizeActivation(act_bits=self.act_bits, minimum=0.0, maximum=0.0, mode=0, quant_scale_max=self.quant_scale_max, quant_id=self.quant_id),
                        nn.AvgPool2d(2)
                    )
                elif self.bn:
                    self.pre_process = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        QuantizeActivation(act_bits=self.act_bits, minimum=0.0, maximum=0.0, mode=0, quant_scale_max=self.quant_scale_max, quant_id=self.quant_id),
                        nn.Dropout(self.dropout),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        QuantizeActivation(act_bits=self.act_bits, minimum=0.0, maximum=0.0, mode=0, quant_scale_max=self.quant_scale_max, quant_id=self.quant_id),
                        nn.Dropout(self.dropout),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        QuantizeActivation(act_bits=self.act_bits, minimum=0.0, maximum=0.0, mode=0, quant_scale_max=self.quant_scale_max, quant_id=self.quant_id),
                        nn.AvgPool2d(2)
                    )
                elif self.bias:
                    self.pre_process = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        QuantizeActivation(act_bits=self.act_bits, minimum=0.0, maximum=0.0, mode=0, quant_scale_max=self.quant_scale_max, quant_id=self.quant_id),
                        nn.Dropout(self.dropout),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        QuantizeActivation(act_bits=self.act_bits, minimum=0.0, maximum=0.0, mode=0, quant_scale_max=self.quant_scale_max, quant_id=self.quant_id),
                        nn.Dropout(self.dropout),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        QuantizeActivation(act_bits=self.act_bits, minimum=0.0, maximum=0.0, mode=0, quant_scale_max=self.quant_scale_max, quant_id=self.quant_id),
                        nn.AvgPool2d(2)
                    )
                else:
                    self.pre_process = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.ReLU(inplace=True),
                        QuantizeActivation(act_bits=self.act_bits, minimum=0.0, maximum=0.0, mode=0, quant_scale_max=self.quant_scale_max, quant_id=self.quant_id),
                        nn.Dropout(self.dropout),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.ReLU(inplace=True),
                        QuantizeActivation(act_bits=self.act_bits, minimum=0.0, maximum=0.0, mode=0, quant_scale_max=self.quant_scale_max, quant_id=self.quant_id),
                        nn.Dropout(self.dropout),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.ReLU(inplace=True),
                        QuantizeActivation(act_bits=self.act_bits, minimum=0.0, maximum=0.0, mode=0, quant_scale_max=self.quant_scale_max, quant_id=self.quant_id),
                        nn.AvgPool2d(2)
                    )
            else:
                if self.bn and self.bias:
                    self.pre_process = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.Dropout(self.dropout),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.Dropout(self.dropout),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.AvgPool2d(2)
                    )
                elif self.bn:
                    self.pre_process = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.Dropout(self.dropout),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.Dropout(self.dropout),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.AvgPool2d(2)
                    )
                elif self.bias:
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

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, dropout=self.dropout)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, dropout=self.dropout)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, dropout=self.dropout)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, dropout=self.dropout)
        if dataset == 'CIFAR10' or dataset == 'CIFAR100':
            self.classifier = nn.Sequential(nn.Linear(512 * 2 * 2, labels, bias=False))
        self._initialize_weights()

    def _initialize_weights(self):
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

    def _make_layer(self, block, planes, num_blocks, stride, dropout):
        if num_blocks == 0:
            return nn.Sequential()
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_planes, planes, stride, dropout, self.bn, self.bias, self.quant_act, self.act_bits, self.quant_scale_max, self.quant_id))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.pre_process(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(x.size(0), -1)
        out = self.classifier(out)
        return out


def ResNet20(labels=10, dropout=0.2, dataset='CIFAR10', bn=False, bias=False, quant_act=False, act_bits=1, quant_scale_max=0.8, quant_id='log'):
    return ResNet(block=BasicBlock, num_blocks=[2, 2, 2, 2], labels=labels, dropout=dropout, dataset=dataset, bn=bn,
                  bias=bias, quant_act=quant_act, act_bits=act_bits, quant_scale_max=quant_scale_max, quant_id=quant_id)