import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuantizeActivation(nn.Module):

    def __init__(self, act_bits, minimum, maximum, mode, quant_scale_max, quant_id):
        super().__init__()
        self.act_bits = act_bits
        if torch.cuda.is_available():
            self.register_buffer("minimum", torch.tensor([minimum]).cuda())
            self.register_buffer("maximum", torch.tensor([maximum]).cuda())
        else:
            self.register_buffer("minimum", torch.tensor([minimum]))
            self.register_buffer("maximum", torch.tensor([maximum]))
        self.quant_scale_max = quant_scale_max
        self.quant_id = quant_id
        self.mode = mode

    def percentile(self, t, q):
        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        result = t.view(-1).kthvalue(k).values.item()
        return result

    def quantize(self, x):
        if self.quant_id == 'lvl':
            eps = 1e-9
            y = (x - self.minimum) / (self.maximum - self.minimum)
            temp = torch.abs(y)
            temp[y > eps] = torch.log2(temp[y > eps]).round()
            temp[temp >= 1] = 0
            temp[torch.logical_and(y != 0, temp > (- self.act_bits))] = 2 ** temp[torch.logical_and(y != 0, temp > (-self.act_bits))]
            temp[temp < 0] = 0
            temp = temp * (self.maximum - self.minimum) + self.minimum
        elif self.quant_id == 'log':
            eps = 1e-9
            y = (x - self.minimum) / (self.maximum - self.minimum)
            temp = torch.abs(y)
            temp[y > eps] = torch.log2(temp[y > eps]).round()
            temp[temp >= 1] = 0
            temp[torch.logical_and(y != 0, temp > (1 - 2 ** self.act_bits))] = 2 ** temp[torch.logical_and(y != 0, temp > (1 - 2 ** self.act_bits))]
            temp[temp < 0] = 0
            temp = temp * (self.maximum - self.minimum) + self.minimum
        elif self.quant_id == 'uniform':
            temp = (x - self.minimum) / (self.maximum - self.minimum) * (2 ** self.act_bits - 1)
            temp = temp.round()
            temp = temp / (2 ** self.act_bits - 1) * (self.maximum - self.minimum) + self.minimum
        return temp

    def forward(self, x):
        out = x
        if self.mode == 0:
            out.data = self.quantize(x)
        elif self.mode == 1:
            if self.minimum.item() > torch.min(x).item():
                self.minimum = torch.min(x)
            if self.maximum.item() < self.percentile(x.view(-1), self.quant_scale_max):
                if torch.cuda.is_available():
                    self.maximum = torch.tensor([self.percentile(x.view(-1), self.quant_scale_max)]).cuda()
                else:
                    self.maximum = torch.tensor([self.percentile(x.view(-1), self.quant_scale_max)])
        elif self.mode == 2:
            if self.minimum.item() > torch.min(x).item():
                self.minimum = torch.min(x)
            if 1.5 * self.maximum.item() > self.percentile(x.view(-1), self.quant_scale_max):
                if torch.cuda.is_available():
                    self.maximum = torch.tensor([self.percentile(x.view(-1), self.quant_scale_max)]).cuda()
                else:
                    self.maximum = torch.tensor([self.percentile(x.view(-1), self.quant_scale_max)])
        return out
