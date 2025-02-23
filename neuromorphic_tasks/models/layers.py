import torch
import torch.nn as nn
import torch.nn.functional as F


class TensorNormalization(nn.Module):
    def __init__(self, mean, std):
        super(TensorNormalization, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.mean = mean
        self.std = std

    def forward(self, X):
        return normalizex(X, self.mean, self.std)


def normalizex(tensor, mean, std):
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    if mean.device != tensor.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    return tensor.sub(mean).div(std)


class SeqToANNContainer(nn.Module):
    # This code is form spikingjelly https://github.com/fangwei123456/spikingjelly
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)

class Layer(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding, spike_in_channels, spike_out_channels,
                 multi_channel, first_layer):
        super(Layer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
            nn.BatchNorm2d(out_plane)
        )
        self.act = LIFSpike(spike_out_channels=spike_out_channels, multi_channel=multi_channel)
        self.spike_in_channels = spike_in_channels
        self.spike_out_channels = spike_out_channels
        self.multi_channel = multi_channel
        self.first_layer = first_layer
        if torch.cuda.is_available():
            self.register_buffer("ops", torch.zeros(1).cuda())
        else:
            self.register_buffer("ops", torch.zeros(1))

    def forward(self, x):
        if self.multi_channel and not self.first_layer:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            temp_x = torch.zeros(x.size(0), x.size(1), x.size(3), x.size(4), x.size(5)).to(device)
            for in_ch in range(self.spike_in_channels):
                temp_x += (2 ** in_ch) * x[:, :, in_ch, ...]
        else:
            temp_x = x
        self.ops += torch.count_nonzero(temp_x)
        temp_x = self.fwd(temp_x)
        x = self.act(temp_x)
        return x


class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama, multi_channel):
        if multi_channel:
            if torch.cuda.is_available():
                out = torch.zeros_like(input[:, ::2, ...]).cuda()
            else:
                out = torch.zeros_like(input[:, ::2, ...])
            cond1 = input[:, ::2, ...]
            cond2 = input[:, 1::2, ...]
            out[torch.logical_and(cond1 > 0, cond2 >= 0)] = 1.0
            L = torch.tensor([gama, multi_channel])
        else:
            out = (input > 0).float()
            L = torch.tensor([gama, multi_channel])

        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        multi_channel = others[1].item()
        grad_input = grad_output.clone()
        if multi_channel:
            grad = torch.zeros_like(input)
            for x in range(input.shape[1] // 2 - 1): 
                grad[:, 2 * x, ...] = (1 / gama) * (1 / gama) * ((gama - torch.abs(input[:, 2 * x, ...])).clamp(min=0)) * grad_input[:, x, ...]
                grad[:, 2 * x + 1, ...] = - (1 / gama) * (1 / gama) * ((gama - torch.abs(input[:, 2 * x + 1, ...])).clamp(min=0)) * grad_input[:, x, ...]
            grad[:, 2 * (x + 1), ...] = (1 / gama) * (1 / gama) * ((gama - torch.abs(input[:, 2 * (x + 1), ...])).clamp(min=0)) * grad_input[:, x + 1, ...]
            return grad, None, None
        else:
            tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
            grad_input = grad_input * tmp
            return grad_input, None, None


class LIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, gama=1.0, spike_out_channels=1, multi_channel=0):
        super(LIFSpike, self).__init__()
        self.act = ZIF.apply
        self.thresh = thresh
        self.tau = tau
        self.gama = gama
        self.spike_out_channels = spike_out_channels
        self.multi_channel = multi_channel

    def forward(self, x):
        if self.multi_channel:
            mem = 0
            spike_pot = []
            T = x.shape[1]
            for t in range(T):
                mem = mem * self.tau + x[:, t, ...]
                rst = torch.zeros_like(mem)
                temp_mem = []

                if self.spike_out_channels == 1:
                    temp_mem.append(mem / self.thresh - 1.0)
                    temp_mem.append(mem / self.thresh - 1.0)
                    rst += (temp_mem[0] > 0).float()
                else:
                    for out_ch in range(self.spike_out_channels - 1):
                        temp_mem.append(mem / ((2 ** out_ch) * self.thresh) - 1.0)
                        temp_mem.append(1.0 - mem / (self.thresh * (2 ** (out_ch + 1))))
                        rst = torch.logical_or(rst, (torch.logical_and(temp_mem[2 * out_ch] > 0, temp_mem[2 * out_ch + 1] >= 0))).float()
                    temp_mem.append(mem / ((2 ** (out_ch + 1)) * self.thresh) - 1.0)
                    temp_mem.append(mem / ((2 ** (out_ch + 1)) * self.thresh) - 1.0)
                    rst = torch.logical_or(rst, (2 ** (out_ch + 1)) * (temp_mem[2 * (out_ch + 1)] > 0)).float()
                spike = self.act(torch.stack(temp_mem, dim=1), self.gama, self.multi_channel)
                mem = (1 - rst) * mem
                spike_pot.append(spike)
        else:
            mem = 0
            spike_pot = []
            T = x.shape[1]
            for t in range(T):
                mem = mem * self.tau + x[:, t, ...]
                spike = self.act(mem - self.thresh, self.gama, self.multi_channel)
                mem = (1 - spike) * mem
                spike_pot.append(spike)
        return torch.stack(spike_pot, dim=1)


def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(1, T, 1, 1, 1)
    return x
