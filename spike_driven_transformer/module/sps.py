import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
from timm.models.layers import to_2tuple
from . import mlif

class MS_SPS(nn.Module):
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=4,
        in_channels=2,
        embed_dims=256,
        pooling_stat="1111",
        spike_mode="lif",
        mlif_channels=1,
        scaled=False,
    ):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.pooling_stat = pooling_stat
        self.spike_mode = spike_mode
        self.mlif_channels = mlif_channels

        self.C = in_channels
        self.H, self.W = (
            self.image_size[0] // patch_size[0],
            self.image_size[1] // patch_size[1],
        )
        self.num_patches = self.H * self.W
        self.proj_conv = nn.Conv2d(
            in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        if spike_mode == "lif":
            self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "mlif":
            self.proj_lif = mlif.MultiStepMLIFNode(tau=2.0, detach_reset=True, channels=mlif_channels, backend="torch", scaled=scaled)
        elif spike_mode == "fast_mlif":
            self.proj_lif = mlif.FastMultiStepMLIFNode(tau=2.0, detach_reset=True, channels=mlif_channels, backend="cupy", scaled=scaled)
        elif spike_mode == "plif":
            self.proj_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv1 = nn.Conv2d(
            embed_dims // 8,
            embed_dims // 4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        if spike_mode == "lif":
            self.proj_lif1 = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "mlif":
            self.proj_lif1 = mlif.MultiStepMLIFNode(tau=2.0, detach_reset=True, channels=mlif_channels, backend="torch", scaled=scaled)
        elif spike_mode == "fast_mlif":
            self.proj_lif1 = mlif.FastMultiStepMLIFNode(tau=2.0, detach_reset=True, channels=mlif_channels, backend="cupy", scaled=scaled)
        elif spike_mode == "plif":
            self.proj_lif1 = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv2 = nn.Conv2d(
            embed_dims // 4,
            embed_dims // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        if spike_mode == "lif":
            self.proj_lif2 = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "mlif":
            self.proj_lif2 = mlif.MultiStepMLIFNode(tau=2.0, detach_reset=True, channels=mlif_channels, backend="torch", scaled=scaled)
        elif spike_mode == "fast_mlif":
            self.proj_lif2 = mlif.FastMultiStepMLIFNode(tau=2.0, detach_reset=True, channels=mlif_channels, backend="cupy", scaled=scaled)
        elif spike_mode == "plif":
            self.proj_lif2 = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.maxpool2 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv3 = nn.Conv2d(
            embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        if spike_mode == "lif":
            self.proj_lif3 = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "mlif":
            self.proj_lif3 = mlif.MultiStepMLIFNode(tau=2.0, detach_reset=True, channels=mlif_channels, backend="torch", scaled=scaled)
        elif spike_mode == "fast_mlif":
            self.proj_lif3 = mlif.FastMultiStepMLIFNode(tau=2.0, detach_reset=True, channels=mlif_channels, backend="cupy", scaled=scaled)
        elif spike_mode == "plif":
            self.proj_lif3 = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.maxpool3 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.rpe_conv = nn.Conv2d(
            embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        if spike_mode == "lif":
            self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "mlif":
            self.rpe_lif = mlif.MultiStepMLIFNode(tau=2.0, detach_reset=True, channels=mlif_channels, backend="torch", scaled=scaled)
        elif spike_mode == "fast_mlif":
            self.rpe_lif = mlif.FastMultiStepMLIFNode(tau=2.0, detach_reset=True, channels=mlif_channels, backend="cupy", scaled=scaled)
        elif spike_mode == "plif":
            self.rpe_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )

    def forward(self, x, hook=None):
        T, B, _, H, W = x.shape
        ratio = 1
        if hook is not None:
            hook[self._get_name() + "_proj_conv"] = {"tensor": x.detach()}
        x = self.proj_conv(x.flatten(0, 1))  # have some fire value
        if hook is not None:
            hook[self._get_name() + "_proj_conv"]["flops"] = torch.prod(torch.tensor(list(self.proj_conv.weight.shape))) * x.shape[-1] * x.shape[-2]
        x = self.proj_bn(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif(x)

        x = x.flatten(0, 1).contiguous()
        if self.pooling_stat[0] == "1":
            x = self.maxpool(x)
            ratio *= 2

        if hook is not None:
            hook[self._get_name() + "_proj_conv1"] = {"tensor": x.detach()}
        x = self.proj_conv1(x)
        if hook is not None:
            hook[self._get_name() + "_proj_conv1"]["flops"] = torch.prod(torch.tensor(list(self.proj_conv1.weight.shape))) * x.shape[-1] * x.shape[-2]
        x = self.proj_bn1(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif1(x)

        x = x.flatten(0, 1).contiguous()
        if self.pooling_stat[1] == "1":
            x = self.maxpool1(x)
            ratio *= 2

        if hook is not None:
            hook[self._get_name() + "_proj_conv2"] = {"tensor": x.detach()}
        x = self.proj_conv2(x)
        if hook is not None:
            hook[self._get_name() + "_proj_conv2"]["flops"] = torch.prod(torch.tensor(list(self.proj_conv2.weight.shape))) * x.shape[-1] * x.shape[-2]
        x = self.proj_bn2(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif2(x)

        x = x.flatten(0, 1).contiguous()
        if self.pooling_stat[2] == "1":
            x = self.maxpool2(x)
            ratio *= 2
        
        if hook is not None:
            hook[self._get_name() + "_proj_conv3"] = {"tensor": x.detach()}
        x = self.proj_conv3(x)
        if hook is not None:
            hook[self._get_name() + "_proj_conv3"]["flops"] = torch.prod(torch.tensor(list(self.proj_conv3.weight.shape))) * x.shape[-1] * x.shape[-2]
        x = self.proj_bn3(x)
        if self.pooling_stat[3] == "1":
            x = self.maxpool3(x)
            ratio *= 2

        x_feat = x
        x = self.proj_lif3(x.reshape(T, B, -1, H // ratio, W // ratio).contiguous())

        x = x.flatten(0, 1).contiguous()
        
        if hook is not None:
            hook[self._get_name() + "_rpe_conv"] = {"tensor": x.detach()}
        x = self.rpe_conv(x)
        if hook is not None:
            hook[self._get_name() + "_rpe_conv"]["flops"] = torch.prod(torch.tensor(list(self.rpe_conv.weight.shape))) * x.shape[-1] * x.shape[-2]  
        x = self.rpe_bn(x)
        x = (x + x_feat).reshape(T, B, -1, H // ratio, W // ratio).contiguous()

        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W), hook
