import random
from models.layers import *
import math

class VGGSNNwoAP(nn.Module):
    def __init__(self, spike_channels, multi_channel, classes):
        super(VGGSNNwoAP, self).__init__()
        self.features = nn.Sequential(
            Layer(2, 64, 3, 1, 1, spike_channels, spike_channels, multi_channel, False),
            Layer(64, 128, 3, 2, 1, spike_channels, spike_channels, multi_channel, False),
            Layer(128, 256, 3, 1, 1, spike_channels, spike_channels, multi_channel, False),
            Layer(256, 256, 3, 2, 1, spike_channels, spike_channels, multi_channel, False),
            Layer(256, 512, 3, 1, 1, spike_channels, spike_channels, multi_channel, False),
            Layer(512, 512, 3, 2, 1, spike_channels, spike_channels, multi_channel, False),
            Layer(512, 512, 3, 1, 1, spike_channels, spike_channels, multi_channel, False),
            Layer(512, 512, 3, 2, 1, spike_channels, spike_channels, multi_channel, False),
        )
        W = int(48 / 2 / 2 / 2 / 2)
        if torch.cuda.is_available():
            self.register_buffer("classifier_ops", torch.zeros(1).cuda())
        else:
            self.register_buffer("classifier_ops", torch.zeros(1))
        self.classifier = SeqToANNContainer(nn.Linear(512 * W * W, classes))
        self.spike_channels = spike_channels
        self.multi_channel = multi_channel

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        if self.multi_channel:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            temp_x = torch.zeros(x.size(0), x.size(1), x.size(3), x.size(4), x.size(5)).to(device)
            for in_ch in range(self.spike_channels):
                temp_x += (2 ** in_ch) * x[:, :, in_ch, ...]
        else:
            temp_x = x 
        x = torch.flatten(temp_x, 2)
        self.classifier_ops += torch.count_nonzero(x)
        x = self.classifier(x)
        return x
