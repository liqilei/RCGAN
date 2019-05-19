import torch.nn as nn
from .blocks import ConvBlock


class CoGAN_G(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(CoGAN_G, self).__init__()

        self.common = nn.Sequential(
            ConvBlock(in_channels, 256, kernel_size=5, bias=True, valid_padding=False, act_type='lrelu',
                      norm_type='bn'),
            ConvBlock(256, 128, kernel_size=5, bias=True, valid_padding=False, act_type='lrelu',
                      norm_type='bn'),
            ConvBlock(128, 64, kernel_size=3, bias=True, valid_padding=False, act_type='lrelu',
                      norm_type='bn')
        )
        self.conv1_a = ConvBlock(64, 32, kernel_size=3, bias=True, valid_padding=False, act_type='lrelu',
                                 norm_type='bn')
        self.conv1_b = ConvBlock(64, 32, kernel_size=3, bias=True, valid_padding=False, act_type='lrelu',
                                 norm_type='bn')
        self.conv2_a = ConvBlock(32, out_channels, kernel_size=1, bias=True, act_type='tanh', norm_type=None)
        self.conv2_b = ConvBlock(32, out_channels, kernel_size=1, bias=True, act_type='tanh', norm_type=None)

    def forward(self, x):
        x = self.common(x)
        out1 = self.conv2_a(self.conv1_a(x))
        out2 = self.conv2_b(self.conv1_b(x))
        return out1, out2


class CoGAN_D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(CoGAN_D, self).__init__()

        self.conv1_a = ConvBlock(in_channels, 32, kernel_size=3, stride=2, valid_padding=True, bias=True,
                                 act_type='lrelu', norm_type=None)
        self.conv1_b = ConvBlock(in_channels, 32, kernel_size=3, stride=2, valid_padding=True, bias=True,
                                 act_type='lrelu', norm_type=None)
        self.conv2_a = ConvBlock(32, 64, kernel_size=3, stride=2, valid_padding=False, bias=True, act_type='lrelu',
                                 norm_type='bn')
        self.conv2_b = ConvBlock(32, 64, kernel_size=3, stride=2, valid_padding=False, bias=True, act_type='lrelu',
                                 norm_type='bn')
        self.common = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, stride=2, valid_padding=False, bias=True, act_type='lrelu',
                      norm_type='bn'),
            ConvBlock(128, 256, kernel_size=3, stride=2, valid_padding=False, bias=True,
                      act_type='lrelu', norm_type='bn')
        )
        self.linear = nn.Linear(6 * 6 * 256, out_channels)

    def forward(self, input_list):
        assert len(input_list) == 2, 'size dose not match'
        input1 = input_list[0]
        input2 = input_list[1]
        input1 = self.conv2_a(self.conv1_a(input1))
        input2 = self.conv2_b(self.conv1_b(input2))
        out1 = self.common(input1)
        out2 = self.common(input2)
        out1 = out1.view(out1.size(0), -1)
        out2 = out2.view(out2.size(0), -1)
        out1 = self.linear(out1)
        out2 = self.linear(out2)
        return out1, out2
