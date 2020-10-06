import torch.nn as nn
from .helpers import get_norm_layer_1D, get_norm_layer_2D, get_nonlinear_layer


################################################################################
# Basic Building Blocks
################################################################################
def Conv1d(in_filters, out_filters, kernel_size=3, stride=1, padding=0, dilation=1,
           bias=True, norm_type='batch', nonlinear_type='LeakyReLU', spectral=True):
    layers = []
    if spectral:
        layers += [nn.utils.spectral_norm(nn.Conv1d(in_filters, out_filters, kernel_size=kernel_size,
                                                    stride=stride, padding=padding, dilation=dilation, bias=bias))]
    else:
        layers += [nn.Conv1d(in_filters, out_filters, kernel_size=kernel_size, stride=stride,
                             padding=padding, dilation=dilation, bias=bias)]

    if norm_type is not None:
        layers += [get_norm_layer_1D(norm_type)(out_filters)]

    if nonlinear_type is not None:
        layers += [get_nonlinear_layer(nonlinear_type)]
    return nn.Sequential(*layers)


def Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=0, dilation=1, bias=True,
           coord_conv=False, norm_type='batch', nonlinear_type="LeakyReLU", spectral=True):
    layers = []
    if spectral:
        layers += [nn.utils.spectral_norm(nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size,
                                                    stride=stride, padding=padding, dilation=dilation, bias=bias))]
    else:
        layers += [nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride,
                             padding=padding, dilation=dilation, bias=bias)]

    if norm_type is not None:
        layers += [get_norm_layer_2D(norm_type)(out_filters)]

    if nonlinear_type is not None:
        layers += [get_nonlinear_layer(nonlinear_type)]
    return nn.Sequential(*layers)


class UpConv2d(nn.Module):
    def __init__(self, infilters, outfilters, kernel_size=3, scale=2, padding=1,
                 norm_type='instance', nonlinear_type='LeakyReLU'):
        super(UpConv2d, self).__init__()
        self.scale = scale
        self.up = nn.functional.interpolate
        self.conv1 = Conv2d(infilters, outfilters, kernel_size=kernel_size, stride=1,
                            padding=padding, norm_type=norm_type, nonlinear_type=nonlinear_type)

    def forward(self, x):
        x = self.up(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        x = self.conv1(x)
        return x

class ResnetBlock(nn.Module):
    '''
    Reimplementation of the basic resnet block for convenience
    '''
    def __init__(self, in_filters, out_filters, kernel_size=3, stride=1, padding=1,
                 norm_type='instance', nonlinear_type='LeakyReLU'):
        super(ResnetBlock, self).__init__()
        self.conv1 = Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride,
                            padding=padding, norm_type=norm_type, nonlinear_type=nonlinear_type)
        self.conv2 = Conv2d(out_filters, out_filters, kernel_size=kernel_size, stride=1,
                            padding=1, norm_type=norm_type, nonlinear_type=None)
        self.downsample = Conv2d(in_filters, out_filters, kernel_size=1, stride=stride,
                                 padding=0, norm_type=norm_type, nonlinear_type=None)
        self.nonlinear = get_nonlinear_layer(nonlinear_type)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.nonlinear(out)
        return out
