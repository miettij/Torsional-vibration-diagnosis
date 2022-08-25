import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.functional import relu

class ConvLayer(nn.Module):
    def __init__(self,
                in_channels, out_channels,
                kernel_size=3, stride=1,
                padding=1, dilation = 1, use_bn = True, bias = False):
        super(ConvLayer, self).__init__()

        padding = int(dilation*(kernel_size-1)/2)

        self.conv = nn.Conv1d(in_channels,
                             out_channels,
                             kernel_size=kernel_size,
                             dilation = dilation,
                             stride=stride,
                             padding=padding,
                             bias=bias)
        self.use_bn = use_bn
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):

        out = self.conv(x)
        if self.use_bn:
            out = self.bn(out)

        return out

class ResBlock(nn.Module):
    def __init__(self,in_channels, out_channels,
                kernel_size = 64, dilation=1, padding = 1, stride = 2, bias = False):
        super(ResBlock,self).__init__()
        self.proposal_gate = ConvLayer(in_channels = in_channels,
                                        out_channels = out_channels,
                                        kernel_size = kernel_size,
                                        dilation = dilation,
                                        stride = stride,
                                        padding = padding,
                                        bias = bias)

        self.control_gate = ConvLayer(in_channels = in_channels,
                                        out_channels = out_channels,
                                        kernel_size = kernel_size,
                                        dilation = dilation,
                                        stride = stride,
                                        padding = padding,
                                        bias = bias)

        # Tähän tulee batchnorm
        self.residual_connection = ConvLayer(in_channels = in_channels,
                                        out_channels = out_channels,
                                        kernel_size = 1,
                                        dilation = dilation,
                                        stride = stride,
                                        padding = padding,
                                        use_bn = False,
                                        bias = bias)

    def forward(self,x):
        prop = torch.tanh(self.proposal_gate(x))
        contrl = torch.sigmoid(self.control_gate(x))
        residual = self.residual_connection(x)

        out = prop*contrl

        return out+residual

class SRDCNN(nn.Module):
    def __init__(self, in_channels, n_classes, args):
        super(SRDCNN,self).__init__()

        self.model_type = 'SRDCNN'

        self.layer1 = ResBlock(in_channels,
                                out_channels = 32,
                                dilation = 1,
                                kernel_size = 64,
                                padding = int(1*(64-1)/2),
                                bias = args.bias)

        self.layer2 = ResBlock(in_channels = 32,
                                out_channels = 32,
                                dilation = 2,
                                kernel_size = 32,
                                padding = int(2*(32-1)/2),
                                bias = args.bias)

        self.layer3 = ResBlock(in_channels = 32,
                                out_channels = 64,
                                dilation = 4,
                                kernel_size = 16,
                                padding = int(4*(16-1)/2),
                                bias = args.bias)

        self.layer4 = ResBlock(in_channels = 64,
                                out_channels = 64,
                                dilation = 8,
                                kernel_size = 8,
                                padding = int(8*(8-1)/2),
                                bias = args.bias)

        self.layer5 = ResBlock(in_channels = 64,
                                out_channels = 64,
                                dilation = 16,
                                kernel_size = 4,
                                padding = int(16*(4-1)/2),
                                bias = args.bias)

        self.n_features = 64*64

        self.fc1 = nn.Linear(self.n_features,100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100,n_classes)
        self.bn2 = nn.BatchNorm1d(n_classes)

    def reset_parameters(self,layer):
        nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        if layer.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(layer.bias, -bound, bound)

    def forward(self,x, verbose = False):
        out = self.layer1(x)
        if verbose:
            print("Shape of out1: ", out.shape)
        out = self.layer2(out)
        if verbose:
            print("Shape of out2: ", out.shape)
        out = self.layer3(out)
        if verbose:
            print("Shape of out3: ", out.shape)
        out = self.layer4(out)
        if verbose:
            print("Shape of out4: ", out.shape)
        out = self.layer5(out)
        if verbose:
            print("Shape of out5: ", out.shape)
        out = out.view(-1,self.n_features).contiguous()
        if verbose:
            print("Shape after reshape: ",out.shape)
        out = self.bn1(self.fc1(out))
        if verbose:
            print("Shape after f1: ",out.shape)
        out = self.bn2(self.fc2(out))
        if verbose:
            print("Shape after f2: ",out.shape)

        return out
