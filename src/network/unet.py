import logging
from torch import nn
import torch.nn.functional as F
import torch

logger = logging.getLogger(__name__)


class DoubleConv(nn.Module):
    """
    Double Convolution module consisting of 2 convolution layers.
    The output of each convolution layer is passed to a ReLU layer, followed by a
    a batch normalization layer.
    """

    def __init__(self, in_channels, out_channels, dropout=True):
        super(DoubleConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if dropout:
            self.dropout = nn.Dropout2d(0.25)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        """ Forward pass
        """

        x = self.bn1(F.relu(self.conv1(x)))

        x = self.bn2(F.relu(self.conv2(x)))

        if self.dropout is not None:
            x = self.dropout(x)
        return x


class Encoder(nn.Module):
    """
    The Encoder Module calls the double convolution module followed by the MaxPool
    module. The module returns two tensor: double conv output and pooling output.
    The double conv output is passed to the decoder side and the pooling output is
    passed to the next encoder module that operates at a different spatial level.
    """

    def __init__(self, in_channels, out_channels, pooling, dropout=True):
        super(Encoder, self).__init__()

        self.double_conv = DoubleConv(in_channels, out_channels, dropout)

        if pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = nn.Identity()

    def forward(self, x):
        x = self.double_conv(x)

        double_conv_out = x
        x = self.pool(x)
        return x, double_conv_out


class Decoder(nn.Module):
    """
    Decoder module of the UNet. The module follows the following sequence
    1. Upconvolution is performed on output from previous layer.
    2. This upconv output is concatented with output from encoder of same spatial resolution
    3. This output is passed to the double conv module as input.
    """

    def __init__(self, in_channels, out_channels, dropout=True):
        super(Decoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.upconv = nn.ConvTranspose2d(self.in_channels,
                                         self.out_channels,
                                         kernel_size=2,
                                         stride=2)

        self.double_conv = DoubleConv(in_channels=out_channels * 2,
                                      out_channels=out_channels,
                                      dropout=dropout)

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        diff = from_down.size()[3] - from_up.size()[3]
        if diff:
            half_pad = diff // 2
            from_up = F.pad(from_up,
                            [half_pad, diff - half_pad, half_pad, diff - half_pad])

        x = torch.cat((from_up, from_down), 1)

        x = self.double_conv(x)
        return x


class UNet(nn.Module):
    """
    UNet class implementation
    """

    def __init__(self, num_classes, in_channels=3, depth=6,
                 start_filts=32, dropout=True):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the
                first conv.
        """
        super(UNet, self).__init__()

        logger.info("Model Dropout flag: {}".format(dropout))




        # Initialize encoder modules
        self.encoders = []
        for i in range(depth):
            module_in_channels = in_channels if i == 0 else module_out_channels
            module_out_channels = start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False

            down_conv = Encoder(module_in_channels, module_out_channels,
                                pooling=pooling, dropout=dropout)

            self.encoders.append(down_conv)
        self.encoders = nn.ModuleList(self.encoders)

        # Initialize decoder modules
        self.decoders = []
        for i in range(depth - 1):
            module_in_channels = module_out_channels
            module_out_channels = module_in_channels // 2
            # dropout = False if i == (depth - 2) else True
            up_conv = Decoder(module_in_channels, module_out_channels,
                              dropout=dropout)
            self.decoders.append(up_conv)
        self.decoders = nn.ModuleList(self.decoders)

        self.conv_final = nn.Conv2d(module_out_channels,
                                    num_classes,
                                    kernel_size=1,
                                    stride=1)

        self.initialize_cnn_weights()

    def initialize_cnn_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Xavier initialization for CNN weights
                nn.init.xavier_normal_(module.weight)
                # 0 initialization for bias
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        encoder_double_conv_outs = []

        for i, encoder in enumerate(self.encoders):
            x, double_conv_out = encoder(x)
            encoder_double_conv_outs.append(double_conv_out)

        # Remove the last term and reverse list containing encoder outputs at each level
        encoder_double_conv_outs = encoder_double_conv_outs[-2::-1]
        for decoder, double_conv_out in zip(self.decoders, encoder_double_conv_outs):
            x = decoder(double_conv_out, x)

        x = self.conv_final(x)
        return x
