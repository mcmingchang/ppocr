from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
from ocr.modeling.backbones.rec_svtrnet import Block, ConvBNLayer


class Im2Seq(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == 1
        x = x.squeeze(2)
        x = x.permute([0, 2, 1])  # (NTC)(batch, width, channels)
        return x


class EncoderWithRNN(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithRNN, self).__init__()
        self.out_channels = hidden_size * 2
        self.lstm = nn.LSTM(
            in_channels, hidden_size, direction='bidirectional', num_layers=2)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x


class EncoderWithFC(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithFC, self).__init__()
        self.out_channels = hidden_size
        self.fc = nn.Linear(in_channels, hidden_size)

    def forward(self, x):
        x = self.fc(x)
        return x


class EncoderWithSVTR(nn.Module):
    def __init__(
            self,
            in_channels,
            dims=64,  # XS
            depth=2,
            hidden_dims=120,
            use_guide=False,
            num_heads=8,
            qkv_bias=True,
            mlp_ratio=2.0,
            drop_rate=0.1,
            attn_drop_rate=0.1,
            drop_path=0.,
            qk_scale=None):
        super(EncoderWithSVTR, self).__init__()
        self.depth = depth
        self.use_guide = use_guide
        self.conv1 = ConvBNLayer(
            in_channels, in_channels // 8, padding=1, act=nn.SiLU)
        self.conv2 = ConvBNLayer(
            in_channels // 8, hidden_dims, kernel_size=1, act=nn.SiLU)

        self.svtr_block = nn.ModuleList([
            Block(
                dim=hidden_dims,
                num_heads=num_heads,
                mixer='Global',
                HW=None,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=nn.SiLU,
                attn_drop=attn_drop_rate,
                drop_path=drop_path,
                norm_layer='nn.LayerNorm',
                epsilon=1e-05,
                prenorm=False) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dims, eps=1e-6)
        self.conv3 = ConvBNLayer(
            hidden_dims, in_channels, kernel_size=1, act=nn.SiLU)
        # last conv-nxn, the input is concat of input tensor and conv3 output tensor
        self.conv4 = ConvBNLayer(
            2 * in_channels, in_channels // 8, padding=1, act=nn.SiLU)

        self.conv1x1 = ConvBNLayer(
            in_channels // 8, dims, kernel_size=1, act=nn.SiLU)
        self.out_channels = dims
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.2)
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.one_()

    def forward(self, x):
        # for use guide
        if self.use_guide:
            z = x.clone()
            z.stop_gradient = True
        else:
            z = x
        # for short cut
        h = z
        # reduce dim
        z = self.conv1(z)
        z = self.conv2(z)
        # SVTR global block
        B, C, H, W = z.shape
        z = z.flatten(2).permute(0, 2, 1)
        for blk in self.svtr_block:
            z = blk(z)
        z = self.norm(z)
        # last stage
        z = z.reshape([0, H, W, C]).permute(0, 3, 1, 2)
        z = self.conv3(z)
        z = torch.concat((h, z), dim=1)
        z = self.conv1x1(self.conv4(z))
        return z


class SequenceEncoder(nn.Module):
    def __init__(self, in_channels, encoder_type, hidden_size=48, **kwargs):
        super(SequenceEncoder, self).__init__()
        self.encoder_reshape = Im2Seq(in_channels)
        self.out_channels = self.encoder_reshape.out_channels
        self.encoder_type = encoder_type
        if encoder_type == 'reshape':
            self.only_reshape = True
        else:
            support_encoder_dict = {
                'reshape': Im2Seq,
                'fc': EncoderWithFC,
                'rnn': EncoderWithRNN,
                'svtr': EncoderWithSVTR
            }
            assert encoder_type in support_encoder_dict, '{} must in {}'.format(
                encoder_type, support_encoder_dict.keys())
            if encoder_type == "svtr":
                self.encoder = support_encoder_dict[encoder_type](
                    self.encoder_reshape.out_channels, **kwargs)
            else:
                self.encoder = support_encoder_dict[encoder_type](
                    self.encoder_reshape.out_channels, hidden_size)
            self.out_channels = self.encoder.out_channels
            self.only_reshape = False

    def forward(self, x):
        if self.encoder_type != 'svtr':
            x = self.encoder_reshape(x)
            if not self.only_reshape:
                x = self.encoder(x)
            return x
        else:
            x = self.encoder(x)
            x = self.encoder_reshape(x)
            return x
