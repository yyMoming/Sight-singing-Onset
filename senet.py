# -*- coding: utf-8 -*-
# @Time    : 2020/9/23 16:27
# @Author  : moMing.yang
# @File    : senet.py
# @Software: PyCharm

import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from collections import OrderedDict
import os
# onset_net_cfg = {
#     'cqt_pad_4': {'conv1': (7, 3), 'pool1': (3, 2), 'conv2': (4, 3), 'pool2': (3, 1), 'fc1': 1176},
# }
onset_net_cfg = {
    'cqt_pad_3': {'conv1': (25, 3), 'pool1': (3, 2), 'conv2': (7, 2), 'pool2': (3, 1), 'fc1': 1050},
    'cqt_pad_4': {'conv1': (25, 3), 'pool1': (3, 2), 'conv2': (7, 3), 'pool2': (3, 1), 'fc1': 1050},
    'cqt_pad_5': {'conv1': (25, 3), 'pool1': (3, 3), 'conv2': (7, 3), 'pool2': (3, 1), 'fc1': 1050},
    'cqt_pad_7': {'conv1': (25, 7), 'pool1': (3, 3), 'conv2': (7, 3), 'pool2': (3, 1), 'fc1': 1050},}

fc1 = {
    '87': 42000,
    '9': 1050
}

class bilstmOnsetnet(nn.Module):
    def __init__(self,
                 pad_length=4,
                 out_size=256,
                 spec_style='cqt',
                 dual_channel=False):
        super(bilstmOnsetnet, self).__init__()
        nchannel = 2 if dual_channel else 1  # 是否双通道
        self.config = onset_net_cfg['{}_pad_{}'.format(spec_style, pad_length)]  # 选择卷积网络

        self.features = nn.Sequential(
            nn.Conv2d(nchannel, 21, kernel_size=self.config['conv1']),
            nn.BatchNorm2d(21),  # 归一化权重
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=self.config['pool1'], stride=self.config['pool1']),
            nn.Conv2d(21, 42, kernel_size=self.config['conv2']),
            nn.BatchNorm2d(42),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=self.config['pool2'], stride=self.config['pool2'])
        )
        self.FC1 = nn.Linear(self.config['fc1'], 512)
        self.FC2 = nn.Linear(512, out_size)
        self.drop = nn.Dropout()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.config['fc1'])
        x = self.drop(x)
        x = F.relu(self.FC1(x))
        x = self.drop(x)
        x = self.FC2(x)
        return torch.sigmoid(x)
class onsetnet(nn.Module):
    """docstring for onsetnet"""

    def __init__(self,
                 pad_length=4,
                 out_size=256,
                 spec_style='cqt',
                 dual_channel=False):
        super(onsetnet, self).__init__()
        nchannel = 2 if dual_channel else 1  # 是否双通道
        self.config = onset_net_cfg['{}_pad_{}'.format(spec_style, pad_length)]  # 选择卷积网络

        self.features = nn.Sequential(
            nn.Conv2d(nchannel, 21, kernel_size=self.config['conv1']),
            nn.BatchNorm2d(21),  # 归一化权重
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=self.config['pool1'], stride=self.config['pool1']),
            nn.Conv2d(21, 42, kernel_size=self.config['conv2']),
            nn.BatchNorm2d(42),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=self.config['pool2'], stride=self.config['pool2'])
        )
        self.fc1 = nn.Linear(self.config['fc1'], 256)
        self.fc2 = nn.Linear(256, out_size)
        self.drop = nn.Dropout()
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.config['fc1'])
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return torch.sigmoid(x) #py3.7中改为torch.sigmoid


'''
	尝试在原网络上加入S-E
	调换一下BN和ReLU的位置
'''
class SeLayer(nn.Module):
	def __init__(self, channel, reduction=16):
		super(SeLayer, self).__init__()
		self.avg = nn.AdaptiveAvgPool2d(1)
		self.features = nn.Sequential(
			nn.Linear(channel, math.ceil(channel / reduction), bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(math.ceil(channel / reduction), channel, bias=False),
			nn.Sigmoid()
		)
	def forward(self, x):
		b, c, _, _ = x.size()
		y = self.avg(x).view(b, c)
		y = self.features(y).view(b, c, 1, 1)
		return x * y.expand_as(x)

'''
	CBAM module
'''
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM_onsetnet(nn.Module):
    def __init__(self, out_size=256, pad_length=4, spec_style='cqt', dual_channel=False):
        super(CBAM_onsetnet, self).__init__()
        nchannel = 2 if dual_channel else 1  # 是否双通道
        self.config = onset_net_cfg['{}_pad_{}'.format(spec_style, pad_length)]
        # self.conv1 = nn.Sequential(OrderedDict([
        #     ('conv1', nn.Conv2d(nchannel, 21, kernel_size=self.config['conv1'])),
        #     ('bn1', nn.BatchNorm2d(21)),
        #     ('relu1', nn.ReLU(inplace=True)),
        #     ('maxpool1', nn.MaxPool2d(kernel_size=self.config['pool1'], stride=self.config['pool1']))
        # ]))
        # self.conv2 = nn.Sequential(OrderedDict([
        #     ('conv2', nn.Conv2d(21, 42, kernel_size=self.config['conv2'])),
        #     ('bn2', nn.BatchNorm2d(42)),
        #     ('relu2', nn.ReLU(inplace=True)),
        #     ('maxpool2', nn.MaxPool2d(kernel_size=self.config['pool2'], stride=self.config['pool2']))
        #
        # ]))
        self.conv1 = nn.Conv2d(nchannel, 21, kernel_size=self.config['conv1'])
        self.conv2 = nn.Conv2d(21, 42, kernel_size=self.config['conv2'])
        self.ca1 = ChannelAttention(21)
        self.ca2 = ChannelAttention(42)
        self.sa = SpatialAttention()
        self.layer1 = nn.Sequential(OrderedDict([
            ('bn1', nn.BatchNorm2d(21)),
            ('relu1', nn.ReLU(inplace=True)),
            ('maxpool1', nn.MaxPool2d(kernel_size=self.config['pool1'], stride=self.config['pool1']))
        ]))
        self.layer2 = nn.Sequential(OrderedDict([
            ('bn2', nn.BatchNorm2d(42)),
            ('relu2', nn.ReLU(inplace=True)),
            ('maxpool2', nn.MaxPool2d(kernel_size=self.config['pool2'], stride=self.config['pool2']))
        ]))
        self.fc1 = nn.Linear(self.config['fc1'], 256)
        self.fc2 = nn.Linear(256, out_size)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout()

    def forward(self, x):
        conv1_out = self.conv1(x)
        ca_ = self.ca1(conv1_out)
        ca_out = ca_.expand_as(conv1_out) * conv1_out
        sa_ = self.sa(ca_out)
        sa_out = sa_.expand_as(conv1_out) * ca_out
        out = self.layer1(sa_out)

        conv2_out = self.conv2(out)
        ca_ = self.ca2(conv2_out)
        ca_out = ca_.expand_as(conv2_out) * conv2_out
        sa_ = self.sa(ca_out)
        sa_out = sa_.expand_as(conv2_out) * conv2_out
        out = self.layer2(sa_out)

        # out = self.drop(out.view(-1, self.config['fc1']))
        out = self.drop(out)
        out = out.view(out.shape[0], -1)
        try:
            out = self.fc1(out)
        except:
            print("input_size:", out.size(), "fc1:", self.fc1)
            exit(233)
        self.relu(out)
        out = self.drop(out)
        out = self.fc2(out)
        return torch.sigmoid(out)

class SE_onsetnet(nn.Module):
	def __init__(self, pad_length=4, spec_style='cqt', dual_channel=False):
		super(SE_onsetnet, self).__init__()
		nchannel = 2 if dual_channel else 1  # 是否双通道
		self.config = onset_net_cfg['{}_pad_{}'.format(spec_style, pad_length)]  # 选择卷积网络

		self.features = nn.Sequential(
			nn.Conv2d(nchannel, 21, kernel_size=self.config['conv1']),
            SeLayer(21),
			nn.BatchNorm2d(21),  # 归一化权重
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=self.config['pool1'], stride=self.config['pool1']),
			nn.Conv2d(21, 42, kernel_size=self.config['conv2']),
            SeLayer(42),
			nn.BatchNorm2d(42),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=self.config['pool2'], stride=self.config['pool2'])
		)
		self.fc1 = nn.Linear(self.config['fc1'], 256)
		self.fc2 = nn.Linear(256, 1)
		self.drop = nn.Dropout()

	def forward(self, x):
		x = self.features(x)
		# print(x.size())

		x = x.view(-1, self.config['fc1'])
		x = self.drop(x)

		x = F.relu(self.fc1(x))

		x = self.drop(x)
		x = self.fc2(x)
		return torch.sigmoid(x)

class bi_lstm(nn.Module):
    '''
        params:
        embbdings: 是否需要将输入序列中每个元素大小重嵌入
        input_dim：序列中每个元素的大小
        hidden_dim：序列隐藏层中元素的大小
        output_dim：序列输出的每个元素的大小
    '''
    def __init__(self,
                 cfg,
                 input_dim,
                 hidden_dim,
                 num_layers,
                 output_dim,
                 max_len=40,
                 dropout=0.5):
        super(bi_lstm, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.max_len = max_len
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, \
                           dropout=dropout, batch_first=True, bidirectional=True)
        self.output = nn.Linear(2 * self.hidden_dim, self.output_dim) #最后输出的概率是1维
        self.fc = nn.Linear((cfg.MODEL.INPUT_SIZE[0] - 8) * self.output_dim, cfg.MODEL.OUTPUT_SIZE[0])
    def forward(self, input):
        out, _ = self.rnn(input)
        out2 = self.output(out)
        # out2 = out2.squeeze(dim=2)
        out2 = out2.view(out2.shape[0], -1)
        # out_prob = torch.sigmoid(out2.squeeze(dim=2))
        one_frame_prob = self.fc(out2)
        return one_frame_prob

class bi_lstm_cbam_onsetnet(nn.Module):
    def __init__(self, cfg, lstm_input_dim, lstm_hidden_dim, lstm_num_layers, lstm_output_dim, conv_out_size=256,):
        super(bi_lstm_cbam_onsetnet, self).__init__()
        self.lstm = bi_lstm(cfg=cfg, input_dim=lstm_input_dim, hidden_dim=lstm_hidden_dim, num_layers=lstm_num_layers, output_dim=lstm_output_dim)
        self.conv_stack = CBAM_onsetnet(out_size=256, pad_length=4, spec_style='cqt', dual_channel=False)
        self.input_size = cfg.MODEL.INPUT_SIZE
    def seperate_(self, x):
        n_frames = []

        for i in range(87 - 8):
            n_frames.append(x[:, :, :, i: i + 9])
        return torch.cat(n_frames, dim=0)

    def forward(self, x):
        lstm_inputs = []
        for frame in x:
            input = self.seperate_(frame.unsqueeze(dim=0))
            # conv_inputs.append(input.unsqueeze_(dim=1))
            conv_out = self.conv_stack(input).squeeze(dim=1).squeeze(dim=1)
            lstm_inputs.append(conv_out.unsqueeze(dim=0))
        inputs = torch.cat(lstm_inputs, dim=0)

        lstm_out = self.lstm(inputs)
        return lstm_out

class bi_lstm_onsetnet(nn.Module):
    def __init__(self,
                 cfg,
                 lstm_input_dim,
                 lstm_hidden_dim,
                 lstm_num_layers,
                 lstm_output_dim,
                 pre_trained_onsetnet=None,
                 conv_out_size=256,):
        super(bi_lstm_onsetnet, self).__init__()
        self.lstm = bi_lstm(cfg=cfg, input_dim=lstm_input_dim, hidden_dim=lstm_hidden_dim, num_layers=lstm_num_layers, output_dim=lstm_output_dim)
        self.conv_stack = bilstmOnsetnet(out_size=conv_out_size, pad_length=4, spec_style='cqt', dual_channel=False)
        self.input_size = cfg.MODEL.INPUT_SIZE
        self.fc = nn.Linear(256, 1) # 4-27尝试取消LSTM网络，进行13帧-》5帧训练
        if pre_trained_onsetnet is not None:
            assert isinstance(pre_trained_onsetnet, str)
            save_model_dict = torch.load(pre_trained_onsetnet)
            onsetnet_dict = self.conv_stack.state_dict()
            state_dict = {k: v for k, v in save_model_dict.items() if k in onsetnet_dict}
            onsetnet_dict.update(state_dict)
            self.conv_stack.load_state_dict(onsetnet_dict)
            # self.fc.load_state_dict(onsetnet_dict.fc1)

    def seperate_(self, x):
        n_frames = []

        for i in range(self.input_size[0] - 8):
            n_frames.append(x[:, :, :, i: i + 9])
        return torch.cat(n_frames, dim=0)

    def forward(self, x):
        lstm_inputs = []
        for frame in x:
            input = self.seperate_(frame.unsqueeze(dim=0))
            # conv_inputs.append(input.unsqueeze_(dim=1))
            conv_out = self.conv_stack(input).squeeze(dim=1).squeeze(dim=1)
            # fc_out = self.fc(conv_out)
            # fc_out = torch.sigmoid(fc_out)
            lstm_inputs.append(conv_out.unsqueeze(dim=0))
        inputs = torch.cat(lstm_inputs, dim=0)
        lstm_out = self.lstm(inputs)
        return torch.sigmoid(lstm_out)
        # return inputs
def temp():
    import logging
    logger = logging.getLogger(__name__)
    logger.info(" I am from senet.py named temp function")


if __name__ == '__main__':
    from train import parse_args
    from utils import default, utils
    from torchsummaryX import summary
    args = parse_args()
    cfg = default.update_cfg(default._C, args)
    # model = onsetnet(pad_length=4,
    #                      out_size=1,
    #                      spec_style='cqt',
    #                      dual_channel=False)
    model = utils.get_model(cfg, is_training=False, model_name=args.model_type)
    # model = nn.DataParallel(model)
    # model.load_state_dict(torch.load("/home/data/ywm_data/bi_lstm_onsetnet_output/output/fold_0/bilstm/bi_lstm_87×267_adam_lr1e-5.fold_0/2021-06-29-12-51/best.pth"))
    # torch.save(model, "/home/data/ywm_data/Models/87to1.pth")
    summary(model.to('cuda'), torch.randn(1, 1, 87, 267).to('cuda'))
    # model = bi_lstm_onsetnet(cfg, lstm_input_dim=256, lstm_hidden_dim=512, lstm_num_layers=2, lstm_output_dim=128)
    # # load_state = torch.load("/home/yangweiming/MUSIC/onset/onset_train/output/fold_0/bilstm/bi_lstm_87×267_adam_lr1e-6.fold_0/2021-05-01-10-57/epoch_55.pth")
    # # model.load_state_dict(load_state)
    # input = torch.randn(256, 1, 267, 87)
    # # summary(model, input)
    # state_dict = torch.load("/home/yangweiming/MUSIC/onset/onset_train/output/fold_0/bilstm/bi_lstm_87×267_adam_lr1e-7.fold_0/2021-05-07-19-00/epoch_55.pth")
    # state_dict = {x.replace("module.", ""): v for x, v in state_dict.items()}
    # model.load_state_dict(state_dict)
    # out = model(input)
    # print(out.size())
    # # for name, param in model.state_dict().items():
    # #     print(name, ":\t", param.size())
    # # summary(model, torch.randn(1, 1, 267, 87).cuda())



