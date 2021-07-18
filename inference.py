# -*- coding: utf-8 -*-
# @Time    : 2021/4/21 11:13
# @Author  : Moming.Yang
# @File    : inference.py

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
import sys
sys.path.append(os.path.join(os.getcwd(), 'lib'))
import yaml
import argparse
from utils import default, utils
import warnings
import numpy as np
import torch
import logging
from dataset.rnn_dataset import rnn_dataset
from senet import bi_lstm_onsetnet
from tensorboardX import SummaryWriter
warnings.filterwarnings('ignore')
from yacs.config import CfgNode as CN
import matplotlib.pyplot as plt
import librosa
from train import parse_args

def test(cfg: CN, model: torch.nn.DataParallel, wav_file: str, anno:str):
    # files = (wav_file, anno)
    y, sr = librosa.load(wav_file[0], sr=44100)
    y_cqt = librosa.core.cqt(y, sr=sr, bins_per_octave=36, n_bins=288, fmin=librosa.note_to_hz('A0'))
    y_cqt = librosa.amplitude_to_db(np.abs(y_cqt), ref=np.max)
    data = rnn_dataset(cfg, wav_file, anno, is_training=False)
    n_frames = len(data)
    output_length = cfg.MODEL.OUTPUT_SIZE[0]
    pad_length = cfg.MODEL.OUTPUT_SIZE[0] // 2
    rnn_probs = np.zeros(n_frames + pad_length * 2)
    data2, spec = utils.data_proc(wav_file[0], cfg)
    plt.figure(figsize=(12, 8))
    plt.subplot(311)
    plt.pcolormesh(data[0][0][0], cmap='jet')
    plt.subplot(312)
    plt.pcolormesh(data2[0], cmap='jet')
    plt.subplot(313)
    plt.pcolormesh(spec[:, :87], cmap='jet')
    plt.show()
    # model.eval()
    # for i in range(n_frames):
    #     x = torch.from_numpy(np.expand_dims(data[i][0], axis=0)).float().cuda()
    #     out = model(x)
    #     rnn_probs[i: i + output_length] += out.squeeze(dim=0).detach().cpu().numpy()
    # rnn_probs = rnn_probs[pad_length: -pad_length] / output_length
    # rnn_probs[pad_length: -pad_length] /= output_length
    # for i in range(pad_length):
    #     rnn_probs[i] /= (i + 1)
    #     rnn_probs[-(i + 1)] /= (i + 1)
    # plt.figure(figsize=(20, 8))
    # plt.subplot(211)
    # plt.pcolormesh(y_cqt, cmap='gist_gray')
    # plt.subplot(212)
    # plt.plot(rnn_probs)
    #
    # plt.show()

def main():
    args = parse_args()
    cfg = default.update_cfg(default._C, args)
    model = bi_lstm_onsetnet(lstm_input_dim=256, lstm_hidden_dim=512, lstm_num_layers=2, lstm_output_dim=128)

    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load("/home/yangweiming/MUSIC/onset/onset_train/output/fold_0/bilstm/bi_lstm_87×267_adam_lr1e-7.fold_0/best.pth"))
    model = model.cuda()
    test(cfg, model, ["/home/data/ywm_data/train_data/shuffle_data/sep_dataset/fold_0/valid/1087.wav"], \
            ["/home/data/ywm_data/train_data/shuffle_data/sep_dataset/fold_0/valid/1087.txt"])

if __name__ == '__main__':
    main()