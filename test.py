'''
 @Time : 2021/1/19 6:07 下午 
 @Author : Moming_Y
 @File : test.py 
 @Software: PyCharm
'''
import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'lib'))
import yaml
import argparse
from utils import default, utils
from pathlib import Path
import time
import warnings
import torch
import logging
from dataset.rnn_dataset import rnn_dataset
from torch.utils.data.dataloader import DataLoader
from senet import bi_lstm_onsetnet, bi_lstm_cbam_onsetnet
from tensorboardX import SummaryWriter
warnings.filterwarnings('ignore')
from lib.core.functions import train, validate
from lib.core.losses import Cross_entropy
import numpy as np
# from train import parse_args
from validate import parse_args
from senet import onsetnet
import matplotlib.pyplot as plt
import pylab
from torchsummaryX import summary
import multiprocessing
import mir_eval
import numpy as np
import pandas as pd
def main():
    args = parse_args()
    cfg = default.update_cfg(default._C, args)
    if cfg.DATASET.DIR:
        train_data_path = Path(cfg.DATASET.ROOT) / cfg.DATASET.DIR / cfg.DATASET.TRAIN_SET
        valid_data_path = Path(cfg.DATASET.ROOT) / cfg.DATASET.DIR / cfg.DATASET.VALID_SET
    else:
        train_data_path = Path(cfg.DATASET.ROOT) / cfg.DATASET.TRAIN_SET
        valid_data_path = Path(cfg.DATASET.ROOT) / cfg.DATASET.VALID_SET
    train_files = utils.get_dataset_filename(train_data_path)

    test_files = utils.get_dataset_filename(valid_data_path)
    start_time = time.time()
    train_dataset = rnn_dataset(cfg, *train_files, is_training=False)
    print("train_dataset_pos:", len(train_dataset.data_pos), "train_dataset_neg:", len(train_dataset.data_neg))
    test_dataset = rnn_dataset(cfg, *test_files, is_training=True)
    print("train_dataset_valid_pos:", len(test_dataset.data_pos), "train_dataset_valid_neg:", len(test_dataset.data_neg))

    scatters = {}
    count = 0
    for x, y in train_dataset:
        one_num = (y == 1).sum()
        zero_num = y.size - one_num
        count += 1
        if one_num not in scatters:
            scatters[one_num] = 1
        else:
            scatters[one_num] += 1
    scatters = {x: y / count for x, y in scatters.items()}
    print(scatters)

    scatters = {}
    count = 0
    for x, y in test_dataset:
        one_num = (y == 1).sum()
        zero_num = y.size - one_num
        count += 1
        if one_num not in scatters:
            scatters[one_num] = 1
        else:
            scatters[one_num] += 1
    scatters = {x: y / count for x, y in scatters.items()}
    print(scatters)

def test():
    model = onsetnet(out_size=1)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load("/home/yangweiming/MUSIC/onset/onset_train/output/fold_0/bilstm/bi_lstm_9×267_adam_lr1e-3.fold_0/2021-04-29-16-21/best.pth"))
    print(model)
    print(summary(model, torch.randn(1, 1, 267, 9)))

def check_intervals(root_path):
    ids = os.listdir(root_path)
    diff = np.array([[1, 1, 1]])
    for id in ids:
        print(id)
        label_file = os.path.join(root_path, id, id + '.txt')
        est_file = os.path.join("/home/yangweiming/MUSIC/onset/onset_train/pred/", id + '_est.txt')
        label_onset = np.loadtxt(label_file)
        est_onset = np.expand_dims(np.loadtxt(est_file), axis=1)
        label_onset_offset = label_onset[:, :2]
        est_onset_offset = np.concatenate([est_onset, est_onset + 0.03], axis=1)
        print("\tonset_precision:", mir_eval.transcription.onset_precision_recall_f1(label_onset_offset, est_onset_offset, onset_tolerance=0.2))
        match_idxs = mir_eval.transcription.match_note_onsets(label_onset_offset, est_onset_offset, onset_tolerance=0.2)
        match_idxs = np.asarray(match_idxs)

        match_est_onsets = np.expand_dims(est_onset_offset[:, 0][match_idxs[:, 1]], axis=1)
        match_label_onsets = np.expand_dims(label_onset_offset[:, 0][match_idxs[:, 0]], axis=1)
        onset_diff = np.abs(match_est_onsets - match_label_onsets)
        match_result = np.concatenate([match_label_onsets, match_est_onsets, onset_diff], axis=1)
        diff = np.append(diff, match_result, axis=0)
        # print(1)
        # break
    diff = diff[1:]
    diff_pd = pd.DataFrame(diff, columns=["label_onset", "pred_onset", "diff"])
    diff_pd.to_csv("onset_align_0.2_.csv")
    # np.savetxt("onset_align.txt", diff, fmt='%.4f', delimiter='\t')

# def pth_to_onnx():
#     net = bi_lstm_cbam_onsetnet()
if __name__ == '__main__':
    from validate import predictor_onset
    # id = "5116"
    # wav_file = os.path.join("/home/yangweiming/MUSIC/onset/ywmfiles/127_data/", id, id + '.mp3')
    # args = parse_args()
    # cfg = default.update_cfg(default._C, args)
    # # logger_file = os.path.join(args.model_path)
    # predictor = predictor_onset(args.model_path, cfg, model_name=args.model)
    # predictor.predict(wav_file)
    # onset_pred = predictor.onset_pred
    # np.savetxt("./{}.onset_pred.txt".format(id), onset_pred, fmt="%.5f")
    # args = parse_args()
    # cfg = default.update_cfg(default._C, args)
    # model = utils.get_model(cfg, model_name=args.model, is_training=False)
    # if args.model != "old_onsetnet":
    #     model = torch.nn.DataParallel(model)
    #
    # state_dict = torch.load(args.model_path)
    # # for x, y in state_dict.items():
    # #     print(x, y.size())
    # # exit(11)
    # model.load_state_dict(state_dict)
    # torch.save(model, '/home/data/ywm_data/Models/' + 'bilstm_onsetnet_87to1.pth')
    # multiprocessing.freeze_support()
    # root_path = "/home/yangweiming/MUSIC/onset/ywmfiles/127_data/"
    # check_intervals(root_path)
