# -*- coding: utf-8 -*-
# @Time    : 2021/6/30 19:30
# @Author  : Moming.Yang
# @File    : onset_check.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import torch
from utils import default, utils
from validate import parse_args
from validate import predictor_onset
import librosa
import mir_eval
import pandas as pd
import matplotlib
import math
i = 0

matplotlib.use('Agg')

import matplotlib.pyplot as plt
def config_model():
    args = parse_args()
    cfg = default.update_cfg(default._C, args)
    model = utils.get_model(cfg, model_name=args.model, is_training=False)
    if args.model != "old_onsetnet":
        model = torch.nn.DataParallel(model)

    state_dict = torch.load(args.model_path)
    # for x, y in state_dict.items():
    #     print(x, y.size())
    # exit(11)
    model.load_state_dict(state_dict)

def visualize(id:str, wav_file:str, onset_file:str):
    args = parse_args()
    cfg = default.update_cfg(default._C, args)
    # logger_file = os.path.join(args.model_path)
    predictor = predictor_onset(args.model_path, cfg, model_name=args.model)
    predictor.predict(wav_file)
    onset_frame = predictor.onset_frame
    y, sr = librosa.load(wav_file, sr=44100)
    # stft_y = librosa.core.stft(y, hop_length=512, win_length=2048)
    cqt = librosa.core.cqt(y, bins_per_octave=36, n_bins=288, sr=sr, fmin=27.5)
    # amp_y = librosa.amplitude_to_db(np.abs(stft_y), ref=np.max)
    amp_y = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    onsets = np.round(np.loadtxt(onset_file)[:, 0] * 44100).astype(np.int)

    plt.figure(figsize=(20, 8))
    plt.subplot(211)
    plt.pcolormesh(amp_y)
    for x in onset_frame:
        plt.axvline(x, c='red', ls="--")
    plt.subplot(212)
    plt.plot(y)
    plt.xlim([0, len(y)])
    for x in onsets:
        plt.axvline(x, c='red', ls="--")
    # fig = plt.gcf()
    plt.savefig("./images/" + id + "_" + args.model + ".png")
    # plt.show()

def get_pitch_contour(wav_file:str, pYin_res_path :str):
    '''

    :param wav_file:
    :param pYin_res_path:
    :return: pyin的音高估计
    '''
    y, sr = librosa.load(wav_file, sr=44100)
    n_frames = math.ceil(len(y) / 256)
    frames = np.zeros((n_frames))
    pyin_pd = pd.read_csv(pYin_res_path, names=[0, 1], index_col=None)
    time = pyin_pd.iloc[:, 0].values
    f0 = pyin_pd.iloc[:, 1].values
    frame_idx = np.round(time * 44100 / 256).astype(np.int)
    frames[frame_idx] = f0
    frames = frames[::2]
    # f0_save_path = os.path.splitext(wav_file)[0] + '.pyin.txt'
    # np.savetxt(f0_save_path, frames, fmt="%.5f")
    return frames

def statistic_lack_and_multi(id:str, wav_file:str, onset_file:str, pitch_file:str):
    global lack_onset_sum
    global noisy_multi_onset
    global same_note_onset
    global  predictor_onset
    predictor.predict(wav_file)
    onset_time = predictor.onset_time
    est_interval = np.concatenate([np.expand_dims(onset_time, axis=1), np.expand_dims(onset_time + 0.3, axis=1)], axis=1)
    y, sr = librosa.load(wav_file, sr=44100)
    # stft_y = librosa.core.stft(y, hop_length=512, win_length=2048)
    cqt = librosa.core.cqt(y, bins_per_octave=36, n_bins=288, sr=sr, fmin=27.5)
    # amp_y = librosa.amplitude_to_db(np.abs(stft_y), ref=np.max)
    amp_y = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    ref_interval = np.loadtxt(onset_file)[:, :2]

    pitches = get_pitch_contour(wav_file, pitch_file)
    matching = mir_eval.transcription.match_note_onsets(ref_interval, est_interval, onset_tolerance=0.05)
    lack_num =  len(ref_interval) - len(matching)
    multi_num = len(est_interval) - len(matching)
    matching = np.array(matching)

    index = matching[:, 1]
    est_onset_index = np.arange(len(est_interval))
    multi_onset_index = np.setdiff1d(est_onset_index, index)
    multi_onset = np.round(onset_time[multi_onset_index] * 44100 / 512).astype(np.int)
    noisy_multi = 0
    for onset in multi_onset:
        if pitches[onset] == 0:
            noisy_multi += 1
            print("[{}]\tnoisy onset:{}".format(id, onset))
    same_note_multi_onset = len(multi_onset_index) - noisy_multi
    print("[{}]\tlack_onset_num:{}\tnoisy_onset_num:{}\tsame_note_onset_num:{}".format(id, lack_num, noisy_multi, same_note_multi_onset))
    lack_onset_sum += lack_num
    noisy_multi_onset += noisy_multi
    same_note_onset += same_note_multi_onset

lack_onset_sum = 0
noisy_multi_onset = 0
same_note_onset = 0
args = parse_args()
cfg = default.update_cfg(default._C, args)
# logger_file = os.path.join(args.model_path)
predictor = predictor_onset(args.model_path, cfg, model_name=args.model)
if __name__ == '__main__':
    import os
    path = "/home/yangweiming/MUSIC/onset/ywmfiles/Pyin_127_data/"
    # path = "/home/yangweiming/MUSIC/onset/ywmfiles/Korea/"
    # path = "/home/yangweiming/MUSIC/onset/ywmfiles/ismir2014_dataset/"
    # pyin_path = "/home/yangweiming/MUSIC/onset/ywmfiles/Pyin_127_data/"
    # pyin_path = "/home/yangweiming/MUSIC/onset/ywmfiles/Korea/"
    # pyin_path = "/home/yangweiming/MUSIC/onset/ywmfiles/ismir2014_dataset/"
    pyin_path = "/home/yangweiming/MUSIC/onset/ywmfiles/Pyin_127_data/"
    dirs = os.listdir(path)
    for id in dirs:
        wav_file = os.path.join(path, id, id + '.mp3')
        onset_file = os.path.join(path, id, id + '.txt')
        pyin_file = os.path.join(pyin_path, id, id + '.pyin.csv')
        # visualize(id, wav_file, onset_file)
        statistic_lack_and_multi(id, wav_file, onset_file, pyin_file)
    print("[总共{}]\tlack_onset_num:{}\tnoisy_onset_num:{}\tsame_note_onset_num:{}".format(path, lack_onset_sum, noisy_multi_onset, same_note_onset))