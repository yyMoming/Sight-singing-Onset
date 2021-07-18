# -*- coding: utf-8 -*-
# @Time    : 2021/3/10 14:42
# @Author  : Moming.Yang
# @File    : validate.py

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import sys
sys.path.append(os.path.join(os.getcwd(), 'lib'))
import yaml
from core.functions import AverageMeter
import argparse
from utils import default, utils
import warnings
import torch
from dataset.rnn_dataset import rnn_dataset
from senet import bi_lstm_onsetnet, CBAM_onsetnet, bi_lstm_cbam_onsetnet
from tensorboardX import SummaryWriter
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import numpy as np
import mir_eval
import librosa
import logging
use_cuda = torch.cuda.is_available()
logger_file = ""
def parse_args():
    parser = argparse.ArgumentParser(description='Train onset network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--is_annotated',
                        help='if test dataset is annotated',
                        action='store_true',
                        default=False)
    parser.add_argument('--model',
                        help='choose one model',
                        # type=str,
                        choices=['bi_lstm_onsetnet', 'CBAM_onsetnet', 'onsetnet', 'old_onsetnet', 'bi_lstm_cbam_onsetnet'],
                        )
    parser.add_argument('--model_path',
                        help='choose the path of the model',
                        type=str,
                        required=True)
    parser.add_argument('--modelDir',
                        help='output model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    args = parser.parse_args()

    return args
    # return parser



class predictor_onset(object):
    '''
        model_path: 待验证模型路径
        cfg: CfgNode, 配置信息
    '''
    def __init__(self,
                 model_path,
                 cfg,
                 pad_length=4,
                 spec_style='cqt',
                 model_name='onsetnet',
                 dual_channel=False,
                 having_nn=False
                 ):
        super(predictor_onset, self).__init__()

        self.pad_length = pad_length
        self.spec_style = spec_style
        self.hopsize_t = cfg.SPEC.HOPSIZE / cfg.SPEC.SR
        self.cfg = cfg
        if not having_nn:
            model = utils.get_model(cfg, model_name=model_name, is_training=False)
            if model_name != "old_onsetnet":
                model = torch.nn.DataParallel(model)
            model = torch.nn.DataParallel(model) #临时
            state_dict = torch.load(model_path)
            # for x, y in state_dict.items():
            #     print(x, y.size())
            # exit(11)
            model.load_state_dict(state_dict)
        else:
            model = torch.load(model_path)
        self.model = model
        if use_cuda:
            self.model = self.model.cuda()
        self.model.eval()

    @property
    def onset_frame(self):
        return self._onset_frame

    @property
    def onset_time(self):
        return self._onset_time

    @property
    def onset_prob(self):
        return self._onset_prob

    @property
    def onset_pred(self):
        return self._onset_pred

    def predict(self, wav_file):
        data, spec = utils.data_proc(wav_file, self.cfg)

        data = torch.from_numpy(data).unsqueeze(dim=1).unsqueeze(dim=1)
        padding_frames = self.cfg.MODEL.OUTPUT_SIZE[0] // 2
        pred_probs = np.zeros(data.shape[0] + padding_frames * 2)
        # if data.shape[0] > 100:
        #     predictions = []
        #     for idx in range(0, data.shape[0], 100):
        #         input = data[idx: idx + 100]
        #         if use_cuda:
        #             input = input.cuda()
        #         output = self.model(input)
        #         predictions.append(output)
        #     predictions = torch.cat(predictions, dim=0)
        # else:
        #     if use_cuda:
        #         data = data.cuda()
        #     predictions = self.model(data)
        # predictions = predictions.detach().cpu().numpy()
        # for idx, pred in enumerate(predictions):
        #     pred_probs[idx: idx + self.cfg.MODEL.OUTPUT_SIZE[0]] += predictions[idx]
        for idx, frame in enumerate(data):
            # plt.pcolormesh(frame, cmap="gist_gray")
            # plt.title(idx)
            # plt.show()
            if use_cuda:
                frame = frame.cuda()
            output = self.model(frame)
            output = output.squeeze(dim=0).detach().cpu().numpy()
            pred_probs[idx: idx + self.cfg.MODEL.OUTPUT_SIZE[0]] += output

        if padding_frames:
            final_preds = pred_probs[padding_frames: -padding_frames] / self.cfg.MODEL.OUTPUT_SIZE[0]
        else:
            final_preds = pred_probs
        self._onset_pred = final_preds
        self.post_process(self._onset_pred)
        return spec

    def post_process(self, pred=None):
        if pred is None:
            pred = self._onset_pred
        self._onset_prob = np.where(pred > self.cfg.TEST.THRESHOLD)[0]
        onset_prob = self._onset_prob.copy()
        onset_frame = []
        i = 0
        while i < len(onset_prob):
            candi_frame = []
            j = i
            while j < len(onset_prob):
                if (onset_prob[j] - onset_prob[i]) <= 15:
                    candi_frame.append(onset_prob[j])
                else:
                    break
                j += 1
            maxprob, max_onset = pred[candi_frame[0]], candi_frame[0]
            for frame in candi_frame:
                if pred[frame] > maxprob:
                    max_onset = frame
                    maxprob = pred[max_onset]
            onset_frame.append(max_onset)
            i = j
        self._onset_time = np.array(onset_frame) * self.hopsize_t
        self._onset_frame = np.array(onset_frame)
        return self._onset_time


def combine_onset(ref_onset):
    '''
    合并间隔<10ms的onset
    '''
    ref_onset_new = [[ref_onset[0]]]
    onset_diff = [y - x for (x, y) in zip(ref_onset, ref_onset[1:])]
    for (i, diff) in enumerate(onset_diff):
        if diff < 0.01:
            ref_onset_new[-1].append(ref_onset[i + 1])
        else:
            ref_onset_new.append([ref_onset[i + 1]])
    ref_onset = []
    for item in ref_onset_new:
        sum = 0
        for item2 in item:
            sum += item2
        ref_onset.append(sum / len(item))
    return np.array(ref_onset)

def evaluate_onset(ref_filepath,
                   est_filepath,
                   onset_combined=True,
                   onset_tolerance=0.05,):
    '''
    return the P,R,F between prediction and annotation
    :param ref_filepath:
    :param est_filepath:
    :param onset_combined:
    :param onset_tolerance:
    :return:
    '''
    ref_onsets = np.loadtxt(ref_filepath)[:, 0]
    est_onsets = np.loadtxt(est_filepath)
    est_onsets = est_onsets.reshape(-1)
    onset_scores = mir_eval.onset.evaluate(ref_onsets, est_onsets, window=onset_tolerance)
    return onset_scores

def main(filename):
    args = parse_args()
    cfg = default.update_cfg(default._C, args)
    writer_path = os.path.join(cfg.LOG_DIR, 'test_images')
    if not os.path.exists(writer_path):
        os.mkdir(writer_path)
    writer = SummaryWriter(writer_path)
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    # 加载模型
    if args.model == 'bi_lstm_onsetnet':
        model = eval('bi_lstm_onsetnet')(lstm_input_dim=256, lstm_hidden_dim=512, lstm_num_layers=2, lstm_output_dim=128)
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(torch.load(args.model_path))

    # 加载数据
    if args.is_annotated:
        data_path = cfg.DATA_DIR
        files = utils.get_dataset_filename(data_path)
        dataset = rnn_dataset(cfg, *files)
        for (input, labels) in dataset:
            if use_cuda:
                input = input.cuda()
                labels = labels.cuda()
            output = model(input)
    else:
        data, spec = utils.data_proc(filename, cfg)

        data = torch.from_numpy(data).unsqueeze(dim=1).unsqueeze(dim=1)
        padding_frames = cfg.MODEL.INPUT_SIZE[0] // 2
        pred_probs = np.zeros(data.shape[0] + padding_frames * 2)
        for idx, frame in enumerate(data):
            if use_cuda:
                frame = frame.cuda()
            output = model(frame)
            output = output.squeeze(dim=0).detach().cpu().numpy()
            pred_probs[idx: idx + cfg.MODEL.OUTPUT_SIZE[0]] += output

        final_preds = pred_probs[padding_frames: -padding_frames] / cfg.MODEL.OUTPUT_SIZE[0]

        # onset_anno_path = os.path.splitext(filename)[0] + '.txt'
        # onset_labels = np.round(np.loadtxt(onset_anno_path, dtype=np.float) * 44100 / 512)[:, 0]
        # plt.figure(figsize=(20,8))
        #
        # plt.subplot(2, 1, 1)
        # plt.pcolormesh(spec, cmap='gist_gray')
        # for onset in onset_labels:
        #     plt.axvline(onset, c='r', ls='--')
        # plt.subplot(2, 1, 2)
        #
        # plt.plot(final_preds)
        # for onset in onset_labels:
        #     plt.axvline(onset, c='r', ls='--')
        # plt.xlim([0, len(final_preds)])
        # plt.title('bilstm-onset:' + os.path.split(filename)[1])
        # plt.show()

def visualization(predictor, spec, filename, args):
    final_preds = predictor.onset_pred
    onset_time = predictor.onset_time
    onset_frames = predictor.onset_frame

    onset_anno_path = os.path.splitext(filename)[0] + '.txt'
    pred_save_file = '_est.txt'
    np.savetxt(pred_save_file, onset_time, fmt='%.4f')
    scores = evaluate_onset( onset_anno_path, pred_save_file, onset_tolerance=0.05)
    onset_time = np.loadtxt(onset_anno_path, dtype=np.float)
    onset_labels = np.round(onset_time * 44100 / 512)[:, 0]
    # print("onset_labels:", onset_time[:, 0])
    plt.figure(figsize=(20,8))

    plt.subplot(2, 1, 1)
    plt.pcolormesh(librosa.amplitude_to_db(spec), cmap='gist_gray')
    for onset in onset_labels:
        plt.axvline(onset, c='r', ls='--')
    plt.xticks(onset_labels, onset_labels.astype(np.int))
    plt.title(args.model_path)
    plt.subplot(2, 1, 2)

    plt.plot(final_preds)
    for onset in onset_labels:
        plt.axvline(onset, c='r', ls='--')
    for onset in onset_frames:
        plt.axvline(onset, c='y', ls='-')
    plt.xticks(onset_labels, onset_labels.astype(np.int))
    plt.xlim([0, len(final_preds)])
    plt.title(args.model + os.path.split(filename)[1])
    fig = plt.gcf()
    # fig.savefig(os.path.split(filename)[1] + '.png')
    plt.show()
    print(scores)
def evaluate_all(root_path: str,
                 prediction_save_path: str = './pred',
                 onset_tolerance: float = 0.05,
                 ):
    '''
    evaluation for test dataset
    :param root_path: test dataset path
    :param prediction_save_path: the model outputs to the path
    :return: averages of P,R,F
    '''
    global logger_file
    args = parse_args()
    cfg = default.update_cfg(default._C, args)
    # logger_file = os.path.join(args.model_path)
    predictor = predictor_onset(args.model_path, cfg, model_name=args.model)
    Precision = AverageMeter()
    Recall = AverageMeter()
    Fscore = AverageMeter()
    ids = os.listdir(root_path)
    for id in ids:
        wav_file = os.path.join(root_path, id, id + '.wav')
        label_file = os.path.join(root_path, id, id + '.txt')
        pred_save_file = os.path.join(prediction_save_path, id + '_est.txt')
        if not os.path.exists(wav_file):
            wav_file = os.path.splitext(wav_file)[0] + '.mp3'
        if not os.path.exists(prediction_save_path):
            os.mkdir(prediction_save_path)
        predictor.predict(wav_file)
        onset_time = predictor.onset_time
        np.savetxt(pred_save_file, onset_time, fmt='%.4f')
        try:
            scores = evaluate_onset(label_file, pred_save_file, onset_tolerance=onset_tolerance)
            Precision.update(scores['Precision'], 1)
            Recall.update(scores['Recall'], 1)
            Fscore.update(scores['F-measure'], 1)

            print('{0}\tp:{1:.4f} \t r:{2:.4f} \t f:{3:.4f}'.format(id, scores['Precision'], \
                                                  scores['Recall'], scores['F-measure']))
        except:
            print('%s---------------\n'%(id))
    print("all measure averages: \tp:{:.4f} \t r:{:.4f} \t f:{:.4f}".format(Precision.avg, Recall.avg, Fscore.avg))
    print("model_name", args.model_path)
    print(root_path)

def model_test(
               root_path: str,
               prediction_save_path: str = './pred',
               onset_tolerance: float = 0.05,):
    global logger_file

    args = parse_args()
    prediction_save_path = '/home/data/ywm_data/Models/' + os.path.split(args.cfg)[1][:-5]
    cfg = default.update_cfg(default._C, args)
    # logger_file = os.path.join(args.model_path)
    predictor = predictor_onset(args.model_path, cfg, model_name=args.model, having_nn=True)
    Precision = AverageMeter()
    Recall = AverageMeter()
    Fscore = AverageMeter()
    ids = os.listdir(root_path)
    for id in ids:
        wav_file = os.path.join(root_path, id, id + '.wav')
        label_file = os.path.join(root_path, id, id + '.txt')
        pred_save_file = os.path.join(prediction_save_path, id + '_est.txt')
        if os.path.exists(pred_save_file):
            continue
        if not os.path.exists(wav_file):
            wav_file = os.path.splitext(wav_file)[0] + '.mp3'
        if not os.path.exists(prediction_save_path):
            os.mkdir(prediction_save_path)
        predictor.predict(wav_file)
        onset_time = predictor.onset_time
        np.savetxt(pred_save_file, onset_time, fmt='%.4f')
        try:
            scores = evaluate_onset(label_file, pred_save_file, onset_tolerance=onset_tolerance)
            Precision.update(scores['Precision'], 1)
            Recall.update(scores['Recall'], 1)
            Fscore.update(scores['F-measure'], 1)

            print('{0}\tp:{1:.4f} \t r:{2:.4f} \t f:{3:.4f}'.format(id, scores['Precision'], \
                                                                    scores['Recall'], scores['F-measure']))
        except:
            print('%s---------------\n' % (id))
    print("all measure averages: \tp:{:.4f} \t r:{:.4f} \t f:{:.4f}".format(Precision.avg, Recall.avg, Fscore.avg))
    print("model_name", args.model)
    print(root_path)
    
if __name__ == '__main__':
    # model_test("/home/yangweiming/MUSIC/onset/ywmfiles/ismir2014_dataset/")
    # model_test("/home/yangweiming/MUSIC/onset/ywmfiles/127_data")
    # args = parse_args()
    # cfg = default.update_cfg(default._C, args)
    # predictor = predictor_onset(args.model_path, cfg, model_name=args.model)
    # ids = [101666,65793,65898,6770,70191,70430,70488,7462]
    # for id in ids:
    #     id = str(id)
    #     wav_file = os.path.join("/home/yangweiming/MUSIC/onset/ywmfiles/127_data", id, id+'.mp3')
    #     spec = predictor.predict(wav_file)
    #     onset_pred = predictor.onset_pred
    #     visualization(predictor, spec, wav_file, args)
    # prediction_save_path = "/home/data/ywm_data/train_data/onsetPrediction/05_11_best"
    # evaluate_all("/home/yangweiming/MUSIC/onset/ywmfiles/sep_dataset/fold_0_5_5/train")
    # evaluate_all("/home/yangweiming/MUSIC/onset/ywmfiles/ismir2014_dataset/")
    evaluate_all("/home/yangweiming/MUSIC/onset/ywmfiles/127_data/")
    # evaluate_all("/home/yangweiming/MUSIC/onset/ywmfiles/Korea/")
    # evaluate_all("/home/data/ywm_data/train_data/shuffle_data/sep_dataset/fold_0/test_dir")
    # dirs = os.listdir("/home/yangweiming/MUSIC/onset/ywmfiles/127_data/")
    #
    # args = parse_args()
    # cfg = default.update_cfg(default._C, args)
    # predictor = predictor_onset(args.model_path, cfg, model_name=args.model)
    #
    # for id in dirs:
    #     # if int(id) in ids:
    #     #     continue
    #     print("----------", id, "--------------")
    #     wav_file = os.path.join("/home/yangweiming/MUSIC/onset/ywmfiles/127_data/",id, id + '.wav')
    #     if not os.path.exists(wav_file):
    #         wav_file = os.path.splitext(wav_file)[0] +'.mp3'
    #     spec = predictor.predict(wav_file)
    #     onset_pred = predictor.onset_pred
    #     visualization(predictor, spec, wav_file, args)
    #     onset_time = predictor.onset_time
    #     print(onset_time)
    # onset_anno_path = os.path.splitext(wav_file)[0] + '.txt'
    # ref_onsets = np.loadtxt(onset_anno_path)[:, 0]
    # scores = mir_eval.onset.evaluate(ref_onsets, onset_time, window=0.05)
    # print(1)
    




