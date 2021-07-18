'''
 @Time : 2021/1/22 9:45 下午 
 @Author : Moming_Y
 @File : dataset.py 
 @Software: PyCharm
'''

import logging
import time
from pathlib import Path
import os
import tqdm
from torch.utils.data import Dataset
import torch
import librosa
import numpy as np
import multiprocessing
from multiprocessing import Process, Manager
import matplotlib.pyplot as plt
import matplotlib
class rnn_dataset(Dataset):
    def __init__(self, cfg, *data, is_training):
        self.cfg = cfg
        self.data_files = data[0]
        self.anno_files = data[1]
        manager = Manager()
        # 20210412 修改
        self.data_pos = manager.list()
        self.data_neg = manager.list()
        self.anno_pos = manager.list()
        self.anno_neg = manager.list()
        self.multiprocess_load()
        self.data_pos = np.concatenate(list(self.data_pos), axis=0)
        self.data_neg = np.concatenate(list(self.data_neg), axis=0)
        self.anno_pos = np.concatenate(list(self.anno_pos), axis=0)
        self.anno_neg = np.concatenate(list(self.anno_neg), axis=0)
        if is_training:
            if cfg.DATASET.NEG_POS_RATE:
                sample_neg_num = int(len(self.data_pos) * cfg.DATASET.NEG_POS_RATE)

                assert(sample_neg_num < len(self.data_neg))
                indexes = np.random.permutation(np.arange(len(self.data_neg)))[: sample_neg_num]
            else:
                indexes = np.random.permutation(np.arange(len(self.data_neg)))
            self.data_neg = self.data_neg[indexes]
            self.anno_neg = self.anno_neg[indexes]
            self.data = np.concatenate([self.data_neg, self.data_pos], axis=0)
            self.anno = np.concatenate([self.anno_neg, self.anno_pos], axis=0)
            random_index = np.random.permutation(np.arange(len(self.anno)))
            self.data = self.data[random_index]
            self.anno = self.anno[random_index]
        else:
            self.data = np.concatenate([self.data_neg, self.data_pos], axis=0)
            self.anno = np.concatenate([self.anno_neg, self.anno_pos], axis=0)
            random_index = np.random.permutation(np.arange(len(self.anno)))
            self.data = self.data[random_index]
            self.anno = self.anno[random_index]
        print(self.data.shape, self.anno.shape)

        # self.data, self.anno = self.cal_sepc_anno(data[0], data[1])
        # for data, anno in  zip(self.data, self.anno):
        #     print(data.shape, anno.shape)
        # self.data = np.concatenate(self.data, axis=0)
        # self.anno = np.concatenate(self.anno, axis=0)
        # print(self.anno.shape, self.data.shape)
        # print(self.anno[0].shape, self.data[0].shape)

    def multiprocess_load(self):
        # pass
        # cpu_count = multiprocessing.cpu_count()
        cpu_count = 1
        data_file_list = []
        anno_file_list = []
        process_list = []
        if len(self.data_files) > cpu_count:
            for i in range(0, len(self.data_files), len(self.data_files) // cpu_count):
                data_file_list.append(self.data_files[i: i + len(self.data_files) // cpu_count])
                anno_file_list.append(self.anno_files[i: i + len(self.data_files) // cpu_count])

            for i in range(cpu_count):
                p  = Process(target=self.process_cal_spec_anno, name=str(i), args=(data_file_list[i], anno_file_list[i], i))
                p.start()
                process_list.append(p)

            for p in process_list:
                p.join()
        else:
            self.process_cal_spec_anno(self.data_files, self.anno_files, 1)



    def process_cal_spec_anno(self, data_list, anno_list, num, spec_style='cqt'):
        # print(num, ":", os.getpid())
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
        n_frame = self.cfg.MODEL.INPUT_SIZE[0]
        n_bins = self.cfg.MODEL.INPUT_SIZE[1]
        output_size = self.cfg.MODEL.OUTPUT_SIZE[0]
        input_pad_length = n_frame // 2 if n_frame > 2 else n_frame
        output_pad_length = output_size // 2 if output_size > 2 else 0
        spec_data_pos, anno_data_pos = [], []
        spec_data_neg, anno_data_neg = [], []
        for file, anno in zip(data_list, anno_list):
            try:
                y, sr = librosa.load(file, sr=44100)
                # logging.info("{} 已读".format(file))
            except Exception as e:
                logging.info("不能读取文件：{}, 原因是：{}".format(file, e))
                continue
            y_spec = np.abs(eval('librosa.core.' + spec_style)(y, sr=sr, hop_length=512, n_bins=n_bins, bins_per_octave=36,
                                                               fmin=librosa.note_to_hz('A0'))).astype(np.float16)
            y_spec_frame = np.pad(y_spec, pad_width=((0, 0), (input_pad_length, input_pad_length)), mode='constant')
            anno_frame = np.round(np.loadtxt(anno) * 44100 / 512)[:, 0] + output_pad_length

            anno_frame = anno_frame.astype(np.int) - 1  # indexes need to be substract 1
            anno_frames = np.empty(shape=1)
            for onset in anno_frame:    # 增加正样本
                onset_loc = np.array([onset - 2, onset - 1, onset, onset + 1, onset + 2])
                onset_loc = onset_loc.clip(min=0, max=y_spec.shape[1] - 1)
                anno_frames = np.append(anno_frames, np.unique(onset_loc), axis=0)
            anno_frames = anno_frames[1:].astype(np.int)
            annos = np.zeros(y_spec.shape[1] + output_pad_length * 2) if output_size != 1 else np.zeros(y_spec.shape[1])
            annos[anno_frames] = 1
            data_pos, anno_pos = [], []
            data_neg, anno_neg = [], []

            for i in range(y_spec.shape[1]):
                input_spec = y_spec_frame[:, i: i + n_frame]

                smax = np.max(input_spec)  # max in one frame
                smin = np.min(input_spec)
                input_spec = (input_spec - smin) / \
                             (smax - smin + 1e-9)  # normalize to 0-1

                if annos[i: i + output_size].sum() >= 1:
                    data_pos.append(input_spec)
                    anno_pos.append(annos[i: i + output_size])
                    # plt.figure()
                    # plt.pcolormesh(librosa.amplitude_to_db(y_spec_frame[:, i: i + n_frame], ref=np.max), cmap='gist_gray')
                    # index = np.where(annos[i: i + output_size] == 1)
                    # print(i, index)
                    # for _x in index[0]:
                    #     plt.axvline(_x + input_pad_length - output_pad_length, c='red', ls='--', lw=5)
                    # plt.show()

                else:
                    data_neg.append(input_spec)
                    anno_neg.append(annos[i: i + output_size])
            self.data_pos.append(np.expand_dims(data_pos, axis=1))
            self.anno_pos.append(anno_pos)
            self.data_neg.append(np.expand_dims(data_neg, axis=1))
            self.anno_neg.append(anno_neg)
        #     spec_data_pos.append(np.expand_dims(data_pos, axis=1))
        #     anno_data_pos.append(anno_pos)
        #     spec_data_neg.append(np.expand_dims(data_neg, axis=1))
        #     anno_data_neg.append(anno_neg)
        # spec_data_pos = np.concatenate(spec_data_pos, axis=0)
        # anno_data_pos = np.concatenate(anno_data_pos, axis=0)
        # spec_data_neg = np.concatenate(spec_data_neg, axis=0)
        # anno_data_neg = np.concatenate(anno_data_neg, axis=0)
        # self.data_pos.append(spec_data_pos)
        # self.anno_pos.append(anno_data_pos)
        # self.data_neg.append(spec_data_neg)
        # self.anno_neg.append(anno_data_neg)
        # print("num:", num, "is end")

    def cal_sepc_anno(self, data_list, anno_list, spec_style='cqt'):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
        n_frame = self.cfg.MODEL.INPUT_SIZE[0]
        n_bins = self.cfg.MODEL.INPUT_SIZE[1]
        output_size = self.cfg.MODEL.OUTPUT_SIZE[0]
        input_pad_length = n_frame // 2
        output_pad_length = output_size // 2
        spec_data, anno_data = [], []
        for file, anno in zip(data_list, anno_list):
            try:
                y, sr = librosa.load(file, sr=44100)
                logging.info("{} 已读".format(file))
            except Exception as e:
                logging.info("不能读取文件：{}, 原因是：{}".format(file, "RuntimeError"))
                continue
            y_spec = np.abs(eval('librosa.core.' + spec_style)(y, hop_length=512, n_bins=n_bins, bins_per_octave=36, fmin=librosa.note_to_hz('A0')))
            y_spec_frame = np.pad(y_spec, pad_width=((0, 0), (input_pad_length, input_pad_length)), mode='reflect')
            anno_frame = np.round(np.loadtxt(anno) * 44100 / 512)[:, 0] + output_pad_length
            anno_frame = anno_frame.astype(np.int)

            annos = np.zeros(y_spec.shape[1] + output_size)
            try:
                annos[anno_frame] = 1
            except Exception as e:
                logging.info("标注的帧数大于anno总帧数：file:{0}, 总帧数：{1}, index：{2}".format(file, annos.shape, anno_frame))
            data, anno = [], []

            for i in range(y_spec.shape[1]):
                data.append(y_spec_frame[:,i: i + n_frame])
                anno.append(annos[i: i + output_size])
                # if(i % 200 == 0):
                #     plt.pcolormesh(y_spec[:, i: i + n_frame], cmap='gist_gray')
                #     plt.plot(np.pad(annos[i: i + output_size], (4, 4)))
                #     plt.show()
            spec_data.append(np.expand_dims(data, axis=1))
            anno_data.append(np.expand_dims(anno, axis=1))
        # spec_data = np.expand_dims(spec_data, axis=0)
        # anno_data = np.expand_dims(anno_data, axis=0)
        return spec_data, anno_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.anno[item]

if __name__ == '__main__':
    rnn_dataset("sdf", "dfsf", "fdsaf")