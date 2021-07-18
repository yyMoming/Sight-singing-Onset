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
from torch.utils.data.dataset import Dataset
import torch
import librosa
import numpy as np
import multiprocessing
from multiprocessing import Process, Manager

class rnn_dataset(Dataset):
    def __init__(self, cfg, *data):
        self.cfg = cfg
        self.data_files = data[0]
        self.anno_files = data[1]
        manager = Manager()
        self.data = manager.list()
        self.anno = manager.list()
        self.multiprocess_load()
        self.data = np.concatenate(list(self.data), axis=0)
        self.anno = np.concatenate(list(self.anno), axis=0)
        # print(self.data.shape, self.anno.shape)

        # self.data, self.anno = self.cal_sepc_anno(data[0], data[1])
        # for data, anno in  zip(self.data, self.anno):
        #     print(data.shape, anno.shape)
        # self.data = np.concatenate(self.data, axis=0)
        # self.anno = np.concatenate(self.anno, axis=0)
        # print(self.anno.shape, self.data.shape)
        # print(self.anno[0].shape, self.data[0].shape)

    def multiprocess_load(self):
        # pass
        cpu_count = multiprocessing.cpu_count()
        # cpu_count = 1
        data_file_list = []
        anno_file_list = []
        process_list = []

        for i in range(0, len(self.data_files), len(self.data_files) // cpu_count):
            data_file_list.append(self.data_files[i: i + len(self.data_files) // cpu_count])
            anno_file_list.append(self.anno_files[i: i + len(self.data_files) // cpu_count])

        for i in range(cpu_count):
            p  = Process(target=self.process_cal_spec_anno, name=str(i), args=(data_file_list[i], anno_file_list[i], i))
            p.start()
            process_list.append(p)

        for p in process_list:
            p.join()



    def process_cal_spec_anno(self, data_list, anno_list, num, spec_style=self.cfg.SPEC_STYLE):
        # print(num, ":", os.getpid())
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
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
                logging.info("不能读取文件：{}, 原因是：{}".format(file, e))
                continue
            y_spec = np.abs(eval('librosa.core.' + spec_style)(y, hop_length=512, n_bins=n_bins, bins_per_octave=36,
                                                               fmin=librosa.note_to_hz('A0')))
            y_spec_frame = np.pad(y_spec, pad_width=((0, 0), (input_pad_length, input_pad_length)), mode='reflect')
            anno_frame = np.round(np.loadtxt(anno) * 44100 / 512)[:, 0] + output_pad_length
            anno_frame = anno_frame.astype(np.int)
            annos = np.zeros(y_spec.shape[1] + output_size)
            annos[anno_frame] = 1
            data, anno = [], []

            for i in range(y_spec.shape[1]):
                data.append(y_spec_frame[:, i: i + n_frame])
                anno.append(annos[i: i + output_size])
                # if(i % 200 == 0):
                #     plt.pcolormesh(y_spec[:, i: i + n_frame], cmap='gist_gray')
                #     plt.plot(np.pad(annos[i: i + output_size], (4, 4)))
                #     plt.show()
            spec_data.append(np.expand_dims(data, axis=1))
            anno_data.append(np.expand_dims(anno, axis=1))

        spec_data = np.concatenate(spec_data, axis=0)
        anno_data = np.concatenate(anno_data, axis=0)
        self.data.append(spec_data)
        self.anno.append(anno_data)
        # print("num:", num, "is end")

    def cal_sepc_anno(self, data_list, anno_list, spec_style=self.cfg.SPEC_STYLE):
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

