'''
 @Time : 2021/1/16 11:37 上午 
 @Author : Moming_Y
 @File : utils.py 
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
from senet import onsetnet, CBAM_onsetnet, bi_lstm_onsetnet, bilstmOnsetnet, bi_lstm_cbam_onsetnet
def create_logger(cfg, cfg_name):
    logging.basicConfig(level=logging.NOTSET, format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    root_output_dir = Path(cfg.OUTPUT_DIR)
    if not root_output_dir.exists():
        print("==> creating root_output_dir: {}".format(root_output_dir))
        root_output_dir.mkdir()

    # 数据集来源
    dataset_dir = Path(cfg.DATASET.ROOT) / cfg.DATASET.DIR
    if not dataset_dir.exists():
        logging.error(u"用于训练和测试的数据集不存在：{}".format(dataset_dir))
        exit(1)
    # 模型名称
    model_name = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0] + '.' + cfg.DATASET.DIR
    # 最终存储的outputdir
    curtime = time.strftime("%Y-%m-%d-%H-%M")
    final_output_dir = root_output_dir / cfg.DATASET.DIR / model_name / cfg_name/ curtime
    # print("==> creating final_output_dir: {}".format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / model_name / cfg_name / curtime
    checkpoint_log_dir = Path(cfg.LOG_DIR) / 'checkpoints' / curtime
    # print("==> creating tensorboard_log_dir: {}".format(tensorboard_log_dir))
    # print("==> creating checkpoint_log_dir: {}".format(checkpoint_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_log_dir.mkdir(parents=True, exist_ok=True)
    # curtime = time.strftime("%Y-%m-%d-%H-%M")
    # log_file = Path(cfg.LOG_DIR) / '{}_{}.log'.format(cfg_name, curtime)
    # log_file = Path(cfg.LOG_DIR) / '{}_.log'.format(cfg_name)
    log_file = final_output_dir / '{}_{}.log'.format(cfg_name, curtime)
    fh = logging.FileHandler(filename=log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.info("==> creating final_output_dir: {}".format(final_output_dir))
    logger.info("==> creating tensorboard_log_dir: {}".format(tensorboard_log_dir))
    logger.info("==> creating checkpoint_log_dir: {}".format(checkpoint_log_dir))

    return logger, str(final_output_dir), str(tensorboard_log_dir), str(checkpoint_log_dir)

def get_dataset_filename(path:Path):
    # valid_dataset_path = Path(cfg.DATASET.ROOT) / cfg.DATASET.VALID_SET
    train_dataset_path = path
    # test_dataset_files = [valid_dataset_path / x for x in valid_dataset_path.iterdir() if x.endswith('.wav') or x.endswith('.mp3')]
    train_dataset_files = [str(train_dataset_path / x) for x in train_dataset_path.iterdir() if str(x).endswith('.wav') or str(x).endswith('.mp3')]

    # test_anno_files = [os.path.splitext(x)[0] + '.txt' for x in test_dataset_files]
    train_anno_files = [os.path.splitext(x)[0] + '.txt' for x in train_dataset_files]

    return (train_dataset_files, train_anno_files)

def get_optimizier(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=cfg.TRAIN.LR,
                                    momentum=cfg.TRAIN.MOMENTUM,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                    nesterov=cfg.TRAIN.NESTEROV
                                    )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=cfg.TRAIN.LR)

    return optimizer

def save_checkpoint(states, is_best, checkpoint_file,
                    filename='checkpoint.pth'):
    torch.save(states, checkpoint_file)

def get_model(cfg, is_training=False, model_name: str = "onsetnet"):
    assert model_name in ['onsetnet', 'CBAM_onsetnet', "bi_lstm_onsetnet", 'old_onsetnet', 'bi_lstm_cbam_onsetnet']
    if model_name == 'onsetnet':
        model = bilstmOnsetnet(pad_length=4,
                         out_size=cfg.MODEL.OUTPUT_SIZE[0],
                         spec_style=cfg.SPEC.SPEC_STYLE,
                         dual_channel=False)
    elif model_name == 'old_onsetnet':
        model = onsetnet(pad_length=4,
                         out_size=cfg.MODEL.OUTPUT_SIZE[0],
                         spec_style=cfg.SPEC.SPEC_STYLE,
                         dual_channel=False)
    elif model_name == 'CBAM_onsetnet':
        model = CBAM_onsetnet(pad_length=4,
                         out_size=cfg.MODEL.OUTPUT_SIZE[0],
                         spec_style=cfg.SPEC.SPEC_STYLE,
                         dual_channel=False)
    elif model_name == "bi_lstm_onsetnet":
        if is_training:
            model = bi_lstm_onsetnet(cfg,
                                     lstm_input_dim=cfg.MODEL.BI_LSTM.lstm_input_dim,
                                     lstm_hidden_dim=cfg.MODEL.BI_LSTM.lstm_hidden_dim,
                                     lstm_num_layers=cfg.MODEL.BI_LSTM.lstm_num_layers,
                                     lstm_output_dim=cfg.MODEL.BI_LSTM.lstm_output_dim,
                                     pre_trained_onsetnet=cfg.MODEL.BI_LSTM.pre_trained_onsetnet)
        else:
            model = bi_lstm_onsetnet(cfg,
                                     lstm_input_dim=cfg.MODEL.BI_LSTM.lstm_input_dim,
                                     lstm_hidden_dim=cfg.MODEL.BI_LSTM.lstm_hidden_dim,
                                     lstm_num_layers=cfg.MODEL.BI_LSTM.lstm_num_layers,
                                     lstm_output_dim=cfg.MODEL.BI_LSTM.lstm_output_dim,)
    elif model_name == 'bi_lstm_cbam_onsetnet':
        model = bi_lstm_cbam_onsetnet(cfg,
                                     lstm_input_dim=cfg.MODEL.BI_LSTM.lstm_input_dim,
                                     lstm_hidden_dim=cfg.MODEL.BI_LSTM.lstm_hidden_dim,
                                     lstm_num_layers=cfg.MODEL.BI_LSTM.lstm_num_layers,
                                     lstm_output_dim=cfg.MODEL.BI_LSTM.lstm_output_dim,)
    return model

def data_proc(filename, cfg):
    import matplotlib.pyplot as plt
    n_frame = cfg.MODEL.INPUT_SIZE[0]
    n_bins = cfg.MODEL.INPUT_SIZE[1]
    output_size = cfg.MODEL.OUTPUT_SIZE[0]
    input_pad_length = n_frame // 2
    output_pad_length = output_size // 2
    spec_data, anno_data = [], []
    y, sr = librosa.load(filename, sr=44100, mono=True)
    y_spec = np.abs(eval('librosa.core.' + 'cqt')(y, sr=sr, hop_length=512, n_bins=n_bins, bins_per_octave=36,
                                                       fmin=librosa.note_to_hz('A0')))
    # y_spec = librosa.core.cqt(y, sr=sr, hop_length=512,
    #                        fmin=librosa.note_to_hz('A0'),
    #                        n_bins=n_bins,
    #                        bins_per_octave=36)
    # plt.figure()
    # plt.pcolormesh(librosa.amplitude_to_db(y_spec), cmap='gist_gray')
    # plt.title("origin")
    # plt.show()
    y_spec_frame = np.pad(y_spec, pad_width=((0, 0), (input_pad_length, input_pad_length)), mode='constant')
    # plt.pcolormesh(y_spec_frame, cmap='gist_gray')
    # plt.title("ywm")
    # plt.show()
    data = []
    for i in range(y_spec.shape[1]):
        input_spec = y_spec_frame[:, i: i + n_frame]
        smax = np.max(input_spec)  # max in one frame
        smin = np.min(input_spec)
        input_spec = (input_spec - smin) / \
                     (smax - smin + 1e-9)  # normalize to 0-1
        # plt.pcolormesh(input_spec, cmap='gist_gray')
        # plt.show()
        data.append(np.expand_dims(input_spec, axis=0))
        # plt.figure()
        # plt.pcolormesh(librosa.amplitude_to_db(librosa.amplitude_to_db(y_spec_frame[:, i: i + n_frame])), cmap='gist_gray')
        # plt.axvline(input_pad_length)
        # plt.show()
    # y_spec = librosa.amplitude_to_db(y_spec, ref=np.max)
    return np.concatenate(data, axis=0), y_spec

def split_train_files(train_files: tuple):
    train_files_list = []
    length = len(train_files[0])
    for i in range(0, length, 30):
        train_files_list.append((train_files[0][i: i+30], train_files[1][i: i+30]))
    return train_files_list

if __name__ == '__main__':
    create_logger()