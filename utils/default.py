'''
 @Time : 2021/1/15 5:32 下午 
 @Author : Moming_Y
 @File : default.py 
 @Software: PyCharm
'''

import os
from yacs.config import CfgNode as CN
import argparse

_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.GPUS = (3, )
_C.WORKERS = 12
_C.PRINT_FREQ = 100
_C.PIN_MEMORY = True
_C.AUTO_RESUME = False
_C.DATA_DIR = ''

_C.LOSS = CN()
_C.LOSS.ALPHA = 1.0
# audio to sepc params

_C.SPEC = CN()
_C.SPEC.HOPSIZE = 512
_C.SPEC.SR = 44100
_C.SPEC.SPEC_STYLE = 'cqt'
# cudnn params

_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# dataset
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DIR = ''
_C.DATASET.VALID_SET = ''
_C.DATASET.TRAIN_SET = ''
_C.DATASET.NEG_POS_RATE = 0.0

# NETWORK common params
_C.MODEL = CN()
_C.MODEL.INIT_WEIGHT = True
_C.MODEL.NAME = 'bilstm_onset'
_C.MODEL.INPUT_SIZE = [87, 267]
_C.MODEL.OUTPUT_SIZE = [79, 1]
_C.MODEL.PRETRAINED = ''
_C.MODEL.CBAM_ONSETNET = CN(new_allowed=True)
_C.MODEL.BI_LSTM = CN(new_allowed=True)
# train
_C.TRAIN = CN()
_C.TRAIN.THRESHOLD = 0.5
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.001

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

# test
_C.TEST = CN()
_C.TEST.BATCH_SIZE_PER_GPU = 32
_C.TEST.THRESHOLD = 0.5
from sacred import Experiment
ex = Experiment()


def update_cfg(cfg, args):
    cfg = _C
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    # print(cfg.TEST.BATCH_SIZE_PER_GPU)
    # opt = ["MODEL.CBAM_ONSETNET.config.conv1", "(24, 7)"]
    cfg.merge_from_list(args.opts)
    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir

    if args.dataDir:
        cfg.DATA_DIR = args.dataDir



    cfg.MODEL.PRETRAINED = os.path.join(
        cfg.LOG_DIR, "checkpoints", cfg.MODEL.PRETRAINED
    )

    # if cfg.TEST.MODEL_FILE:
    #     cfg.TEST.MODEL_FILE = os.path.join(
    #         cfg.DATA_DIR, cfg.TEST.MODEL_FILE
    #     )
    cfg.freeze()
    # print(cfg)

    return cfg

if __name__ == '__main__':
    from utils import get_model
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default="/home/yangweiming/MUSIC/onset/onset_train/bi_lstm_87×267_adam_lr1e-7.yaml", type=str)
    parser.add_argument('opts', help='modify config options using the command-line', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--modelDir', help='modify config options using the command-line', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--logDir', help='modify config options using the command-line', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--dataDir', help='modify config options using the command-line', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args("WORKERS 4 DATASET.DIR kfold1".split(" "))
    cfg = update_cfg(cfg=_C, args=args)
    get_model(cfg, is_training=True, model_name='bi_lstm_onsetnet')


