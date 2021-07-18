# -*- coding: utf-8 -*-
# @Time    : 2021/1/15 16:27
# @Author  : moMing.yang
# @File    : train.py
# @Software: PyCharm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import sys
sys.path.append(os.path.join(os.getcwd(), 'lib'))
import yaml
import numpy as np
import argparse
from utils import default, utils
from pathlib import Path
import time
import warnings
import torch
import logging
from dataset.rnn_dataset import rnn_dataset
from torch.utils.data.dataloader import DataLoader
from senet import bi_lstm_onsetnet
import multiprocessing
from tensorboardX import SummaryWriter
warnings.filterwarnings('ignore')
from lib.core.functions import train, validate
from lib.core.losses import Cross_entropy


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
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
    parser.add_argument('--checkpointDir',
                        help='the checkpoint file to the last training',
                        type=str,
                        default=''
                        )
    parser.add_argument('--model_type',
                        help="the name of model name like 'onsetnet', 'CBAM_onsetnet' and so on",
                        type=str,
                        default="bi_lstm_onsetnet",
                        choices=['onsetnet', 'CBAM_onsetnet', "bi_lstm_onsetnet", 'old_onsetnet', 'bi_lstm_cbam_onsetnet'],
                        )
    args = parser.parse_args()

    return args

def main():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler('./log1.log', mode='w')
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    args = parse_args()
    cfg = default.update_cfg(default._C, args)
    logger, final_output_dir, tensorboard_log_dir, checkpoint_log_dir = utils.create_logger(cfg, args.cfg)
    logger.info(cfg)
    # cudnn加载

    writer_dict = {
        'writer': SummaryWriter(log_dir=tensorboard_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    torchSeed = np.random.randint(1, 50000)
    torch.manual_seed(torchSeed)
    npSeed = np.random.randint(1, 50000)
    np.random.seed(npSeed)
    logger.info("torchSeed:{0}".format(torchSeed))
    logger.info("npSeed:{0}".format(npSeed))
    # model = bi_lstm_onsetnet(cfg, lstm_input_dim=256, lstm_hidden_dim=512, lstm_num_layers=2, lstm_output_dim=128, pre_trained_onsetnet='./Models/onset.pth')
    model = utils.get_model(cfg, model_name=args.model_type, is_training=True)

    # 数据加载
    if cfg.DATASET.DIR:
        train_data_path = Path(cfg.DATASET.ROOT) / cfg.DATASET.DIR / cfg.DATASET.TRAIN_SET
        valid_data_path = Path(cfg.DATASET.ROOT) / cfg.DATASET.DIR / cfg.DATASET.VALID_SET
    else:
        train_data_path = Path(cfg.DATASET.ROOT) / cfg.DATASET.TRAIN_SET
        valid_data_path = Path(cfg.DATASET.ROOT) / cfg.DATASET.VALID_SET
    train_files = utils.get_dataset_filename(train_data_path)
    train_files = list(train_files)
    # # start_time = time.time()
    # train_file_list = utils.split_train_files(train_files)

    # train_dataset = rnn_dataset(cfg, *train_files, is_training=True)
    # train_dataloader = DataLoader(train_dataset,
    #                               batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
    #                               shuffle=True,
    #                               num_workers=cfg.WORKERS,
    #                               pin_memory=cfg.PIN_MEMORY)
    # print("加载训练数据耗时：", time.time() - start_time)
    start_time = time.time()
    valid_files = utils.get_dataset_filename(valid_data_path)
    valid_dataset = rnn_dataset(cfg, is_training=False, *valid_files)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
                                  shuffle=False,
                                  num_workers=cfg.WORKERS,
                                  pin_memory=cfg.PIN_MEMORY)
    print("加载验证数据耗时：", time.time() - start_time)
    # exit(111)
    writer = writer_dict['writer']
    # sample = torch.randn(1, 1, 267, 87)
    # writer.add_graph(model, sample)

    model = torch.nn.DataParallel(model).cuda()

    # 设置 optimizer
    last_epoch = -1
    optimizer = utils.get_optimizier(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH

    F1 = 0.0
    loss_alpha = cfg.LOSS.ALPHA
    if cfg.AUTO_RESUME and args.checkpointDir:
        checkpoint_file = os.path.join(args.checkpointDir, 'checkpoints.pth')
        if not os.path.exists(checkpoint_file):
            logger.info("待加载的 checkpoint file '{}' 不存在".format(checkpoint_file))
            exit(1)
        logger.info("==> 加载 checkpoint '{}".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        F1 = checkpoint['best_F1']
        last_epoch = checkpoint['epoch']
        loss_alpha = checkpoint['loss_alpha']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("==> 加载完毕 checkpoint '{}' (epoch {}) ".format(checkpoint_file, begin_epoch))
        criterion = Cross_entropy(alpha=checkpoint['loss_alpha'], eps=1e-7, size_average=True)
    else:
        criterion = Cross_entropy(alpha=cfg.LOSS.ALPHA, eps=1e-7, size_average=True)
        checkpoint_file = os.path.join(checkpoint_log_dir, 'checkpoints.pth')

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.LR_STEP, \
                                                        gamma=cfg.TRAIN.LR_FACTOR, last_epoch=last_epoch, )

    # 训练

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        lr_scheduler.step()
        randomPerm = np.random.permutation(len(train_files[0]))
        train_files[0] = np.array(train_files[0])[randomPerm].tolist()
        train_files[1] = np.array(train_files[1])[randomPerm].tolist()
        # start_time = time.time()
        train_file_list = utils.split_train_files(train_files)
        model = train(cfg, logger, train_file_list, model, criterion, optimizer, epoch,
              final_output_dir, tensorboard_log_dir, writer_dict)

        F1_cur, model = validate(cfg, logger, valid_dataloader, model, criterion, optimizer, epoch,
                          final_output_dir, tensorboard_log_dir, writer_dict)
        print("F1_cur:", F1_cur)
        if (epoch % 5 == 0):
            model_name = os.path.join(str(final_output_dir), 'epoch_{}.pth'.format(epoch))
            torch.save(model.state_dict(), model_name)
        if F1_cur > F1:
            F1 = F1_cur
            best_model_file = os.path.join(str(final_output_dir), 'best.pth')
            torch.save(model.state_dict(), best_model_file)
            is_best = True
        else:
            is_best = False

        logger.info("==> saving checkpoint to {}".format(checkpoint_file))
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'best_F1': F1,
            'optimizer': optimizer.state_dict(),
            'loss_alpha': loss_alpha,
        }, is_best, checkpoint_file)

    writer_dict['writer'].close()
if __name__ == '__main__':
    # with open('/home/ywm/MUSIC/Codes/bilstm_onset/bi_lstm_87×288_adam_lr1e-3.yaml', 'r') as f:
    #     y = yaml.load(f)
    #     print(y)
    multiprocessing.freeze_support()
    main()








