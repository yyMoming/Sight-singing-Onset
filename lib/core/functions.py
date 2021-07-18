'''
 @Time : 2021/1/22 10:15 下午 
 @Author : Moming_Y
 @File : functions.py 
 @Software: PyCharm
'''

import torch
import torch.nn as nn
import os
import numpy
import logging
import time
from .evaluate import *
from dataset.rnn_dataset import rnn_dataset
logger = logging.getLogger(__name__)
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = (self.sum) / self.count if self.count != 0 else 0

use_cuda = torch.cuda.is_available()

def train(cfg, logger, train_file_list, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    Precision = AverageMeter()
    Recall = AverageMeter()
    F1_score = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (x, y) in enumerate(train_file_list):
        start_ = time.time()

        train_dataset = rnn_dataset(cfg, x, y, is_training=True)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                      batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
                                      shuffle=True,
                                      num_workers=cfg.WORKERS,
                                      pin_memory=cfg.PIN_MEMORY)
        print("第 {0} 轮训练加载训练数据{1}耗时： {2}".format(epoch, i, time.time() - start_))
        for i, (input, label) in enumerate(train_loader):

            if(use_cuda):
                input = input.cuda()
                label = label.cuda()
            data_time.update(time.time() - end)
            input = input.float()
            label = label.float()
            # compute output
            output = model(input)

            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            '''
                梯度截断，还未使用
            '''
            # for tag, value in model.named_parameters():
            #     value.grad.data.clamp_()
            #     writer.add_scalar(tag, value.detach().cpu().numpy().mean(), global_steps)
            #     writer.add_scalar(tag + 'grad', value.grad.detach().cpu().numpy().mean(), global_steps)
            optimizer.step()
            losses.update(loss.item(), input.size()[0])

            P, R, F1 = eval("eval_{}_train".format(cfg.MODEL.NAME))(cfg, output, label)
            Precision.update(P, 1)
            Recall.update(R, 1)
            F1_score.update(F1, 1)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfg.PRINT_FREQ == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                      'P {p.val:.3f} ({p.avg:.3f}) \t' \
                      'R {r.val:.3f} ({r.avg:.3f}) \t' \
                      'F1 {f.val:.3f} ({f.avg:.3f}) '.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    speed=input.size(0) / batch_time.val,
                    data_time=data_time, loss=losses, p=Precision, r=Recall, f=F1_score)

                logger.info(msg)

                writer = writer_dict['writer']
                global_steps =  writer_dict['train_global_steps']
                writer.add_scalar('train_losses', losses.val, global_steps)
                writer.add_scalar('train_F1', F1_score.val, global_steps)
                for tag, value in model.named_parameters():
                    writer.add_scalar(tag, value.detach().cpu().numpy().mean(), global_steps)
                    writer.add_scalar(tag + 'grad', value.grad.detach().cpu().numpy().mean(), global_steps)
                writer_dict['train_global_steps'] += 1
    return model

def validate(cfg, logger, valid_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    Precision = AverageMeter()
    Recall = AverageMeter()
    F1_score = AverageMeter()

    # switch to train mode
    model.eval()

    end = time.time()

    for i, (input, label) in enumerate(valid_loader):
        if (use_cuda):
            input = input.cuda()
            label = label.cuda()
        input = input.float()
        label = label.float()
        data_time.update(time.time() - end)

        # compute output
        output = model(input)

        loss = criterion(output, label)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        losses.update(loss.item(), input.size()[0])

        P, R, F1 = eval("eval_{}_train".format(cfg.MODEL.NAME))(cfg, output, label)
        Precision.update(P, 1)
        Recall.update(R, 1)
        F1_score.update(F1, 1)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'P {p.val:.3f} ({p.avg:.3f}) \t' \
                  'R {r.val:.3f} ({r.avg:.3f}) \t' \
                  'F1 {f.val:.3f} ({f.avg:.3f}) '.format(
                epoch, i, len(valid_loader), batch_time=batch_time,
                speed=input.size(0) / batch_time.val,
                data_time=data_time, loss=losses, p=Precision, r=Recall, f=F1_score)

            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_losses', losses.val, global_steps)
            writer.add_scalar('valid_F1', F1_score.val, global_steps)
            writer_dict['valid_global_steps'] += 1
    # with torch.no_grad():
    #     for i, (input, label) in enumerate(valid_loader):
    #         if (use_cuda):
    #             input = input.cuda()
    #             label = label.cuda()
    #         input = input.float()
    #         label = label.float()
    #         data_time.update(time.time() - end)
    #
    #         # compute output
    #         output = model(input)
    #
    #         loss = criterion(output, label)
    #         # optimizer.zero_grad()
    #         # loss.backward()
    #         # optimizer.step()
    #         losses.update(loss.item(), input.size()[0])
    #
    #         P, R, F1 = eval("eval_{}_train".format(cfg.MODEL.NAME))(cfg, output, label)
    #         Precision.update(P, 1)
    #         Recall.update(R, 1)
    #         F1_score.update(F1, 1)
    #
    #         batch_time.update(time.time() - end)
    #         end = time.time()
    #
    #         if i % cfg.PRINT_FREQ == 0:
    #             msg = 'Epoch: [{0}][{1}/{2}]\t' \
    #                   'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
    #                   'Speed {speed:.1f} samples/s\t' \
    #                   'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
    #                   'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
    #                   'P {p.val:.3f} ({p.avg:.3f}) \t' \
    #                   'R {r.val:.3f} ({r.avg:.3f}) \t' \
    #                   'F1 {f.val:.3f} ({f.avg:.3f}) '.format(
    #                 epoch, i, len(valid_loader), batch_time=batch_time,
    #                 speed=input.size(0) / batch_time.val,
    #                 data_time=data_time, loss=losses, p=Precision, r=Recall, f=F1_score)
    #
    #             logger.info(msg)
    #
    #             writer = writer_dict['writer']
    #             global_steps = writer_dict['valid_global_steps']
    #             writer.add_scalar('valid_losses', losses.val, global_steps)
    #             writer.add_scalar('valid_F1', F1_score.val, global_steps)
    #             writer_dict['valid_global_steps'] += 1

    return F1_score.val, model