'''
 @Time : 2021/1/23 12:08 上午 
 @Author : Moming_Y
 @File : evaluate.py 
 @Software: PyCharm
'''

import numpy as np
import torch

def eval_bilstm_train(cfg, output, labels):
    output1 = np.where(output.detach().cpu().numpy() > cfg.TRAIN.THRESHOLD, 1, 0)
    labels = labels.detach().cpu().numpy()
    TP = np.where((output1 == labels) & (output1 == 1))[0].shape[0]
    TN = np.where((output1 == labels) & (output1 == 0))[0].shape[0]
    FP = np.where(output1 == 1)[0].shape[0] - TP
    FN = np.where(output1 == 0)[0].shape[0] - TN
    P = TP / (TP + FP) if (TP + FP) != 0.0 else 0.0
    R = TP / (TP + FN) if (TP + FN) != 0.0 else 0.0
    F1_Score = 2 * P * R / (P + R) if (P + R) != 0.0 else 0.0

    return P, R, F1_Score

def eval_(output, labels, cfg = 0.5):
    output1 = np.where(output.detach().cpu().numpy() > cfg, 1, 0)
    labels = labels.detach().cpu().numpy()
    TP = np.where((output1 == labels) & (output1 == 1))[0].shape[0]
    TN = np.where((output1 == labels) & (output1 == 0))[0].shape[0]
    FP = np.where(output1 == 1)[0].shape[0] - TP
    FN = np.where(output1 == 0)[0].shape[0] - TN
    P = TP / (TP + FP) if (TP + FP) != 0 else 0
    R = TP / (TP + FN) if (TP + FN) != 0 else 0
    F1_Score = 2 * P * R / (P + R) if (P + R) != 0 else 0

    return P, R, F1_Score


if __name__ == '__main__':
    y = torch.randn(10)
    y = torch.from_numpy(np.where(y > 0, 1, 0))
    x = torch.randn(10)
    # x = torch.from_numpy(np.where(x > 0, 1, 0))
    res = eval_(x, y)