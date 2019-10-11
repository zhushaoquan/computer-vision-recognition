# !/usr/bin/python3
# -*- coding:utf-8 -*-
# @time : 2019.09.06
# @IDE : pycharm
# @auto : jeff_hua
# @github : https://github.com/Jeffer-hua

import torch
from torch import nn
import os
import shutil


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    lr = lr[0]

    return lr


# 计算top-N acc
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# 迁移训练 : 冻结除最后一层全连接层外所有网络层
def init_extract_model(model, output_classes):
    for param in model.parameters():
        param.requires_grad = False
    fc_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(fc_inputs, output_classes),
    )
    return model


# 迁移训练 : 微调所有层
def init_finetune_model(model, output_classes):
    fc_inputs = model.fc.in_features
    model.fc = nn.Linear(fc_inputs, output_classes)
    return model


# 等间隔下降学习率
def scheduler_StepLR():
    pass


# 保存模型l
def save_checkpoint(model_state, save_model_dir, is_best, epoch):
    save_model_path = os.path.join(save_model_dir, 'epoch_{}.pth.tar'.format(epoch))
    torch.save(model_state, save_model_path)
    if is_best:
        best_model_path = os.path.join(save_model_dir, 'best.pth.tar')
        shutil.copyfile(save_model_path, best_model_path)
