# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time: 2021/3/6 4:40 PM
@Author: Qinyang Lu
"""
import torch
import torch.nn as nn

import numpy as np
import cv2
from sklearn import metrics


def get_metrics(scores, labels):
    fpr, tpr, ths = metrics.roc_curve(labels, scores)
    auc = round(metrics.auc(fpr, tpr), 3)

    rightindex = tpr - fpr
    index = np.argmax(rightindex)
    th = ths[index]

    labels_l = np.array(labels)
    scores_l = np.array(scores)
    scores_l[scores_l > th] = 1
    scores_l[scores_l != 1] = 0
    tp = np.sum(scores_l * labels_l)
    a = labels_l + scores_l
    sub = labels_l - scores_l

    tn = np.where(a == 0)[0].shape[0]
    fp = np.where(sub == -1)[0].shape[0]

    acc = (tp + tn) / labels_l.shape[0]
    recall = tp / np.sum(labels_l)
    precision = tp / np.sum(scores_l)
    specificity = tn / (tn + fp)

    return acc, recall, precision, specificity, auc, th


def eval_cls(net, dataloader, gpu):
    net.eval()
    labels = []
    scores = []
    losses = 0

    criterion = nn.CrossEntropyLoss()
    i = 0
    for i, (img, label) in enumerate(dataloader):
        if gpu:
            img = img.cuda()
            label = label.cuda()

        logits = net(img)
        label_flat = label.view(-1)

        loss = criterion(logits, label_flat.long())
        losses = losses + loss.item()

        predictions_ = torch.softmax(logits, dim=1)
        scores.append(round(predictions_[0, 1].item(), 4))
        labels.append(label.item())

    acc, recall, precision, specificity, auc, th = get_metrics(scores, labels)

    net.train()

    return acc, recall, precision, specificity, auc, th, losses / (i+1)
