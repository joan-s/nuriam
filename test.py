#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 18:47:13 2022

@author: joan
"""
import os
import logging
import numpy as np
from sklearn.metrics import jaccard_score
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader
from utils import get_logger, load_checkpoint, net_to_device, RunningAverage, \
    get_device
from model import UNet3D
from snapshot_dataset import SnapshotDataset



def accuracy(cm):
    return np.trace(cm)/np.sum(cm)

def precision(cm):
    return np.diag(cm)/cm.sum(axis=0)

def recall(cm):
    return np.diag(cm)/cm.sum(axis=1)

def compute_metrics(ytrue, ypred):
    cm = confusion_matrix(y_true, y_pred)
    per_class_iou = jaccard_score(y_true, y_pred, average=None)
    # next metrics are derived from the confusion matrix
    return cm, per_class_iou, per_class_iou.mean(), accuracy(cm), \
        precision(cm), recall(cm)
        

logger = get_logger('test', level=logging.INFO)
experiment_id = '968615'
epoch = 48  
gpu = '7'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
# now we are using a single GPU
config, model_state_dict, _, _ = load_checkpoint(experiment_id, epoch, logger)
snapshots_test = config['snapshots_test']
select_features = config['select_features']

test_dataset = SnapshotDataset(snapshots_test, select_features=select_features)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
snapshot_size = test_dataset.size()
logger.info('{} test snapshots of size : {}'\
            .format(len(snapshots_test), snapshot_size))

num_features = snapshot_size[-1]
num_classes = config['num_classes']
num_groups = config['num_groups']
net = UNet3D(num_features, num_classes, num_groups=num_groups, final_sigmoid=False)
net = net_to_device(net, logger)                                                                                                                                        
net.load_state_dict(model_state_dict)

device = get_device()
running_avg_confusion_matrix = RunningAverage()
running_avg_per_class_iou = RunningAverage()
net.eval()
for nbatch, (features, labels) in enumerate(test_dataloader):
    # each batch is a whole snapshot
    features = features.to(device)
    out = net(features)
    num_samples = len(features)
    
    pred = torch.argmax(out, dim=1).to('cpu').numpy().astype(np.uint8)
    np.savez('results/prediction_snapshot_{}_experiment_{}.npz'\
             .format(nbatch, experiment_id), pred=pred)
    y_pred = pred.flatten()
    y_true = labels.long().numpy().flatten()
    cm, per_class_iou, mean_iou, acc, prec, rec = \
        compute_metrics(y_true, y_pred)
    logger.info(
        'snapshot {}\n'
        'confusion matrix\n{}\n'
        'accuracy {}\n'
        'precision {}\n'
        'recall {}\n'
        'per-class-IoU {}\n'
        'mean IoU {}'\
        .format(snapshots_test[nbatch], cm, acc, prec, rec, 
                per_class_iou, mean_iou))
    running_avg_confusion_matrix.update(cm, n=1)
    running_avg_per_class_iou.update(per_class_iou, num_samples)

global_per_class_iou = running_avg_per_class_iou.avg
global_mean_iou = global_per_class_iou.mean()
global_conf_mat = running_avg_confusion_matrix.sum
global_accuracy = accuracy(global_conf_mat)
global_precision = precision(global_conf_mat)
global_recall = recall(global_conf_mat)
logger.info(
    'Global metrics :\n'
    'confusion matrix\n{}\n'
    'accuracy {}\n'
    'precision {}\n'
    'recall {}\n'
    'per-class-IoU {}\n'
    'mean IoU {}'\
    .format(global_conf_mat, global_accuracy, global_precision, global_recall,
            global_per_class_iou, global_mean_iou))

