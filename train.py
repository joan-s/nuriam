#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 23:40:22 2022

@author: joan
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from snapshot_dataset import SnapshotDataset
from model import UNet3D
import logging
from utils import get_logger, RunningAverage
from sklearn.metrics import jaccard_score
from sklearn.metrics import confusion_matrix


logger = get_logger('train',level=logging.INFO)


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')
    return device
    

def net_to_device(net):  
    device = get_device()      
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:            
            net_parallel = torch.nn.DataParallel(net) 
            logger.info("Using", torch.cuda.device_count(), 'GPUs')
            return net_parallel.to(device)
        else:
            logger.info("Using just one GPU")
            return net.to(device)
    else:
        logger.warning('CUDA not available, using CPU !')
        return net.to(device)
    

#
# Parameters and input data
#
if True:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    # use only the first gpu, no parallelization
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" 
    # use 4 gpus in parallel
    
if False:
    snapshots_train = ['snapshots/fake/snapshot0.npz',
                       'snapshots/fake/snapshot1.npz',
                       'snapshots/fake/snapshot2.npz',]
    snapshots_test = ['snapshots/fake/snapshot3.npz',
                      'snapshots/fake/snapshot4.npz',]
    crop_size = (32, 32, 32)
    stride = (8, 8, 8)
    select_features = [] # all
    batch_size = 10
    learning_rate = 1e-4
    num_epochs = 1
else:
    snapshots_train = ['snapshots/dataset_snapshot_1.npz',
                       'snapshots/dataset_snapshot_2.npz',
                       'snapshots/dataset_snapshot_3.npz',]
    snapshots_test = ['snapshots/dataset_snapshot_4.npz',]
    crop_size = (32, 32, 32)
    stride = (8, 8, 8)
    select_features = [0, 1, 2, 3, 4 ,5, 6, 7]
    batch_size = 10
    learning_rate = 1e-4
    num_epochs = 1

#
# Make datasets of train and validation and corresponding dataloaders
# from existing snapshots
# TODO: make a valitation dataset-dataloader for evaluation during training
#
train_dataset = SnapshotDataset(snapshots_train, crop_size, stride, 
                                select_features=select_features)
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
logger.info('There is a total of {} train samples of size {}'\
            .format(len(train_dataloader), list(crop_size) + [len(select_features)]))

snapshot_size = train_dataset.size()
num_features = snapshot_size[-1]
logger.info('snapshot size : {}'.format(snapshot_size))

test_dataset = SnapshotDataset(snapshots_test, select_features=select_features)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#
# Make network and move it into the GPU(s)
#
num_classes = 3
net = UNet3D(num_features, num_classes, num_groups=8, final_sigmoid=False)
# in_channels must be divisible by num_groups
logger.debug(net)
net = net_to_device(net)

#
# Loss function
#
loss_func = torch.nn.CrossEntropyLoss()

#
# Optimizer
#
optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate) 

#
# Train loop
# TODO: evaluate every n steps or after each epoch
# TODO: define evaluation metric(s), see scikit-learn and papers (dice, mIoU)
#
device = get_device()
for epoch in range(1, num_epochs+1):
    logger.info('\nEpoch {}'.format(epoch))
    net.train()
    for nbatch, (features, labels) in enumerate(train_dataloader):
        logger.debug('net input size : {}'.format(features.size()))
        features = features.to(device)
        labels = labels.to(device)
        out = net(features)
        logger.debug('net output size : {}'.format(out.size()))
        loss = loss_func(out, labels.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.info('batch {}, loss {}'.format(nbatch, loss.item()))

        #if nbatch==1:
        #    break
        
    # evaluation on the validation set
    running_avg_confusion_matrix = RunningAverage()
    running_avg_per_class_iou = RunningAverage()
    net.eval()
    for nbatch, (features, labels) in enumerate(test_dataloader):
        features = features.to(device)
        out = net(features)
        probs = torch.nn.functional.softmax(out, dim=1)
        num_samples = len(features)
        
        y_pred = torch.argmax(probs, dim=1).to('cpu').numpy().flatten()
        y_true = labels.long().to('cpu').numpy().flatten()
        cm = confusion_matrix(y_true, y_pred)
        running_avg_confusion_matrix.update(cm, n=1)
        jac = jaccard_score(y_true, y_pred, average=None)
        running_avg_per_class_iou.update(jac, num_samples)
    
    per_class_iou = running_avg_per_class_iou.avg
    mean_iou = running_avg_per_class_iou.avg.mean()
    conf_mat = running_avg_confusion_matrix.sum
    # next metrics are derived from the confision matrix
    accuracy = np.trace(conf_mat)/np.sum(conf_mat)
    precision = np.diag(conf_mat)/conf_mat.sum(axis=0)
    recall = np.diag(conf_mat)/conf_mat.sum(axis=1)
    logger.info(
        'validation epoch {} :\n'
        'confusion matrix\n{}\n'
        'accuracy {}\n'
        'precision {}\n'
        'recall {}\n'
        'per-class-IoU {}\n'
        'mean IoU {}'\
        .format(epoch, conf_mat, accuracy, precision, recall,
                        per_class_iou, mean_iou))
    
    


