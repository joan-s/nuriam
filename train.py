#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 23:40:22 2022

@author: joan

To launch a training experiment do:
nohup python train.py --config template_experiment.yaml > results/template.out &

To stop it:
nvidia-smi, to look for the process number under column PID
kill -9 process number
nvidia-smi again to check is dead, or top or ps -ax
"""

import os
import numpy as np
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from snapshot_dataset import SnapshotDataset
from model import UNet3D
import logging
from utils import get_logger, RunningAverage
from sklearn.metrics import jaccard_score
from sklearn.metrics import confusion_matrix


logger = get_logger('train',level=logging.INFO)


def load_config():
    parser = argparse.ArgumentParser(description='UNet3D for fluid-gas segmentation')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))
    config['device'] = get_device()
    return config



def get_device():
    if torch.cuda.is_available(): # is there any visible GPU ?
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')
    return device


def net_to_device(net):  
    device = get_device()      
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            net_parallel = torch.nn.DataParallel(net) 
            logger.info('Using {} GPUs'.format(torch.cuda.device_count()))
            return net_parallel.to(device)
        else:
            logger.info('Using just one GPU')
            return net.to(device)
    else:
        logger.warning('CUDA not available, using CPU !')
        return net.to(device)
    

#
# Parameters and input data
#
config = load_config()
logger.info(config)
os.environ["CUDA_VISIBLE_DEVICES"] = config['cuda_visible_devices']

    
#
# Make datasets of train and validation and corresponding dataloaders
# from existing snapshots
# TODO: make a valitation dataset-dataloader for evaluation during training
#
snapshots_train = config['snapshots_train']
snapshots_test = config['snapshots_test']
crop_size = config['crop_size']
stride = config['stride']
select_features = config['select_features']
batch_size = config['batch_size']

train_dataset = SnapshotDataset(snapshots_train, crop_size, stride, 
                                select_features=select_features)
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
logger.info('There is a total of {} mini-batches of {} samples of size {}'\
            .format(len(train_dataloader), batch_size, \
                    list(crop_size) + [len(select_features)]))

snapshot_size = train_dataset.size()
num_features = snapshot_size[-1]
logger.info('snapshot size : {}'.format(snapshot_size))

test_dataset = SnapshotDataset(snapshots_test, select_features=select_features)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#
# Make network and move it into the GPU(s)
#
num_classes = config['num_classes']
num_groups = config['num_groups']
net = UNet3D(num_features, num_classes, num_groups=num_groups, final_sigmoid=False)
# in_channels=num_features must be divisible by num_groups
logger.debug(net)
net = net_to_device(net)


#
# Loss function
#
device = get_device()
class_weights = torch.tensor(config['class_weights']).to(device)
loss_func = torch.nn.CrossEntropyLoss(weight=class_weights)

#
# Optimizer
#
learning_rate = float(config['learning_rate'])
optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate) 


#
# Train loop
#
num_epochs = config['num_epochs']
for epoch in range(1, num_epochs+1):
    logger.info('Epoch {}'.format(epoch))
    net.train()
    for nbatch, (features, labels) in enumerate(train_dataloader):
        logger.debug('net input size : {}'.format(features.size()))
        features = features.to(device)
        labels = labels.to(device)
        labels = labels.long()
        out = net(features)
        logger.debug('net output size : {}'.format(out.size()))
        
        loss = loss_func(out, labels)      
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.info('batch {}, loss {}'.format(nbatch, loss.item()))
        
        #if nbatch==5: break
        # to debug evaluation code below soon
        
    # evaluation on the validation set
    running_avg_confusion_matrix = RunningAverage()
    running_avg_per_class_iou = RunningAverage()
    net.eval()
    for nbatch, (features, labels) in enumerate(test_dataloader):
        features = features.to(device)
        out = net(features)
        num_samples = len(features)
        
        y_pred = torch.argmax(out, dim=1).to('cpu').numpy().flatten()
        y_true = labels.long().numpy().flatten()
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
    
    


