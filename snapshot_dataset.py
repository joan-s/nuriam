#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 11:23:56 2022

@author: joan
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import logging
from utils import get_logger


logger = get_logger('SnapshotDataset', level=logging.INFO)

# We make a custom dataset that is a number of crops of a single snapshot,
# being each crop of the same size and extracted at random positions from the
# snapshot volume.
# TODO: This dataset reads all the given snapshots and keeps them in memory in 
# order to crop one of them each time a sample is requested by the dataloader. 
# When the number of snapshots and/or their size will be much greater this may be
# impossible. Another SnapshotDataset class should be done that reads the crops 
# previously computed and saved into files.
class SnapshotDataset(Dataset):
    def __init__(self, filenames_snapshots, crop_size=None, stride=None,  
                 data_key='x', label_key='y', select_features=[]):
        self.filenames_snapshots = filenames_snapshots
        self.select_features = select_features
        # we may want to test classification with a subset of features, or
        # select some because unet3d imposes constraints on its number
        self.snapshots = []
        self.labels = []
        self.snapshot_size = None
        for fn in self.filenames_snapshots:
            f = np.load(fn)
            snapshot = f[data_key]
            label = f[label_key]
            if self.snapshot_size is None:
                self.snapshot_size = snapshot.shape
            else:
                assert self.snapshot_size == snapshot.shape, \
                    'for {} data shape is different from first snapshot: '\
                    '{} , {}'.format(fn, self.snapshot_size, snapshot.shape)
                # all snapshots must have the same shape
            assert self.snapshot_size[:3] == label.shape, \
                'for {} data and labels have different shapes: {} , {}'\
                .format(fn, self.snapshot_size[:3], label.shape)
            self.snapshots.append(snapshot)
            self.labels.append(label)
            logger.info('loaded snapshot {}'.format(fn))
            
        self.snapshot_size = list(self.snapshot_size)
        if self.select_features:
            self.snapshot_size[-1] = len(self.select_features)

        assert (crop_size is None and stride is None) \
            or (crop_size is not None and stride is not None), \
            'crop_size and stride must be both None or not None'
        if crop_size is None:
            self.crop_size = self.snapshot_size[:3]
            self.stride = self.crop_size
            # there will be just as much samples as snapshots, each one being
            # a whole snapshot
        else:
            self.crop_size = crop_size
            self.stride = stride
        
        self.num_snapshots = len(self.snapshots)
        assert np.all(np.array(self.snapshot_size[:3]) - np.array(self.crop_size) >= 0)
        # make sure we can make a crop
        
        # compute the upper left coordinates of all patches
        w = np.arange(0, self.num_snapshots)
        x = np.arange(0, self.snapshot_size[0] - self.crop_size[0] + 1, step=self.stride[0])
        y = np.arange(0, self.snapshot_size[1] - self.crop_size[1] + 1, step=self.stride[1])
        z = np.arange(0, self.snapshot_size[2] - self.crop_size[2] + 1, step=self.stride[2])
        w, x, y, z = np.meshgrid(w,x,y,z)
        self.w = w.ravel()
        self.x = x.ravel()
        self.y = y.ravel()
        self.z = z.ravel()
        
        
    def size(self):
        return self.snapshot_size

    def __len__(self):
        return len(self.w)

    def __getitem__(self, idx):
        # TODO: add simple data augmentation in the form o flip and 3d 90 deg. 
        # rotations but take into account that due to anisotropy (different 
        # spacing in x, y, z) not all rotations will be allowed
        # See https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/augment/transforms.py
        w0 = self.w[idx]
        x0 = self.x[idx]
        y0 = self.y[idx]
        z0 = self.z[idx]
        x1 = x0 + self.crop_size[0]
        y1 = y0 + self.crop_size[1]
        z1 = z0 + self.crop_size[2]
        logger.debug('{}, {}, {}, {} - {}, {}, {}, {}'\
                     .format(w0, x0, y0, z0, w0, x1, y1, z1 ))
        crop = self.snapshots[w0][x0:x1, y0:y1, z0:z1, :]
        label = self.labels[w0][x0:x1, y0:y1, z0:z1]
        if self.select_features:
            # keep only the indexed features (last dimension) of the crop
            crop = crop[:, :, :, self.select_features]

        crop = np.transpose(crop, [3, 0, 1, 2]) # pytorch wants features x height x width x depth
        return torch.tensor(crop, dtype=torch.float32), \
               torch.tensor(label, dtype=torch.float32)

            

# Create fake snaphots data + groundtruth, to be used for train and test
def make_fake_snapshots(num_snapshots, snapshot_size, num_classes):
    logger.info('making {} snapshots with\n\t{} classes\n\t{} features per voxel'
                '\n\tsize {}'.format(num_snapshots, num_classes, snapshot_size[-1],
                                      snapshot_size))
    for i in range(num_snapshots):
        snapshot = np.random.uniform(size=tuple(snapshot_size))
        label = np.random.randint(low=0, high=num_classes, size=snapshot_size[:3])
        fn = os.path.join('snapshots', 'fake', 'snapshot{}.npz'.format(i))
        np.savez_compressed(fn, x=snapshot, y=label)
        logger.info('made and saved snapshot {} of size {}'.format(fn, snapshot_size))



def get_batches(train_dataloader, test_dataloader):
    # test getting a batch
    print('\n{} train samples per epoch'.format(len(train_dataloader)))
    for nbatch, (train_features, train_labels) in enumerate(train_dataloader):
        print('train batch {}'.format(nbatch))
        print('features shape: {}'.format(train_features.size()))
        print('labels shape: {}'.format(train_labels.size()))
        if nbatch==5:
            break
    
    print('\n{} test samples'.format(len(test_dataloader)))
    # should be only as much batches as snapshots
    for nbatch, (test_features, test_labels) in enumerate(test_dataloader):
        print('test batch {}'.format(nbatch))
        print('features shape: {}'.format(test_features.size()))
        print('labels shape: {}'.format(test_labels.size()))
    
    logger.info('{} test batches'.format(nbatch+1))


def test_fake_snapshots():
    logger.info('Test fake snapshots')
    # creates a number of random snapshots plus corresponding groundtruth 
    # labels 0,1
    num_classes = 3
    num_features = 16
    snapshot_size = (64, 64, 64, num_features) # (128, 128, 128, 20)
    num_snapshots = 5
    make_fake_snapshots(num_snapshots, snapshot_size, num_classes)
    
    snapshots_train = ['snapshots/fake/snapshot0.npz',
                       'snapshots/fake/snapshot1.npz',
                       'snapshots/fake/snapshot2.npz',]
    snapshots_test = ['snapshots/fake/snapshot3.npz',
                      'snapshots/fake/snapshot4.npz',]
    crop_size = (32, 32, 32)
    stride = (8, 8, 8)
    # a train sample is a crop of the snapshot volume of features and the  
    # corresponding groundtruth (label) volume
    batch_size = 3
    train_dataset = SnapshotDataset(snapshots_train, crop_size, stride)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    
    test_dataset = SnapshotDataset(snapshots_test)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    get_batches(train_dataloader, test_dataloader)
    
    
def test_real_snapshots():
    logger.info('Test real snapshots')
    snapshots_train = ['snapshots/dataset_snapshot_1.npz',
                       'snapshots/dataset_snapshot_2.npz',
                       'snapshots/dataset_snapshot_3.npz',]
    snapshots_test = ['snapshots/dataset_snapshot_4.npz',]
    crop_size = (32, 32, 32)
    stride = (8, 8, 8)
    batch_size = 3
    train_dataset = SnapshotDataset(snapshots_train, crop_size, stride)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    
    test_dataset = SnapshotDataset(snapshots_test)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    get_batches(train_dataloader, test_dataloader)



def test_select_features():
    logger.info('Test select features')
    snapshots = ['snapshots/dataset_snapshot_1.npz',]
    crop_size = (2, 3, 4)
    stride = (8, 8, 8)
    dataset1 = SnapshotDataset(snapshots, crop_size, stride)    
    dataset2 = SnapshotDataset(snapshots, crop_size, stride, select_features=[3,2])
    
    data1, _ = dataset1.__getitem__(3)
    data2, _ = dataset2.__getitem__(3) # third sample
    logger.info('data1 shape {}'.format(data1.shape))
    logger.info('data2 shape {}'.format(data2.shape))
    logger.info('these should be equal :\n{}\n{}'.format(data1[3,:,:,:], data2[0,:,:,:]))
    logger.info('these should be equal :\n{}\n{}'.format(data1[2,:,:,:], data2[1,:,:,:]))
    assert np.array_equal(data1[3,:,:,:], data2[0,:,:,:])
    assert np.array_equal(data1[2,:,:,:], data2[1,:,:,:])
    

if __name__ == '__main__':
    test_fake_snapshots()
    test_real_snapshots()
    test_select_features()
    
    
 
