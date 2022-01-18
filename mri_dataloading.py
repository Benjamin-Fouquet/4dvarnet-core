import torchio as tio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import multiprocessing
import pytorch_lightning as pl
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
from typing import Sequence, Optional, Callable, Iterable, Dict
from torchio.data import Subject
from torch.utils.data import Dataset
from new_dataloading import FourDVarNetDataset, FourDVarNetDataModule
import kornia



# q_data = torch.load('data/sample_batch_4dvarnet.torch') #batch size = 2

# mri_path = '/home/benjamin/Documents/Datasets/HCP/'

# percentage = 50

# num_workers = multiprocessing.cpu_count()

# epochs = 200

# device = [2] if torch.cuda.is_available() else []

# subjects = []

#  #only accept lists

# dataloader = DataLoader(subjects_dataset, num_workers=num_workers)

#tio SubjectsDataset superseed of torch.utils.dataset
#dataloader is classical torch dataloader

class MriDataset(Dataset):
    def __init__(self, subject_ds):
        self.subject_ds = subject_ds
    def __len__(self):
        return len(self.subject_ds)
    def __getitem__(self, idx):
        subject_item = self.subject_ds[idx]
        oi_item = kornia.gaussian_blur(subject_item['t2'])
        obs_mask_item = subject_item['rn_mask']
        obs_item = subject_item['rn_t2']
        gt_item = subject_item['t2']
        return oi_item, obs_mask_item, obs_item, gt_item


class MriDataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
    def setup(self):
        mri_path = '/home/benjamin/Documents/Datasets/HCP/'

        percentage = 50

        num_workers = multiprocessing.cpu_count()

        epochs = 200

        device = [2] if torch.cuda.is_available() else []

        subjects = []
        subject = tio.Subject(

            t2=tio.ScalarImage(mri_path + '100307_T2.nii.gz'),

            label=tio.LabelMap(mri_path + '100307_mask.nii.gz'),
        )

        transforms = [
            #centré réduit
            tio.RescaleIntensity(out_min_max=(0, 1)),

            tio.RandomAffine(),

        ]

        def create_rn_mask(subject, percentage):
            shape = subject.t2.shape
            rn_mask = torch.FloatTensor(np.random.choice([1, 0], size=shape, p=[(percentage * 0.01), 1 - ((percentage * 0.01))]))
            undersampling = rn_mask * subject.t2.data
            subject.add_image(tio.LabelMap(tensor=rn_mask, affine=subject.t2.affine), 'rn_mask')
            subject.add_image(tio.ScalarImage(tensor=undersampling, affine=subject.t2.affine), 'rn_t2')
            return None


        subjects = [subject]

        for subject in subjects:
            create_rn_mask(subject, percentage=percentage)

        transform = tio.Compose(transforms)

        subjects_dataset = tio.SubjectsDataset(subjects, transform=transform)

        mri_dataset = MriDataset(subjects_dataset)
        #later, split with random_split from torch
        self.train_ds = mri_dataset
        self.val_ds = mri_dataset
        self.test_ds = mri_dataset 

    def train_dataloader(self):
        return DataLoader(self.train_ds)
    def val_dataloader(self):
        return DataLoader(self.val_ds) 
    def test_dataloader(self):
        return DataLoader(self.test_ds)  
