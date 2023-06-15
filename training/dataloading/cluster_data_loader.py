import numpy as np
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset

class nnUNetClusterDataLoader2D(nnUNetDataLoaderBase):
    def generate_train_batch(self):
