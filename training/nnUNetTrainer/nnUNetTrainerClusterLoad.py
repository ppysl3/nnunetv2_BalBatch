from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import numpy as np
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoDA import nnUNetTrainerNoDA
from nnunetv2.training.dataloading.cluster_data_loader import nnUNetClusterDataLoader2D
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
import sys
from typing import Union, Tuple, List
from batchgenerators.transforms.abstract_transforms import AbstractTransform
class nnUNetTrainerClusterLoad(nnUNetTrainerNoDA):
    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        
        #From NoDA's version of plain_dataloaders, adding back in case its overwritten
        initial_patch_size=self.configuration_manager.patch_size,
        dim=dim
        
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        #We only want to modify the training loader
        if dim == 2:
            dl_tr = nnUNetClusterDataLoader2D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None)
            dl_val = nnUNetDataLoader2D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None)
        else:
            print("UNSUITABLE DIMENSIONS")
            sys.exit()
        return dl_tr, dl_val
