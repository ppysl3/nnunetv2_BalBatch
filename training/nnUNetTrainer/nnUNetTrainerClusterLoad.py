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
            print("ONE!!!!!!!!!!!!!!!!!!")
            dl_tr = nnUNetClusterDataLoader2D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None)
            print("TWOOOOOOOOOOOOOOOOOOOOO")
            dl_val = nnUNetDataLoader2D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None)
            print("THREEEEEEEEEEEEEEEEEEEE")
        else:
            print("UNSUITABLE DIMENSIONS")
            sys.exit()
        return dl_tr, dl_val
    def get_dataloaders(self):
            # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
            # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
            patch_size = self.configuration_manager.patch_size
            dim = len(patch_size)

            # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
            # outputs?
            deep_supervision_scales = self._get_deep_supervision_scales()

            rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
                self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

            # training pipeline
            tr_transforms = self.get_training_transforms(
                patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
                order_resampling_data=3, order_resampling_seg=1,
                use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
                is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
                regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
                ignore_label=self.label_manager.ignore_label)

            # validation pipeline
            val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                            is_cascaded=self.is_cascaded,
                                                            foreground_labels=self.label_manager.foreground_labels,
                                                            regions=self.label_manager.foreground_regions if
                                                            self.label_manager.has_regions else None,
                                                            ignore_label=self.label_manager.ignore_label)

            dl_tr, dl_val = self.get_plain_dataloaders(initial_patch_size, dim)

            allowed_num_processes = get_allowed_n_proc_DA()
            if allowed_num_processes == 0:
                mt_gen_train = SingleThreadedAugmenter(dl_tr, tr_transforms)
                mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
            else:
                mt_gen_train = LimitedLenWrapper(self.num_iterations_per_epoch, data_loader=dl_tr, transform=tr_transforms,
                                                num_processes=allowed_num_processes, num_cached=6, seeds=None,
                                                pin_memory=self.device.type == 'cuda', wait_time=0.02)
                mt_gen_val = LimitedLenWrapper(self.num_val_iterations_per_epoch, data_loader=dl_val,
                                            transform=val_transforms, num_processes=max(1, allowed_num_processes // 2),
                                            num_cached=3, seeds=None, pin_memory=self.device.type == 'cuda',
                                            wait_time=0.02)
            return mt_gen_train, mt_gen_val

