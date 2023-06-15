import numpy as np
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D

'''
We need to edit reset such that it iniialises the cluster indicies
We need to edit get_indicies such that it selects N indicies from N clusters.
We should enforce that N must be divisible by batch size


''''

class nnUNetClusterDataLoader2D(nnUNetDataLoader2D):
        def reset(self):
        assert self.indices is not None

        self.current_position = self.thread_id * self.batch_size

        self.was_initialized = True

        # no need to shuffle if we are returning infinite random samples
        #if not self.infinite and self.shuffle:
        #    self.rs.shuffle(self.indices)

        self.last_reached = False

    def get_indices(self):
        # if self.infinite, this is easy
        if self.infinite:
            return np.random.choice(self.indices, self.batch_size, replace=True, p=self.sampling_probabilities)

        if self.last_reached:
            self.reset()
            raise StopIteration

        if not self.was_initialized:
            self.reset()

        indices = []

        for b in range(self.batch_size):
            if self.current_position < len(self.indices):
                indices.append(self.indices[self.current_position])

                self.current_position += 1
            else:
                self.last_reached = True
                break

        if len(indices) > 0 and ((not self.last_reached) or self.return_incomplete):
            self.current_position += (self.number_of_threads_in_multithreaded - 1) * self.batch_size
            return indices
        else:
            self.reset()
            raise StopIteration

    @abstractmethod
