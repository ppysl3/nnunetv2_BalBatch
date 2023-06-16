import numpy as np
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
import random
'''
We need to edit reset such that it iniialises the cluster indicies
We need to edit get_indicies such that it selects N indicies from N clusters.
We should enforce that N must be divisible by batch size


'''

class nnUNetClusterDataLoader2D(nnUNetDataLoader2D):
    def reset(self):
        assert self.indices is not None
        #print(self.indices)
        self.current_position = self.thread_id * self.batch_size

        self.was_initialized = True

        # no need to shuffle if we are returning infinite random samples
        #if not self.infinite and self.shuffle:
        #    self.rs.shuffle(self.indices)

        self.last_reached = False

        #clusters=np.load(r"C:\Users\ppysl3\OneDrive - The University of Nottingham\Postgraduate\Year 1\nnUNETAlterationTests\TCL200-8preds.npy")
        clusters=np.load(r"/home/ppysl3/TotalAutomationHam3ClusterExperiment3MainLesions/TCLModels/NumpyFiles/200-8preds.npy")
        #print(clusters)
        listofzeros=[]
        listofones=[]
        listoftwos=[]
        listofthrees=[]
        listoffours=[]
        listoffives=[]
        listofsixes=[]
        listofsevens=[]
        for idx, value in enumerate(clusters):
            if value==0:
                listofzeros.append(idx)
            elif value==1:
                listofones.append(idx)
            if value==2:
                listoftwos.append(idx)
            elif value==3:
                listofthrees.append(idx)
            if value==4:
                listoffours.append(idx)
            elif value==5:
                listoffives.append(idx)
            elif value==6:
                listofsixes.append(idx)
            elif value==7:
                listofsevens.append(idx)
        random.shuffle(listofzeros)
        random.shuffle(listofones)
        random.shuffle(listoftwos)
        random.shuffle(listofthrees)
        random.shuffle(listoffours)
        random.shuffle(listoffives)
        random.shuffle(listofsixes)
        random.shuffle(listofsevens)
        actualarray=[]
        actualarray.append(listofzeros)
        actualarray.append(listofones)
        actualarray.append(listoftwos)
        actualarray.append(listofthrees)
        actualarray.append(listoffours)
        actualarray.append(listoffives)
        actualarray.append(listofsixes)
        actualarray.append(listofsevens)
        counters=np.zeros(8, dtype=int)
        counters=list(counters)
        #print(actualarray)
        self.actualarray=actualarray
        self.counters=counters
    def get_indices(self):
        print("GET")
        # if self.infinite, this is easy
        #if self.infinite:
        #    return np.random.choice(self.indices, self.batch_size, replace=True, p=self.sampling_probabilities)
        
        

        if self.last_reached:
            self.reset()
            arraytot=self.actualarray
            counters=self.counters
            raise StopIteration

        if not self.was_initialized:
            self.reset()
            arraytot=self.actualarray
            counters=self.counters
            #print("INIT")
            #print(arraytot)
        #Get our array from above
        #arraytot=list(self.actualarray)
        numarray=len(arraytot)
        #print(arraytot)
        indices = []

        for b in range(0, self.batch_size, numarray):
            #print("b"+str(b))
            #print(self.batch_size)
            #print(numarray)
            for num, array in enumerate(arraytot):
                #print(num)
                #print(array)
                if self.current_position < len(self.indices):
                    counter=counters[num]
                    if counter==0:
                        numselect=counter
                        random.shuffle(array)
                        arraytot[num]=array
                    else:
                        numselect=(counter % len(array))
                    counters[num]=counter+1
                    #print("numsel"+str(numselect))
                    #print(array)
                    numberchosen=array[numselect]   
                    indices.append(numberchosen)

                    self.current_position += 1
                else:
                    self.last_reached = True
                    break
        print(len(indices))
        if len(indices) > 0 and ((not self.last_reached) or self.return_incomplete):
            self.current_position += (self.number_of_threads_in_multithreaded - 1) * self.batch_size
            return indices
        else:
            self.reset()
            raise StopIteration
