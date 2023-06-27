import numpy as np
from nnunetv2.training.dataloading.mod4cluster_base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
import random
import sys

#We need to edit reset such that it iniialises the cluster indicies
#We need to edit get_indicies such that it selects N indicies from N clusters.
#We should enforce that N must be divisible by batch size

class nnUNetClusterDataLoader2D(nnUNetDataLoaderBase):
    def determine_shapes(self):
        # load one case
        print("RunningClusterLoader")
        data, seg, properties = self._data.load_case(self.indices[0])
        num_color_channels = data.shape[0]
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, seg.shape[0], *self.patch_size)
        return data_shape, seg_shape
    def generate_train_batch(self):
        print("GENERATETRAINBATCH")
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []
        print(selected_keys)
        for j, current_key in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)
            data, seg, properties = self._data.load_case(current_key)

            # select a class/region first, then a slice where this class is present, then crop to that area
            if not force_fg:
                if self.has_ignore:
                    selected_class_or_region = self.annotated_classes_key
                else:
                    selected_class_or_region = None
            else:
                # filter out all classes that are not present here
                eligible_classes_or_regions = [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) > 0]

                # if we have annotated_classes_key locations and other classes are present, remove the annotated_classes_key from the list
                # strange formulation needed to circumvent
                # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
                tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
                if any(tmp):
                    if len(eligible_classes_or_regions) > 1:
                        eligible_classes_or_regions.pop(np.where(tmp)[0][0])

                selected_class_or_region = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
                    len(eligible_classes_or_regions) > 0 else None
            if selected_class_or_region is not None:
                selected_slice = np.random.choice(properties['class_locations'][selected_class_or_region][:, 1])
            else:
                selected_slice = np.random.choice(len(data[0]))

            data = data[:, selected_slice]
            seg = seg[:, selected_slice]

            # the line of death lol
            # this needs to be a separate variable because we could otherwise permanently overwrite
            # properties['class_locations']
            # selected_class_or_region is:
            # - None if we do not have an ignore label and force_fg is False OR if force_fg is True but there is no foreground in the image
            # - A tuple of all (non-ignore) labels if there is an ignore label and force_fg is False
            # - a class or region if force_fg is True
            class_locations = {
                selected_class_or_region: properties['class_locations'][selected_class_or_region][properties['class_locations'][selected_class_or_region][:, 1] == selected_slice][:, (0, 2, 3)]
            } if (selected_class_or_region is not None) else None

            # print(properties)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg if selected_class_or_region is not None else None,
                                               class_locations, overwrite_class=selected_class_or_region)

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)
        return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': selected_keys}
    
    
    
    
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
        listoffives=[0,1,2,3,4,5,6,7,8,9]
        listoftwos=[10,11]
        random.shuffle(listofzeros)
        random.shuffle(listofones)
        random.shuffle(listoftwos)
        random.shuffle(listofthrees)
        random.shuffle(listoffours)
        random.shuffle(listoffives)
        random.shuffle(listofsixes)
        random.shuffle(listofsevens)
        print("!!SHUFFLE!!")
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
        if self.last_reached:
            self.reset()
            arraytot=self.actualarray
            counters=self.counters
            raise StopIteration
        if not self.was_initialized:
            self.reset()
            arraytot=self.actualarray
            counters=self.counters
        arraytot=self.actualarray
        counters=self.counters
            #print("INIT")
            #print(arraytot)
        #Get our array from above
        #arraytot=list(self.actualarray)
        numarray=len(arraytot)
        #print(arraytot)
        tempindices = []
        indices=[]
        if self.batch_size % len(arraytot) != 0:
            raise Exception ("BATCH SIZE ERROR: Batch size must be divisble by number of clusters, number of clusters is " + str(len(arraytot)))
        if len(self.indices)  % self.batch_size != 0:
            raise Exception("BATCH SIZE ERROR: Number of images must be divisible by batch size")
        currentprogress=0
        while currentprogress < self.batch_size:
            print(counters[0])
            if self.last_reached==True:
                break
            for num, array in enumerate(arraytot):
                if self.current_position < len(self.indices):
                    counter=counters[num]
                    if counter==0:
                        #This is a redundant bit of code which ensures that newly initiated arrays are shuffled before any selection.
                        #Also sets the first numselect, and the modulo operator will go crazy at 0 otherwise.
                        numselect=counter
                        print("ShuffleDueToZeroCounter")
                        random.shuffle(array)
                        arraytot[num]=array
                    else:
                        numselect=(counter % len(array))
                    counters[num]=counter+1
                    numberchosen=array[numselect]   
                    tempindices.append(numberchosen)
                    currentprogress=currentprogress+1
                    self.current_position += 1
                else:
                    print("LAST REACHED")
                    self.last_reached = True
                    break
                if numselect+1==len(array):
                        #This is here to shuffle when getting to the end of an array.
                        print("Shuffle after next batch for "+str(num))
                        random.shuffle(array)
                        arraytot[num]=array
        #periodically update self.counters
        self.counters=counters
        self.actualarray=arraytot
        for i in tempindices:
            indices.append(self.indices[i])
        indices=np.array(indices)
        #sys.exit()
        if len(indices) > 0 and ((not self.last_reached) or self.return_incomplete):
            self.current_position += (self.number_of_threads_in_multithreaded - 1) * self.batch_size
            if self.current_position == len(self.indices):
                print("RESETTING: New batch should be incoming")
                self.reset()
            return indices
        else:
            self.reset()
            raise StopIteration

if __name__ == '__main__':
    folder = '/media/fabian/data/nnUNet_preprocessed/Dataset004_Hippocampus/2d'
    ds = nnUNetDataset(folder, None, 1000)  # this should not load the properties!
    dl = nnUNetDataLoader2D(ds, 366, (65, 65), (56, 40), 0.33, None, None)
    a = next(dl)
