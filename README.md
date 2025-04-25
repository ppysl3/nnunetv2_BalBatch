# nnunetv2
Modified for Cluster Loading.<br>
<br>
Instructions being developed.
<br>
Installation <br>
First, install the original nnUNETV2 normally using python 3.9 in a conda environment <br>
You will require torch version 2.0.1+cu117 <br>
Ensure, using your own test set, that nnUNETV2 is working correctly <br>
Then, replace the nnUNET package files with the ones contained in this repo. You'll find at the top level, the folders are named identically. <br>
Steps to run:<br>
1: Under training/dataloading, point line 131 in cluster_data_loader to the full imagesTr (I.E All images in the full dataset) <br>
2: Insert your cluster file inside the DatasetXXX_NAME folder in nnUNET_Preprocessed<br>
3: Run using the nnUNetTrainerClusterLoad trainer.<br>
<br>
Your cluster file should be as following:<br>
<br>
For your entire imagesTr, your cluster file should be a 1 by N vertical array of cluster assignments, saved as a numpy file.<br>
For example, if you had 5 images, and 2 cluster classes, your array could look like:<br>
<br>
0<br>
0<br>
1<br>
1<br>
0<br>
<br>
The file should be saved out with the ending "preds.npy", for example, "EXAMPLEpreds.npy".<br>
