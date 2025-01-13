# nnunetv2
Modified for Cluster Loading.<br>
<br>
Steps to run:<br>
1: Point line 131 in cluster_data_loader to the full imagesTr (I.E All images in the full dataset) <br>
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
