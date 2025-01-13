# nnunetv2
Modified for Cluster Loading.

Steps to run:
1: Point line 131 in cluster_data_loader to the full imagesTr (I.E All images in the full dataset) 
2: Insert your cluster file inside the DatasetXXX_NAME folder in nnUNET_Preprocessed
3: Run using the nnUNetTrainerClusterLoad trainer.

Your cluster file should be as following.
