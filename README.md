# nnunetv2
Code is as standard on first commit, except to a change on line 197 of run_training that comments out the post-train validation, which runs but then crashes when the distributed training ends, resulting in slurm never finishing.
