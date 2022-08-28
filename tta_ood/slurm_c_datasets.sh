#!/bin/bash

#SBATCH --job-name=create_svhn_c                                            # Job name
#SBATCH --output=/mnt/qb/work/bethge/ldemleitner05/bachelorthesis/tta_ood/99_slurm/%x_%j.out      # File to which STDOUT will be written
#SBATCH --error=/mnt/qb/work/bethge/ldemleitner05/bachelorthesis/tta_ood/99_slurm/%x_%j.err       # File to which STDERR will be written

#SBATCH --partition=cpu-preemptable                              # Partition to submit to
#SBATCH --time=0-05:15                                                  # Runtime in D-HH:MM

#SBATCH --ntasks=1                                                      # Number of tasks (see below)
#SBATCH --nodes=1                                                       # Ensure that all cores are on one machine
#SBATCH --mem=16G                                                       # Memory pool for all cores
# #SBATCH --gres=gpu:1                                                    # Request one GPU
#SBATCH --cpus-per-task=8                                               # Evi: always choose 8 cpus per gpu

# from Evi: SBATCH --cpus-per-task=32
# from Evi: SBATCH --array=13-14                                        # maps 0 to 3 to SLURM_ARRAY_TASK_ID below  #507-513

# dont need this SBATCH --array=1                                                     # should match number of corruptions in input txt file

start_time=$SECONDS

# include information about the job in the output
scontrol show job=$SLURM_JOB_ID

# run the actual command
srun singularity exec \
--bind /mnt/qb/work/bethge/ldemleitner05/ \
/mnt/qb/work/bethge/ldemleitner05/ba2.sif python3 /mnt/qb/work/bethge/ldemleitner05/bachelorthesis/tta_ood/create_c_dataset.py \
    --ds SVHN \
    --dd /mnt/qb/work/bethge/ldemleitner05/bachelorthesis/tta_ood/02_data/ \

elapsed=$(( SECONDS - start_time ))
echo "Time elapsed: $elapsed"

echo DONE.