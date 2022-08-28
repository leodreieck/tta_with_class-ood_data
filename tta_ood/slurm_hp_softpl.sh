#!/bin/bash

#SBATCH --job-name=spl_ooddetection_lr0.003                             # Job name #tent_cifar100c_100
#SBATCH --output=/mnt/qb/work/bethge/ldemleitner05/bachelorthesis/tta_ood/99_slurm/%x_%j.out      # File to which STDOUT will be written
#SBATCH --error=/mnt/qb/work/bethge/ldemleitner05/bachelorthesis/tta_ood/99_slurm/%x_%j.err       # File to which STDERR will be written

#SBATCH --partition=gpu-2080ti-preemptable                              # Partition to submit to
#SBATCH --time=0-3:00                                                  # Runtime in D-HH:MM

#SBATCH --ntasks=1                                                      # Number of tasks (see below)
#SBATCH --nodes=1                                                       # Ensure that all cores are on one machine
#SBATCH --mem=24G                                                       # Memory pool for all cores
#SBATCH --gres=gpu:1                                                    # Request one GPU
#SBATCH --cpus-per-task=8                                               # Evi: always choose 8 cpus per gpu

# from Evi: SBATCH --cpus-per-task=32
# from Evi: SBATCH --array=13-14                                        # maps 0 to 3 to SLURM_ARRAY_TASK_ID below  #507-513

#SBATCH --array=1-12                                                # should match number of corruptions in input txt file

params=($(cat 03_cfgs/hp_settings_ooddetection_lr0.003.txt | sed ${SLURM_ARRAY_TASK_ID:-1}'q;d') )
corruption_par=${params[0]}
svhn_par=${params[1]}
svhnc_par=${params[2]}
cifar100_par=${params[3]}
cifar100c_par=${params[4]}
lr_par=${params[5]}
pl_t_par=${params[6]}
ood_m_par=${params[7]}
ood_t_par=${params[8]}
echo "corruption_par $corruption_par"
echo "ood_t_par $ood_t_par"

start_time=$SECONDS

# include information about the job in the output
scontrol show job=$SLURM_JOB_ID

# run the actual command
srun singularity exec \
--bind /mnt/qb/work/bethge/ldemleitner05/ \
--nv \
/mnt/qb/work/bethge/ldemleitner05/ba1.sif python3 /mnt/qb/work/bethge/ldemleitner05/bachelorthesis/tta_ood/cifar10c.py \
    --cfg 03_cfgs/softpl.yaml \
    DATA_DIR 02_data \
    SAVE_DIR 04_output/output_leo \
    CKPT_DIR 05_ckpt \
    CORRUPTION.TYPE [$corruption_par] \
    CORRUPTION.SEVERITY [1,2,3,4,5] \
    N_EPOCHS 6 \
    OPTIM.LR $lr_par \
    MODEL.CREATE_EMBEDDINGS False \
    CORRUPTION.SVHN_samples $svhn_par \
    CORRUPTION.SVHNC_samples $svhnc_par \
    CORRUPTION.CIFAR100_samples $cifar100_par \
    CORRUPTION.CIFAR100C_samples $cifar100c_par \
    MODEL.OOD_METHOD $ood_m_par \
    MODEL.OOD_THRESHOLD $ood_t_par \


elapsed=$(( SECONDS - start_time ))
echo "Time elapsed: $elapsed"

echo DONE.