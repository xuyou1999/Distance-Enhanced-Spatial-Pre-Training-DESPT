#!/bin/bash

#SBATCH --job-name=experiment
#SBATCH --output=slurm_log/experiement_output_%j.txt
#SBATCH --partition=mcs.gpu.q
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=4
#SBATCH --gres=gpu:4

# Load modules or software if needed
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
# Execute the script or command
python execution_tune_pretrain.py