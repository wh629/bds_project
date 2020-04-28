#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=v100_sxm2_4,p40_4,p100_4,v100_pci_2,k80_8,k80_4
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=wh629@nyu.edu
#SBATCH --mem=16000
#SBATCH --job-name=bds
#SBATCH --output=/scratch/wh629/bds/project/%j.out
module purge
module load anaconda3/5.3.1
module load cuda/10.1.105
module load gcc/6.3.0

NETID=wh629
source activate /scratch/${NETID}/nlu/env

PROJECT=/scratch/wh629/bds/project
export BDS_DATA_DIR=${PROJECT}/data
export BDS_RESULTS_DIR=${PROJECT}/results

echo python $COMMAND
python $COMMAND
