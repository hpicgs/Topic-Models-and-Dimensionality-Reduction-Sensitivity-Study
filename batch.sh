#!/bin/bash
#SBATCH -A doellner
#SBATCH --partition=magic
#SBATCH --mem=40G
#SBATCH --array=1-9999
#SBATCH --exclude=cx23,cx27,cx28
#SBATCH --time=2000
#SBATCH --constraint="ARCH:X86"
#SBATCH --job-name=dr_benchmark_small
#SBATCH --cpus-per-task=4

#SBATCH --mail-user=Tim.Cech@hpi.de   # email address
#SBATCH --mail-type=ALL
#SBATCH --output=./status_experiment_distances/"log_perplexity-%j_%a.txt" # job standard output file (%j replaced by job id, %a by array task)
#SBATCH --error=./status_experiment_distances/"err_training-%j_%a.txt" # job standard error file (%j replaced by job id, %a by array task)


line=$(sed -n ${SLURM_ARRAY_TASK_ID}p < ./slurm_test/parameters.csv)
rec_column1=$(cut -d',' -f1 <<< "$line")
rec_column2=$(cut -d',' -f2 <<< "$line")
if [ ! -f "$rec_column1" ]; then
  echo python3 $rec_column2
  srun --container-image=./python-ml-15-02.sqsh --container-name=python-ml_batch27 --container-mounts=/hpi/fs00/share/fg-doellner/tim.cech/slurm_test:/home/tim.cech/slurm_test --container-workdir=/home/tim.cech/slurm_test python3 $rec_column2
fi
