#!/bin/bash -l
#SBATCH --job-name="Full_scale_NEURON"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=claudia.casellato@unipv.it
#SBATCH --time=01:00:00
#SBATCH --nodes=20
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=36
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=mc
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source $HOME/bsb-daint/load.sh
source $HOME/nest-2.18.0-install/bin/nest_vars.sh
srun bsb simulate stim_on_MFs --hdf5="300x_200z.hdf5"


