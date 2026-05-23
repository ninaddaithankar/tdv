#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION

### NODE/CPU/MEM/GPU  ###

#SBATCH --mem-bind=verbose,local
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem-per-gpu=118G
#SBATCH --cpus-per-gpu=72