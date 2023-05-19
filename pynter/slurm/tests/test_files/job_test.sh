#!/bin/sh
#SBATCH -A projecttest0000
#SBATCH --job-name=test
#SBATCH --array=1-2%1
#SBATCH --mail-user=test@pynter.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --output=out.%j
#SBATCH --error=err.%j
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --mem-per-cpu=3500

module purge
ml intel/2020.4 
ml intelmpi/2020.4 
ml fftw/3.3.10 

if [ ! -f POSCAR_initial ] ; then
    cp POSCAR POSCAR_initial
fi
if [ -f CONTCAR ]
then
    cp CONTCAR POSCAR
fi

srun /home/test/code

convergence.py > convergence.txt
if  grep -q 'Electronic convergence: True' convergence.txt  = true  && grep -q 'Ionic convergence: True' convergence.txt  = true; then
    automation.py
    scancel ${SLURM_ARRAY_JOB_ID}_*
fi