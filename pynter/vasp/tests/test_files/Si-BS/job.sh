#!/bin/sh
#SBATCH -A project0000
#SBATCH --job-name=Si-BS_PBE-el-str_3
#SBATCH --mail-user=test@pynter.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --output=out.%j
#SBATCH --error=err.%j
#SBATCH --time=00:30:00
#SBATCH -p deflt
#SBATCH --exclusive
#SBATCH --mem-per-cpu=2400
#SBATCH -C avx2
ml intel/2019.2 
ml intel/2019.3 
ml intelmpi/2019.3 
ml fftw/3.3.8 

srun /home/vasp-5-3-3

automation_vasp.py --contcar --chgcar --wavecar --check-kpoints --error-check
