#!/bin/bash

#SBATCH --nodes=1                       # here the number of nodes
#SBATCH --ntasks=1                      # here total number of mpi tasks
#SBATCH --ntasks-per-node=1             # here ppn = number of process per node
#SBATCH --cpus-per-task=1               # number of cores per node
#SBATCH -p sequana_cpu_dev               # target partition
#SBATCH --threads-per-core=1
#SBATCH -J NPB_BT-MZ                       # job name
#SBATCH --time=00:10:00                 # time limit
#SBATCH --exclusive                     # to have exclusive use of your nodes

echo "Cluster configuration:"
echo "==="
echo "Partition: " $SLURM_JOB_PARTITION
echo "Number of nodes: " $SLURM_NNODES
echo "Number of MPI processes: " $SLURM_NTASKS " (" $SLURM_NNODES " nodes)"
echo "Number of MPI processes per node: " $SLURM_NTASKS_PER_NODE
echo "Number of threads per MPI process: " $SLURM_CPUS_PER_TASK
echo "NPB Benchmark: " $1
echo "Bechmark class problem: " $2

###################################
#           COMPILER              #
###################################
module load sequana/current
module load git/2.23_sequana
module load cmake/3.23.2_sequana
module load python/3.8.2_sequana
alias python='python3.8'
alias python3='python3.8'
module load openmpi/gnu/4.1.2+cuda-11.2_sequana

workdir=/scratch/cenapadrjsd/rpsouto
version=v0.17.1
partition=sequana
spackdir=${workdir}/spack/${partition}/${version}
. ${spackdir}/share/spack/setup-env.sh

export SPACK_USER_CONFIG_PATH=${workdir}/.spack/${partition}/${version}

spack load scalasca

DIR=$PWD

bench=${1}
class=${2}
execfile="${bench}.${class}.$SLURM_NTASKS"
BIN=$DIR/${execfile}

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd $DIR

scalasca -analyze -s srun --resv-ports -n $SLURM_NTASKS $BIN
#scan -s srun --resv-ports -n $SLURM_NTASKS $BIN

scorepdirorig="scorep_${bench}_${SLURM_NTASKS}x${SLURM_CPUS_PER_TASK}_sum"
scorepdirdest="scorep_${bench}_${class}_sum_MPI-${SLURM_NTASKS}_OMP-${SLURM_CPUS_PER_TASK}_JOBID-${SLURM_JOBID}"
mv $scorepdirorig $scorepdirdest
mv slurm-${SLURM_JOBID}.out $scorepdirdest/

#pós-processamento contendo análise mais detalhada
scalasca -examine -s $scorepdirdest
