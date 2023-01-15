#!/bin/bash

#SBATCH --nodes=1                       # here the number of nodes
#SBATCH --ntasks=1                      # here total number of mpi tasks
#SBATCH --ntasks-per-node=1             # here ppn = number of process per node
#SBATCH --cpus-per-task=1               # number of cores per node
#SBATCH -p cpu_dev              # target partition
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
module load openmpi/gnu/2.0.4.2
module load scalasca/2.4_openmpi_gnu
module load papi/5.5.1.0
module load papi-devel/5.5.1.0

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
