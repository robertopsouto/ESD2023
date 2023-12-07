#!/bin/bash

#SBATCH --nodes=1                       # here the number of nodes
#SBATCH --ntasks=1                      # here total number of mpi tasks
#SBATCH --cpus-per-task=1               # number of cores per node
#SBATCH -p sequana_cpu_dev              # target partition
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
module load python/3.9.1_sequana
alias python='python3.9'
alias python3='python3.9'
module load gcc/8.3_sequana
module load openmpi/gnu/2.1.6-gcc-8.3-cuda_sequana

workdir=/scratch/cenapadrjsd/rpsouto
version=v0.17.1
partition=sequana
spackdir=${workdir}/spack/${partition}/${version}
. ${spackdir}/share/spack/setup-env.sh

export SPACK_USER_CONFIG_PATH=${workdir}/.spack/${partition}/${version}

spack load hpctoolkit

bench=${1}
class=${2}
executable="${bench}.${class}.$SLURM_NTASKS"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMPI_MCA_opal_warn_on_missing_libcuda=0

#srun --resv-ports -n $SLURM_NTASKS \
mpirun -n $SLURM_NTASKS \
hpcrun -t -e CPUTIME \
./${executable}

hpcstruct ${executable}

hpcprof \
 -I ../BT-MZ/+ \
 -S ${executable}.hpcstruct hpctoolkit-${executable}-measurements-${SLURM_JOBID}

hpctoolkitresultdir=profiling/hpctoolkit/NUMNODES-$SLURM_JOB_NUM_NODES/${bench}_${class}_MPI-${SLURM_NTASKS}_OMP-${SLURM_CPUS_PER_TASK}_JOBID-${SLURM_JOBID}

mkdir -p ${hpctoolkitresultdir}

mv slurm-${SLURM_JOBID}.out ${hpctoolkitresultdir}/
mv hpctoolkit-${executable}-database-${SLURM_JOBID} ${hpctoolkitresultdir}/
mv hpctoolkit-${executable}-measurements-${SLURM_JOBID} ${hpctoolkitresultdir}/
mv ${executable}.hpcstruct ${hpctoolkitresultdir}/

