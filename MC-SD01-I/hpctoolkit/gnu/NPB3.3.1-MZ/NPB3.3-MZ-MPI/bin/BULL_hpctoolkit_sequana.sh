#!/bin/bash

#SBATCH --nodes=1                       # here the number of nodes
#SBATCH --ntasks=1                      # here total number of mpi tasks
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
module load sequana/current
module load python/3.9.1_sequana
module load openmpi/gnu/4.1.2+cuda-11.2_sequana

#Load Spack v0.18.1 and HPCToolkit - BEGIN

workdir=${SCRATCH}/invmultifis
version=v0.18.1
partition=sequana
spackdir=${workdir}/tools/spack/${partition}/${version}
. ${spackdir}/share/spack/setup-env.sh

export SPACK_USER_CONFIG_PATH=${workdir}/.spack/${partition}/${version}
export SPACK_USER_CACHE_PATH=${SPACK_USER_CONFIG_PATH}/tmp

spack load hpctoolkit@2022.05.15

#Load Spack v0.18.1 and HPCToolkit - END

bench=${1}
class=${2}
executable="${bench}.${class}.$SLURM_NTASKS"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mpirun -np $SLURM_NTASKS \
hpcrun -e CPUTIME -t \
./${executable}

hpcstruct ${executable}

hpcprof \
 -I ../BT-MZ/+ \
 -S ${executable}.hpcstruct hpctoolkit-${executable}-measurements-${SLURM_JOBID}

hpctoolkitresultdir=hpctoolkit/${bench}_${class}/NUMNODES-$SLURM_JOB_NUM_NODES/MPI-${SLURM_NTASKS}/OMP-${SLURM_CPUS_PER_TASK}/JOBID-${SLURM_JOBID}
mkdir -p ${hpctoolkitresultdir}

mv slurm-${SLURM_JOBID}.out ${hpctoolkitresultdir}/
mv hpctoolkit-${executable}-database-${SLURM_JOBID} ${hpctoolkitresultdir}/
mv hpctoolkit-${executable}-measurements-${SLURM_JOBID} ${hpctoolkitresultdir}/
mv ${executable}.hpcstruct ${hpctoolkitresultdir}/

