#!/bin/bash
#SBATCH --mail-user=koen.essers@ru.nl	#!
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --output=output_slurm/R-%x_%A_%a.out	#!
#SBATCH --error=output_slurm/R-%x_%A_%a.err	#!
#SBATCH -p proj_stev
#SBATCH --mem=8G
#SBATCH -N 1 -n 8 
#SBATCH --time=0-48:00:00


idx=${SLURM_ARRAY_TASK_ID}	# this is the index of the array of jobs your have pushed with one sbatch request

MODEL_DIRS=($(ls -d /vol/astro8/onnop/kessers/tpagb-mass-transfer/mesa-models/binary-tpagb-grid-2/runs/*))
WORK_DIR=${MODEL_DIRS[$idx]}	# your will once again need to change your_path to the place where your MESA models are, and change your_model to the directory name of that MESA model and appended to that the corresponding index (i.e. if you push an array of 2 jobs, you will have needed to create beforehand for example your_model_0 and your_model_1 yourself, these specific index values can be defined with the sbatch commend, see COMA guide for more info) 

echo $WORK_DIR

if [ -z "$WORK_DIR" ]; then
    echo "ERROR: WORK_DIR is empty"
    exit 1
fi

#-- initialisation
set -x
SCRATCH_DIR=/scratch/kessers/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}	#!
mkdir -p $SCRATCH_DIR || exit 1
cd $SCRATCH_DIR || exit 1

echo $SCRATCH_DIR


#-- copying code
scp -vr coma01:$WORK_DIR/* ./ 2>/dev/null

#-- code 
export MESA_DIR=/vol/astro8/onnop/kessers/mesa/mesa-r23.05.1/   	   	#! change this to the directory of your MESA installation, you will have had to install MESA in your directory on astro8 beforehand
export OMP_NUM_THREADS=8
export MESASDK_ROOT=/vol/astro8/onnop/kessers/mesa/mesasdk-21.4.1/       #! change this to the directory of your MESA SDK, which you will have had to setup in your directory on astro9 beforehand
source $MESASDK_ROOT/bin/mesasdk_init.sh
mkdir caches
export MESA_CACHES_DIR=$SCRATCH_DIR/caches
./clean && ./mk
./rn | tee terminal_output.txt
rm -r caches

#-- finalisation 
SIM_EXITSTAT=$?
rsync -avrP $SCRATCH_DIR/* coma01:$WORK_DIR 2>/dev/null
rm -rf $SCRATCH_DIR
exit $SIM_EXITSTAT
