#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --account=ie-idi
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=256000
#SBATCH --job-name="train_paintings"
#SBATCH --output=out/Wikiart/32769006.out
#SBATCH --mail-user=hmlangvi@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "We are running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "The job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

export CUDA_VISIBLE_DEVICES=0,1

echo "CUDA_VISIBLE DEVICES: $CUDA_VISIBLE_DEVICES"

CUDA_LAUNCH_BLOCKING=1

module purge
module load Python/3.9.6-GCCcore-11.2.0

# Create environment if it doesn't exist
if [ ! -d "env" ]; then
  python3 -m venv env
fi

source env/bin/activate

# Install required dependencies
pip install --upgrade pip

song_id=32769006

echo "--------------Starting python script----------------"
quadrant=$(python3 Get_songs_in_quadrants.py ${song_id})
echo "Quadrant: $quadrant"
folder=./datasets/All/${quadrant}
python3 select_modern_and_not_images.py ${folder}
echo "-----------------------COMPLETE-------------------"
uname -a
