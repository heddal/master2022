#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --account=ie-idi
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=256000
#SBATCH --job-name="train_paintings"
#SBATCH --output=out/landscapes/B/32734677.out
#SBATCH --mail-user=hmlangvi@stud.ntnu.no
#SBATCH --mail-type=ALL

echo "Lets go"

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
module load CUDA/11.3.1
module load GCCcore/11.2.0
module load Python/3.9.6-GCCcore-11.2.0

# Create environment if it doesn't exist
if [ ! -d "env" ]; then
  python3 -m venv env
fi

source env/bin/activate

# Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

song_id=32734677
results_folder=./results/landscapes/B/${song_id}

#!./scripts/train_cyclegan.sh
echo "--------------Starting python script----------------"
quadrant=$(python3 Get_songs_in_quadrants.py ${song_id})
#quadrant=1
echo "Quadrant: $quadrant"
python3 test.py --dataroot ./datasets/landscapes/B/${quadrant} --name photo2art_hedda --model test --no_dropout --results_dir ${results_folder}
python3 select_random_image.py ${results_folder}
echo "-----------------------COMPLETE-------------------"
uname -a
