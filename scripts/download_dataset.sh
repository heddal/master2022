
module purge
module load Python/3.9.6-GCCcore-11.2.0
# Create environment if it doesn't exist
if [ ! -d "env" ]; then
  python3 -m venv env
fi

source env/bin/activate
pip install gdown
URL=https://drive.google.com/file/d/1s4vJh6u3xYGpQ7yc6IvHqQZGTXw9MR7M/view?usp=sharing
echo "downloading from $URL"
gdown $URL
