
module purge
module load Python/3.9.6-GCCcore-11.2.0
# Create environment if it doesn't exist
if [ ! -d "env" ]; then
  python3 -m venv env
fi

source env/bin/activate
pip install gdown
URL_ID=1Pfy0fGrNxXH-tZ10ZU-MsgA5YejkRCB3
echo "downloading from $URL_ID"
gdown $URL_ID
