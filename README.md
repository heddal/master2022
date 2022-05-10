# Master's Thesis 2022

This is the code for a Master's Thesis using Russell's Quadrants to unite auditory and visual stimuli. Much of the code is from [Zhu et al.'s unpaired image-to-iamage translation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). The repository is extended with new code with connects song with images based on their metadata. 


# Details from the README of the original CycleGAN repository
## Prerequisites

- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Acknowledgments

The image-to-image translation code used in this project is retrieved from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). 

## Getting Started

### Installation

- Clone this repo:

```bash
git clone https://github.com/heddal/master2022
cd master2022
```

- Install [PyTorch](http://pytorch.org) and 0.4+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.
  - For Docker users, we provide the pre-built Docker image and Dockerfile. Please refer to our [Docker](docs/docker.md) page.
  - For Repl users, please click [![Run on Repl.it](https://repl.it/badge/github/junyanz/pytorch-CycleGAN-and-pix2pix)](https://repl.it/github/junyanz/pytorch-CycleGAN-and-pix2pix).

### CycleGAN train/test

- Download a CycleGAN dataset (e.g. maps):

```bash
bash ./datasets/download_cyclegan_dataset.sh maps
```

- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.
- To log training progress and test images to W&B dashboard, set the `--use_wandb` flag with train and test script
- Train a model:

```bash
#!./scripts/train_cyclegan.sh
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```

To see more intermediate results, check out `./checkpoints/maps_cyclegan/web/index.html`.

- Test the model:

```bash
#!./scripts/test_cyclegan.sh
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```

- The test results will be saved to a html file here: `./results/maps_cyclegan/latest_test/index.html`.

### Apply a pre-trained model (CycleGAN)

- You can download a pretrained model (e.g. horse2zebra) with the following script:

```bash
bash ./scripts/download_cyclegan_model.sh horse2zebra
```

- The pretrained model is saved at `./checkpoints/{name}_pretrained/latest_net_G.pth`. Check [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/scripts/download_cyclegan_model.sh#L3) for all the available CycleGAN models.
- To test the model, you also need to download the horse2zebra dataset:

```bash
bash ./datasets/download_cyclegan_dataset.sh horse2zebra
```

- Then generate the results using

```bash
python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout
```

- The option `--model test` is used for generating results of CycleGAN only for one side. This option will automatically set `--dataset_mode single`, which only loads the images from one set. On the contrary, using `--model cycle_gan` requires loading and generating results in both directions, which is sometimes unnecessary. The results will be saved at `./results/`. Use `--results_dir {directory_path_to_save_result}` to specify the results directory.

- For pix2pix and your own models, you need to explicitly specify `--netG`, `--norm`, `--no_dropout` to match the generator architecture of the trained model. See this [FAQ](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md#runtimeerror-errors-in-loading-state_dict-812-671461-296) for more details.


## [Docker](docs/docker.md)

We provide the pre-built Docker image and Dockerfile that can run this code repo. See [docker](docs/docker.md).

## [Datasets](docs/datasets.md)

Download pix2pix/CycleGAN datasets and create your own datasets.

## [Training/Test Tips](docs/tips.md)

Best practice for training and testing your models.

## [Frequently Asked Questions](docs/qa.md)

Before you post a new question, please first look at the above Q & A and existing GitHub issues.

## Custom Model and Dataset

If you plan to implement custom models and dataset for your new applications, we provide a dataset [template](data/template_dataset.py) and a model [template](models/template_model.py) as a starting point.

## [Code structure](docs/overview.md)

To help users better understand and use our code, we briefly overview the functionality and implementation of each package and each module.
