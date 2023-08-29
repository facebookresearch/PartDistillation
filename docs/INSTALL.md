# Installation

## Requirements
- Linux with Python $\ge$ 3.6
- PyTorch $\ge$ 1.8 and torchvision that matches the PyTorch version. Please install them together at [pytorch.org](https://pytorch.org/). Note, please check PyTorch version matches that is required by Detectron2.
- Detectron2: Please follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).
- PyDenseCRF: PartDistillation uses PyDenseCRF. Please follow [PyDenseCRF](https://github.com/lucasb-eyer/pydensecrf) to install.
- Detic-dependency: PartDistillation uses Detic. Please follow [Detic](https://github.com/facebookresearch/Detic/blob/main/docs/INSTALL.md) to install properly. 

## Example conda environment setup
*NOTE: `CUDA_HOME` must be defined and points to the directory of the installed CUDA toolkit.*

```
conda create --name part_distillation python=3.9 -y
conda activate part_distillation
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -c nvidia 
pip install git+https://github.com/lucasb-eyer/pydensecrf.git

cd YOUR_WORKING_DIRECTORY 
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .

cd ..
git clone git@github.com:facebookresearch/PartDistillation.git # Change it later to public repo. 
cd PartDistillation
pip install -r requirements.txt 
cd part_distillation/modeling/pixel_decoder/ops 
sh make.sh # CUDA_HOME must be defined and points to the directory of the installed CUDA toolkit.

# detic
cd ../../../..
git clone https://github.com/facebookresearch/Detic.git --recurse-submodules
cd Detic
pip install -r requirements.txt
```
