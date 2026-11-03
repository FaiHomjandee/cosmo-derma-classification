#!/bin/bash
# Build a conda env "vimreal" for the official Vision Mamba (hustvl/Vim) + this project's pipeline.
# Run on a GPU node (the CUDA kernels need nvcc):   bash vim_real/install_vim_env.sh 2>&1 | tee vim_real/install.log
set -e
cd "$(dirname "$0")"
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda 2>/dev/null || true
command -v nvcc >/dev/null || { echo "ERROR: nvcc not found - load a CUDA module first (module avail cuda)"; exit 1; }
CUDA_MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
if [ "$CUDA_MAJOR" = "11" ]; then CU=cu118; else CU=cu121; fi
echo "nvcc CUDA major $CUDA_MAJOR -> PyTorch wheels $CU"

source ~/anaconda3/etc/profile.d/conda.sh
conda env list | grep -q '^vimreal ' || conda create -y -n vimreal python=3.10.13
conda activate vimreal

pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/$CU
# keep these exact versions for every later pip install (the CUDA kernels are built against them)
printf "torch==2.1.1\ntorchvision==0.16.1\n" > torch_constraints.txt
export PIP_CONSTRAINT="$PWD/torch_constraints.txt"
[ -d Vim ] || git clone https://github.com/hustvl/Vim.git
# Only what the classification model needs. The full vim_requirements.txt also pins the
# segmentation stack (mmcv 1.3.8, mmsegmentation) whose build fails with current setuptools.
pip install "setuptools<70" wheel packaging ninja
pip install numpy==1.26.2 timm==0.4.12 einops==0.7.0 transformers==4.35.2 triton==2.1.0

# CUDA kernels (compiled locally)
export MAX_JOBS=8
# --no-build-isolation: the setup.py files import torch, so they must see the env's torch
pip install --no-build-isolation -e Vim/causal-conv1d
pip install --no-build-isolation -e Vim/mamba-1p1p1

# packages used by the CoSMO classification pipeline
pip install pandas==2.1.3 scikit-learn==1.3.2 scipy==1.11.4 smote-variants "grad-cam==1.5.0" vit-pytorch imageio opencv-python-headless \
            umap-learn tensorboard matplotlib scikit-image huggingface_hub

# ImageNet-pretrained Vim-Tiny weights
python - <<'PY'
from huggingface_hub import list_repo_files, hf_hub_download
repo = "hustvl/Vim-tiny-midclstok"
pth = [f for f in list_repo_files(repo) if f.endswith(".pth")]
print("files in", repo, ":", pth)
p = hf_hub_download(repo, pth[0], local_dir="weights")
print("saved", p)
PY
python -c "import torch; assert torch.__version__.startswith('2.1.1'), torch.__version__; print('torch', torch.__version__)"
echo "INSTALL DONE"
