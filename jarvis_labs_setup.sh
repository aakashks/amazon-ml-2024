echo "alias ll='ls -hlAF'" >> ~/.bashrc

# /root is not persistent so, instead install conda on /home
sudo rm -rf /root/miniconda3

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

echo "export PATH=~/miniconda3/bin:$PATH" >> ~/.bashrc
~/miniconda3/bin/conda init bash
source ~/.bashrc

# create a new conda environment
conda create -n vl python=3.10 -y

echo "conda activate vl" >> ~/.bashrc
source ~/.bashrc

pip install vllm
pip install uv

uv pip install jupyterlab notebook seaborn pandas gpustat

# Qwen 2 VL
uv pip install qwen_vl_utils
uv pip install git+https://github.com/huggingface/transformers.git@21fac7abba2a37fae86106f87fcf9974fd1e3830

cd ~ # takes to /home

# git clone
# cd into dir
#
# python file.py
