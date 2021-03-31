# update for Ubuntu 18.04
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubunu1804/x86_64/7fa2af80.pub
sudo apt update && sudo apt install libnccl2 libnccl-dev
