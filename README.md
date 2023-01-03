# Re-NeRF
- an example implementation of Re:NeRF, a 3D compression method designed for NeRF techniques 
- this repository is tailered for TensoRF (project page: https://github.com/apchenstu/TensoRF) but can be modified to fit other NeRFs

## Download TensoRF

https://github.com/apchenstu/TensoRF

## Prep

Install environment(from https://github.com/apchenstu/TensoRF) :
```
conda create -n TensoRF python=3.8
conda activate TensoRF
pip install torch torchvision
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard
```
Dataset

* [Synthetic-NeRF](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) 
* [Synthetic-NSVF](https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NSVF.zip)
* [Tanks&Temples](https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip)
* [Forward-facing](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)


pretrained models available on TensoRF repo

https://1drv.ms/u/s!Ard0t_p4QWIMgQ2qSEAs7MUk8hVw?e=dc6hBm

## Run our compression method on a pretrained model   

python run_pruning.py --config <path_to_config> --ckpt <path_to_checkpoint>

## Example run 

1. download lego dataset: https://drive.google.com/drive/folders/1lrDkQanWtTznf48FCaW5lX9ToRdNDF1a?usp=sharing

2. move the dataset to ./data/nerf_synthetic/lego

3. download pretrained lego TensoRF model: https://onedrive.live.com/?authkey=%21AKpIQCzsxSTyFXA&cid=0C624178FAB774B7&id=C624178FAB774B7%21296&parId=C624178FAB774B7%21230&o=OneUp 

4. python run_pruning.py --config config/lego.txt --ckpt lego.th
