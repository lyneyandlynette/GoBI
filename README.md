# Go Beyond Imagination 
Implementation of [GoBI](https://arxiv.org/pdf/2308.13661.pdf): Go Beyond Imagination: Maximizing Episodic Reachability with World Models (ICML 2023)

## Installation
```
# Install Instructions
conda create -n gobi python=3.7
conda activate gobi 
pip install -r requirements.txt
```
## Collect Data
```
. ./collect_data.sh
```
## Train GoBI on MiniGrid
```
OMP_NUM_THREADS=1 python main.py --model gobi --env MiniGrid-MultiRoom-N7-S8-v0 --intrinsic_reward_coef=0.01 --total_frames 8000000 --simu_step 1 --intrinsic_decay 0.0000006
```

This codebase is built on [NovelD](https://github.com/tianjunz/NovelD).
