#!/usr/bin/env zsh

source ~/.zshrc
conda activate python3.12

script=plot_cls.py

# python $script --method eLORETA
# python $script --method sLORETA
# python $script --method eLORETA-filter
python $script --method eLORETA-newCov