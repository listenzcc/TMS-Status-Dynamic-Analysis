#!/usr/bin/env zsh

source ~/.zshrc
conda activate python3.12

script=plot-stc-singlemap.py
method=MNE-64
conditions=(T80 T100 T120 Sham)
states=(0 1 2 3)

for condition in "${conditions[@]}"; do
    for state in "${states[@]}"; do
        python $script -m $method -c $condition -s $state -p
    done
done