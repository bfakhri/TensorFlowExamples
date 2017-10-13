#!/bin/sh 
tmux new-session -s 'train' -d 'python byp_mnist.py'
tmux split-window -h 'watch -n 0.1 nvidia-smi'
tmux split-window -v 'htop'
tmux rename-window 'Monitor'
tmux new-window 'tensorboard --logdir=./logs/1 --port=12345'
tmux rename-window 'Tensorboard'
tmux select-window -t 0
tmux -2 attach-session -d 
