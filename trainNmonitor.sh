#!/bin/sh 
tmux new-session -s 'train' -d 'python test.py'
tmux split-window -v 'watch -n 0.1 nvidia-smi'
tmux split-window -h 'htop'
tmux rename-window 'Monitor'
tmux new-window 'tensorboard --logdir=/tmp/tbdir/1 --port=12345'
tmux rename-window 'Tensorboard'
tmux select-window -t 0
tmux -2 attach-session -d 
