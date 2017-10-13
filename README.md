# Tensorflow Examples 

Testing regularization that maintains spatial organization between layers.

# Running

trainNmonitor.sh
Running this launches a tmux session with the following windows and panes:
1) Training Monitor
    - tensorflow training session
    - nvidia-smi (for monitoring gpu)
    - htop (for monitoring cpu)
2) Tensorboard - if running locally, connect via browser using "localhost:12345"

