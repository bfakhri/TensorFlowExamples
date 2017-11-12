# Tensorflow Examples 

A collection of Tensorflow examples for deep learning (mostly). 
Examples use tensorboard for the visualization of training performance, outputs, activations, etc.

# Structure

The project is divided into directories named for the category of example. Each directory has a "template" file and subdirectories use that file as a template. 

# Running

In all directories there is a script called trainNmonitor.sh
Running this launches a tmux session with the following windows and panes:
1) Training Monitor
	- tensorflow training session
	- nvidia-smi (for monitoring gpu)
	- htop (for monitoring cpu)
2) Tensorboard - if running locally, connect via browser using "localhost:12345"



![alt text](https://github.com/bfakhri/tf/blob/master/images/monitor.png "Training Monitor")
![alt text](https://github.com/bfakhri/tf/blob/master/images/tb_graph.png "TB Graph")
![alt text](https://github.com/bfakhri/tf/blob/master/images/tb_imgs.png "TB Images")
