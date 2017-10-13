# About

Architecture

Conv1 -> Conv2 -> FC1 -> FC2 -> FC3
                   \-------------/

Some outputs of neurons in FC1 completely bypass FC2 and go straight to FC3.

Slows learning compared to the architecture below but final accuracies are comparable.

Conv1 -> Conv2 -> FC1 -> FC2
 
