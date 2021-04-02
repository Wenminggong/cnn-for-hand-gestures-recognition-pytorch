# CNN for Hand Gestures Recognition with pytorch
This repo contains the code for classifying customized hand gestures using cnn. The hand gestures are designed for Human-Robot Interaction in some special scenes (e.g. outer space).

## Dependencies
This code requires the following: 
* python 3.7.6
* pytorch 1.7.1
* torchvision 0.8.2


## Data Set
The hand_gestures_data_set_nju data set consists of 2140 color images containing 20 classes. The images were collected in different backgrounds and light conditions, and with different image size. An example image is shown as follow:
![](https://github.com/Wenminggong/cnn-for-hand-gestures-recognition-pytorch/blob/main/hand_gestures_data_set_nju/hand/1.%E6%8E%A7%E5%88%B6%E5%BC%80%E5%A7%8B/%E4%BA%AE/B_wall_F_1.1.jpg)

To get the complete hand_gestures_data_set_nju data set for academic research, please email to [mjliu@smail.nju.edu.cn](mjliu@smail.nju.edu.cn).

## CNN Architecture
The architecture of the designed cnn is shown as follow:
![](https://github.com/Wenminggong/cnn-for-hand-gestures-recognition-pytorch/blob/main/cnn_architecture.PNG)

## Experimental Results
The experimental results are shown as follows:
![](https://github.com/Wenminggong/cnn-for-hand-gestures-recognition-pytorch/blob/main/saves/training_loss.png "training loss")
![](https://github.com/Wenminggong/cnn-for-hand-gestures-recognition-pytorch/blob/main/saves/test_acc.png "test accuracy")
