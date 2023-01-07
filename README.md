# Faster-R-CNN
#It is a modified version of Faster-RCNN from TYUT (Taiyuan University of Technology) SMART OPTICS Group
#Produced by: Qiang Liu, Yongyang Sun, Wenbo Liu, Lichun Sun, Rui Sun, Heng Zhang, Kaiyang Li and Peng Jia
#The code could be adopted to any detection tasks with Faster-RCNN architecture for astronomical data
#with the following options:
#Configurable parameters that would affect structure of neural networks and loss functions
#Backbone: 
#           VGG 
#           ResNet
#           LeNet, should be included in ./cfgs yaml files
#Detection Target: 
#           extended targets: return position (bounding boxes) and type
#           point targets: return position (centre) and type
#Data Type:
#           Number of Channels: would affect the backbone neural network and the main structure of the neural network
#Data Augmentation:
#           Methods that are used to carry out data augmentation, including: spatial shift, rotation, channel shift, zoom
#           Need to be further modified and added by GeLi 