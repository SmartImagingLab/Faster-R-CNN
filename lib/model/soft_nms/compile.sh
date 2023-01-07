#!/bin/bash
echo 'Compiling NMS module...'
(cd /home/lab30201/lq/Faster_RCNN/FPN_SH/fpn_sh_soft/lib/model/soft_nms; python3 setup_linux.py build_ext --inplace)
#echo 'Compiling bbox module...'
#(cd lib/bbox; python setup_linux.py build_ext --inplace)
#echo 'Compiling chips module...'
#(cd lib/chips; python setup.py)
#echo 'Compiling coco api...'
#(cd lib/dataset/pycocotools; python setup_linux.py build_ext --inplace)
echo 'All Done!'