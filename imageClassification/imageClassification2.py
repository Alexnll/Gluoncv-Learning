# https://gluon-cv.mxnet.io/build/examples_classification/dive_deep_cifar10.html
# Gluoncv Tutorials
# Image Classification
# 2. Dive Deep into Training with CIFAR10
from __future__ import division
import argparse, time, loggin, random, math
import numpy as np
import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, TrainingHistory


# get the net model
def get_net(model_input='cifar_resnet2_v1', classes_input = 10, pretrained_or_not = False, ctx_used=mx.cpu()):
    # decide using GPU or CPU and the GPU number
    # num_gpu = 1
    # ctx_used = [mx.gpu(i) for i in range(num_gpu)]
    
    # get the model with 10 output classes, without pre-trained weights
    net = get_model(model_input, classes=classes_input, pretrained=pretrained_or_not)
    net.initialize(mx.init.Xavier(), ctx=ctx_used)
    return net

# Data Augmentation and Data loader
# Assuming that, for the same object, photos under different 
# composition, lighting condition, or color should all yield the same prediction.
def transf():
    