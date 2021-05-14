import torch
import numpy as np
from modules import *
import matplotlib.pyplot as plt


def test_linear(dout):
    linear = Linear(3,4)
    linear.forward([2,2,3])
    return linear.backward(dout)

def test_ce():
    ce = CrossEntropy()
    ce.forward(x=[2,2,3,3],y=[2,3,2,3])
    return ce.backward(x=[2,2,3,3],y=[2,3,2,3])

def test_sm(dout):
    sm = SoftMax()
    sm.forward([2,2,3,3])
    return sm.backward(dout)

def test_rl(dout):
    rl = ReLU()
    rl.forward([2,2,3,3])
    return rl.backward(dout)


if __name__ == '__main__':
    # dout_ce = test_ce()
    # print('test_ce: dout',dout_ce)
    # dout_sm = test_sm(dout_ce)
    # print('test_sm: dout',dout_sm)
    # dout_rl = test_rl(dout_sm)
    # print('test_rl: dout',dout_rl)
    # dout_li = test_linear(dout_rl)
    # print('test_li: dout',dout_li)
    a=[[2,1],[4,3]]
    print(a[:,0])

