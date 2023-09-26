#!/usr/bin/python
#encoding:utf-8
'''
Project 'Columbus' cross feature function modules
author: Guowei Xu
Date: July 11, 2018
所有函数返回值要么是单个变量，要么为一个numpy array
'''
import pandas as pd
import numpy as np

#elementwise乘积, 也可按需求添加点积
def getMul(x1, x2):
    return x1*x2

def getDiv(x1, x2):
    div_result = np.nan if x2 == 0 else x1/x2
    return div_result

#返回一个shape和输入一样的矩阵，矩阵每一位都是x1和x2取大者
def getMax(x1, x2):
    result = np.zeros(x1.shape)
    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):
            result[i][j] = max(x1[i][j], x2[i][j])
    return result

#返回一个shape和输入一样的矩阵，矩阵每一位都是x1和x2取较小者
def getMin(x1, x2):
    result = np.zeros(x1.shape)
    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):
            result[i][j] = min(x1[i][j], x2[i][j])
    return result
