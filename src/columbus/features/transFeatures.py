#!/usr/bin/python
#encoding:utf-8
'''
Project 'Columbus' trans feature function modules
author: Guowei Xu
Date: July 11, 2018
所有函数返回值要么是单个变量，要么为一个numpy array
'''
import pandas as pd
import numpy as np

def getPow2(x):
    return np.power(x,2)

def getPow3(x):
    return np.power(x, 3)

def getSqrt(x):
    # assert (x >=0).all()
    # fill the out range number with np.nan
    inner = x.copy()
    inner = pd.Series([np.nan if y <0 else y for y in x])
    return np.power(inner, 0.5)

def getLog2(x):
    # assert (x>0).all()
    inner = x.copy()
    inner = pd.Series([np.nan if y <=0 else y for y in x])
    return np.log2(inner)

def getLog10(x):
    # assert (x>0).all()
    inner = x.copy()
    inner = pd.Series([np.nan if y <=0 else y for y in x])
    return np.log10(inner)

def getLn(x):
    # assert (x>0).all()
    inner = x.copy()
    inner = pd.Series([np.nan if y <=0 else y for y in x])
    return np.log(inner)

def getCos(x):
    return np.cos(x)

def getSin(x):
    return np.sin(x)

def getTan(x):
    return np.tan(x)

def getSigmoid(x):
    return 1.0/(1+np.exp(-x))

def get25Percent(x):
    return np.percentile(x, 25)

def get50Percent(x):
    return np.percentile(x, 50)

def get75Percent(x):
    return np.percentile(x, 75)

def getStd(x):
    return np.std(x)

def getVar(x):
    return np.var(x)

def getMin(x):
    return np.min(x)

def getMax(x):
    return np.max(x)