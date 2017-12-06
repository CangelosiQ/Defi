 # -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 10:33 2017

@author: Quentin

"""

__all__ = ['big_iter','imports','line','title']


def title(t):
    n=len(t)
    print('\n        %s\n            %s\n        %s\n' %('='*(n+8),t,'='*(n+8)))
    
def big_iter(i,info=None):
    print('__________________________________________________________________________________________________________________________________________')
    print('                   Iteration ',i,'                                           ',info)
    print('__________________________________________________________________________________________________________________________________________')

    
def imports():
    import numpy as np
    import matplotlib.pyplot as plt
    import copy
    import sys
    import time
    import datetime
    plt.close("all")
    start=time.time()
    plt.rc('text',usetex=True)
    #orig_stdout=sys.stdout
    #f=open('out.txt','w')
    #sys.stdout=f
    return np, plt, start, time, sys, datetime, copy

def line():
    print('__________________________________________________________________________________________________________________________________________\n')