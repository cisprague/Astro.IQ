import sys
sys.path.append('../../')
from numpy import *
from Trajectory import *
import multiprocessing
import itertools
from numba import jit
'''
>> python Drive_MLP.py
'''

def main(context, nnodes, nlayers, si):
    model = Point_Lander_Drag(si)
    dim = (nnodes, nlayers)
    model.controller = Neural(model, context, dim).Control
    tf = 100
    npts = 100
    s, c = model.Propagate.Neural(si, tf, nots, False)
    save()

if __name__ == '__main__':
    pass
