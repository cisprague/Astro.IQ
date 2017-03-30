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
    sifl = array([[-3003.20255298,  3610.03013931,   -75.5395321 ,   -65.83635707,
    7525.01646505],
    [-3318.68671679,  3201.16686938,   -23.41968182,   -60.97279329,
    7341.96179948],
    [-3309.01321848,  2842.84146239,    25.84812311,   -49.81414984,
    7158.90713391]])
    sicl = array([[ -243.53707153,  4296.31407193,  -142.61335252,  -119.98680334,
    6675.1642641 ],
    [ -835.75013923,  3693.99375614,   -91.83984475,  -117.27870268,
    6530.46150148],
    [-1177.33712024,  3118.44557095,   -43.6791286 ,  -109.21858109,
    6385.75873886]])

    structf = [
        (10, 1),
        (10, 2),
        (20, 1),
        (20, 2),
        (20, 3),
        (20, 5)
    ]

    for arg in itertools.product(sifl, structf):
        si = arg[0]
        struc = arg[1]
        nnodes = struc[0]
        nlayers = struc[1]
        a = ('Mars_Far', nnodes, nlayers, si)
        p = multiprocessing.Process(target=main, args=a)
        p.start()
