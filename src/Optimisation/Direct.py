from PyGMO.problem import base

class Trapezoidal(base):
    def __init__(self, problem, nsegs):
        nnodes = nseg + 1
        dim    = 1 + (problem.sdim + problem.cdim)*nnodes
        cdim   = problem.sdim*nseg + 2*problem.sdim - 1
        base.__init__(problem, dim, 0, 1, cdim, 0, 1e-8)
        problem.set_bounds(
            problem.tlb + (problem.slb + self.clb)*nnodes,
            problem.tub + (problem.sub + self.cub)*nnodes
        )
