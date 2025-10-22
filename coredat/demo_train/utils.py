import collections.abc
from itertools import repeat
from neuron import h

def pair(x):
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, 2))

def setup_hpc(model, morph):
    cell = getattr(h, model)()
    nl = h.Import3d_Neurolucida3()
    nl.quiet = 1
    nl.input(morph)
    imprt = h.Import3d_GUI(nl, 0)
    imprt.instantiate(cell)
    cell.indexSections(imprt)
    cell.geom_nsec()
    cell.geom_nseg()
    cell.delete_axon()
    cell.insertChannel()
    cell.init_rc()
    cell.biophys()
    return cell

class IdManager:
    def __init__(self):
        self.gid_start = 0
        self.sid_start = 0
        self.sid2param = {}

    def alloc_gid(self, n):
        gids = [i for i in range(self.gid_start, self.gid_start + n)]
        self.gid_start += n
        return gids

    def alloc_sid(self, n):
        sids = [i for i in range(self.sid_start, self.sid_start + n)]
        self.sid_start += n
        return sids
