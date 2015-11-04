

"""
Generate input and test sequence for the temporal order task.

Description
-----------
The input has 6 channels. At any time all channels are 0 except for one
which has value 1 (i.e. the 6 channels are used for a one-hot encoding
of a 6 possible symbols).

The first two channels are reserved for symbols {A, B}, the others
to {c,d,e,f}. At one random position `p0` in [1, L/10] either A or B
is showed. The same happens at a second position `p1` in [5*L/10, 6*L/10].
At all other position a random symbol from {c,d,e,f} is used.

At the end of the sequence one has to predict the order in which the
symbols where provided (either AA, AB, BA or BB).

Author: Razvan Pascanu
contact: r.pascanu@gmail
"""

import numpy

class TempOrderTask(object):
     def __init__(self, rng, floatX):
        self.rng = rng
        self.floatX = floatX
        self.nin = 6
        self.nout = 4
        self.classifType='lastSoftmax'

     def generate(self, batchsize, length):
        l = length
        p0 = self.rng.randint(int(l*.1), size=(batchsize,)) + int(l*.1)
        v0 = self.rng.randint(2, size=(batchsize,))
        p1 = self.rng.randint(int(l*.1), size=(batchsize,)) + int(l*.5)
        v1 = self.rng.randint(2, size=(batchsize,))
        targ_vals = v0 + v1*2
        vals  = self.rng.randint(4, size=(l, batchsize))+2
        vals[p0, numpy.arange(batchsize)] = v0
        vals[p1, numpy.arange(batchsize)] = v1
        data = numpy.zeros((l, batchsize, 6), dtype=self.floatX)
        targ = numpy.zeros((batchsize, 4), dtype=self.floatX)
        data.reshape((l*batchsize, 6))[numpy.arange(l*batchsize),
                                    vals.flatten()] = 1.
        targ[numpy.arange(batchsize), targ_vals] = 1.
        return data, targ


if __name__ == '__main__':
    print 'Testing temp Order task generator ..'
    task = TempOrderTask(numpy.random.RandomState(123), 'float32')
    seq, targ = task.generate(3, 25)
    assert seq.dtype == 'float32'
    assert targ.dtype == 'float32'
    print 'Sequence 0'
    print '----------'
    print seq[:,0,:]
    print 'Target:', targ[0]
    print
    print 'Sequence 1'
    print '----------'
    print seq[:,1,:]
    print 'Target:', targ[1]
    print
    print 'Sequence 2'
    print '----------'
    print seq[:,2,:]
    print 'Target', targ[2]
