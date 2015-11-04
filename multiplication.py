
import numpy

class MulTask(object):
    def __init__(self, rng, floatX):
        self.rng = rng
        self.floatX = floatX
        self.nin = 2
        self.nout = 1
        self.classifType = 'lastLinear'

    def generate(self, batchsize, length):
        l = self.rng.randint(int(length*.1))+length
        p0 = self.rng.randint(int(l*.1), size=(batchsize,))
        p1 = self.rng.randint(int(l*.4), size=(batchsize,)) + int(l*.1)
        data = self.rng.uniform(size=(l, batchsize, 2)).astype(self.floatX)
        data[:,:,0] = 0.
        data[p0, numpy.arange(batchsize), numpy.zeros((batchsize,),
                                                      dtype='int32')] = 1.
        data[p1, numpy.arange(batchsize), numpy.zeros((batchsize,),
                                                      dtype='int32')] = 1.

        targs = (data[p0, numpy.arange(batchsize),
                     numpy.ones((batchsize,), dtype='int32')] * \
                 data[p1, numpy.arange(batchsize),
                      numpy.ones((batchsize,), dtype='int32')])
        return data, targs.astype(self.floatX).reshape((-1,1))

if __name__ == '__main__':
    print 'Testing mul task generator ..'
    multask = MulTask(numpy.random.RandomState(123), 'float32')
    seq, targ = multask.generate(3, 25)
    assert seq.dtype == 'float32'
    assert targ.dtype == 'float32'
    print 'Seq_0'
    print seq[:,0,:]
    print 'Targ0', targ[0]
    print
    print 'Seq_1'
    print seq[:,1,:]
    print 'Targ1', targ[1]
    print
    print 'Seq_2'
    print seq[:,2,:]
    print 'Targ2', targ[2]
