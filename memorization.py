
import numpy

class MemTask(object):
    def __init__(self,
                  rng,
                  floatX,
                  n_values = 5,
                  n_pos = 10,
                  generate_all = False):
        self.rng = rng
        self.floatX = floatX
        self.dim = n_values**n_pos
        self.n_values = n_values
        self.n_pos = n_pos
        self.generate_all = generate_all
        if generate_all:
            self.data = numpy.zeros((n_pos, self.dim, n_values+2))
            for val in xrange(self.dim):
                tmp_val = val
                for k in xrange(n_pos):
                    self.data[k, val, tmp_val % n_values] = 1.
                    tmp_val = tmp_val // n_values
        self.nin = self.n_values + 2
        self.nout = n_values + 1
        self.classifType = 'softmax'
        self.report = 'all'


    def generate(self, batchsize, length):

        if self.generate_all:
            batchsize = self.dim
        input_data = numpy.zeros((length + 2*self.n_pos,
                                  batchsize,
                                  self.n_values + 2),
                                 dtype=self.floatX)
        targ_data = numpy.zeros((length + 2*self.n_pos,
                                 batchsize,
                                 self.n_values+1),
                                dtype=self.floatX)
        targ_data[:-self.n_pos,:, -1] = 1
        input_data[self.n_pos:,:, -2] = 1
        input_data[length + self.n_pos, :, -2] = 0
        input_data[length + self.n_pos, :, -1] = 1

        if not self.generate_all:
            self.data = numpy.zeros((self.n_pos, batchsize, self.n_values+2))
            for val in xrange(batchsize):
                tmp_val = self.rng.randint(self.dim)
                for k in xrange(self.n_pos):
                    self.data[k, val, tmp_val % self.n_values] = 1.
                    tmp_val = tmp_val // self.n_values
        input_data[:self.n_pos, :, :] = self.data
        targ_data[-self.n_pos:, :, :] = self.data[:,:,:-1]
        return input_data, targ_data.reshape(((length +
                                               2*self.n_pos)*batchsize, -1))

if __name__ == '__main__':
    print 'Testing memorization task generator ..'
    task = MemTask(numpy.random.RandomState(123),
                   'float32')
    seq, targ = task.generate(3, 25)
    assert seq.dtype == 'float32'
    assert targ.dtype == 'float32'
    print 'Seq_0'
    print seq[:,0,:].argmax(axis=1)
    print 'Targ0'
    print targ.reshape((25+2*10, 3, -1))[:,0,:].argmax(1)
    print
    print 'Seq_1'
    print seq[:,1,:].argmax(axis=1)
    print 'Targ1'
    print targ.reshape((25+2*10, 3, -1))[:,1,:].argmax(1)
    print
    print 'Seq_2'
    print seq[:,2,:].argmax(axis=1)
    print 'Targ2'
    print targ.reshape((25+2*10, 3, -1))[:,2,:].argmax(1)
