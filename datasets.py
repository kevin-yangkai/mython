import cPickle
import gzip
import sys
import numpy as np


def load_imdb(imdb_dataset='datasets/imdb.pkl', nb_words=None, skip_top=0, maxlen=None, test_split=0.2, seed=113,
              start_char=1, oov_char=2, index_from=3):
    if imdb_dataset.endswith(".gz"):
        f = gzip.open(imdb_dataset, 'rb')
    else:
        f = open(imdb_dataset, 'rb')
    X, labels = cPickle.load(f)
    f.close()
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(labels)
    if start_char is not None:
        X = [[start_char] + [w + index_from for w in x] for x in X]
    elif index_from:
        X = [[w + index_from for w in x] for x in X]
    if maxlen:
        new_X = []
        new_labels = []
        for x, y in zip(X, labels):
            if len(x) < maxlen:
                new_X.append(x)
                new_labels.append(y)
        X = new_X
        labels = new_labels
    if not nb_words:
        nb_words = max([max(x) for x in X])
    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters: 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        X = [[oov_char if (w >= nb_words or w < skip_top) else w for w in x] for x in X]
    else:
        nX = []
        for x in X:
            nx = []
            for w in x:
                if (w >= nb_words or w < skip_top):
                    nx.append(w)
            nX.append(nx)
        X = nX
    X_train = X[:int(len(X) * (1 - test_split))]
    y_train = labels[:int(len(X) * (1 - test_split))]
    X_test = X[int(len(X) * (1 - test_split)):]
    y_test = labels[int(len(X) * (1 - test_split)):]
    return (X_train, y_train), (X_test, y_test)


def load_mnist(mnist_dataset='datasets/mnist.pkl.gz'):
    if mnist_dataset.endswith(".gz"):
        f = gzip.open(mnist_dataset, 'rb')
    else:
        f = open(mnist_dataset, 'rb')
    if sys.version_info < (3,):
        data = cPickle.load(f)
    else:
        data = cPickle.load(f, encoding="bytes")
    f.close()
    return data  # (X_train, y_train), (X_test, y_test)


def load_reuters(reuters_dataset='datasets/reuters.pkl', nb_words=None, skip_top=0, maxlen=None, test_split=0.1,
                 seed=113,
                 start_char=1, oov_char=2, index_from=3):
    f = open(reuters_dataset, 'rb')
    X, labels = cPickle.load(f)
    f.close()
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(labels)
    if start_char is not None:
        X = [[start_char] + [w + index_from for w in x] for x in X]
    elif index_from:
        X = [[w + index_from for w in x] for x in X]
    if maxlen:
        new_X = []
        new_labels = []
        for x, y in zip(X, labels):
            if len(x) < maxlen:
                new_X.append(x)
                new_labels.append(y)
        X = new_X
        labels = new_labels
    if not nb_words:
        nb_words = max([max(x) for x in X])
    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters: 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        X = [[oov_char if (w >= nb_words or w < skip_top) else w for w in x] for x in X]
    else:
        nX = []
        for x in X:
            nx = []
            for w in x:
                if (w >= nb_words or w < skip_top):
                    nx.append(w)
            nX.append(nx)
        X = nX
    X_train = X[:int(len(X) * (1 - test_split))]
    y_train = labels[:int(len(X) * (1 - test_split))]
    X_test = X[int(len(X) * (1 - test_split)):]
    y_test = labels[int(len(X) * (1 - test_split)):]
    return (X_train, y_train), (X_test, y_test)


def view_pkl(datasets_path='datasets/xxx.pkl'):
    pkl_file = open(datasets_path)
    datasets = cPickle.load(pkl_file)
    pkl_file.close()
    print len(datasets)
    print len(datasets[0])
    print len(datasets[1])
    print datasets[0][2]
    print datasets[1][2]
    print(len(datasets[0][2]))
    print datasets[1][20000:20500]
    return 0


def view_mnist(mnist_dir='datasets/mnist.pkl.gz'):
    dataset = load_mnist(mnist_dir)
    print len(dataset)
    print len(dataset[0])
    print len(dataset[1])
    print len(dataset[0][0])
    print len(dataset[0][1])
    print dataset[0][1][1]
    print dataset[0][0][1]
    return 0
    # view_pkl(datasets_path='datasets/imdb.pkl')
    # view_mnist()
