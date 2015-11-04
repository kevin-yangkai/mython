__author__ = 'zhangxulong'
import cPickle


def load_pkl(pkl_str):
    file_pkl = open(pkl_str)
    pkl_data = cPickle.load(file_pkl)
    return pkl_data


def decode_pkl(load_pkl_data):
    print load_pkl_data
    print len(load_pkl_data)
    data = load_pkl_data[0]
    target = load_pkl_data[1]
    data_item = data[0]

    print data_item
    print len(data[2])
    print"===================="
    print len(load_pkl_data)
    print"===================="
    print len(data)
    print len(target)
    print target
    return 0


def main():
    data = load_pkl('imdb.pkl')
    decode_pkl(data)
    return 0


main()
