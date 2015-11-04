import os
import cPickle

__author__ = 'zhangxulong'


def read_txt_segments(txt_dir):
    count = 0
    txt = open(txt_dir, 'r')
    list_of_all_the_lines = txt.readlines()
    for line in list_of_all_the_lines:
        count += 1
    print '\033[1;31;40m'
    print "there are %i lines in txt: %s" % (count, txt_dir)
    return list_of_all_the_lines


def generate_singer_label(txt_dir):
    dict_singer_label = {}
    singers = []
    all_lines = read_txt_segments(txt_dir)
    for line in all_lines:
        wav_dir = line.split('@')[0]
        singer_name = wav_dir.split('.')[0].split('/')[-1].split('_')[0]
        singers.append(singer_name)
    only_singers = sorted(list(set(singers)))
    for item, singer in enumerate(only_singers):
        dict_singer_label[singer] = item
    # print dict_singer_label
    return dict_singer_label


def trans_string_to_list(string='[1,2,3]'):
    string = string.split('[')[1]
    string = string.split(']')[0]
    string = string.split(',')
    string_list = []
    for item in string:
        newitem = float(item)
        string_list.append(newitem)
    return string_list


def get_timbres(segments='string'):
    timbres = []
    timbre_right = segments.split("u'timbre':")
    timbre_right = timbre_right[1:]
    for item in timbre_right:
        timbre = item.split(", u'pitches'")[0]
        timbres.append(timbre)
    return timbres


def build_dataset():
    dataset = []
    data = []
    target = []
    txt_dir = 'sliced_save_segments.txt'
    all_lines = read_txt_segments(txt_dir)
    dict_singer_label = generate_singer_label(txt_dir)
    for line in all_lines:
        wav_dir = line.split('@')[0]
        segments = line.split('@')[1]
        singer_name = wav_dir.split('.')[0].split('/')[-1].split('_')[0]
        singer_label = dict_singer_label[singer_name]
        timbres = get_timbres(segments)
        for timbre_item in timbres:
            timbre = trans_string_to_list(timbre_item)
            data.append(timbre)
            target.append(singer_label)
    dataset.append(data)
    dataset.append(target)
    pickle_dataset(dataset, 'segment_timbre_dataset.pkl')
    return 0


def pickle_dataset(dataset, dir):
    pkl_file = file(dir, 'wb')
    cPickle.dump(dataset, pkl_file, True)
    pkl_file.close()
    return 0


def load_dataset(pkl_dir):
    pkl_file = file(pkl_dir, 'rb')
    dataset = cPickle.load(pkl_file)
    pkl_file.close()
    return dataset


def view_dataset_pkl(dir='segment_timbre_dataset.pkl'):
    dataset = load_dataset(dir)

    print 'dataset is len %i' % len(dataset)
    print 'data len is %i' % len(dataset[0])
    print 'target len is %i ' % len(dataset[1])
    print 'data 1:5=========='
    print dataset[0][1:5]
    print 'target 1:5=========='
    print dataset[1][1:5]
    print len( dataset[0][1])
    return 0

# build_dataset()
view_dataset_pkl()