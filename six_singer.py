import os
import cPickle

__author__ = 'zhangxulong'
import numpy
import wave
from matplotlib import pylab

from pyechonest import track, config


def draw_segments_from_echonest(wav_dir, starts_point):
    song = wave.open(wav_dir, "rb")
    params = song.getparams()
    nchannels, samplewidth, framerate, nframes = params[:4]  # format info
    song_data = song.readframes(nframes)
    song.close()
    wave_data = numpy.fromstring(song_data, dtype=numpy.short)
    wave_data.shape = -1, 1
    wave_data = wave_data.T
    time = numpy.arange(0, nframes) * (1.0 / framerate)
    len_time = len(time)
    time = time[0:len_time]
    pylab.plot(time, wave_data[0])
    num_len = len(starts_point)
    pylab.plot(starts_point, [1] * num_len, 'ro')
    pylab.xlabel("time")
    pylab.ylabel("wav_data")
    pylab.show()
    return 0


def from_segments_get_timbre(wav_dir, segments):
    timbre_pitches_loudness = []
    starts_point = []
    for segments_item in segments:
        timbre = segments_item['timbre']

        start = segments_item['start']
        starts_point.append(start)

        timbre_pitches_loudness.append(timbre)

    # plot the segments seg#####################################
    # draw_segments_from_echonest(wav_dir, starts_point)

    return timbre_pitches_loudness


def get_timbre(wav_dir):
    # from echonest capture the timbre and pitches loudness et.al.
    config.ECHO_NEST_API_KEY = "BPQ7TEP9JXXDVIXA5"  # daleloogn my api key
    f = open(wav_dir)
    print "process%s====================================================================" % wav_dir
    # t = track.track_from_file(f, 'wav')
    t = track.track_from_file(f, 'wav', 256, force_upload=True)
    # if not with force_upload it will timed out for sockets
    t.get_analysis()
    segments = t.segments  # list of dicts :timing,pitch,loudness and timbre for each segment
    timbre = from_segments_get_timbre(wav_dir, segments)
    return timbre


def save_echonest_data_to_txt(wav_dirs):
    save_txt = 'save_six_segments.txt'
    for parent, dirnames, filenames in os.walk(wav_dirs):
        for filename in filenames:
            song_dir = os.path.join(parent, filename)
            if not os.path.exists(save_txt):
                segments_file = open(save_txt, 'w')
                segments_file.close()
            segments_file = open(save_txt, 'r')
            all_lines = segments_file.readlines()
            segments_file.close()
            dirs = []
            for line_item in all_lines:
                dir_song = line_item.split('@')[0]
                dirs.append(dir_song)
            if song_dir in dirs:
                pass
            else:
                segments = get_timbre(song_dir)
                lines = song_dir + '@' + str(segments) + '\r\n'
                segments_file = open(save_txt, 'a', )
                segments_file.write(lines)
                segments_file.close()
    return 0


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

def trans_string_to_list_no_bracet(string='1,2,3'):

    string = string.split(',')
    string_list = []
    for item in string:
        newitem = float(item)
        string_list.append(newitem)
    return string_list

def timbres_to_list(timbres='[[1,2,3],[4,5,6]]'):
    timbres_list = []
    split_left_bracket = timbres.split('[')
    for left in split_left_bracket:
        if left !='':
            number_string=left.split(']')[0]
            string_list=trans_string_to_list_no_bracet(number_string)
            timbres_list.append(string_list)
        else:
            pass

    #split_right_bracket = split_left_bracket.split(']')


    return timbres_list


def build_dataset():
    dataset = []
    data = []
    target = []
    txt_dir = 'save_six_segments.txt'
    all_lines = read_txt_segments(txt_dir)
    dict_singer_label = generate_singer_label(txt_dir)
    for line in all_lines:
        wav_dir = line.split('@')[0]
        timbres = line.split('@')[1]

        timbres_list = timbres_to_list(timbres)
        singer_name = wav_dir.split('.')[0].split('/')[-1].split('_')[0]
        singer_label = dict_singer_label[singer_name]

        for timbre_item in timbres_list:


            data.append(timbre_item)
            target.append(singer_label)
    dataset.append(data)
    dataset.append(target)
    pickle_dataset(dataset, 'six_timbre_dataset.pkl')
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


def view_dataset_pkl(dir='six_timbre_dataset.pkl'):
    dataset = load_dataset(dir)

    print 'dataset is len %i' % len(dataset)
    print 'data len is %i' % len(dataset[0])
    print 'target len is %i ' % len(dataset[1])
    print 'data 1:5=========='
    print dataset[0][1:5]
    print 'target 1:5=========='
    print dataset[1][1:5]
    print len(dataset[0][1])
    return 0


# build_dataset()
view_dataset_pkl()
