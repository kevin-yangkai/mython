# in this file, i will get the singger vocal audio from the pkl dataset
# depend on ffmpeg it need add the untrust ppa to ubuntu
# sudo add-apt-repository ppa:mc3man/trusty-media
# sudo apt-get update
# sudo apt-get dist-upgrade
__author__ = 'zhangxulong'
# use for collect songs but here we use a dataset and this code will be the util to handle the dataset.
import cPickle
import wave
import gzip
import scipy
import scipy.io.wavfile
from time import time
from matplotlib import pylab
import scipy.io.wavfile
import os

import numpy as np
from numpy import *
from scikits.talkbox.features import mfcc
from pydub import AudioSegment
from pyechonest import track, config
import numpy

import mfcc_diy
from feature_extraction import feature_reduction_union_list


class Dataset:
    """Slices, shuffles and manages a small dataset for the HF optimizer."""

    def __init__(self, data, batch_size, number_batches=None, targets=None):
        '''SequenceDataset __init__

  data : list of lists of numpy arrays
    Your dataset will be provided as a list (one list for each graph input) of
    variable-length tensors that will be used as mini-batches. Typically, each
    tensor is a sequence or a set of examples.
  batch_size : int or None
    If an int, the mini-batches will be further split in chunks of length
    `batch_size`. This is useful for slicing subsequences or provide the full
    dataset in a single tensor to be split here. All tensors in `data` must
    then have the same leading dimension.
  number_batches : int
    Number of mini-batches over which you iterate to compute a gradient or
    Gauss-Newton matrix product. If None, it will iterate over the entire dataset.
  minimum_size : int
    Reject all mini-batches that end up smaller than this length.'''
        self.current_batch = 0
        self.number_batches = number_batches
        self.items = []
        if targets is None:
            if batch_size is None:
                # self.items.append([data[i][i_sequence] for i in xrange(len(data))])
                self.items = [[data[i]] for i in xrange(len(data))]
            else:
                # self.items = [sequence[i:i+batch_size] for sequence in data for i in xrange(0, len(sequence), batch_size)]

                for sequence in data:
                    num_batches = sequence.shape[0] / float(batch_size)
                    num_batches = numpy.ceil(num_batches)
                    for i in xrange(int(num_batches)):
                        start = i * batch_size
                        end = (i + 1) * batch_size
                        if end > sequence.shape[0]:
                            end = sequence.shape[0]
                        self.items.append([sequence[start:end]])
        else:
            if batch_size is None:
                self.items = [[data[i], targets[i]] for i in xrange(len(data))]
            else:
                for sequence, sequence_targets in zip(data, targets):
                    num_batches = sequence.shape[0] / float(batch_size)
                    num_batches = numpy.ceil(num_batches)
                    for i in xrange(int(num_batches)):
                        start = i * batch_size
                        end = (i + 1) * batch_size
                        if end > sequence.shape[0]:
                            end = sequence.shape[0]
                        self.items.append([sequence[start:end], sequence_targets[start:end]])

        if not self.number_batches:
            self.number_batches = len(self.items)
        self.num_min_batches = len(self.items)
        self.shuffle()

    def shuffle(self):
        numpy.random.shuffle(self.items)

    def iterate(self, update=True):
        for b in xrange(self.number_batches):
            yield self.items[(self.current_batch + b) % len(self.items)]
        if update: self.update()

    def update(self):
        if self.current_batch + self.number_batches >= len(self.items):
            self.shuffle()
            self.current_batch = 0
        else:
            self.current_batch += self.number_batches


def load_audio(song_dir):
    file_format = song_dir.split('.')[1]
    if 'wav' == file_format:
        song = wave.open(song_dir, "rb")
        params = song.getparams()
        nchannels, samplewidth, framerate, nframes = params[:4]  # format info
        song_data = song.readframes(nframes)
        song.close()
        wave_data = numpy.fromstring(song_data, dtype=numpy.short)
        wave_data.shape = -1, 2
        wave_data = wave_data.T
    else:
        raise NameError("now just support wav format audio files")
    return wave_data


def pickle_dataset(dataset, out_pkl='dataset.pkl'):
    pkl_file = file(out_pkl, 'wb')
    cPickle.dump(dataset, pkl_file, True)
    pkl_file.close()
    return 0


def build_song_set(songs_dir):
    songs_dataset = []
    for parent, dirnames, filenames in os.walk(songs_dir):
        for filename in filenames:
            song_dir = os.path.join(parent, filename)
            audio = load_audio(song_dir)
            # change the value as your singer name level in the dir
            # eg. short_wav/new_wav/dataset/singer_name so I set 3
            singer = song_dir.split('/')[1]
            # this value depends on the singer file level in the dir
            songs_dataset.append((audio, singer))
    pickle_dataset(songs_dataset, 'songs_audio_singer.pkl')
    return 0


def build_dataset(dataset_dir):
    dataset = {"author": "zhangxulong"}
    dict_singer_label = {"aerosmith": 0, "beatles": 1, "creedence_clearwater_revival": 2, "cure": 3,
                         "dave_matthews_band": 4, "depeche_mode": 5, "fleetwood_mac": 6, "garth_brooks": 7,
                         "green_day": 8, "led_zeppelin": 9, "madonna": 10, "metallica": 11, "prince": 12, "queen": 13,
                         "radiohead": 14, "roxette": 15, "steely_dan": 16, "suzanne_vega": 17, "tori_amos": 18,
                         "u2": 19}
    singers = []
    singers_label = []
    song_feature = []
    for parent, dirnames, filenames in os.walk(dataset_dir):
        for filename in filenames:
            song_dir = os.path.join(parent, filename)
            song_feature = feature_reduction_union_list(song_dir)
            # song = load_audio(song_dir)
            song_feature.append(song_feature)
            # change the value as your singer name level in the dir
            # eg. short_wav/new_wav/dataset/singer_name so I set 3
            singer = song_dir.split('/')[1]
            # this value depends on the singer file level in the dir
            singers.append(singer)
            singers_label.append(dict_singer_label[singer])
    dataset['singers'] = singers
    dataset['data'] = numpy.array(song_feature)
    dataset['singers_label'] = numpy.array(singers_label)
    dataset['dict_singer_label'] = dict_singer_label
    pickle_dataset(dataset)
    return 0


def load_data(pkl_dir='dataset.pkl'):
    pkl_dataset = open(pkl_dir, 'rb')
    dataset = cPickle.load(pkl_dataset)
    return dataset


def test():
    build_dataset('dataset_short5')
    return 0


test()


def get_mono_left_right_audio(wavs_dir='mir1k-Wavfile'):
    for parent, dirnames, filenames in os.walk(wavs_dir):
        for filename in filenames:
            audio_dir = os.path.join(parent, filename)
            mono_sound_dir = 'mono/' + audio_dir
            if not os.path.exists(os.path.split(mono_sound_dir)[0]):
                os.makedirs(os.path.split(mono_sound_dir)[0])
            left_audio_dir = 'left_right/' + os.path.splitext(mono_sound_dir)[0] + '_left.wav'
            if not os.path.exists(os.path.split(left_audio_dir)[0]):
                os.makedirs(os.path.split(left_audio_dir)[0])
            right_audio_dir = 'left_right/' + os.path.splitext(mono_sound_dir)[0] + '_right.wav'
            if not os.path.exists(os.path.split(right_audio_dir)[0]):
                os.makedirs(os.path.split(right_audio_dir)[0])
            sound = AudioSegment.from_wav(audio_dir)
            mono = sound.set_channels(1)
            left, right = sound.split_to_mono()
            mono.export(mono_sound_dir, format='wav')
            left.export(left_audio_dir, format='wav')
            right.export(right_audio_dir, format='wav')
    return 0


def get_right_voice_audio(wavs_dir='mir1k-Wavfile'):
    for parent, dirnames, filenames in os.walk(wavs_dir):
        for filename in filenames:
            audio_dir = os.path.join(parent, filename)
            right_audio_dir = 'right_voices/' + os.path.splitext(audio_dir)[0] + '_right.wav'
            if not os.path.exists(os.path.split(right_audio_dir)[0]):
                os.makedirs(os.path.split(right_audio_dir)[0])
            sound = AudioSegment.from_wav(audio_dir)
            left, right = sound.split_to_mono()
            right.export(right_audio_dir, format='wav')
    return 0


def draw_wav(wav_dir):
    print "begin draw_wav ==feature_extraction.py=="
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
    pylab.xlabel("time")
    pylab.ylabel("wav_data")
    pylab.show()
    return 0


def get_mfcc(wav_dir):
    sample_rate, audio = scipy.io.wavfile.read(wav_dir)
    # ceps, mspec, spec = mfcc(audio, nwin=256, fs=8000, nceps=13)
    ceps, mspec, spec = mfcc_diy.mfcc(audio, nwin=8000, fs=8000, nceps=13)
    mfccs = ceps
    return mfccs


def get_raw(wav_dir):
    sample_rate, audio = scipy.io.wavfile.read(wav_dir)
    return audio


def pickle_dataset(dataset):
    pkl_file = file('dataset.pkl', 'wb')
    cPickle.dump(dataset, pkl_file, True)
    pkl_file.close()
    return 0


def load_dataset(pkl_dir):
    pkl_file = file(pkl_dir, 'rb')
    dataset = cPickle.load(pkl_file)
    pkl_file.close()
    return dataset


def filter_nan_inf(mfccss):
    filter_nan_infs = []
    for item in mfccss:
        new_item = []
        for ii in item:
            if numpy.isinf(ii):
                ii = 1000
            elif numpy.isnan(ii):
                ii = -11
            else:
                ii = ii
            new_item.append(ii)
        filter_nan_infs.append(new_item)
    new_mfcc = numpy.array(filter_nan_infs)
    return new_mfcc


def get_timbre_pitches_loudness(wav_dir):
    # from echonest capture the timbre and pitches loudness et.al.
    config.ECHO_NEST_API_KEY = "BPQ7TEP9JXXDVIXA5"  # daleloogn my api key
    f = open(wav_dir)
    print "process%s====================================================================" % wav_dir
    # t = track.track_from_file(f, 'wav')
    t = track.track_from_file(f, 'wav', 256, force_upload=True)
    # if not with force_upload it will timed out for sockets
    t.get_analysis()
    segments = t.segments  # list of dicts :timing,pitch,loudness and timbre for each segment
    # print'=========-----------------=================='
    # print segments
    # print'=========-----------------=================='
    # flag_test = 1
    # print 'echonest segments %i' % len(segments)
    timbre_pitches_loudness = from_segments_get_timbre_pitch_etal(wav_dir, segments)
    timbre_pitches_loudness_file_txt = open('timbre_pitches_loudness_file.txt', 'a')
    timbre_pitches_loudness_file_txt.write(wav_dir + '\r\n')
    timbre_pitches_loudness_file_txt.write(str(timbre_pitches_loudness))
    timbre_pitches_loudness_file_txt.close()
    return segments


def from_segments_get_timbre_pitch_etal(wav_dir, segments):
    timbre_pitches_loudness = []
    starts_point = []
    for segments_item in segments:
        timbre = segments_item['timbre']
        pitches = segments_item['pitches']
        loudness_start = segments_item['loudness_start']
        loudness_max_time = segments_item['loudness_max_time']
        loudness_max = segments_item['loudness_max']
        # print"###########################"
        durarion = segments_item['duration']
        start = segments_item['start']
        starts_point.append(start)
        # print durarion
        # print"###########################"
        segments_item_union = timbre + pitches + [loudness_start, loudness_max_time, loudness_max]
        timbre_pitches_loudness.append(segments_item_union)
    ####
    ##plot the segments seg
    draw_segments_from_echonest(wav_dir, starts_point)
    ####
    return timbre_pitches_loudness


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


def generate_singer_label(wavs_dir):
    dict_singer_label = {}
    singers = []
    for parent, dirnames, filenames in os.walk(wavs_dir):
        for filename in filenames:
            singer_name = filename.split('_')[0]
            singers.append(singer_name)
    only_singers = sorted(list(set(singers)))
    for item, singer in enumerate(only_singers):
        dict_singer_label[singer] = item
    # print dict_singer_label
    return dict_singer_label


def build_dataset(wavs_dir):
    print 'from %s build dataset==============' % wavs_dir
    dataset = []
    data = []
    target = []
    dict_singer_label = generate_singer_label(wavs_dir)
    for parent, dirnames, filenames in os.walk(wavs_dir):
        for filename in filenames:
            song_dir = os.path.join(parent, filename)
            print"get mfcc of %s ====" % filename
            song_feature = get_mfcc(song_dir)

            song_feature = filter_nan_inf(song_feature)  # feature=======================a song mfcc vector
            singer = filename.split('_')[0]  # this value depends on the singer file level in the dir
            singer_label = dict_singer_label[singer]  # target class====================
            # song_mfcc_sum_vector = mfcc_sum_vector(song_feature)
            #  feature=======================a song mfcc vector sum
            songs_mfcc_vecto_link = []
            for vector_item in song_feature:
                vector_item = [x for x in vector_item]
                # songs_mfcc_vecto_link.extend(vector_item)
                # data.append(songs_mfcc_vecto_link)  # feature just a frame
                data.append(vector_item)
                target.append(singer_label)

    dataset.append(data)
    # print data[1:50]
    dataset.append(target)
    print 'pkl_to dataset.pkl'
    pickle_dataset(dataset)
    return 0


def slice_wav_beigin_one_end_one(wav_dir):
    new_dir = 'sliced/' + wav_dir
    if not os.path.exists(os.path.split(new_dir)[0]):
        os.makedirs(os.path.split(new_dir)[0])
    audio = AudioSegment.from_wav(wav_dir)
    one_seconds = 1 * 1000
    first_five_seconds = audio[one_seconds:-2000]
    first_five_seconds.export(new_dir, format='wav')
    return 0


def slice_wavs_dirs(dirs):
    for parent, dirnames, filenames in os.walk(dirs):
        for filename in filenames:
            song_dir = os.path.join(parent, filename)
            slice_wav_beigin_one_end_one(song_dir)
    return 0


def view_pkl_dataset(dataset_dir='dataset.pkl'):
    dataset = load_dataset(dataset_dir)
    data_num = len(dataset[0])
    print data_num
    max_long = 0
    for item in dataset[0]:
        long = len(item)
        if long > max_long:
            max_long = long
    print max_long
    print max_long / 13
    print 'let us check out the target '
    singgers_set = set(dataset[1])
    print(singgers_set)

    return 0


def save_echonest_data_to_txt(wav_dirs):
    for parent, dirnames, filenames in os.walk(wav_dirs):
        for filename in filenames:
            song_dir = os.path.join(parent, filename)
            if not os.path.exists('save_segments.txt'):
                segments_file = open('save_segments.txt', 'w')
                segments_file.close()
            segments_file = open('save_segments.txt', 'r')
            all_lines = segments_file.readlines()
            segments_file.close()
            dirs = []
            for line_item in all_lines:
                dir_song = line_item.split('@')[0]
                dirs.append(dir_song)
            if song_dir in dirs:
                pass
            else:
                segments = get_timbre_pitches_loudness(song_dir)
                lines = song_dir + '@' + str(segments) + '\r\n'
                segments_file = open('save_segments.txt', 'a', )
                segments_file.write(lines)
                segments_file.close()

    return 0


def try_except_goon(program):
    try:
        program
    except Exception, ex:
        try_except_goon(program)
    return 0


print '\033[1;31;40m'
print 'start===================================='
start = time()

finish = time()
print '\033[1;33;40m'
print"build dataset takes %.2f seconds" % (finish - start)
print '\033[0m'


def draw_wav(wav_dir):
    print "begin draw_wav ==feature_extraction.py=="
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

    pylab.xlabel("time")
    pylab.ylabel("wav_data")
    pylab.show()
    return 0


def get_mfcc(wav_dir):
    sample_rate, audio = scipy.io.wavfile.read(wav_dir)
    ceps, mspec, spec = mfcc(audio, nwin=256, nfft=512, fs=8000, nceps=13)
    mfccs = ceps
    return mfccs


def get_pca(mata, length):
    # PCA reduce the dimension
    meanVal = mean(mata, axis=0)
    stdVal = std(mata)
    rmmeanMat = (mata - meanVal) / stdVal
    covMat = cov(rmmeanMat, rowvar=0)
    eigval, eigvec = linalg.eig(covMat)
    maxnum = argsort(-eigval, axis=0)  # sort descend
    tfMat = eigvec[:, maxnum[0:length]]  # top length
    finalData = dot(rmmeanMat, tfMat)  #
    recoMat = finalData * tfMat.T * stdVal + meanVal
    return finalData, recoMat


def pickle_dataset(dataset):
    pkl_file = file('dataset.pkl', 'wb')
    cPickle.dump(dataset, pkl_file, True)
    pkl_file.close()
    return 0


def filter_nan_inf(mfccss):
    filter_nan_infs = []
    for item in mfccss:
        new_item = numpy.nan_to_num(item)
        filter_nan_infs.append(new_item)
    new_mfcc = numpy.array(filter_nan_infs)
    return new_mfcc


def mfcc_sum_vector(song_feature):
    sums = []
    for mfcc_item in song_feature:
        sums.append(sum(mfcc_item))
    return sums


def build_dataset(datasets_dir):
    print 'from %s build dataset==============' % datasets_dir
    dataset = []
    data = []
    target = []
    dict_singer_label = {"aerosmith": 0, "beatles": 1, "creedence_clearwater_revival": 2, "cure": 3,
                         "dave_matthews_band": 4, "depeche_mode": 5, "fleetwood_mac": 6, "garth_brooks": 7,
                         "green_day": 8, "led_zeppelin": 9, "madonna": 10, "metallica": 11, "prince": 12, "queen": 13,
                         "radiohead": 14, "roxette": 15, "steely_dan": 16, "suzanne_vega": 17, "tori_amos": 18,
                         "u2": 19}

    for parent, dirnames, filenames in os.walk(datasets_dir):
        for filename in filenames:
            song_dir = os.path.join(parent, filename)
            print"get mfcc of %s ====" % filename
            song_feature = get_mfcc(song_dir)
            mfcc_file = open('mfcc.txt', 'a')
            mfcc_file.write(song_dir)
            mfcc_file.write(str(song_feature) + '\r\n')
            print 'filter mfcc nan inf'
            song_feature = filter_nan_inf(song_feature)  # feature=======================a song mfcc vector
            fprint = open('mfcc_feature.txt', 'a')
            fprint.write(str(song_feature) + '\r\n')
            fprint.close()
            singer = song_dir.split('/')[1]  # this value depends on the singer file level in the dir
            singer_label = dict_singer_label[singer]  # target class====================
            # song_mfcc_sum_vector = mfcc_sum_vector(song_feature)
            #  feature=======================a song mfcc vector sum
            for vector_item in song_feature:
                data.append(vector_item)  # feature just a frame
                target.append(singer_label)
    dataset.append(data)
    dataset.append(target)
    print 'pkl_to dataset.pkl'
    pickle_dataset(dataset)
    return 0


def load_dataset(pkl_dir):
    pkl_file = file(pkl_dir, 'rb')
    dataset = cPickle.load(pkl_file)
    pkl_file.close()
    return dataset


def split_dataset(dataset):
    data_x = dataset['data']
    target_y = dataset['singers_label']
    dict_singer_label = dataset['dict_singer_label']
    total_len = len(data_x)
    train_par = 1
    train_data_x = data_x[0:train_par * total_len]
    train_target_y = target_y[0:train_par * total_len]
    test_data_x = data_x[train_par * total_len:total_len]
    test_target_y = target_y[train_par * total_len:total_len]
    trainset = {}
    testset = {}
    trainset['data'] = train_data_x
    trainset['target'] = train_target_y
    trainset['dict'] = dict_singer_label
    testset['data'] = test_data_x
    testset['target'] = test_target_y
    testset['dict'] = dict_singer_label
    return trainset, testset


def test(dirss):
    return 0


if __name__ == '__main__':
    print 'start===================================='
    start = time()
    build_dataset('dataset_E_vocal')

    finish = time()
    print"build dataset takes %.2f" % (finish - start)
    timefile = open('time.txt', 'a')

    timefile.write(str(finish - start) + '\r\n')


def draw_wav(wav_dir):
    print "begin draw_wav ==feature_extraction.py=="
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

    pylab.xlabel("time")
    pylab.ylabel("wav_data")
    pylab.show()
    return 0


def get_mfcc(wav_dir):
    sample_rate, audio = scipy.io.wavfile.read(wav_dir)
    ceps, mspec, spec = mfcc(audio, nwin=256, fs=8000, nceps=13)
    mfccs = ceps
    return mfccs


def get_pca(mata, length):
    # PCA reduce the dimension
    meanVal = mean(mata, axis=0)
    stdVal = std(mata)
    rmmeanMat = (mata - meanVal) / stdVal
    covMat = cov(rmmeanMat, rowvar=0)
    eigval, eigvec = linalg.eig(covMat)
    maxnum = argsort(-eigval, axis=0)  # sort descend
    tfMat = eigvec[:, maxnum[0:length]]  # top length
    finalData = dot(rmmeanMat, tfMat)  #
    recoMat = finalData * tfMat.T * stdVal + meanVal
    return finalData, recoMat


def pickle_dataset(dataset):
    pkl_file = file('dataset.pkl', 'wb')
    cPickle.dump(dataset, pkl_file, True)
    pkl_file.close()
    return 0


def filter_nan_inf(mfccss):
    filter_nan_infs = []
    for item in mfccss:
        new_item = numpy.nan_to_num(item)
        filter_nan_infs.append(new_item)
    new_mfcc = numpy.array(filter_nan_infs)
    return new_mfcc


def build_dataset(datasets_dir):
    print 'from %s build dataset==============' % datasets_dir
    dataset = {"author": "zhangxulong"}
    dict_singer_label = {"aerosmith": 0, "beatles": 1, "creedence_clearwater_revival": 2, "cure": 3,
                         "dave_matthews_band": 4, "depeche_mode": 5, "fleetwood_mac": 6, "garth_brooks": 7,
                         "green_day": 8, "led_zeppelin": 9, "madonna": 10, "metallica": 11, "prince": 12, "queen": 13,
                         "radiohead": 14, "roxette": 15, "steely_dan": 16, "suzanne_vega": 17, "tori_amos": 18,
                         "u2": 19}
    singers = []
    singers_label = []
    feature = []

    for parent, dirnames, filenames in os.walk(datasets_dir):
        for filename in filenames:
            song_dir = os.path.join(parent, filename)
            print"get mfcc of %s ====" % filename
            song_feature = get_mfcc(song_dir)
            mfcc_file = open('mfcc.txt', 'a')
            mfcc_file.write(song_dir)
            mfcc_file.write(str(song_feature) + '\r\n')
            print 'filter mfcc nan inf'
            song_feature = filter_nan_inf(song_feature)
            song_feature_mat = numpy.mat(song_feature)
            print"get pca of mfcc"
            final_feature, recover_feature = get_pca(song_feature_mat.transpose(), 1)
            final_feature = final_feature.transpose()
            final_feature = numpy.array(final_feature)[0].tolist()
            fprint = open('feature_print', 'a')
            feature_print = str(final_feature)
            fprint.write(feature_print + '\r\n')
            fprint.close()
            feature.append(final_feature)
            # TODO change the value as your singer name level in the dir
            # TODO eg. short_wav/new_wav/dataset/singer_name so I set 3
            singer = song_dir.split('/')[1]  # this value depends on the singer file level in the dir
            singers.append(singer)
            singers_label.append(dict_singer_label[singer])
    dataset['singers'] = singers
    dataset['data'] = numpy.array(feature)
    dataset['singers_label'] = numpy.array(singers_label)
    dataset['dict_singer_label'] = dict_singer_label
    print 'pkl_to dataset.pkl'
    pickle_dataset(dataset)
    return 0


def load_dataset(pkl_dir):
    pkl_file = file(pkl_dir, 'rb')
    dataset = cPickle.load(pkl_file)
    pkl_file.close()
    return dataset


def split_dataset(dataset):
    data_x = dataset['data']
    target_y = dataset['singers_label']
    dict_singer_label = dataset['dict_singer_label']
    total_len = len(data_x)
    train_par = 1
    train_data_x = data_x[0:train_par * total_len]
    train_target_y = target_y[0:train_par * total_len]
    test_data_x = data_x[train_par * total_len:total_len]
    test_target_y = target_y[train_par * total_len:total_len]
    trainset = {}
    testset = {}
    trainset['data'] = train_data_x
    trainset['target'] = train_target_y
    trainset['dict'] = dict_singer_label
    testset['data'] = test_data_x
    testset['target'] = test_target_y
    testset['dict'] = dict_singer_label
    return trainset, testset


def test(dirss):
    return 0


if __name__ == '__main__':
    print 'start===================================='
    start = time()
    build_dataset('dataset_short5')
    finish = time()
    print"build dataset takes %.2f" % (finish - start)
    timefile = open('time.txt', 'a')
    timefile.write(str(finish - start) + '\r\n')


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
