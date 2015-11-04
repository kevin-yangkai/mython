"""
Feature extraction.
Siddharth Sigia
Feb,2014
C4DM
"""
import subprocess
import os
import wave
import pylab
import scipy.io.wavfile

import tables
import numpy
from pyechonest import track
from pyechonest import config
from scikits.talkbox.features import mfcc
from scikits.talkbox.linpred import lpc

from spectrogram import SpecGram
from dimension_reduction import pca

__author__ = 'zhangxulong'


# extract the feature used for sid
# here used  timbre, pitches and loudness by Echo Nest API

def turn_api():
    number = 0
    # number = random.randint(0, 2)

    if number == 0:
        config.ECHO_NEST_API_KEY = "BPQ7TEP9JXXDVIXA5"  # daleloogn my api key
    elif number == 1:
        config.ECHO_NEST_API_KEY = "GEUJCAK8YWE7XFJYK"  # zhangxulong my api key
    else:
        config.ECHO_NEST_API_KEY = "IYM0ZNGMKC1NZHQHH "  # zhangxulong my api key
    return 0


def draw_wave(wave_data, nframes, framerate):
    time = numpy.arange(0, nframes) * (1.0 / framerate)
    len_time = len(time)
    time = time[0:len_time]
    pylab.subplot(211)
    pylab.plot(time, wave_data[0])
    pylab.subplot(212)
    pylab.plot(time, wave_data[1], c="r")
    pylab.xlabel("time")
    pylab.ylabel("wav_data")
    pylab.show()
    return 0


def get_timbre_pitches_loudness(wav_dir):
    turn_api()
    f = open(wav_dir)
    print "process%s====================================================================" % wav_dir
    # t = track.track_from_file(f, 'wav')
    t = track.track_from_file(f, 'wav', 256, force_upload=True)
    # if not with force_upload it will timed out for sockets
    t.get_analysis()
    segments = t.segments  # list of dicts :timing,pitch,loudness and timbre for each segment
    timbre_pitches_loudness = []
    # flag_test = 1
    for segments_item in segments:
        timbre = segments_item['timbre']
        pitches = segments_item['pitches']
        loudness_start = segments_item['loudness_start']
        loudness_max_time = segments_item['loudness_max_time']
        loudness_max = segments_item['loudness_max']
        segments_item_union = timbre + pitches + [loudness_start, loudness_max_time, loudness_max]
        timbre_pitches_loudness.append(segments_item_union)
    timbre_pitches_loudness_file_txt = open('timbre_pitches_loudness_file.txt', 'a')
    timbre_pitches_loudness_file_txt.write('\r\n' + wav_dir + '\r\n')
    timbre_pitches_loudness_file_txt.write(str(timbre_pitches_loudness))
    timbre_pitches_loudness_file_txt.close()
    return timbre_pitches_loudness


def get_mfcc(wav_dir):
    sample_rate, audio = scipy.io.wavfile.read(wav_dir)
    ceps, mspec, spec = mfcc(audio, nwin=256, fs=8000, nceps=13)
    mfccs = ceps
    return mfccs


def get_lpc(wav_dir):
    sample_rate, audio = scipy.io.wavfile.read(wav_dir)
    audio_array = numpy.array(audio)
    lpcs_a, lpcs_e, lpcs_k = lpc(audio_array, 12)
    return list(lpcs_a)


def generate_union_feature_matrix(wav_dir):
    union_feature = []
    timbre_pitches_loudness = get_timbre_pitches_loudness(wav_dir)
    mfccs = get_mfcc(wav_dir)
    lpcs = get_lpc(wav_dir)
    timbre_pitches_loudness_flag = len(timbre_pitches_loudness) - 1
    lpcs_flag = len(lpcs) - 1
    initial_value = 0
    list_length27 = 27
    list_length13 = 13
    timbre_pitches_loudness_list = [initial_value] * list_length27
    lpc_list = [initial_value] * list_length13
    for index, mfccs_item in enumerate(mfccs):
        if timbre_pitches_loudness_flag > 0 and lpcs_flag > 0:
            union_feature.append(timbre_pitches_loudness[index] + list(mfccs_item) + list(lpcs[index]))
        elif timbre_pitches_loudness_flag > 0 > lpcs_flag:
            union_feature.append(timbre_pitches_loudness[index] + list(mfccs_item) + lpc_list)
        else:
            union_feature.append(timbre_pitches_loudness_list + list(mfccs_item) + lpc_list)
        timbre_pitches_loudness_flag -= 1
        lpcs_flag -= 1
    matrix_list = []
    for union_feature_item in union_feature:
        matrix_list.append(union_feature_item[0:52])
    union_feature_matrix = numpy.mat(matrix_list)  # ndim cannot over 41
    return union_feature_matrix


def draw_wav(wav_dir):
    song = wave.open(wav_dir, "rb")
    params = song.getparams()
    nchannels, samplewidth, framerate, nframes = params[:4]  # format info
    song_data = song.readframes(nframes)
    song.close()
    wave_data = numpy.fromstring(song_data, dtype=numpy.short)
    wave_data.shape = -1, 2
    wave_data = wave_data.T
    draw_wave(wave_data, nframes, framerate)
    return 0


def feature_reduction_union_list(wav_dir):
    # get the union feature list 53 bit and the first 27 are the pca of (timbre and pitch and loudness)
    # and the middle 13 are the  pca of mfcc and the last 13 are the lpc
    ndim = 1
    matrix_timbre = numpy.mat(get_timbre_pitches_loudness(wav_dir))
    matrix_timbre = matrix_timbre.transpose()
    final_timbre, recover_timbre = pca(matrix_timbre, ndim)
    matrix_mfcc = numpy.mat(get_mfcc(wav_dir))
    matrix_mfcc = matrix_mfcc.transpose()
    final_mfcc, recover_mfcc = pca(matrix_mfcc, ndim)
    lpcs = get_lpc(wav_dir)
    matrix_lpcs = numpy.mat(lpcs)
    final_lpcs = matrix_lpcs.transpose()
    final_timbre_list = final_timbre.transpose().tolist()[0]
    final_mfcc_list = final_mfcc.transpose().tolist()[0]
    final_lpcs_list = final_lpcs.transpose().tolist()[0]
    feature_reduction_union_lists = []
    feature_reduction_union_lists.extend(final_timbre_list)
    feature_reduction_union_lists.extend(final_mfcc_list)
    feature_reduction_union_lists.extend(final_lpcs_list)
    return feature_reduction_union_lists


def read_wav(filename):
    bits_per_sample = '16'
    cmd = ['sox', filename, '-t', 'raw', '-e', 'unsigned-integer', '-L', '-c', '1', '-b', bits_per_sample, '-', 'pad',
           '0', '30.0', 'rate', '22050.0', 'trim', '0', '30.0']
    cmd = ' '.join(cmd)
    print cmd
    raw_audio = numpy.fromstring(subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0],
                                 dtype='uint16')
    max_amp = 2. ** (int(bits_per_sample) - 1)
    raw_audio = (raw_audio - max_amp) / max_amp
    return raw_audio


def calc_specgram(x, fs, winSize, ):
    spec = SpecGram(x, fs, winSize)
    return spec.specMat


def make_4tensor(x):
    assert x.ndim <= 4
    while x.ndim < 4:
        x = numpy.expand_dims(x, 0)
    return x


class FeatExtraction():
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.list_dir = os.path.join(self.dataset_dir, 'lists')
        self.get_filenames()
        self.feat_dir = os.path.join(self.dataset_dir, 'features')
        self.make_feat_dir()
        self.h5_filename = os.path.join(self.feat_dir, 'feats.h5')
        self.make_h5()
        self.setup_h5()
        self.extract_features()
        self.close_h5()

    def get_filenames(self, ):
        dataset_files = os.path.join(self.list_dir, 'audio_files.txt')
        self.filenames = [l.strip() for l in open(dataset_files, 'r').readlines()]
        self.num_files = len(self.filenames)

    def make_feat_dir(self, ):
        if not os.path.exists(self.feat_dir):
            print 'Making output dir.'
            os.mkdir(self.feat_dir)
        else:
            print 'Output dir already exists.'

    def make_h5(self, ):
        if not os.path.exists(self.h5_filename):
            self.h5 = tables.openFile(self.h5_filename, 'w')
        else:
            print 'Feature file already exists.'
            self.h5 = tables.openFile(self.h5_filename, 'a')

    def setup_h5(self, ):
        filename = self.filenames[0]
        x = read_wav(filename)
        spec_x = calc_specgram(x, 22050, 1024)
        spec_x = make_4tensor(spec_x)
        self.data_shape = spec_x.shape[1:]
        self.x_earray_shape = (0,) + self.data_shape
        self.chunkshape = (1,) + self.data_shape
        self.h5_x = self.h5.createEArray('/', 'x', tables.FloatAtom(itemsize=4), self.x_earray_shape,
                                         chunkshape=self.chunkshape, expectedrows=self.num_files)
        self.h5_filenames = self.h5.createEArray('/', 'filenames', tables.StringAtom(256), (0,),
                                                 expectedrows=self.num_files)
        self.h5_x.append(spec_x)
        self.h5_filenames.append([filename])

    def extract_features(self, ):
        for i in xrange(1, self.num_files):
            filename = self.filenames[i]
            print 'Filename: ', filename
            x = read_wav(filename)
            spec_x = calc_specgram(x, 22050, 1024)
            spec_x = make_4tensor(spec_x)
            self.h5_x.append(spec_x)
            self.h5_filenames.append([filename])

    def close_h5(self, ):
        self.h5.flush()
        self.h5.close()


if __name__ == '__main__':
    test = FeatExtraction('dataset')
