import cPickle
import wave
import gzip
import scipy
import scipy.io.wavfile
from matplotlib import pylab
import scipy.io.wavfile
import os
from numpy import *
from pydub import AudioSegment
from pyechonest import track, config
import numpy
import mfcc_diy


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
    # pickle the file and for long time save
    pkl_file = file(out_pkl, 'wb')
    cPickle.dump(dataset, pkl_file, True)
    pkl_file.close()
    return 0


def build_song_set(songs_dir):
    # save songs and singer lable to pickle file
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


def load_data(pkl_dir='dataset.pkl'):
    # load pickle data file
    pkl_dataset = open(pkl_dir, 'rb')
    dataset = cPickle.load(pkl_dataset)
    pkl_dataset.close()
    return dataset


def get_mono_left_right_audio(wavs_dir='mir1k-Wavfile'):
    # split a audio to left and right channel
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
    # get singer voice from the right channel
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
    '''
    draw the wav audio to show
    '''
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
    # mfccs
    sample_rate, audio = scipy.io.wavfile.read(wav_dir)
    # ceps, mspec, spec = mfcc(audio, nwin=256, fs=8000, nceps=13)
    ceps, mspec, spec = mfcc_diy.mfcc(audio, nwin=8000, fs=8000, nceps=13)
    mfccs = ceps
    return mfccs


def get_raw(wav_dir):
    # raw audio data
    sample_rate, audio = scipy.io.wavfile.read(wav_dir)
    return audio


def filter_nan_inf(mfccss):
    # filter the nan and inf data point of mfcc
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
    print "process:============ %s =============" % wav_dir
    t = track.track_from_file(f, 'wav', 256, force_upload=True)
    t.get_analysis()
    segments = t.segments  # list of dicts :timing,pitch,loudness and timbre for each segment
    timbre_pitches_loudness = from_segments_get_timbre_pitch_etal(wav_dir, segments)
    timbre_pitches_loudness_file_txt = open('timbre_pitches_loudness_file.txt', 'a')
    timbre_pitches_loudness_file_txt.write(wav_dir + '\r\n')
    timbre_pitches_loudness_file_txt.write(str(timbre_pitches_loudness))
    timbre_pitches_loudness_file_txt.close()
    return segments


def draw_segments_from_echonest(wav_dir, starts_point):
    # just draw it and show the difference duration of segments
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


def from_segments_get_timbre_pitch_etal(wav_dir, segments):
    # from segments get the feature you want
    timbre_pitches_loudness = []
    starts_point = []
    for segments_item in segments:
        timbre = segments_item['timbre']
        pitches = segments_item['pitches']
        loudness_start = segments_item['loudness_start']
        loudness_max_time = segments_item['loudness_max_time']
        loudness_max = segments_item['loudness_max']
        durarion = segments_item['duration']
        start = segments_item['start']
        starts_point.append(start)
        segments_item_union = timbre + pitches + [loudness_start, loudness_max_time, loudness_max]
        timbre_pitches_loudness.append(segments_item_union)
        ##plot the segments seg
    draw_segments_from_echonest(wav_dir, starts_point)
    ####
    return timbre_pitches_loudness


def generate_singer_label(wavs_dir):
    # generate the singer to label dict
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
    # it used for cut the wav file head and end
    new_dir = 'sliced/' + wav_dir
    if not os.path.exists(os.path.split(new_dir)[0]):
        os.makedirs(os.path.split(new_dir)[0])
    audio = AudioSegment.from_wav(wav_dir)
    one_seconds = 1 * 1000
    first_five_seconds = audio[one_seconds:-2000]
    first_five_seconds.export(new_dir, format='wav')
    return 0


def slice_wavs_dirs(dirs):
    # in batach to slice
    for parent, dirnames, filenames in os.walk(dirs):
        for filename in filenames:
            song_dir = os.path.join(parent, filename)
            slice_wav_beigin_one_end_one(song_dir)
    return 0


def save_echonest_data_to_txt(wav_dirs):
    # cache the data from the internet
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


def print_color(color="red or yellow"):
    # consloe out color
    if color == 'red':
        print '\033[1;31;40m'
    elif color == 'yellow':
        print '\033[1;33;40m'
    else:
        print '\033[0m'
    return 0


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


def load_imdb(imdb_dataset='datasets/imdb.pkl', nb_words=None, skip_top=0, maxlen=None, test_split=0.2, seed=113,
              start_char=1, oov_char=2, index_from=3):
    if imdb_dataset.endswith(".gz"):
        f = gzip.open(imdb_dataset, 'rb')
    else:
        f = open(imdb_dataset, 'rb')
    X, labels = cPickle.load(f)
    f.close()
    numpy.random.seed(seed)
    numpy.random.shuffle(X)
    numpy.random.seed(seed)
    numpy.random.shuffle(labels)
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
    numpy.random.seed(seed)
    numpy.random.shuffle(X)
    numpy.random.seed(seed)
    numpy.random.shuffle(labels)
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
