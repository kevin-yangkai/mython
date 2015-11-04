import cPickle
from pydub import AudioSegment
from audiofile_read import *  # included in the rp_extract git package
# Rhythm Pattern Audio Extraction Library
from rp_extract import rp_extract


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


def get_feature_timbre_ssd(wav_dir):
    audiofile = wav_dir
    samplerate, samplewidth, wavedata = audiofile_read(audiofile)
    nsamples = wavedata.shape[0]
    nchannels = wavedata.shape[1]
    print "Successfully read audio file:", audiofile
    print samplerate, "Hz,", samplewidth * 8, "bit,", nchannels, "channel(s),", nsamples, "samples"

    features = rp_extract(wavedata,  # the two-channel wave-data of the audio-file
                          samplerate=11025,  # the samplerate of the audio-file
                          extract_ssd=True,  # <== extract this feature!
                          transform_db=True,  # apply psycho-accoustic transformation
                          transform_phon=True,  # apply psycho-accoustic transformation
                          transform_sone=True,  # apply psycho-accoustic transformation
                          fluctuation_strength_weighting=True,  # apply psycho-accoustic transformation
                          skip_leadin_fadeout=1,  # skip lead-in/fade-out. value = number of segments skipped
                          step_width=1)  #
    # plotssd(features['ssd'])
    print len(features['ssd'])
    return features


def get_timbre_ssd(dirs):
    dict_singer_label = generate_singer_label(dirs)
    dataset = []
    data = []
    target = []
    for parent, dirnames, filenames in os.walk(dirs):
        for filename in filenames:
            song_dir = os.path.join(parent, filename)
            features_dict = get_feature_timbre_ssd(song_dir)
            ssd_feature = features_dict['ssd']
            ssd_feature = ssd_feature.tolist()
            singer = filename.split('_')[0]  # this value depends on the singer file level in the dir
            singer_label = dict_singer_label[singer]  # target class====================
            data.append(ssd_feature)
            target.append(singer_label)
    dataset.append(data)
    # print data[1:50]
    dataset.append(target)
    return dataset


def pickle_dataset(dataset):
    pkl_file = file('ssd_6singers_dataset.pkl', 'wb')
    cPickle.dump(dataset, pkl_file, True)
    pkl_file.close()
    return 0

pickle_dataset(get_timbre_ssd('aa'))

print 'ok'

