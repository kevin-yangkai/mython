from pydub import AudioSegment
import subprocess
import os


# readme mv title should not have space like that "I Love pytho.mpeg" u can change space to _
def transe_mpeg_to_wav():
    mvs_dir = 'MV'
    for parent, dirnames, filenames in os.walk(mvs_dir):
        for filename in filenames:
            mv_dir = os.path.join(parent, filename)

            audio_dir = 'audio/' + os.path.splitext(mv_dir)[0] + '.wav'
            if not os.path.exists(os.path.split(audio_dir)[0]):
                os.makedirs(os.path.split(audio_dir)[0])
            command = 'ffmpeg -i ' + mv_dir + ' -ab 160k -ac 2 -ar 44100 -vn ' + audio_dir
            subprocess.call(command, shell=True)

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


print 'okay'
