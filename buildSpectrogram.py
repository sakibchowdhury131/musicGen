import librosa
import matplotlib.pyplot as plt



def getSpectrogram(audio_fpath):
  #sampling_rate, data=read_wav(audio_fpath)
  x, sr = librosa.load(audio_fpath)
  librosa.display.waveplot(x, sr=sr)
  X = librosa.stft(x)
  Xdb = librosa.amplitude_to_db(abs(X))
  plt.figure(figsize=(14, 5))
  librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
  plt.colorbar()





def getLogMelSpec(audio_path):
  #librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
  x, sr = librosa.load(audio_path)
  #ipd.Audio(x, rate = sr)
  X = librosa.stft(x)
  X = librosa.feature.melspectrogram(x)
  Xdb = librosa.amplitude_to_db(abs(X))
  #plt.figure(figsize=(14, 5))

  #librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
  #plt.colorbar()
  return Xdb