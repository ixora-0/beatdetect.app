# parameters of histograms
SAMPLE_RATE = 22050  # hz
N_FFT = 1024
N_STFT = int((N_FFT // 2) + 1)
F_MIN = 30  # hz
F_MAX = 11000  # hz
N_MELS = 128
MEL_SCALE = "slaney"
HOP_LENGTH = 441
