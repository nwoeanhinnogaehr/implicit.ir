import wave
import struct
import numpy as np
from stft import *

impulse_length = 65536 # all must be same length
impulse_stft = STFT(impulse_length,2,2)

impulse_files = ["output/impulse{:03}.wav".format(i) for i in range(16)]
impulses = []
impulse_spectra = []
for filename in impulse_files:
    with wave.open(filename, 'rb') as reader:
        n = reader.getnframes()
        assert(n == impulse_length)
        nchan = reader.getnchannels()
        frames = reader.readframes(n*nchan)
        out = struct.unpack_from("%dh" % n*nchan, frames)
        split = [out[i::nchan] for i in range(nchan)]
        audio = np.array(split, dtype=float)/32767
        impulses.append(audio)
        spec = np.fft.rfft(audio)
        spec /= np.max(abs(spec))
        impulse_spectra.append(spec)

# which impulse to select at frame t
def select_buf(t):
    return t%len(impulses)

time = 0
def process(i,o):
    global time
    for x in impulse_stft.forward(i):
        if len(impulses) > 0:
            idx = select_buf(time)
            x *= impulse_spectra[idx]*8
        impulse_stft.backward(x)
        time += 1
    impulse_stft.pop(o)
    o/=4

