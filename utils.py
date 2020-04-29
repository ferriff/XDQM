import struct
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as dsp
import sys
from numba import jit

def clear():
    plt.cla()
    plt.clf()

def show():
    plt.show(block=False)

def plot(h, ylog = True):
    clear()
    plt.yscale('log' if ylog else 'linear')
    plt.step(*h.data)
    show()

from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

class DataIter:
    '''Class to read cross data from Run 2 DAQ'''
    def __init__(self, file, chunck_size):
        if isinstance(file, str):
            self.fname = file
            self.f = None
        else:
            self.f = file
            self.fname = None

        self.chunck_size = chunck_size

    def __iter__(self):
        if self.f is None:
            self.f = open(self.fname, "rb")

        self.f.seek(0, 2)
        self.fsize = self.f.tell()
        self.f.seek(0, 0)

        self.ichunck = 0
        s1 = self.f.read(1)
        s3 = self.f.read(3)
        self.endianness = 'little' if 'l' in s3.decode() else 'big'
        self.nbits = int.from_bytes(s1, byteorder = self.endianness)
        s = self.f.read(4)
        self.sampling_freq = struct.unpack('f', s)[0]
        print('# Header --- endiannes: %s  nbits: %d  sampling_frequency: %f' % (self.endianness, self.nbits, self.sampling_freq))
        return self

    def __next__(self):
        chunck =  np.fromfile(self.f, dtype = np.dtype(np.uint32), count = self.chunck_size)
        self.ichunck += 1
        if len(chunck) == 0:
            raise StopIteration
        else:
            return chunck

    def progress(self):
        return self.ichunck * 4. * self.chunck_size / self.fsize

class Hist1D(object):

    def __init__(self, nbins, xlow, xhigh):
        self.nbins = nbins
        self.xlow  = xlow
        self.xhigh = xhigh
        self.hist, edges = np.histogram([], bins=nbins, range=(xlow, xhigh))
        self.bins = (edges[:-1] + edges[1:]) / 2.

    def fill(self, arr):
        hist, edges = np.histogram(arr, bins=self.nbins, range=(self.xlow, self.xhigh))
        self.hist += hist

    @property
    def data(self):
        return self.bins, self.hist

def make_filter(nped, nrise, npeak):
    return np.array([-1./nped]*nped + [0]*nrise + [1./npeak] * npeak)

@jit(nopython=True)
def peak_finder(a, thr = 0.):
    return np.array([ a[i] if a[i-1] < a[i] and a[i] > a[i+1] and a[i] > thr else 0 for i in range(len(a) - 2) ])

def volt(raw):
    scale = 10.24/(1<<31)
    return raw*scale-10.24

h = Hist1D(10000, 0., 1)
sig_filter = make_filter(50, 10, 1)


def loop():

    reader = DataIter('/home/pgras/work/cupid/data/Th232/000059_20200327T131240_001_000.bin', 2000*3600)

    ns = 0
    for i, data in enumerate(reader):
        #signal = peak_finder(np.convolve(sig_filter, volt(data), mode='valid'), 0.001)
        signal = peak_finder(butter_bandpass_filter(volt(data)-volt(data[0]), 0.5, 50, 2000, 3) , 0.001)
        h.fill(signal[np.where(signal > 0)])
        print("%.1f%%" % (reader.progress() * 100.))
        ns += len(signal)
        #if i > 10:
        #break

    print(ns)
    print(ns/2000/3600.)
    return h
