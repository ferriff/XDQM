import struct
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as dsp
import sys
import os
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

class DataReader:
    '''Class to read cross data from Run 2 DAQ.
Usage: h = DataReader(['file1', 'file2', 'file3',..], nsamples_in_a_chunck)
       for chunk in h:
            ...analyse chunk, a numpy array with time samples...
            print("Progress: %.1f%%" % h.progress()*100.)
'''
    def __init__(self, files, chunck_size):
        '''Initialises the data reader. The list of files to read (name or file objects)
is provided in the parameter files and the number of sample is each returned
data chunk in the parameter chunck_size'''
        
        if isinstance(files, list) or isinstance(files, tuple):
            self.files = files            
        else:
            self.files = (files,)

        self.chunck_size = chunck_size
        self._init_tot_size()
        self._init_counters()
        
    def _init_counters(self):
        self.ifile = -1
        self.last_chunck_size = 0
        self.nread = 0        
        
    def _init_tot_size(self):
        self.tot_bytes = 0
        for f in self.files:
            if isinstance(f, str):
                st = os.stat(f)
            else:
                st = os.fstat(f)
            self.tot_bytes += st.st_size

    def _getfile(file):
        if isinstance(file, str):
            return open(file, "rb")
        else:
            return file

    def _read_header(self):
        header = self.f.read(8)
        if len(header) < 8:
            print("Skipping empty file %s" % f.name, file=sys.stderr)
            return None
        
        nbits, endianness, self.sampling_freq  = struct.unpack('<ccxxf', header)
        self.nbits = int(nbits[0])
        self.endianness = 'little' if endianness == 'l' else 'big'

        print('# Header --- endiannes: %s  nbits: %d  sampling_frequency: %d' % (self.endianness, self.nbits, self.sampling_freq))
        return self.f

    def __next__(self):
        toread = self.chunck_size * 4
        blocks = []
        while toread > 0 and self.f:
            b =  self.f.read(toread)
            self.nread += len(b)
            if len(b) > 0:
                blocks.append(np.frombuffer(b, dtype=np.uint32))
                toread -= len(b)
            else: #end of file
                self._nextfile()

        if len(blocks) == 0:
            raise StopIteration
                
        chunck =  np.concatenate(blocks)
        for i in range(len(blocks)):
            blocks[i] = None
        self.last_chunk_size = len(chunck)
        
        return chunck
       
    def _nextfile(self):
        self.ifile += 1

        if self.ifile >= len(self.files):
            self.f = None
        else:
            self.f = DataReader._getfile(self.files[self.ifile])
            self._read_header()
            
        return self.f
        
    def __iter__(self):
        self._init_counters()
        self._nextfile()
        return self

    def progress(self):
        return self.nread / self.tot_bytes

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

@jit(nopython=True)
def peak_finder_window(a, width, thr = 0.):
    peak_vals = []
    peak_pos = []
    stream = [0.]
    i = 1
    while i < (len(a) - 1):
        if a[i-1] < a[i] and a[i] > a[i+1] and (a[i] > thr or thr is None):
            j = i
            #peak found, open a window to look for a higher peak
            k = 0
            while (i + k) < (len(a) - 1) and k < width:
                if a[i+k] > a[j]:
                    j = i + k
                    
                k += 1
            #end while
            
            peak_pos.append(j)
            peak_vals.append(a[j])
            stream.extend([0] * (j-i) + [a[j]] + [0] * (i + width - j))
            i += width
        else:
            stream.append(0)
        i += 1

    stream.extend([0] * (len(a) - len(stream)))
        
    return np.array(stream)
                

def volt(raw):
    scale = 10.24/(1<<31)
    return raw*scale-10.24

h = Hist1D(10000, 0., 1)
sig_filter = make_filter(50, 10, 1)


def loop():

#    reader = DataReader(['/home/pgras/work/cupid/data/Th232/000059_20200327T131240_001_000.bin',
#                       '/home/pgras/work/cupid/data/Th232/000059_20200327T131240_002_000.bin'],
#                      2000*3600)

    
#    reader = DataReader('/home/pgras/work/cupid/data/Th232/000059_20200327T131240_001_000.bin',
#                      2000*3600)

    reader = DataReader('/home/pgras/work/cupid/data/Th232/000059_20200327T131240_001_000.bin',
                      2000*3600)

#    reader = DataReader('/home/pgras/work/cupid/data/Th232/000059_20200327T131240_001_000.bin',
#                        2000*10)

    ns = 0
    for i, data in enumerate(reader):
        #signal = peak_finder(np.convolve(sig_filter, volt(data), mode='valid'), 0.001)
        signal = butter_bandpass_filter(volt(data)-volt(data[0]), 3, 300, 2000, 5)
        peaks = peak_finder_window(signal, 20, 0.002)
        h.fill(peaks[np.where(peaks > 0)])
        print("%.1f%%" % (reader.progress() * 100.))
        ns += len(signal)
        #if i > 10:

    print(ns)
    print(ns/2000/3600.)
    return h, signal, peaks, data
