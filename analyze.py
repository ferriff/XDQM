import numpy as np
import pickle
import struct

from numba import jit
from scipy import signal

# DQM modules
import configure as cfg
import filt_ana 
import accumulator

from utils import volt, DataReader


@jit(nopython=True)
def read_data_old_daq(filename):
    return np.fromfile(filename, dtype=np.dtype(np.int16))


def read_data(filename, nsamples = -1):
    '''Read a binary file from the Cross DAQ'''
    f = open(filename, 'rb')
    # first 32 bits: endianness << 8 + NBits
    #   where Nbits tells if data are written with 32 or 25 bits
    s1 = f.read(1)
    s3 = f.read(3)
    endianness = None
    if 'l' in s3.decode():
        #print('# -->', s3)
        endianness = 'little'
    else:
        endianness = 'big'
    nbits = int.from_bytes(s1, byteorder=endianness)
    # second 32 bits: sampling frequency encoded as float
    s = f.read(4)
    cfg.params.sampling_freq = struct.unpack('f', s)[0]
    # data stream of 32 bits: data << 8 + flags
    print('# Header --- endiannes: %s  nbits: %d  sampling_frequency: %f' % (endianness, nbits, cfg.params.sampling_freq))
    #return np.fromfile(filename, dtype=np.dtype('i3, i1'))
    return np.fromfile(f, dtype=np.dtype(np.uint32), count = nsamples)



def file_info(filename, dump = False, dump_count = 1000):
    '''Read a binary file from the Cross DAQ.'''
    f = open(filename, 'rb')
    # first 32 bits: endianness << 8 + NBits
    #   where Nbits tells if data are written with 32 or 25 bits
    s1 = f.read(1)
    s3 = f.read(3)
    endianness = None
    if 'l' in s3.decode():
        #print('# -->', s3)
        endianness = 'little'
    else:
        endianness = 'big'
    nbits = int.from_bytes(s1, byteorder=endianness)
    # second 32 bits: sampling frequency encoded as float
    s = f.read(4)
    cfg.params.sampling_freq = struct.unpack('f', s)[0]
    # data stream of 32 bits: data << 8 + flags
    print('# Header --- endiannes: %s  nbits: %d  sampling_frequency: %f' % (endianness, nbits, sampling_freq))
    # dump the first dump_count samples of the file
    if dump:
        d = np.fromfile(f, dtype=np.dtype(np.uint32), count = dump_count)
        d = volt(d)
        for j, s in enumerate(d):
            print(j, s)



def load_amplitude_reco_weights(filename):
    '''Load a weight set for the amplitude reconstruction from file <filename>.'''
    return pickle.load(open(filename, 'rb'))



import sys
#@jit(cache=True, nopython=True)
def compute_pulse_weights(stream, window, filename='computed_weights.pkl'):
    '''Compute a weight set for the amplitude reconstruction and dump it in pickle
format to <filename>. Thresholds are hardcoded and inspection of the 
pulse selected is highly recommended.'''
    # find a window with one pulse only
    print('entering compute_pulse_weights')
    sL =  len(stream) - window
    threshold = 0.3 # (volt(20.) - volt(0)) * 100.
    rise = 10
    whalf = int(window / 2)
    i = 0
    while i < sL:
        #print('##', sL, threshold, rise, i, stream[i + rise], stream[i + rise] - stream[i])
        if stream[i + rise] - stream[i] > threshold:
            # found one peak in m
            m = np.argmax(stream[i:i + window])
            print('found peak at ', m)
            ##for j, s in enumerate(stream[max(0, i + m - window):i + m + window]):
            ##    print('##', j, s)
            ##print("##\n##\n")
            fir = stream[max(0, i + m - whalf):i + m + whalf]
            for j, s in enumerate(fir):
                print('##', j, s)
            print("##\n##\n")
            fir = np.concatenate(([np.average(fir[m - 20:m - 17])] * 5, fir[whalf - 2:whalf + 4])) # only pedestal and maximum
            fir = fir - np.min(fir)
            fir = fir / np.max(fir)
            fir = fir - np.average(fir)
            fir = fir / np.sum(fir**2)
            fir = np.concatenate((fir[0:5], 20 * [0], fir[5:]))
            for j, s in enumerate(fir):
                print('##', j, s)
            break
        i = i + int(window / 4.)
    print('exiting compute_pulse_weights')
    # save the pulse
    pickle.dump(fir, open(filename, 'wb'))
    sys.exit(1)



def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)



@jit(cache=True, nopython=True)
def baseline(stream, window):
    '''Compute the baseline as minimum sample value in a given window'''
    baseline, baseline_min = [], []
    i = 0
    sL =  len(stream) - window
    while i < sL:
        m = np.argmin(stream[i:i + window])
        v = stream[i + m]
        baseline.append(i + m)
        baseline_min.append(v)
        i = i + window
    return baseline, baseline_min



@jit(cache=True, nopython=True)
def pulse_shapes(stream, peaks, window):
    '''For every point in <peaks>, collect the pulse shape of <stream> in the <window>
point - 0.25 * window size, point + 0.75 * window_size'''
    s = []
    lp = -len(peaks)
    for p in peaks[max(-100, lp):]:
        d = stream[p - int(window * 0.25):p + int(window * 0.75)]
        d = d - np.min(d)
        #d = d / np.max(d)
        s.append([d, np.max(d)])
    return s



@jit(cache=True, nopython=True)
def amplitude_reco_weights(pulse, weights):
    '''Compute the amplitude of <pulse> using the given set of <weights>'''
    weights = cfg.ampl_reco_weights
    return np.dot(pulse, weights)



@jit(cache=True, nopython=True)
def find_peaks(stream, weights, window=200, threshold=0.01, rise=10):
    '''Find the peaks in <stream> by triggering on a difference of <threshold> volts between two
samples apart of <rise> samples, in a given <window>'''
    peaks, peaks_max = [], []
    wM = np.argmax(weights)
    wL = len(weights)
    i = wL + 1
    sL =  len(stream) - window
    while i < sL:
        if stream[i + rise] - stream[i] > threshold:
            m = np.argmax(stream[i:i + window])
            # reconstruct amplitude with bandpass filter
            # FIXME: remove weights, which are retrieved via cfg from the amplitude reco function
            # and support different pulse reconstructions, e.g.
            # if reco = 'weights':
            # elif reco = 'maximum':
            # elif reco = 'filter_salca':
            #ampl = amplitude_reco_weights(stream[i + m - wM:i + m - wM + wL], weights)
            ampl = np.dot(stream[i + m - wM:i + m - wM + wL], weights)
            #print('# max found:', i, m, ampl)
            peaks.append(i + m)
            peaks_max.append(ampl)
            i = i + window
        else:
            i = i + 1
    return peaks, peaks_max



def compute_rate(peaks, peaks_max, window=1000):
    '''Compute the number of peaks in a given sliding window'''
    r = []
    if len(peaks) == 0:
        return r
    tmp_peaks = []
    mP =  peaks[-1]
    pL = len(peaks)
    i = 1000
    maxp = 0
    while i < mP:
        # add from right (new elements)
        while maxp < pL:
            if peaks[maxp] < i + window:
                if peaks_max[maxp] > 0:
                    tmp_peaks.append(peaks[maxp])
                maxp = maxp + 1
                #print("adding", peaks[maxp])
            else:
                break
        # remove from left (existing elements)
        tmp_peaks = [x for x in tmp_peaks if x >= i]
        #for j, el in enumerate(tmp_peaks[:]):
        #    if el < i:
        #        #print("removing", el)
        #        print('-->', el, i, j, len(tmp_peaks))
        #        del tmp_peaks[j]
        #    else:
        #        break
        r.append(len(tmp_peaks))
        i = i + 1000
    return r



def analyze(data, acc):
    """Analyze the data in `data' and store quantities in the accumulator `acc'"""

    #fir = load_amplitude_reco_weights('pulse_weights.pkl')
    fir = load_amplitude_reco_weights('computed_weights.pkl')

    print(len(fir), fir)

    signal_processing =  cfg.cfg.get('analysis', 'signal_processing', fallback = '')

    if signal_processing == 'butterworth':
        butterworth = True
        print("Using Butterworth filter for signal processing")
    else:
        butterworth = False

    lfreq_default = cfg.cfg.getfloat('analysis', 'lfreq_default', fallback=3)
    hfreq_default = cfg.cfg.getfloat('analysis', 'hfreq_default', fallback=300)
    thr_default = cfg.cfg.getfloat('analysis', 'threshold_default', fallback=0.01)
    win_default = cfg.cfg.getfloat('analysis', 'peak_search_window', fallback=1.e-3)
    lfreq = []
    thr = []
    hfreq = []
    win = []
    for i in range(len(data)):
        lfreq.append(cfg.cfg.getfloat('analysis', 'lfreq_ch%03d', fallback=lfreq_default))
        hfreq.append(cfg.cfg.getfloat('analysis', 'hfreq_ch%03d', fallback=hfreq_default))
        thr.append(cfg.cfg.getfloat('analysis', 'thr_ch%003d', fallback=thr_default))
        win.append(cfg.cfg.getfloat('analysis', 'peak_search_window_ch%003d', fallback=win_default))

    max_samples = cfg.cfg.getint('data', 'max_samples_per_file', fallback=-1)
        
    ## analyze independent channels
    for i, f in enumerate(data):

        print('Processing file', f, '(%d)' % i)
        # to avoid a too big file loaded in RAM, split the reading in parts
        # and accumulate the results in acc
        #d = read_data(f, max_samples)
        h = DataReader(f, max_samples)
        for d in h:
            cfg.params.sampling_freq = h.sampling_freq
            duration = len(d) / 3.6e3 / cfg.params.sampling_freq
            # skipping runs of less than 28.8 seconds
            if duration < 0.008:
                print("# skipping file/chunk %d (%d samples - %f hours)" % (i, len(d), duration))
                continue
            ###print("# processing file %d (%d samples - %f hours)" % (i, len(d), duration))
            print("Progress: %.1f%%" % (h.progress()*100.))
            d = volt(d)
            suff = '_det%03d' % i
            det = i + 1

            #compute_pulse_weights(d, 200)

            #for j, s in enumerate(d):
            #    if j > 100000:
            #        break
            #    print('#', j, s)

            # amplitude spectrum
            if butterworth:
                #TODO select freq depending on channel type
                peaks, peaks_max = filt_ana.find_peaks_2(d, [lfreq[i], hfreq[i]], cfg.params.sampling_freq, win[i], thr[i])
            else:
                peaks, peaks_max = find_peaks(d * 1., fir)
                
            acc.add(det, 'peak', (peaks, peaks_max))

            # store peak positions and amplitudes for 
            # correlation analysis
            #corr_peaks[1] = (peaks, peaks_max)

            # baseline vs time
            base, base_min = baseline(d * 1., 10000)
            acc.add(det, 'baseline', (base, base_min))

            ## normalized pulse shape
            #shapes = pulse_shapes(d * 1., peaks, 1000)
            #plot_pulse_shapes(shapes, suff, det)

            # power spectral density
            f, Pxx_den = signal.welch(d, cfg.params.sampling_freq, nperseg = 25000)
            acc.add(det, 'fft', (f, Pxx_den))

            ## rate FFT # FIXME: takes quite a long time
            #p = [0] * (peaks[len(peaks) - 1] + 1)
            #for el in peaks:
            #    p[el] = 1.
            ##p = np.abs(np.fft.rfft(p[:10000]))
            ##p = np.abs(np.fft.rfft(rate))
            ##f = np.linspace(0, 1/2, len(p))
            ##plot_fft_rate(f, p, suff)
            #from scipy import signal
            #f, Pxx_den = signal.periodogram(p[:10000], 1)
            #plot_fft_rate(f, Pxx_den, suff)
