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
    print('# Header --- endiannes: %s  nbits: %d  sampling_frequency: %f' % (endianness, nbits, cfg.params.sampling_freq))
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
def remove_signals(stream, threshold, window):
    '''Remove signals by removing <window> from <stream> if one sample is greater than <threshold>'''
    res = []
    i = 0
    sL =  len(stream) - window
    while i < sL:
        m = np.min(stream[i:i + window])
        M = np.max(stream[i:i + window])
        if m < threshold:
            res.extend(stream[i:i + window])
        i = i + window
    return res



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



from bisect import bisect_left

def find_correlation_window_orig(det_a, peaks_a, det_b, peaks_b):
    of = open('correlations_%03d_%03d.dat' % (det_a, det_b), 'a+')
    for a in peaks_a:
        i = bisect.bisect_left(peaks_b, a)
        #print('--> corr', det_a, det_b, a, peaks_b[i-1] - a, peaks_b[i] - a, peaks_b[i+1] - a)
        of.write('%d %d %d %d\n' % (det_a, det_b, a, peaks_b[i] - a))
    of.write('\n\n')
    of.close()


def closest_position(value_list, x):
    """Return the position of the value in `value_list' closest to `x'. Assumes value_list is sorted."""
    i = bisect_left(value_list, x)
    if i == 0:
        return i
    if i == len(value_list):
        return i - 1
    if abs(value_list[i] - x) < abs(x - value_list[i - 1]):
        return i
    else:
        return i - 1

import time
def find_correlation_window(peaks_a, peaks_b):
    deltat = []
    cnt = 0
    if len(peaks_b) == 0:
        return deltat
    for a in peaks_a:
        i = closest_position(peaks_b, a)
        ##if abs(a - peaks_b[i]) > 1e4:
        ##    print('***>', peaks_a[cnt - 10:cnt + 10])
        ##    print('***>', peaks_b[i - 10:i + 10])
        ##    print(a, peaks_b[i], '(%d %d)' % (cnt, i), a - peaks_b[i])
        ##    time.sleep(0.15)
        cnt += 1
        dt = a - peaks_b[i]
        deltat.append(dt)
        if dt < 0: # peaks_b[i] arrives after a
            if i > 0:
                deltat.append(a - peaks_b[i - 1]) # previous peak
        else:      # peaks_b[i] arrives before a
            if i < len(peaks_b) - 1:
                deltat.append(a - peaks_b[i + 1]) # next peak
    return deltat


def compute_correlations(peaks_a, peaks_max_a, peaks_b, peaks_max_b, window=400):
    """For each peak in `peaks_a', find the closest peak in `peaks_b' and store both peaks if their distance is less than `window' samples. Assumes the `peaks_a,b' are sorted. Return the list of peak positions and peak amplitudes."""
    a, am, b, bm = [], [], [], []
    if len(peaks_a) == 0 or len(peaks_b) == 0:
        return a, am, b, bm
    pa, pam, pb, pbm = peaks_a, peaks_max_a, peaks_b, peaks_max_b
    l_pa = len(pa)
    for i in range(l_pa):
        j = closest_position(pb, pa[i])
        if abs(pa[i] - pb[j]) < window:
            a.append(pa[i])
            am.append(pam[i])
            b.append(pb[j])
            bm.append(pbm[j])

    return a, am, b, bm



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
    gain = []
    noise_threshold = []
    for i in range(len(data)):
        lfreq.append(cfg.cfg.getfloat('analysis', 'filter_lfreq_ch%03d' % (i+1), fallback=lfreq_default))
        hfreq.append(cfg.cfg.getfloat('analysis', 'filter_hfreq_ch%03d' % (i+1), fallback=hfreq_default))
        thr.append(cfg.cfg.getfloat('analysis', 'thr_ch%003d' % (i+1), fallback=thr_default))
        win.append(cfg.cfg.getfloat('analysis', 'peak_search_window_ch%003d' % (i+1), fallback=win_default))
        gain.append(cfg.cfg.getfloat('setup', 'gain_ch%03d' % (i+1), fallback=1000))
        noise_threshold.append(cfg.cfg.getfloat('analysis', 'noise_threshold_ch%03d' % (i+1), fallback=2e-6))

    max_samples = cfg.cfg.getint('data', 'max_samples_per_file', fallback=-1)
    n_max_chunk = cfg.cfg.getint('data', 'n_max_chunk', fallback=-1)
        
    ## analyze independent channels
    for i, f in enumerate(data):

        print('Processing file', f, '(%d)' % i)
        # to avoid a too big file loaded in RAM, split the reading in parts
        # and accumulate the results in acc
        #d = read_data(f, max_samples)
        h = DataReader(f, max_samples, n_max_chunk)
        n_samples_read = 0
        for d in h:
            cfg.params.sampling_freq = h.sampling_freq
            duration = len(d) / 3.6e3 / cfg.params.sampling_freq
            # skipping runs of less than 28.8 seconds
            if duration < 0.008:
                print("# skipping file/chunk %d (%d samples - %f hours)" % (i, len(d), duration))
                continue
            ###print("# processing file %d (%d samples - %f hours)" % (i, len(d), duration))
            print("Progress: %.1f%%" % (h.progress()*100.))
            d = volt(d) / gain[i]
            suff = '_det%03d' % i
            det = i + 1

            acc.set_sampling_freq(det, h.sampling_freq)

            #compute_pulse_weights(d, 200)

            #for j, s in enumerate(d):
            #    #if j > 500000:
            #    #    break
            #    print(i, j, s)
            #print('\n')
            #import sys
            #sys.exit(0)

            # amplitude spectrum
            if butterworth:
                #TODO select freq depending on channel type
                peaks, peaks_max = filt_ana.find_peaks_2(d, [lfreq[i], hfreq[i]], cfg.params.sampling_freq, win[i], thr[i])
            else:
                peaks, peaks_max = find_peaks(d * 1., fir)

            peaks = list(np.add(peaks, n_samples_read))
            #print(peaks[:10], '...', peaks[-10:])
            acc.add(det, 'peak', (peaks, peaks_max))

            # store peak positions and amplitudes for 
            # correlation analysis
            #corr_peaks[1] = (peaks, peaks_max)

            # baseline vs time
            base, base_min = baseline(d * 1., 10000)
            base = list(np.add(base, n_samples_read))
            acc.add(det, 'baseline', (base, base_min))

            ## normalized pulse shape
            #shapes = pulse_shapes(d * 1., peaks, 1000)
            #plot_pulse_shapes(shapes, suff, det)

            ## power spectral density -- all
            #f, Pxx_den = signal.welch(d, cfg.params.sampling_freq, nperseg = 25000)
            # power spectral density -- noise only
            dn = remove_signals(d, noise_threshold[i], 10000)
            #print(dn[:10], '...', dn[-10:], len(dn), d[:10], '...', d[-10:], len(d))
            f, Pxx_den = signal.welch(dn, cfg.params.sampling_freq, nperseg = min(25000, int(len(dn) / 2)))
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

            acc.add_analyzed_samples(det, h.last_chunk_size)
            n_samples_read += h.last_chunk_size
            #print('-->', h.last_chunk_size, n_samples_read)

    #import sys
    #sys.exit(0)
