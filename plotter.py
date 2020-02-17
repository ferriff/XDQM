#!/usr/bin/env python3

import configparser
import numpy as np
import numba as nb
import pickle
import struct

import PyGnuplot as gp


def parse_config(filename):
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read(filename)
    return config


#@nb.jit
def read_data(filename):
    return np.fromfile(filename, dtype=np.dtype(np.int16))



def read_data_new_daq(filename):
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
    sampling_freq = struct.unpack('f', s)[0]
    # data stream of 32 bits: data << 8 + flags
    print('# Header --- endiannes: %s  nbits: %d  sampling_frequency: %f' % (endianness, nbits, sampling_freq))
    #return np.fromfile(filename, dtype=np.dtype('i3, i1'))
    return np.fromfile(f, dtype=np.dtype(np.uint32))



def read_pulse_weights(filename):
    return pickle.load(open(filename, 'rb'))



#def running_mean(x, N):
#    return np.convolve(x, np.ones((N,))/N, mode='valid')
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)



@nb.jit(cache=True, nopython=True)
def baseline(stream, window):
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



#@nb.jit(cache=True, nopython=True)
def pulse_shapes(stream, peaks, window):
    s = []
    lp = -len(peaks)
    for p in peaks[max(-100, lp):]:
        d = stream[p - int(window * 0.25):p + int(window * 0.75)]
        d = d - np.min(d)
        #d = d / np.max(d)
        s.append([d, np.max(d)])
    return s



@nb.jit(cache=True, nopython=True)
def find_peaks(stream, weights, window=200, threshold=20., rise=10):
    peaks, peaks_max = [], []
    wM = np.argmax(weights)
    wL = len(weights)
    i = wL
    sL =  len(stream) - window
    while i < sL:
        if stream[i + rise] - stream[i] > threshold:
            m = np.argmax(stream[i:i + window])
            ampl = np.dot(stream[i + m - wM:i + m - wM + wL], weights)
            print('# max found:', i, m, ampl)
            peaks.append(i + m)
            peaks_max.append(ampl)
            i = i + window
        else:
            i = i + 1
    return peaks, peaks_max



def compute_rate(peaks, peaks_max, window=1000):
    r = []
    tmp_peaks = []
    #p = np.array(peaks)
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



def gp_set_defaults():
    gp.c('reset')
    gp.c('set terminal svg size 600,480 font "Helvetica,16" background "#ffffff"')
    #gp.c('set terminal canvas size 600,480 font "Helvetica,16" enhanced mousing standalone background "#ffffff"')
    #gp.c('set terminal pdfcairo enhanced color solid font "Helvetica,16" size 5,4')
    #gp.c('set term pngcairo enhanced font "Helvetica,12"')
    gp.c('set encoding utf8')
    gp.c('set minussign')
    gp.c('set mxtics 5')
    gp.c('set mytics 5')
    gp.c('set grid lc "#bbbbbb"')
    gp.c('set key samplen 1.5')
    gp.c('set tmargin 1.5')
    gp.c('set label "{/=18:Bold CUPID }{/=16:Italic Data Quality Monitoring}{/:Normal \\ }" at graph 0, graph 1.04')
    gp.c('set label "{/=18 Canfranc Run}" at graph 1, graph 1.04 right')
    gp.c('set label 101 "Last updated on ".system("date -u \\"+%a %e %b %Y %R:%S %Z\\"") at screen .975, graph 1 rotate by 90 right font ",12" tc "#999999"')
    gp.c('hour(x) = x / 1000. / 3600.')
    gp.c('odir = "test/"')



def plot_amplitude(maxima, suffix):
    values, edges = np.histogram(maxima, bins=1500, range=(0, 15000), density=False)
    gp_set_defaults()
    gp.c('set out odir."amplitude%s"' % suffix)
    gp.c('set log y')
    gp.c('set ylabel "Events / 10 ADC count"')
    gp.c('set xlabel "Amplitude (ADC count)"')
    gp.s([edges[:-1], values], filename='tmp_amplitude.dat')
    gp.c('plot [][0.1:] "tmp_amplitude.dat" u 1:2 not w histep lt 6')
    gp.c('set out')



def plot_peaks(peaks, peaks_max, suffix):
    gp_set_defaults()
    gp.c('set out odir."peaks%s"' % suffix)
    gp.s([peaks, peaks_max], filename='tmp_peaks.dat')
    gp.c('set auto fix')
    gp.c('set offsets graph 0.1, graph 0.1, graph 0.1, graph 0.1')
    gp.c('set ylabel "Amplitude (ADC count)"')
    gp.c('set xlabel "Time (h)"')
    #gp.c('plot [][-4000:0] "tmp.dat" u (hour($1)):($2) not w l lt 6')
    gp.c('plot [][] "tmp_peaks.dat" u (hour($1)):($2) not w l lt 6')
    gp.c('set out')



def plot_baseline(base, base_min, suffix):
    gp_set_defaults()
    gp.c('set out odir."baseline%s"' % suffix)
    gp.s([base, base_min], filename='tmp_baseline.dat')
    gp.c('set auto fix')
    gp.c('set offsets graph 0.1, graph 0.1, graph 0.1, graph 0.1')
    gp.c('set ylabel "Amplitude (ADC count)"')
    gp.c('set xlabel "Time (h)"')
    #gp.c('plot [][-4000:0] "tmp.dat" u (hour($1)):($2) not w l lt 6')
    gp.c('plot [][] "tmp_baseline.dat" u (hour($1)):($2) not w l lt 6')
    gp.c('set out')



def plot_pulse_shapes(shapes, suffix):

    of = open('tmp_shapes.dat', 'w')
    for cnt, el in enumerate(shapes):
        for i, v in enumerate(el[0]):
            of.write("%d %f %d %f\n" % (i, v, cnt, el[1]))
        of.write("\n\n")

    gp_set_defaults()
    gp.c('set auto fix')
    gp.c('set offsets graph 0.1, graph 0.1, graph 0.1, graph 0.1')
    gp.c('load "blues.pal"')
    gp.c('set ylabel "Amplitude (ADC count)"')
    gp.c('set xlabel "Time (ms)"')
    gp.c('set out odir."/normalized_shapes%s"' % suffix)
    gp.c('plot [][] "tmp.dat" u 1:($2 / $4):3 not w l lt palette')
    gp.c('set out')
    gp.c('set out odir."/shapes%s"' % suffix)
    #gp.c('set log y')
    gp.c('unset colorbox')
    gp.c('plot [][1:] "tmp_shapes.dat" u 1:2:3 not w l lt palette')
    gp.c('set out')



def plot_rate(rate, window, suffix):
    fname = 'tmp_rate.dat'
    gp_set_defaults()
    gp.c('set out odir."rate%s"' % suffix)
    gp.s([rate], filename=fname)
    gp.c('set auto fix')
    gp.c('set offsets graph 0.1, graph 0.1, graph 0.1, graph 0.1')
    gp.c('set ylabel "Rate (Hz)"')
    gp.c('set xlabel "Time (h)"')
    #gp.c('plot [][-4000:0] "tmp.dat" u (hour($1)):($2) not w l lt 6')
    gp.c('plot [][] "'+fname+'" u ($0 / 3600.):($1/'+str(window)+') not w l lt 6')
    gp.c('set out')



def plot_fft_rate(freq, power, suffix):
    fname = 'tmp_fft_rate.dat'
    gp_set_defaults()
    gp.c('set out odir."fft_rate%s"' % suffix)
    gp.s([freq, power], filename=fname)
    gp.c('set auto fix')
    gp.c('set offsets graph 0.1, graph 0.1, graph 0.1, graph 0.1')
    gp.c('set log y')
    gp.c('set ylabel ""')
    gp.c('set xlabel "Frequency (1/min)"')
    #gp.c('plot [][-4000:0] "tmp.dat" u (hour($1)):($2) not w l lt 6')
    gp.c('plot [][] "'+fname+'" u ($1 * 60):2 not w l lt 6')
    gp.c('set out')



def plot_fft_data(freq, power, suffix):
    fname = 'tmp_fft_data.dat'
    gp_set_defaults()
    gp.c('set out odir."fft_data%s"' % suffix)
    gp.s([freq, power], filename=fname)
    gp.c('set auto fix')
    gp.c('set offsets graph 0.1, graph 0.1, graph 0.1, graph 0.1')
    gp.c('set log y')
    gp.c('set ylabel ""')
    gp.c('set xlabel "Frequency (Hz)"')
    #gp.c('plot [][-4000:0] "tmp.dat" u (hour($1)):($2) not w l lt 6')
    gp.c('plot [][] "'+fname+'" u ($1 / 3600.):2 not w l lt 6')
    gp.c('set out')



cfg = parse_config('configuration.cfg')

data = []
#data.append(read_data("/home/ferri/data/double_beta/canfranc/20190615_21h27.BIN2"))
#data.append(read_data("/home/ferri/data/double_beta/canfranc/20190615_21h27.BIN3"))
data.append(read_data_new_daq("/home/ferri/tmp/downloads/000001_20191222T054757_001_001.bin"))

#data.append(read_data_new_daq('/home/ferri/data/double_beta/canfranc/new_daq/CH1_20191211T115653_0.bin'))
#for el, flags in zip(np.array((np.array(data[0]) >> 8) - 0x800000, dtype=np.int32), np.array(data[0] & 0x000000ff)):
#    print(el, flags)
#import sys
#sys.exit(0)

#for i in range(1, 7):
#    print(i)
#    data.append(read_data("/home/ferri/data/double_beta/canfranc/run2/20191205_17h21_.BIN%d" % i))
##import sys
##sys.exit(0)

fir = read_pulse_weights('pulse_weights.pkl')

## do plots for independent channels

for i, d in enumerate(data):

    print("# processing file %d (%d samples - %f hours)" % (i, len(d), len(d) / 3.6e6))
    suff = '_det%03d.svg' % i

    print('#', d)
    d = (d>>8) - 8388608
    print('#', d)
    for j, s in enumerate(d):
        if j > 100000:
            break
        print(j, s)

    # amplitude spectrum
    peaks, peaks_max = find_peaks(d * 1., fir)
    plot_amplitude(peaks_max, suff)
    #break

    # peaks vs time
    plot_peaks(peaks, peaks_max, suff)

    ### baseline vs time
    base, base_min = baseline(d * 1., 10000)
    plot_baseline(base, base_min, suff)

    ### normalized pulse shape
    ##shapes = pulse_shapes(d * 1., peaks, 1000)
    ##plot_pulse_shapes(shapes, suff)

    # power spectra
    # all samples
    p = np.abs(np.fft.rfft(d[len(d) - 100000:len(d)]))
    f = np.linspace(0, 1000/2, len(p))
    plot_fft_data(f, p, suff)

    # rate
    rate = compute_rate(peaks, peaks_max, 100 * 1e3)
    plot_rate(rate, 100, suff)

    ## rate FFT
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

# plots:
# - baseline
# - cumulative pulse shape (best: last N pulses, fading away the old ones)
# - amplitude spectrum
# - light vs. heat channels
# - live streaming

# - average amplitude above a given threshold vs. time
# - trigger rate vs. time
# - baseline rms vs. time

