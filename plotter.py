#!/usr/bin/env python3.7

import configparser
import glob
#import numba as nb # FIXME: not available in the current devuan version
import numpy as np
import os
import sys
import pickle
import struct
import argparse
import PyGnuplot as gp
import filt_ana 

from datetime import datetime

from utils import volt

class Parameters:
    def __init__(self):
        self.sampling_freq = 0

def parse_config(filename):
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read(filename)
    return config


#@nb.jit
def read_data(filename):
    return np.fromfile(filename, dtype=np.dtype(np.int16))

def parse_args():
    '''Parses comand line arguments.'''
    parser = argparse.ArgumentParser(description='CROSS DQM: analysis and plotting. Configuration is read from the file configuration.cfg')
    parser.add_argument('--files', 
                        help='Limit the analysis to a list of files. File paths must be provided in the FILES argument as comma separated list.')
    
    return parser.parse_args()

def read_data_new_daq(filename, nsamples = -1):
    global params
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
    params.sampling_freq = struct.unpack('f', s)[0]
    # data stream of 32 bits: data << 8 + flags
    print('# Header --- endiannes: %s  nbits: %d  sampling_frequency: %f' % (endianness, nbits, params.sampling_freq))
    #return np.fromfile(filename, dtype=np.dtype('i3, i1'))
    return np.fromfile(f, dtype=np.dtype(np.uint32), count = nsamples)



def file_info(filename, dump = False):
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
    if dump:
        d = np.fromfile(f, dtype=np.dtype(np.uint32), count = 1000)
        d = (d>>8)
        for j, s in enumerate(d):
            print(j, s)



def read_pulse_weights(filename):
    return pickle.load(open(filename, 'rb'))

def compute_pulse_weights(stream, peaks, window, filename='computed_weights.pkl'):
    # find a window with one pulse only
    
    # save the pulse
    pickle.save(open(filename, 'w'))



#def running_mean(x, N):
#    return np.convolve(x, np.ones((N,))/N, mode='valid')
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)



#FIXME: reactivate when numba is found @nb.jit(cache=True, nopython=True)
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



#FIXME: reactivate when numba is found @nb.jit(cache=True, nopython=True)
def find_peaks(stream, weights, window=200, threshold=20., rise=10):
    peaks, peaks_max = [], []
    wM = np.argmax(weights)
    wL = len(weights)
    i = wL + 1
    sL =  len(stream) - window
    while i < sL:
        if stream[i + rise] - stream[i] > threshold:
            m = np.argmax(stream[i:i + window])
            ampl = np.dot(stream[i + m - wM:i + m - wM + wL], weights)
            #print('# max found:', i, m, ampl)
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



def analyze(data):

    global params

    fir = read_pulse_weights('pulse_weights.pkl')


    signal_processing =  cfg.get('analysis', 'signal_processing', fallback = '')

    if signal_processing == 'butterworth':
        butterworth = True
        print("Using Butterworth filter for signal processing")
    else:
        butterworth = False
        
    lfreq_default = cfg.getfloat('analysis', 'lfreq_default', fallback=3)
    hfreq_default = cfg.getfloat('analysis', 'hfreq_default', fallback=300)
    thr_default = cfg.getfloat('analysis', 'threshold_default', fallback=None)
    win_default = cfg.getfloat('analysis', 'peak_search_window', fallback=1.e-3)
    lfreq = []
    thr = []
    hfreq = []
    win = []
    for i in range(len(data)):
        lfreq.append(cfg.getfloat('analysis', 'lfreq_ch%03d', fallback=lfreq_default))
        hfreq.append(cfg.getfloat('analysis', 'hfreq_ch%03d', fallback=hfreq_default))
        thr.append(cfg.getfloat('analysis', 'thr_ch%003d', fallback=thr_default))
        win.append(cfg.getfloat('analysis', 'peak_search_window_ch%003d', fallback=win_default))

    max_samples = cfg.getint('data', 'max_samples_per_file', fallback=-1)
        
    ## do plots for independent channels
    for i, f in enumerate(data):

        d = read_data_new_daq(f, max_samples)
        duration = len(d) / 3.6e3 / params.sampling_freq
        # skipping runs of less than 28.8 seconds
        if duration < 0.008:
            print("# skipping file %d (%d samples - %f hours)" % (i, len(d), duration))
            continue
        print("# processing file %d (%d samples - %f hours)" % (i, len(d), duration))
        d = volt(d)
        suff = '_det%03d.svg' % i
        det = i + 1

        #for j, s in enumerate(d):
        #    if j > 100000:
        #        break
        #    print('#', j, s)

        # amplitude spectrum
        if butterworth:
            #TODO select freq depending on channel type
            peaks, peaks_max = filt_ana.find_peaks_2(d, [lfreq[i], hfreq[i]], params.sampling_freq, win[i], thr[i])
        else:
            peaks, peaks_max = find_peaks(d * 1., fir)
            
        plot_amplitude(peaks_max, suff, det)

        # peaks vs time
        plot_peaks(peaks, peaks_max, suff, det)

        # baseline vs time
        base, base_min = baseline(d * 1., 10000)
        plot_baseline(base, base_min, suff, det)

        # normalized pulse shape
        shapes = pulse_shapes(d * 1., peaks, 1000)
        plot_pulse_shapes(shapes, suff, det)

        # power spectra
        # all samples
        p = np.abs(np.fft.rfft(d[len(d) - 100000:len(d)]))
        f = np.linspace(0, 1000/2, len(p))
        plot_fft_data(f, p, suff, det)

        # rate
        rate = compute_rate(peaks, peaks_max, 100 * 1e3)
        plot_rate(rate, 100, suff, det)

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

    # plots:
    # - baseline
    # - cumulative pulse shape (best: last N pulses, fading away the old ones)
    # - amplitude spectrum
    # - light vs. heat channels
    # - live streaming

    # - average amplitude above a given threshold vs. time
    # - trigger rate vs. time
    # - baseline rms vs. time


def detector_name(det):
    global cfg
    return cfg['setup']['ch%03d' % det].replace(' ', '_') # FIXME: better sanitize dir names


def gp_set_defaults():
    global global_odir
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
    gp.c('odir = "' + global_odir + '/"')



def plot_amplitude(maxima, suffix, det):
    fname = 'tmp_amplitude.dat'
    global global_odir
    suff_noext, dummy = os.path.splitext(suffix)
    det_name = detector_name(det)
    xmin = cfg.getfloat("plot", "min_ampl", fallback=0.)
    xmax = cfg.getfloat("plot", "max_ampl", fallback=1.)
    os.makedirs(os.path.join(global_odir, det_name, 'amplitude%s' % suff_noext), exist_ok=True)
    values, edges = np.histogram(maxima, bins=1500, range=(xmin, xmax), density=False)
    gp_set_defaults()
    gp.c('set out odir."%s/amplitude%s/amplitude%s"' % (det_name, suff_noext, suffix))
    gp.c('set log y')
    gp.c('set ylabel "Events / V"')
    gp.c('set xlabel "Amplitude (V)"')
    gp.s([edges[:-1], values], filename=fname)
    gp.c('plot [][0.1:] "'+fname+'" u 1:2 not w histep lt 6')
    gp.c('set out')



def plot_peaks(peaks, peaks_max, suffix, det):
    fname = 'tmp_peaks.dat'
    global global_odir
    suff_noext, dummy = os.path.splitext(suffix)
    det_name = detector_name(det)
    os.makedirs(os.path.join(global_odir, det_name, 'peaks%s' % suff_noext), exist_ok=True)
    gp_set_defaults()
    gp.c('set out odir."%s/peaks%s/peaks%s"' % (det_name, suff_noext, suffix))
    gp.s([peaks, peaks_max], filename=fname)
    gp.c('set auto fix')
    gp.c('set offsets graph 0.1, graph 0.1, graph 0.1, graph 0.1')
    gp.c('set ylabel "Amplitude (V)"')
    gp.c('set xlabel "Time (h)"')
    #gp.c('plot [][-4000:0] "tmp.dat" u (hour($1)):($2) not w l lt 6')
    pmin = np.percentile(peaks_max, 0.025)
    pmax = np.percentile(peaks_max, 0.975)
    gp.c('set y2range [%f:%f]"' % (pmin, pmax))
    gp.c('plot [][] "'+fname+'"'+" u (hour($1)):($2) not w l lc '#555555', '' u (hour($1)):($2) axis x1y2 not w l lt 6")
    gp.c('set out')



def plot_baseline(base, base_min, suffix, det):
    fname = 'tmp_baseline.dat'
    global global_odir
    suff_noext, dummy = os.path.splitext(suffix)
    det_name = detector_name(det)
    os.makedirs(os.path.join(global_odir, det_name, 'baseline%s' % suff_noext), exist_ok=True)
    gp_set_defaults()
    gp.c('set out odir."%s/baseline%s/baseline%s"' % (det_name, suff_noext, suffix))
    gp.s([base, base_min], filename=fname)
    gp.c('set auto fix')
    gp.c('set offsets graph 0.1, graph 0.1, graph 0.1, graph 0.1')
    gp.c('set ylabel "Amplitude (V)"')
    gp.c('set xlabel "Time (h)"')
    #gp.c('plot [][-4000:0] "tmp.dat" u (hour($1)):($2) not w l lt 6')
    pmin = np.percentile(base_min, 0.025)
    pmax = np.percentile(base_min, 0.975)
    gp.c('set y2range [%f:%f]"' % (pmin, pmax))
    #gp.c('plot [][] "'+fname+'" u (hour($1)):($2) not w l lt 6')
    gp.c('plot [][] "'+fname+'"'+" u (hour($1)):($2) not w l lc '#555555', '' u (hour($1)):($2) axis x1y2 not w l lt 6")
    gp.c('set out')



def plot_pulse_shapes(shapes, suffix, det):

    fname = 'tmp_shapes.dat'
    global global_odir
    suff_noext, dummy = os.path.splitext(suffix)
    det_name = detector_name(det)
    os.makedirs(os.path.join(global_odir, det_name, 'shapes%s' % suff_noext), exist_ok=True)
    os.makedirs(os.path.join(global_odir, det_name, 'normalized_shapes%s' % suff_noext), exist_ok=True)
    of = open(fname, 'w')
    for cnt, el in enumerate(shapes):
        for i, v in enumerate(el[0]):
            of.write("%d %f %d %f\n" % (i, v, cnt, el[1]))
        of.write("\n\n")

    gp_set_defaults()
    gp.c('set auto fix')
    gp.c('set offsets graph 0.1, graph 0.1, graph 0.1, graph 0.1')
    gp.c('load "blues.pal"')
    gp.c('set ylabel "Amplitude (V)"')
    gp.c('set xlabel "Time (ms)"')
    gp.c('set out odir."%s/normalized_shapes%s/normalized_shapes%s"' % (det_name, suff_noext, suffix))
    gp.c('plot [][] "'+fname+'" u 1:($2 / $4):3 not w l lt palette')
    gp.c('set out')
    gp.c('set out odir."%s/shapes%s/shapes%s"' % (det_name, suff_noext, suffix))
    #gp.c('set log y')
    gp.c('unset colorbox')
    gp.c('plot [][1:] "'+fname+'" u 1:2:3 not w l lt palette')
    gp.c('set out')



def plot_rate(rate, window, suffix, det):
    fname = 'tmp_rate.dat'
    global global_odir
    suff_noext, dummy = os.path.splitext(suffix)
    det_name = detector_name(det)
    os.makedirs(os.path.join(global_odir, det_name, 'rate%s' % suff_noext), exist_ok=True)
    gp_set_defaults()
    gp.c('set out odir."%s/rate%s/rate%s"' % (det_name, suff_noext, suffix))
    gp.s([rate], filename=fname)
    gp.c('set auto fix')
    gp.c('set offsets graph 0.1, graph 0.1, graph 0.1, graph 0.1')
    gp.c('set ylabel "Rate (Hz)"')
    gp.c('set xlabel "Time (h)"')
    #gp.c('plot [][-4000:0] "tmp.dat" u (hour($1)):($2) not w l lt 6')
    gp.c('plot [][] "'+fname+'" u ($0 / 3600.):($1/'+str(window)+') not w l lt 6')
    gp.c('set out')



def plot_fft_rate(freq, power, suffix, det):
    fname = 'tmp_fft_rate.dat'
    global global_odir
    suff_noext, dummy = os.path.splitext(suffix)
    det_name = detector_name(det)
    os.makedirs(os.path.join(global_odir, det_name, 'fft_rate%s' % suff_noext), exist_ok=True)
    gp_set_defaults()
    gp.c('set out odir."%s/fft_rate%s/fft_rate%s"' % (det_name, suff_noext, suffix))
    gp.s([freq, power], filename=fname)
    gp.c('set auto fix')
    gp.c('set offsets graph 0.1, graph 0.1, graph 0.1, graph 0.1')
    gp.c('set log y')
    gp.c('set ylabel ""')
    gp.c('set xlabel "Frequency (1/min)"')
    #gp.c('plot [][-4000:0] "tmp.dat" u (hour($1)):($2) not w l lt 6')
    gp.c('plot [][] "'+fname+'" u ($1 * 60):2 not w l lt 6')
    gp.c('set out')



def plot_fft_data(freq, power, suffix, det):
    fname = 'tmp_fft_data.dat'
    global global_odir
    suff_noext, dummy = os.path.splitext(suffix)
    det_name = detector_name(det)
    os.makedirs(os.path.join(global_odir, det_name, 'fft_data%s' % suff_noext), exist_ok=True)
    gp_set_defaults()
    gp.c('set out odir."%s/fft_data%s/fft_data%s"' % (det_name, suff_noext, suffix))
    gp.s([freq, power], filename=fname)
    gp.c('set auto fix')
    gp.c('set offsets graph 0.1, graph 0.1, graph 0.1, graph 0.1')
    gp.c('set log y')
    gp.c('set ylabel ""')
    gp.c('set xlabel "Frequency (Hz)"')
    #gp.c('plot [][-4000:0] "tmp.dat" u (hour($1)):($2) not w l lt 6')
    gp.c('plot [][] "'+fname+'" u ($1 / 3600.):2 not w l lt 6')
    gp.c('set out')


def plot_more_recent_than_data(plotdir, datainfo):
    flist = glob.glob(os.path.join(plotdir, 'amplitude*00/*svg*')) # FIXME: better tool...
    plot_time = -1
    if len(flist) > 0:
        plot_time = os.path.getmtime(flist[0])
    data_time = os.path.getmtime(datainfo[0])
    return plot_time > data_time


cfg = parse_config('configuration.cfg')

params = Parameters()

data = []

data_suff = cfg['data']['suffix']
data_root = cfg['data']['root_dir']
analyzed_runs = cfg['data']['log_analyzed_runs']
plot_out_dir_root = cfg['plot']['output_dir']

def ana_dir(data_root, data_suff = '.bin'):
    '''Analyses all data found in the directory data_root. It looks for data files
with suffix data_suff.'''

    global global_odir
    
    # find files
    path_pattern = data_root + '/**/*' + data_suff
    runs = glob.glob(path_pattern, recursive=True)

    if len(runs) == 0:
        print("No data to process while looking for pattern %s." % path_pattern, file=sys.stderr)


    # find directories
    dirs = set()
    for r in runs:
        dirs.add(os.path.dirname(r))

    # map files into dirs
    df = {}
    for d in dirs:
        df[d] = []
    for r in runs:
        df[os.path.dirname(r)].append(os.path.basename(r))

            
    # find run and chunk number
    run_chunk = {}
    for d in df.keys():
        rc = set()
        for r in df[d]:
            v = r.replace(data_suff, '').split('_')
            rc.add((v[0], v[3]))
        run_chunk[d] = sorted(rc)

    # map the files to a given dir, run, chunk
    drc = {}
    for d in run_chunk.keys():
        for rc in run_chunk[d]:
            drc[tuple([d] + list(rc))] = glob.glob(d + '/' + rc[0] + '_*_' + rc[1] + data_suff)

    # ready to analyze runs
    # load list of analyzed runs
    analyzed = {}

    global_odir = ""
    for k in drc.keys():
        if os.path.isfile(analyzed_runs):
            analyzed = pickle.load(open(analyzed_runs, 'rb'))
        # discard runs with different setup # FIXME: has to be improved        
        if len(drc[k][:12]) != 12:
            continue
        # skip analyzed runs
        global_odir = os.path.join(k[0].replace(os.path.dirname(data_root), plot_out_dir_root), k[1], k[2])
        if k in analyzed.keys() and plot_more_recent_than_data(global_odir, drc[k]):
            continue
        os.makedirs(global_odir, exist_ok=True)
        data = []
        for f in drc[k][:12]:
            print(datetime.utcnow().strftime("# %s %Y-%m-%d %H:%M:%S UTC"), ' detected file', f)
            #data.append(read_data(f))
            data.append(f)
            #print('done')
        ####data.append('/mnt/samba/RUNS/RUN2/Xmas_run/000001_20191222T054757_001_001.bin')
        print("Analyzing: " + ", ".join(data))
        analyze(data)
        analyzed[k] = True
        # dump updated list of analyzed runs
        pickle.dump(analyzed, open(analyzed_runs, 'wb'))
        print(datetime.utcnow().strftime("# %s %Y-%m-%d %H:%M:%S UTC"), ' processing done.')

if __name__ == "__main__":
    global global_odir
    args = parse_args()
    if args.files:
        global_odir = "out"
        print("Plots will be stored in the out directory.")
        analyze(args.files.split(","))
    else:
        ana_dir(data_root)

