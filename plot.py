import configure as cfg

import os
import numpy as np
import PyGnuplot as gp


def detector_name(det):
    return cfg.cfg['setup']['ch%03d' % det].replace(' ', '_') # FIXME: better sanitize dir names


def gp_set_terminal(t):
    if   t == 'svg':
        gp.c('set terminal svg size 600,480 font "Helvetica,16" background "#ffffff" enhanced')
    elif t == 'png':
        gp.c('set term pngcairo enhanced font "Helvetica,12"')
    else:
        print('Terminal `%s\' not supported' % t)


def gp_set_defaults():
    gp.c('reset')
    gp.c('set encoding utf8')
    gp.c('set minussign')
    gp.c('set mxtics 5')
    gp.c('set mytics 5')
    gp.c('set grid')
    gp.c('set key samplen 1.5')
    gp.c('set tmargin 1.5')
    gp.c('set label "{/=16:Bold CUPID/CROSS }{/=14:Italic Data Quality Monitoring}{/:Normal \\ }" at graph 0, graph 1.04')
    gp.c('set label "{/=16 Canfranc}" at graph 1, graph 1.04 right')
    gp.c('set label 101 "Last updated on ".system("date -u \\"+%a %e %b %Y %R:%S %Z\\"") at screen .975, graph 1 rotate by 90 right font ",12" tc "#999999"')
    gp.c('hour(x) = x / 1000. / 3600.')
    print(cfg.global_odir)
    gp.c('odir = "' + cfg.global_odir + '/"')



def plot_amplitude(maxima, suffix, det):
    fname = 'tmp_amplitude.dat'
    det_name = detector_name(det)
    xmin = cfg.cfg.getfloat("plot", "ampl_min", fallback=0.)
    xmax = cfg.cfg.getfloat("plot", "ampl_max", fallback=1.)
    xbin = cfg.cfg.getfloat("plot", "ampl_bin", fallback=1000)
    os.makedirs(os.path.join(cfg.global_odir, det_name, 'amplitude%s' % suffix), exist_ok=True)
    values, edges = np.histogram(maxima, bins=1500, range=(xmin, xmax), density=False)
    gp_set_defaults()
    gp.c('set log y')
    gp.c('set ylabel "Events / V"')
    gp.c('set xlabel "Amplitude (V)"')
    gp.s([edges[:-1], values], filename=fname)
    for ext in cfg.cfg['plot']['output_format'].split():
        gp_set_terminal(ext)
        gp.c('set out odir."%s/amplitude%s/amplitude%s.%s"' % (det_name, suffix, suffix, ext))
        gp.c('plot [][] "'+fname+'" u 1:2 not w histep lt 6')
        gp.c('set out')



def plot_peaks(peaks, peaks_max, suffix, det):
    fname = 'tmp_peaks.dat'
    det_name = detector_name(det)
    os.makedirs(os.path.join(cfg.global_odir, det_name, 'peaks%s' % suffix), exist_ok=True)
    gp_set_defaults()
    gp.s([peaks, peaks_max], filename=fname)
    gp.c('set auto fix')
    gp.c('set offsets graph 0.1, graph 0.1, graph 0.1, graph 0.1')
    gp.c('set ylabel "Amplitude (V)\\n{/=12 (range containing 95% of the peaks)}"')
    gp.c('set xlabel "Time (h)"')
    pmin = np.quantile(peaks_max, 0.025, axis = 0)
    pmax = np.quantile(peaks_max, 0.975, axis = 0)
    gp.c('set yrange [%f:%f]' % (pmin, pmax))
    for ext in cfg.cfg['plot']['output_format'].split():
        gp_set_terminal(ext)
        gp.c('set out odir."%s/peaks%s/peaks%s.%s"' % (det_name, suffix, suffix, ext))
        gp.c('plot "'+fname+'"'+" u (hour($1)):($2) not w imp lt 6")
        gp.c('set out')



def plot_baseline(base, base_min, suffix, det):
    fname = 'tmp_baseline.dat'
    det_name = detector_name(det)
    os.makedirs(os.path.join(cfg.global_odir, det_name, 'baseline%s' % suffix), exist_ok=True)
    gp_set_defaults()
    gp.s([base, base_min], filename=fname)
    gp.c('set auto fix')
    gp.c('set offsets graph 0.1, graph 0.1, graph 0.1, graph 0.1')
    gp.c('set ytics nomirror')
    gp.c('set y2tics nomirror tc "#bbbbbb"')
    gp.c('set ylabel "Amplitude (V)"')
    gp.c('set xlabel "Time (h)"')
    pmin = np.quantile(base_min, 0.0025)
    pmax = np.quantile(base_min, 0.9975)
    gp.c('set yrange [%f:%f]' % (pmin, pmax))
    for ext in cfg.cfg['plot']['output_format'].split():
        gp_set_terminal(ext)
        gp.c('set out odir."%s/baseline%s/baseline%s.%s"' % (det_name, suffix, suffix, ext))
        gp.c('plot "'+fname+'"'+" u (hour($1)):($2) axis x1y2 not w l lc '#bcbcbc', '' u (hour($1)):($2) not w l lt 6")
        gp.c('set out')



def plot_pulse_shapes(shapes, suffix, det):

    fname = 'tmp_shapes.dat'
    det_name = detector_name(det)
    os.makedirs(os.path.join(cfg.global_odir, det_name, 'shapes%s' % suffix), exist_ok=True)
    os.makedirs(os.path.join(cfg.global_odir, det_name, 'normalized_shapes%s' % suffix), exist_ok=True)
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
    for ext in cfg.cfg['plot']['output_format'].split():
        gp_set_terminal(ext)
        gp.c('set out odir."%s/normalized_shapes%s/normalized_shapes%s.%s"' % (det_name, suffix, suffix, ext))
        gp.c('plot [][] "'+fname+'" u 1:($2 / $4):3 not w l lt palette')
        gp.c('set out')
        gp.c('set out odir."%s/shapes%s/shapes%s.%s"' % (det_name, suffix, suffix, ext))
        gp.c('unset colorbox')
        gp.c('plot [][1:] "'+fname+'" u 1:2:3 not w l lt palette')
        gp.c('set out')



def plot_rate(rate, window, suffix, det):
    fname = 'tmp_rate.dat'
    det_name = detector_name(det)
    os.makedirs(os.path.join(cfg.global_odir, det_name, 'rate%s' % suffix), exist_ok=True)
    gp_set_defaults()
    gp.s([rate], filename=fname)
    gp.c('set auto fix')
    gp.c('set offsets graph 0.1, graph 0.1, graph 0.1, graph 0.1')
    gp.c('set ylabel "Rate (Hz)"')
    gp.c('set xlabel "Time (h)"')
    for ext in cfg.cfg['plot']['output_format'].split():
        gp_set_terminal(ext)
        gp.c('set out odir."%s/rate%s/rate%s.%s"' % (det_name, suffix, suffix, ext))
        gp.c('plot [][] "'+fname+'" u ($0 / 3600.):($1/'+str(window)+') not w l lt 6')
        gp.c('set out')



def plot_fft_rate(freq, power, suffix, det):
    fname = 'tmp_fft_rate.dat'
    det_name = detector_name(det)
    os.makedirs(os.path.join(cfg.global_odir, det_name, 'fft_rate%s' % suffix), exist_ok=True)
    gp_set_defaults()
    gp.s([freq, power], filename=fname)
    gp.c('set auto fix')
    gp.c('set offsets graph 0.1, graph 0.1, graph 0.1, graph 0.1')
    gp.c('set log y')
    gp.c('set ylabel ""')
    gp.c('set xlabel "Frequency (1/min)"')
    for ext in cfg.cfg['plot']['output_format'].split():
        gp_set_terminal(ext)
        gp.c('set out odir."%s/fft_rate%s/fft_rate%s.%s"' % (det_name, suffix, suffix, ext))
        gp.c('plot [][] "'+fname+'" u ($1 * 60):2 not w l lt 6')
        gp.c('set out')



def plot_fft_data(freq, power, suffix, det):
    fname = 'tmp_fft_data.dat'
    det_name = detector_name(det)
    os.makedirs(os.path.join(cfg.global_odir, det_name, 'fft_data%s' % suffix), exist_ok=True)
    gp_set_defaults()
    gp.s([freq, power], filename=fname)
    gp.c('set auto fix')
    gp.c('set offsets graph 0.1, graph 0.1, graph 0.1, graph 0.1')
    gp.c('set log y')
    gp.c('set mytics 10')
    gp.c('set log x')
    gp.c('set mxtics 10')
    gp.c('set ylabel "Power Spectral Density (V^2 / Hz)"')
    #gp.c('set format y "%L"')
    gp.c('set xlabel "Frequency (Hz)"')
    for ext in cfg.cfg['plot']['output_format'].split():
        gp_set_terminal(ext)
        gp.c('set out odir."%s/fft_data%s/fft_data%s.%s"' % (det_name, suffix, suffix, ext))
        gp.c('plot [1:][] "'+fname+'" u 1:2 not w l lt 6')
        gp.c('set out')
