#
# Copyright 2019-2020 F. Ferri, Ph. Gras

import configure as cfg

import os
import numpy as np
import PyGnuplot as gp
import tempfile
import time

fn_collector = []

def clean_tmpfiles():
    print('Waiting 10 sec to clean temporary files and directories...')
    time.sleep(10)
    for fn in fn_collector:
        if os.path.isfile(fn):
            os.remove(fn)
    if os.path.isdir(cfg.tmpdir):
        os.rmdir(cfg.tmpdir)
    print('temporary files and directories cleaned.')


def detector_name(det):
    return cfg.cfg['setup']['ch%03d' % det].replace(' ', '_').split(':') # FIXME: better sanitize det/dir names?


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
    gp.c('set tmargin 2.5')
    gp.c('set label "{/=16:Bold CUPID/CROSS }{/=14:Italic Data Quality Monitoring}{/:Normal \\ }" at graph 0, graph 1.1')
    gp.c('set label "{/=16 Canfranc}" at graph 1, graph 1.1 right')
    gp.c('set label 101 "Last updated on ".system("date -u \\"+%a %e %b %Y %R:%S %Z\\"") at screen .975, graph 1 rotate by 90 right font ",12" tc "#999999"')
    gp.c('hour(x) = x / ' + str(cfg.params.sampling_freq) + ' / 3600.')
    gp.c('uvolt(x) = x * 1e6')
    gp.c('odir = "' + cfg.global_odir + '/"')
    if cfg.tmpdir == '':
        cfg.tmpdir = tempfile.mkdtemp(prefix='tmp_out_', dir='.')



def plot_amplitude(maxima, suffix, det):
    plot_name = '00_amplitude'
    dummy, fname = tempfile.mkstemp(prefix='tmp_amplitude_%03d' % det, suffix='.dat', dir=cfg.tmpdir, text=True)
    det_name, det_feat = detector_name(det)
    xmin = cfg.cfg.getfloat("plot", "ampl_min", fallback=0.)
    xmax = cfg.cfg.getfloat("plot", "ampl_max", fallback=1.)
    xbin = cfg.cfg.getfloat("plot", "ampl_bin", fallback=1000)
    os.makedirs(os.path.join(cfg.global_odir, det_name, '%s%s' % (plot_name, suffix)), exist_ok=True)
    #values, edges = np.histogram(maxima, bins=1500, range=(xmin, xmax), density=False)
    values, edges = np.histogram(maxima, bins=1500, density=False)
    gp_set_defaults()
    gp.c('set label "' + det_name + ' ' + det_feat + ' - amplitude spectrum" at graph 0, graph 1.04 noenhanced')
    gp.c('set log y')
    gp.c('set log x')
    gp.c('set ylabel "Events / {/Symbol m}V"')
    gp.c('set xlabel "Amplitude ({/Symbol m}V)"')
    gp.s([edges[:-1], values], filename=fname)
    for ext in cfg.cfg['plot']['output_format'].split():
        gp_set_terminal(ext)
        gp.c('set out odir."%s/%s%s/%s%s.%s"' % (det_name, plot_name, suffix, plot_name, suffix, ext))
        gp.c('plot [][0.5:] "'+fname+'" u (uvolt($1)):2 not w histep lt 6')
        gp.c('set out')
    fn_collector.append(fname)
    os.close(dummy)



def plot_peaks(peaks, peaks_max, suffix, det):
    plot_name= '20_peaks'
    if len(peaks) == 0:
        return
    dummy, fname = tempfile.mkstemp(prefix='tmp_peaks_%03d' % det, suffix='.dat', dir=cfg.tmpdir, text=True)
    det_name, det_feat = detector_name(det)
    os.makedirs(os.path.join(cfg.global_odir, det_name, '%s%s' % (plot_name, suffix)), exist_ok=True)
    gp_set_defaults()
    gp.c('set label "' + det_name + ' ' + det_feat + ' - signal peaks" at graph 0, graph 1.04 noenhanced')
    gp.s([peaks, peaks_max], filename=fname)
    gp.c('set auto fix')
    gp.c('set offsets graph 0.1, graph 0.1, graph 0.1, graph 0.1')
    gp.c('set log y')
    gp.c('set ylabel "Amplitude ({/Symbol m}V)"')
    gp.c('set xlabel "Time (h)"')
    for ext in cfg.cfg['plot']['output_format'].split():
        gp_set_terminal(ext)
        gp.c('set out odir."%s/%s%s/%s%s.%s"' % (det_name, plot_name, suffix, plot_name, suffix, ext))
        gp.c('plot "'+fname+'"'+" u (hour($1)):(uvolt($2)) not w p pt 6 ps 0.125 lt 6")
        gp.c('set out')
    fn_collector.append(fname)
    os.close(dummy)



def plot_baseline(base, base_min, suffix, det):
    plot_name = '10_baseline'
    plot_name_zoom = '11_baseline_zoom'
    if len(base) == 0:
        return
    dummy, fname = tempfile.mkstemp(prefix='tmp_baseline_%03d' % det, suffix='.dat', dir=cfg.tmpdir, text=True)
    det_name, det_feat = detector_name(det)
    os.makedirs(os.path.join(cfg.global_odir, det_name, '%s%s' % (plot_name, suffix)), exist_ok=True)
    # zoomed version
    os.makedirs(os.path.join(cfg.global_odir, det_name, '%s%s' % (plot_name_zoom, suffix)), exist_ok=True)
    gp_set_defaults()
    gp.c('set label "' + det_name + ' ' + det_feat + ' - baseline" at graph 0, graph 1.04 noenhanced')
    gp.s([base, base_min], filename=fname)
    gp.c('set auto fix')
    gp.c('set offsets graph 0.1, graph 0.1, graph 0.1, graph 0.1')
    gp.c('set ytics nomirror')
    gp.c('set ylabel "Amplitude ({/Symbol m}V)"')
    gp.c('set xlabel "Time (h)"')
    #gp.c('set yrange [%f:%f]' % (pmin, pmax))
    gp.c('stats "'+fname+'" u (uvolt($2) < -8000 ? NaN : uvolt($2)) nooutput')
    for ext in cfg.cfg['plot']['output_format'].split():
        gp_set_terminal(ext)
        gp.c('set out odir."%s/%s%s/%s%s.%s"' % (det_name, plot_name, suffix, plot_name, suffix, ext))
        gp.c('plot "'+fname+'"'+" u (hour($1)):(uvolt($2)) not w l lt 6")
        gp.c('set out')
        # zoomed version
        gp.c('set out odir."%s/%s%s/%s%s.%s"' % (det_name, plot_name_zoom, suffix, plot_name_zoom, suffix, ext))
        yrange = cfg.cfg.get('plot', 'baseline_range_ch%03d' % det, fallback='[]')
        yr = yrange.strip('[]').split(':')
        if len(yr) == 2:
            for i in range(2):
                if yr[i] != '' and yr[i] != '*':
                    yr[i] = 'uvolt(' + yr[i] + ')'
            yrange = '[' + yr[0] + ':' + yr[1] + ']'
        #gp.c('plot []'+ yrange +' "'+fname+'"'+" u (hour($1)):(uvolt($2)) not w l lt 6")
        gp.c('set label 201 sprintf("y-range sample stddev: %4.2f {/Symbol m}V", STATS_ssd) at graph 1, graph 1.04 right')
        gp.c('plot [][STATS_min:STATS_max] "'+fname+'"'+" u (hour($1)):(uvolt($2)) not w l lt 6")
        gp.c('set out')
    fn_collector.append(fname)
    os.close(dummy)



def plot_pulse_shapes(shapes, suffix, det):
    plot1_name = 'XX_shapes'
    plot2_name = 'XX_normalized_shapes'
    dummy, fname = tempfile.mkstemp(prefix='tmp_shapes_%03d' % det, suffix='.dat', dir=cfg.tmpdir, text=True)
    det_name, det_feat = detector_name(det)
    os.makedirs(os.path.join(cfg.global_odir, det_name, '%s%s' % (plot1_name, suffix)), exist_ok=True)
    os.makedirs(os.path.join(cfg.global_odir, det_name, '%s%s' % (plot2_name, suffix)), exist_ok=True)
    of = open(fname, 'w')
    for cnt, el in enumerate(shapes):
        for i, v in enumerate(el[0]):
            of.write("%d %f %d %f\n" % (i, v, cnt, el[1]))
        of.write("\n\n")

    gp_set_defaults()
    gp.c('set label "' + det_name + ' ' + det_feat + '" at graph 0, graph 1.04 noenhanced')
    gp.c('set auto fix')
    gp.c('set offsets graph 0.1, graph 0.1, graph 0.1, graph 0.1')
    gp.c('load "blues.pal"')
    gp.c('set ylabel "Amplitude ({/Symbol m}V)"')
    gp.c('set xlabel "Time (ms)"')
    for ext in cfg.cfg['plot']['output_format'].split():
        gp_set_terminal(ext)
        gp.c('set out odir."%s/%s%s/%s%s.%s"' % (det_name, plot2_name, suffix, plot2_name, suffix, ext))
        gp.c('plot [][] "'+fname+'" u 1:($2 / $4):3 not w l lt palette')
        gp.c('set out')
        gp.c('set out odir."%s/%s%s/%s%s.%s"' % (det_name, plot1_name, suffix, plot1_name, suffix, ext))
        gp.c('unset colorbox')
        gp.c('plot [][1:] "'+fname+'" u 1:(uvolt($2)):3 not w l lt palette')
        gp.c('set out')
    fn_collector.append(fname)
    os.close(dummy)



def plot_rate(rate, window, suffix, det):
    plot_name = 'XX_rate'
    dummy, fname = tempfile.mkstemp(prefix='tmp_rate_%03d' % det, suffix='.dat', dir=cfg.tmpdir, text=True)
    det_name, det_feat = detector_name(det)
    os.makedirs(os.path.join(cfg.global_odir, det_name, '%s%s' % (plot_name, suffix)), exist_ok=True)
    gp_set_defaults()
    gp.c('set label "' + det_name + ' ' + det_feat + '" at graph 0, graph 1.04 noenhanced')
    gp.s([rate], filename=fname)
    gp.c('set auto fix')
    gp.c('set offsets graph 0.1, graph 0.1, graph 0.1, graph 0.1')
    gp.c('set ylabel "Rate (Hz)"')
    gp.c('set xlabel "Time (h)"')
    for ext in cfg.cfg['plot']['output_format'].split():
        gp_set_terminal(ext)
        gp.c('set out odir."%s/%s%s/%s%s.%s"' % (det_name, plot_name, suffix, plot_name, suffix, ext))
        gp.c('plot [][] "'+fname+'" u ($0 / 3600.):($1/'+str(window)+') not w l lt 6')
        gp.c('set out')
    fn_collector.append(fname)
    os.close(dummy)



def plot_fft_rate(freq, power, suffix, det):
    plot_name = 'XX_fft_rate'
    dummy, fname = tempfile.mkstemp(prefix='tmp_fft_rate_%03d' % det, suffix='.dat', dir=cfg.tmpdir, text=True)
    det_name, det_feat = detector_name(det)
    os.makedirs(os.path.join(cfg.global_odir, det_name, '%s%s' % (plot_name, suffix)), exist_ok=True)
    gp_set_defaults()
    gp.c('set label "' + det_name + ' ' + det_feat + '" at graph 0, graph 1.04 noenhanced')
    gp.s([freq, power], filename=fname)
    gp.c('set auto fix')
    gp.c('set offsets graph 0.1, graph 0.1, graph 0.1, graph 0.1')
    gp.c('set log y')
    gp.c('set ylabel ""')
    gp.c('set xlabel "Frequency (1/min)"')
    for ext in cfg.cfg['plot']['output_format'].split():
        gp_set_terminal(ext)
        gp.c('set out odir."%s/%s%s/%s%s.%s"' % (det_name, plot_name, suffix, plot_name, suffix, ext))
        gp.c('plot [][] "'+fname+'" u ($1 * 60):2 not w l lt 6')
        gp.c('set out')
    fn_collector.append(fname)
    os.close(dummy)



def plot_fft_data(freq, power, suffix, det):
    plot_name = '30_fft_data'
    dummy, fname = tempfile.mkstemp(prefix='tmp_fft_data_%03d' % det, suffix='.dat', dir=cfg.tmpdir, text=True)
    det_name, det_feat = detector_name(det)
    os.makedirs(os.path.join(cfg.global_odir, det_name, '%s%s' % (plot_name, suffix)), exist_ok=True)
    gp_set_defaults()
    gp.c('set label "' + det_name + ' ' + det_feat + '" at graph 0, graph 1.04 noenhanced')
    gp.s([freq, power], filename=fname)
    gp.c('set auto fix')
    gp.c('set offsets graph 0.1, graph 0.1, graph 0.1, graph 0.1')
    gp.c('set log y')
    gp.c('set mytics 10')
    gp.c('set log x')
    gp.c('set mxtics 10')
    #gp.c('set ylabel "Power Spectral Density (V^2 / Hz)"')
    gp.c('set ylabel "Noise FFT (V / √Hz)"')
    gp.c('set format y "10^{%L}"')
    gp.c('set xlabel "Frequency (Hz)"')
    for ext in cfg.cfg['plot']['output_format'].split():
        gp_set_terminal(ext)
        gp.c('set out odir."%s/%s%s/%s%s.%s"' % (det_name, plot_name, suffix, plot_name, suffix, ext))
        gp.c('plot [1:][] "'+fname+'" u 1:(sqrt($2)) not w l lt 6')
        gp.c('set out')
    fn_collector.append(fname)
    os.close(dummy)



def plot_correlations(peaks_a, peaks_max_a, det_a, peaks_b, peaks_max_b, det_b, suffix):
    plot_name = '40_corr_peaks'
    if len(peaks_a) == 0:
        return
    dummy, fname = tempfile.mkstemp(prefix='tmp_corr_peaks_%03d' % det_a, suffix='.dat', dir=cfg.tmpdir, text=True)
    det_a_name, det_a_feat = detector_name(det_a)
    det_b_name, det_b_feat = detector_name(det_b)
    # the output directory corresponds to the detector on the x-axis
    os.makedirs(os.path.join(cfg.global_odir, det_a_name, '%s%s' % (plot_name, suffix)), exist_ok=True)
    gp_set_defaults()
    gp.c('set label "' + det_a_name + ' ' + det_a_feat + ' (x-axis), ' + det_b_name + ' ' + det_b_feat + ' (y-axis)" at graph 0, graph 1.04 noenhanced')
    gp.s([peaks_max_a, peaks_max_b], filename=fname)
    gp.c('set auto fix')
    gp.c('set log x')
    gp.c('set log y')
    gp.c('set offsets graph 0.1, graph 0.1, graph 0.1, graph 0.1')
    gp.c('set xlabel "Amplitude ({/Symbol m}V) - Detector: %s %s"' % (det_a_name, det_a_feat))
    gp.c('set ylabel "Amplitude ({/Symbol m}V) - Detector: %s %s"' % (det_b_name, det_b_feat))
    for ext in cfg.cfg['plot']['output_format'].split():
        gp_set_terminal(ext)
        gp.c('set out odir."%s/%s%s/%s%s.%s"' % (det_a_name, plot_name, suffix, plot_name, suffix, ext))
        gp.c('plot [][] "'+fname+'"'+" u (uvolt($1)):(uvolt($2)) not w p lt 6 pt 7 ps 0.125")
        gp.c('set out')
    fn_collector.append(fname)
    os.close(dummy)


def plot_correlation_deltat(det_a, det_b, deltat, suffix):
    plot_name = '41_corr_deltat'
    if len(deltat) == 0:
        return
    dummy, fname = tempfile.mkstemp(prefix='tmp_corr_deltat_%03d' % det_a, suffix='.dat', dir=cfg.tmpdir, text=True)
    det_a_name, det_a_feat = detector_name(det_a)
    det_b_name, det_b_feat = detector_name(det_b)
    # the output directory corresponds to the detector on the x-axis
    os.makedirs(os.path.join(cfg.global_odir, det_a_name, '%s%s' % (plot_name, suffix)), exist_ok=True)
    gp_set_defaults()
    gp.c('set label "' + det_a_name + ' ' + det_a_feat + ', ' + det_b_name + ' ' + det_b_feat + '" at graph 0, graph 1.04 noenhanced')
    values, edges = np.histogram(np.multiply(deltat, 1e3 / cfg.params.sampling_freq), bins=200, range=(-5000, 5000), density=False) # convert deltat in ms
    gp.s([edges[:-1], values], filename=fname)
    gp.c('set log y')
    gp.c('set offsets graph 0.1, graph 0.1, graph 0.1, graph 0.1')
    gp.c('set xlabel "Peak time %s %s - time of the closest peak of %s %s (ms)" noenhanced' % (det_a_name, det_a_feat, det_b_name, det_b_feat))
    gp.c('set arrow 201 from first -600, graph 0 to first -600, graph 1 nohead lc rgb "#666666" dt "-"')
    gp.c('set arrow 202 from first +600, graph 0 to first +600, graph 1 nohead lc rgb "#666666" dt "-"')
    gp.c('set label 203 "correlation window" at first 0, graph 0.9 center tc rgb "#666666"')
    gp.c('set ylabel "Number of events"')
    for ext in cfg.cfg['plot']['output_format'].split():
        gp_set_terminal(ext)
        gp.c('set out odir."%s/%s%s/%s%s.%s"' % (det_a_name, plot_name, suffix, plot_name, suffix, ext))
        gp.c('plot [-2000:2000] "'+fname+'"'+" u 1:2 not w histep lt 6")
        gp.c('set out')
    fn_collector.append(fname)
    os.close(dummy)


def plot_done(det):
    det_name, det_feat = detector_name(det)
    gp.c('system("touch ".odir."/%s/done")' % det_name)
