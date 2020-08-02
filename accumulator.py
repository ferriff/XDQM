import configure as cfg

import bisect
import numpy as np
import plot      as plt
import analyze   as ana

import pickle

class accumulator:
    """Accumulates plotting information"""

    def __init__(self):
        self.det = {}

    def add(self, det, obs, values):
        """Add to the detector `det' the observable `obs' with values `values'. If `obs' already exists, `values' are extended assuming no time overlap."""
        if det not in self.det.keys():
            self.det[det] = detinfo()
        self.det[det].add(obs, values)

    def add_acc(self, acc):
        """Extend the accumulated information with that contained in `acc'."""
        if self.det.keys() != acc.det.keys():
            print("Error, cannot add two different detinfo objects:\n-->", self.det.keys(), "-->", acc.det.keys())
            import sys
            sys.exit(1)
        for key, val in self.det.values():
            self.det[det].add_det(acc[det])

    def plot(self):
        """Plot the stored information."""
        # plot channels
        for det in self.det.keys():
            suff = '_det%03d' % (det - 1)

            d = self.det[det]

            plt.plot_amplitude(d.peak_max, suff, det)
            # peaks vs time
            plt.plot_peaks(d.peak, d.peak_max, suff, det)
            # baseline
            plt.plot_baseline(d.baseline, d.baseline_min, suff, det)
            # average fft
            plt.plot_fft_data(d.fft_f, d.fft, suff, det)
            # rate for signals above threshold
            rate = ana.compute_rate(d.peak, d.peak_max, 100 * 1e3)
            plt.plot_rate(rate, 100, suff, det)

    def clear(self):
        """Clear detector information."""
        for key, val in self.det.items():
            val.clear()

    def dump(self, filename):
        """Dump the stored information to a file."""
        f = open(filename, 'wb')
        pickle.dump(self, f)

    def load(self, filename):
        """Load the information from a file."""
        f = open(filename, 'rb')
        self = pickle.load(f)

    def print(self):
        for key, val in self.det.values():
            print('Detector', key, '---')
            val.print()



class detinfo:
    """Information on the single detector"""

    def __init__(self):
        self.peak = []
        self.peak_max = []
        self.baseline = []
        self.baseline_min = []
        self.fft = []
        self.fft_f = []
        self.fft_n = 0.
        self.rate = []


    def add_det(self, det):
        """Add to self the information from detinfo `det'."""
        self.add('peak', (det.peak, det.peak_max))
        self.add('baseline', (det.baseline, det.baseline_min))
        self.add('fft', (det.fft, det.fft_f, det.fft_n))


    def add(self, obs, values, n = 1):
        """Add values to the given observable. The information to be added is assumed not to overlap in time. For observables requiring averages, `n' is the size of the set from which `values' have been computed."""
        if   obs == 'peak':
            p, p_m = values
            if len(p) == 0:
                return
            if len(self.peak) == 0 or self.peak[0] < p[0]:
                i  = bisect.bisect_left(self.peak, p[0])
                self.peak = self.peak[:i] + p
                self.peak_max = self.peak_max[:i] + p_m
            else:
                print(self.peak[0], p[0])
                print(self.peak)
                print(p)
                i  = bisect.bisect_left(p, self.peak[0])
                self.peak = p[:i] + self.peak
                self.peak_max = p_m[:i] + self.peak_max
        elif obs == 'baseline':
            b, b_m = values
            if len(b) == 0:
                return
            if len(self.baseline) == 0 or self.baseline[0] < b[0]:
                i = bisect.bisect_left(self.baseline, b[0])
                self.baseline = self.baseline[:i] + b
                self.baseline_min = self.baseline_min[:i] + b_m
            else:
                i = bisect.bisect_left(b, self.baseline[0])
                self.baseline = b[:i] + self.baseline
                self.baseline_min = b_m[:i] + self.baseline_min
        elif obs == 'fft':
            f, pxx = values
            if len(self.fft) == 0:
                self.fft = pxx
                self.fft_f = f
                self.fft_n = n
            else:
                np.add(np.multiply(self.fft, self.fft_n), np.multiply(pxx, n)) / (self.fft_n + n)
            if len(f) != len(self.fft_f) or f[-1] != self.fft_f[-1]:
                print('Error, FFT frequencies across chunks are not identical')
            self.fft_n += n
        else:
            print('detinfo.add: object %s not supported', obs)


    def clear(self):
        self.__init__()


    def print(self):
        print("detinfo:")
        print("     peaks:", len(self.peak))
        print("  baseline:", len(self.baseline))
        print("      rate:", len(self.rate))
        print("       fft:", self.fft_n)
