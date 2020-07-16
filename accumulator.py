import configure as cfg

import bisect
import numpy as np


class accumulator:
    """Accumulates plotting information"""

    def __init__(self):
        self.det = {}

    def add(self, det, obs, values):
        if det not in self.det.keys():
            self.det[det] = detinfo()
        self.det[det].add(obs, values)


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

    def add(self, obs, values):
        """Add values to the given observable"""
        if   obs == 'peak':
            p, p_m = values
            if len(p) == 0:
                return
            i = bisect.bisect_left(self.peak, p[0])
            self.peak = self.peak[:i] + p
            self.peak_max = self.peak_max[:i] + p_m
        elif obs == 'baseline':
            b, b_m = values
            if len(b) == 0:
                return
            i = bisect.bisect_left(self.baseline, b[0])
            self.baseline = self.baseline[:i] + b
            self.baseline_min = self.baseline_min[:i] + b_m
        elif obs == 'fft':
            f, pxx = values
            if len(self.fft) == 0:
                self.fft = pxx
                self.fft_f = f
            else:
                np.add(np.multiply(self.fft, self.fft_n), pxx) / (self.fft_n + 1)
            if len(f) != len(self.fft_f) or f[-1] != self.fft_f[-1]:
                print('Error, FFT frequencies across chunks are not identical')
            self.fft_n += 1
        else:
            print('detinfo.add: object %s not supported', obs)
