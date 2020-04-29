from numba import jit
from scipy.signal import butter, lfilter

# Signal analysis based an Butterworth filtering
#

def butter_bandpass(lowcut, highcut, fs, order = 2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


@jit(nopython=True)
def peak_finder2(s, thr = 0):
    peak_pos = []
    peak_level = []
    for i in range(len(s) - 2):
        if s[i-1] < s[i] and s[i] > s[i+1] and (thr is None or s[i] > thr):
            peak_pos.append(i)
            peak_level.append(s[i])

    return peak_pos, peak_level


def find_peaks_2(samples, freq_win, fs, thr = None):
    s = butter_bandpass_filter(samples-samples[0], freq_win[0], freq_win[1], fs)
    peak_pos = []
    peak_level = []
        
    return peak_finder2(s, thr)
