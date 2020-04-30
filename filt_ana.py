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
def peak_finder(a, thr = 0.):
    return np.array([ a[i] if a[i-1] < a[i] and a[i] > a[i+1] and a[i] > thr else 0 for i in range(len(a) - 2) ])

@jit(nopython=True)
def peak_finder_window(a, width, thr = 0.):
    peak_vals = []
    peak_pos = []
    i = 1
    while i < (len(a) - 1):
        if a[i-1] < a[i] and a[i] > a[i+1] and (a[i] > thr):
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
            i += width

        i += 1

    return peak_pos, peak_vals
                
#@jit(nopython=True)
#def peak_finder2(s, thr = 0):
#    peak_pos = []
#    peak_level = []
#    for i in range(len(s) - 2):
#        if s[i-1] < s[i] and s[i] > s[i+1] and (thr is None or s[i] > thr):
#            peak_pos.append(i)
#            peak_level.append(s[i])
#
#    return peak_pos, peak_level


def find_peaks_2(samples, freq_win, fs, win = 1.e-3, thr = 0.):
    s = butter_bandpass_filter(samples-samples[0], freq_win[0], freq_win[1], fs)
    peak_pos = []
    peak_level = []
    width = int(win/fs + 0.5)

    if thr is None:
        thr = -1.e9
    return peak_finder_window(s, width, thr)
