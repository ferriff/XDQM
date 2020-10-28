#
# Copyright 2019-2020 F. Ferri, Ph. Gras

import configparser
import pickle

def parse_config(filename):
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read(filename)
    return config

class Parameters:
    def __init__(self):
        self.sampling_freq = 0


def init(config_file):

    global cfg
    cfg = parse_config(config_file)

    global cfg_file
    cfg_file = config_file

    global params
    params = Parameters()

    data = []

    global data_suff
    data_suff = cfg['data']['suffix']

    global data_root
    data_root = cfg['data']['root_dir']

    global analyzed_runs
    analyzed_runs = cfg['data']['log_analyzed_runs']

    global plot_out_dir_root
    plot_out_dir_root = cfg['plot']['output_dir']

    global global_odir
    global_odir = "./out/"

    #global noise_threshold
    #noise_threshold = float(cfg['analysis']['noise_threshold'])

    global ampl_reco_weights
    #ampl_reco_weights = pickle.load(open('amplitude_reco_weights.pkl', 'rb'))
    ampl_reco_weights = pickle.load(open('pulse_weights.pkl', 'rb'))

    global tmpdir
    tmpdir = ""
