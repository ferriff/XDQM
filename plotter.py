#!/usr/bin/env python3.7
#
# Copyright 2019-2020 F. Ferri, Ph. Gras

import argparse
import glob
import os
import pickle
import shutil
import sys

from datetime import datetime

# DQM modules
import accumulator
import analyze   as ana
import configure as cfg
import plot      as plt


def is_data_more_recent_than_plot(plotdir, datainfo):
    '''Determine if new data have arrived after a plot has been produced
- basic implementation'''
    #print("# --> plot_dir:", plotdir, "   path:", os.path.join(plotdir, 'amplitude*00/*svg*'))
    flist = glob.glob(os.path.join(plotdir, 'process_configuration.cfg')) # FIXME: better tool...
    plot_time = -1
    if len(flist) > 0:
        plot_time = os.path.getmtime(flist[0])
    if plot_time < 0:
        return False
    data_time = os.path.getmtime(datainfo[0])
    print("# --> plot_time:", plot_time, "   data_time:", data_time)
    return plot_time < data_time


def ana_dir(data_root, acc, detect_only=False, data_suff = '.bin'):
    '''Analyses all data found in the directory data_root. It looks for data files
with suffix data_suff.'''

    # find files
    path_pattern = os.path.join(data_root, '**/*' + data_suff)
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
        # extract dirname and filename from file full path
        df[os.path.dirname(r)].append(os.path.basename(r))

    # find run and chunk number
    run_chunk = {}
    # for each directory in the list
    for d in df.keys():
        rc = set()
        # for each file in the directory
        for f in df[d]:
            v = f.replace(data_suff, '').split('_')
            # store a pair (run_number, chunk_number)
            rc.add((v[0], v[3]))
        # for each directory, store a sorted list of (run_number, chunk_number)
        run_chunk[d] = sorted(rc)

    # reorganize the structure to
    # map the files to a given dir, run, chunk
    drc = {}
    for d in run_chunk.keys():
        for rc in run_chunk[d]:
            # the key   is a tuple (directory, run, chunk)
            # the value is the list of the files in directory corresponding to run and chunk
            drc[tuple([d] + list(rc))] = glob.glob(d + '/' + rc[0] + '_*_' + rc[1] + data_suff)

    if detect_only:
        analyzed = {}
        if os.path.isfile(cfg.analyzed_runs):
            analyzed = pickle.load(open(cfg.analyzed_runs, 'rb'))
        found = False
        print('Adding to the run DB:')
        for k in drc.keys():
            if k not in analyzed.keys():
                print('  ', k)
                analyzed[k] = False
                found = True
        if not found:
            print('no new runs found.')
        pickle.dump(analyzed, open(cfg.analyzed_runs, 'wb'))
        sys.exit(0)


    # ready to analyze runs
    # load list of analyzed runs
    analyzed = {}

    cfg.global_odir = ""
    for k in drc.keys():
        if os.path.isfile(cfg.analyzed_runs):
            analyzed = pickle.load(open(cfg.analyzed_runs, 'rb'))
        ### discard runs with different setup than 24 channels # FIXME: has to be improved        
        if len(drc[k][:24]) != 24:
            print('Warning: skipping files', drc[k])
            continue
        # skip analyzed runs
        cfg.global_odir = os.path.join(k[0].replace(os.path.dirname(data_root), cfg.plot_out_dir_root), k[1], k[2])
        if k in analyzed.keys():
            if analyzed[k] == True and not is_data_more_recent_than_plot(cfg.global_odir, drc[k]):
                continue
        os.makedirs(cfg.global_odir, exist_ok=True)
        data = []
        for f in drc[k][:24]:
            print(datetime.utcnow().strftime("# %s %Y-%m-%d %H:%M:%S UTC"), ' detected file', f)
            data.append(f)
            #print('done')
        print("# Analyzing:\n# --> " + "\n# --> ".join(data))
        ns = ana.analyze(data, acc)
        # plot the run every chunk
        if ns:
            acc.plot()
            acc.dump(os.path.join(cfg.global_odir, 'output.pkl'))
        acc.clear()
        # copy cfg file used in the processing
        shutil.copyfile(cfg.cfg_file, os.path.join(cfg.global_odir, 'process_configuration.cfg'))
        analyzed[k] = True
        # dump updated list of analyzed runs
        pickle.dump(analyzed, open(cfg.analyzed_runs, 'wb'))
        print(datetime.utcnow().strftime("# %s %Y-%m-%d %H:%M:%S UTC"), ' processing done.')


def parse_args():
    '''Parses comand line arguments.'''
    parser = argparse.ArgumentParser(description='CROSS DQM: analysis and plotting. Configuration is read from the file configuration.cfg')
    parser.add_argument('--files', 
                        help='Limit the analysis to a list of files. File paths must be provided in the FILES argument as comma separated list.')
    parser.add_argument('-c', '--cfg', default='configuration.cfg',
                        help='Configuration file to be used')
    parser.add_argument('-d', '--detect', action='store_true', default=False,
                        help='Only finds new files and store them in the run DB with a False (i.e. not processed) flag.')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg.init(args.cfg)
    acc = accumulator.accumulator()
    if args.files:
        cfg.global_odir = "out"
        print("Plots will be stored in the `out' directory.")
        ana.analyze(args.files.split(","), acc)
    else:
        ana_dir(cfg.data_root, acc, detect_only=args.detect)

    #plt.clean_tmpfiles()
