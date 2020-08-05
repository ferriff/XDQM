#!/usr/bin/env python3.7

import glob
import os
import sys
import pickle
import argparse

from datetime import datetime

# DQM modules
import accumulator
import analyze   as ana
import configure as cfg
import plot      as plt


def is_data_more_recent_than_plot(plotdir, datainfo):
    '''Determine if new data have arrived after a plot has been produced
- basic implementation'''
    flist = glob.glob(os.path.join(plotdir, 'amplitude*00/*svg*')) # FIXME: better tool...
    plot_time = -1
    if len(flist) > 0:
        plot_time = os.path.getmtime(flist[0])
    data_time = os.path.getmtime(datainfo[0])
    return plot_time > data_time


def ana_dir(data_root, acc, data_suff = '.bin'):
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

    # ready to analyze runs
    # load list of analyzed runs
    analyzed = {}

    cfg.global_odir = ""
    for k in drc.keys():
        if os.path.isfile(cfg.analyzed_runs):
            analyzed = pickle.load(open(cfg.analyzed_runs, 'rb'))
        # discard runs with different setup than 12 channels # FIXME: has to be improved        
        if len(drc[k][:12]) != 12:
            continue
        # skip analyzed runs
        cfg.global_odir = os.path.join(k[0].replace(os.path.dirname(data_root), cfg.plot_out_dir_root), k[1], k[2])
        if k in analyzed.keys() and not is_data_more_recent_than_plot(cfg.global_odir, drc[k]):
            continue
        os.makedirs(cfg.global_odir, exist_ok=True)
        data = []
        for f in drc[k][:12]:
            print(datetime.utcnow().strftime("# %s %Y-%m-%d %H:%M:%S UTC"), ' detected file', f)
            data.append(f)
            #print('done')
        print("Analyzing:\n--> " + "\n--> ".join(data))
        ana.analyze(data, acc)
        # plot the run every chunk
        acc.plot()
        acc.dump(os.path.join(cfg.global_odir, 'output.pkl'))
        acc.clear()
        analyzed[k] = True
        # dump updated list of analyzed runs
        pickle.dump(analyzed, open(cfg.analyzed_runs, 'wb'))
        print(datetime.utcnow().strftime("# %s %Y-%m-%d %H:%M:%S UTC"), ' processing done.')


def parse_args():
    '''Parses comand line arguments.'''
    parser = argparse.ArgumentParser(description='CROSS DQM: analysis and plotting. Configuration is read from the file configuration.cfg')
    parser.add_argument('--files', 
                        help='Limit the analysis to a list of files. File paths must be provided in the FILES argument as comma separated list.')
    
    return parser.parse_args()


if __name__ == "__main__":
    cfg.init()
    args = parse_args()
    acc = accumulator.accumulator()
    if args.files:
        cfg.global_odir = "out"
        print("Plots will be stored in the `out' directory.")
        ana.analyze(args.files.split(","), acc)
    else:
        ana_dir(cfg.data_root, acc)

    plt.clean_tmpfiles()
