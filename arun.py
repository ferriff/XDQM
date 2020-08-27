#!/usr/bin/env python3.7

import glob
import os
import sys
import pickle
import argparse
import shutil

def parse_args():
    '''Parses comand line arguments.'''
    parser = argparse.ArgumentParser(description='CROSS DQM Analyzed runs logger: list and modify the analyzed runs')
    parser.add_argument('inputfiles', metavar='input_file', type=str, nargs='+', help='an input file database')
    parser.add_argument('--chunks', help='comma separated list of chunks')
    parser.add_argument('--dirs', help='comma separated list of directories')
    parser.add_argument('--in-place', nargs='?', default=False, const=' ', help='edit file in-place (makes backup if suffix [IN-PLACE] supplied)')
    parser.add_argument('--list', action='store_true', default=False, help='list analyzed runs')
    #parser.add_argument('--output', default='analyzed_output.pkl', help='output file')
    parser.add_argument('--output', help='output file')
    parser.add_argument('--runs', help='comma separated list of runs')
    parser.add_argument('--set-done', action='store_true', default=False, help='set the files as done, i.e. processed')
    parser.add_argument('--set-todo', action='store_true', default=False, help='set the files as to do, i.e. not processed')
    parser.add_argument('--verbose', help='print both entries and modifications')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    analyzed, analyzed_all = {}, {}

    for ifile in args.inputfiles:

        if not os.path.isfile(ifile):
            print('Warning: cannot open file', ifile)
            continue

        analyzed.update(pickle.load(open(ifile, 'rb')))

        dirs, runs, chunks = [], [], []
        if args.dirs:
            dirs = args.dirs.split(',')
        if args.runs:
            runs = list(map(int, args.runs.split(',')))
        if args.chunks:
            chunks = list(map(int, args.chunks.split(',')))

        keys = sorted(analyzed.keys(), key=lambda x : x[0])
        for k in keys:
            d, r, c = k[0], int(k[1]), int(k[2])
            if args.verbose:
                print('v>', k[0], k[1], k[2], analyzed[k])
            sel = False

            # select files
            ds, rs, cs = False, False, False
            if d in dirs or not len(dirs):
                ds = True
            if r in runs or not len(runs):
                rs = True
            if c in chunks or not len(chunks):
                cs = True

            if ds and rs and cs:
                sel = True

            if sel:
                if args.list:
                    print(k[0], k[1], k[2], analyzed[k])
                if args.set_done:
                    analyzed[k] = True
                    print('-->', k[0], k[1], k[2], analyzed[k])
                if args.set_todo:
                    analyzed[k] = False
                    print('-->', k[0], k[1], k[2], analyzed[k])

        analyzed_all.update(analyzed)

        # if in-place -> save each file individually
        if args.in_place:
            if args.in_place != ' ':
                # make backup copy
                shutil.copyfile(ifile, ifile + args.in_place)
                print('save each file individually', ifile)
            of = open(ifile, 'wb')
            pickle.dump(analyzed, of)
            of.close()

    if args.output:
        of = open(args.output, 'wb')
        pickle.dump(analyzed_all, of)
        of.close()
