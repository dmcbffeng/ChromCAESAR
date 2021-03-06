import sys
import time
import argparse
import numpy as np
from scipy.stats import zscore
from model import model_fn
from utils import chromosome_sizes, load_epigenetic_data, generate_batches


def train_and_evaluate(args):
    # Build model
    model = model_fn(
        first_layer=[args.inp_kernel, args.inp_window],
        gcn_layers=[args.n_GC_units] * args.n_GC_layers,
        conv_layer_filters=[int(k) for k in args.conv_kernels.split(',')],
        conv_layer_windows=[int(k) for k in args.conv_windows.split(',')],
        nBins=1250,
        lr=args.lr,
        nMarks=args.n_marks,
        verbose=1
    )

    # Load chromosomes
    st_ed_pos = chromosome_sizes[args.cell_line]
    chromosomes = ['chr' + elm for elm in args.chrs.split(',')] if args.chrs.lower() != 'all' else st_ed_pos.keys()

    # Load resolutions
    resolutions = [int(elm) for elm in args.inp_resolutions.split(',')]

    # Load epigenetic data
    epi_names = ['ATAC_seq', 'CTCF', 'H3K4me1', 'H3K4me3',
                 'H3K9ac', 'H3K27ac', 'H3K27me3', 'H3K36me3']
    epigenetic_data = load_epigenetic_data(args.cell_line, list(st_ed_pos.keys()), epi_names)
    # epigenetic_data = load_processed_epigenetic_data_2(chromosomes, epi_names, args.cell_line)

    for i in range(args.epochs):
        print('Epoch', i, ':')
        t1 = time.time()
        for (epi, hics), micros in generate_batches(args.cell_line, chromosomes, resolutions, epi_names,
                                                    epigenetic_data, args.batch_size, args.inp_window):
            t2 = time.time()
            print(' - Loading data:', t2 - t1, 's')
            model.train_on_batch([hics, epi], micros)
            t3 = time.time()
            print(' - Training:', t3 - t2, 's')
            mse = model.evaluate([hics, epi], micros, batch_size=args.batch_size, verbose=0)
            t1 = time.time()
            print(' - Evaluating:', t1 - t3, 's')
            print(' - MSE:', mse)

        v0 = 0
        for (epi, hics), micros in generate_batches(args.cell_line, ['chr2'], resolutions, epi_names,
                                                    epigenetic_data, args.batch_size, args.inp_window):
            mse = model.evaluate([hics, epi], micros, batch_size=args.batch_size, verbose=0)
            print(' - Validation:', v0, mse)
            v0 += 1

        if i % args.checkpoint_frequency == 0:
            model.save_weights('temp_model_{0}.h5'.format(i))


if __name__ == '__main__':
    # sys.stdout = open('log.txt', 'w')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cell_line',
        type=str,
        default='hESC',
        help='cell_type'
    )
    parser.add_argument(
        '--chrs',
        type=str,
        default='1',
        help='"All", "all" or comma-separated chromosome ids'
    )
    parser.add_argument(
        '--inp_resolutions',
        type=str,
        default='5000,1000',
        help='comma-separated input resolutions'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help='learning rate'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=8,
        help='number of epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=50,
        help='batch size'
    )
    parser.add_argument(
        '--n_GC_layers',
        type=int,
        default=3,
        help='number of GC layers'
    )
    parser.add_argument(
        '--n_GC_units',
        type=int,
        default=96,
        help='number of hidden units in GC layers'
    )
    parser.add_argument(
        '--inp_window',
        type=int,
        default=15,
        help='window size of the input conv layer'
    )
    parser.add_argument(
        '--inp_kernel',
        type=int,
        default=96,
        help='number of kernel of the input first conv layer'
    )
    parser.add_argument(
        '--conv_kernels',
        type=str,
        default='64',
        help='comma-separated numbers of conv kernels'
    )
    parser.add_argument(
        '--conv_windows',
        type=str,
        default='15',
        help='comma-separated numbers of conv windows'
    )
    parser.add_argument(
        '--checkpoint_frequency',
        type=int,
        default=1,
        help='how frequent to save model'
    )
    parser.add_argument(
        '--n_marks',
        type=int,
        default=8,
        help='number of epigenetic marks'
    )
    args, _ = parser.parse_known_args()
    train_and_evaluate(args)

