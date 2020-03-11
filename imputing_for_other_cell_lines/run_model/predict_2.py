import sys
import time
import argparse
import numpy as np
from scipy.stats import zscore
from model import model_fn


def load_epigenetic_data(chromosomes, epi_names, cell_type='hESC', verbose=1):
    """
    Load epigenetic data from processed file (.npy files)

    Args:
        epi_names (list): the folder for storing data from different chromosomes
        verbose (int):

    Return:
         epigenetic_data (dict): {chromosome: numpy.array (shape: n_bins * n_channels)}
    """

    epigenetic_data = {}
    res = 200

    for ch in chromosomes:
        epigenetic_data[ch] = None
        for i, k in enumerate(epi_names):
            path = '/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/high_res_map_project_training_data/Epi'
            path = f'{path}/{cell_type}/{ch}/{ch}_{res}bp_{k}.npy'
            s = np.load(path)
            # print(ch, k, s.shape)
            s = zscore(s)
            if verbose:
                print(ch, k, len(s))
            if i == 0:
                epigenetic_data[ch] = np.zeros((len(s), len(epi_names)))
            epigenetic_data[ch][:, i] = s
            # epigenetic_data[ch] = epigenetic_data[ch].T
    return epigenetic_data


def evaluate(args):
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

    model.load_weights('v11_temp_model_7.h5')

    # Use the model to predict chromosome 2
    ch = 'chr2'
    res = 1000  # Input resolution: 1000 bp

    # Load epigenetic data
    epi_names = ['ATAC_seq', 'CTCF', 'H3K4me1', 'H3K4me3',
                 'H3K9ac', 'H3K27ac', 'H3K27me3', 'H3K36me3']
    epigenetic_data = load_epigenetic_data([ch], epi_names, args.cell_line)

    path = '/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/high_res_map_project_training_data'

    for pos in range(100000, 242100000 - 249999, 125000):
        print(pos)
        hic = np.load(f'{path}/HiC/{args.cell_line}/{ch}/{ch}_{res}bp_{pos}_{pos + 250000}.npy')
        epi = epigenetic_data[ch][pos // 200 - (args.inp_window // 2): pos // 200 + 1250 + (args.inp_window // 2), :]
        hics = np.array([hic])
        epis = np.array([epi])
        m = model.predict([hics, epis])[0, :, :]  # Model Input: HiC and Epigenetic data
        np.save(f'outputs/pred_{ch}_{res}bp_{pos}.npy', m)


if __name__ == '__main__':
    # sys.stdout = open('log.txt', 'w')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cell_line',
        type=str,
        default='hESC',
        help='cell type'
    )
    # parser.add_argument(
    #     '--chrs',
    #     type=str,
    #     default='1,4,7,10,13,17,18',
    #     help='"All", "all" or comma-separated chromosome ids'
    # )
    # parser.add_argument(
    #     '--inp_resolutions',
    #     type=str,
    #     default='1000',
    #     help='comma-separated input resolutions'
    # )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help='learning rate'
    )
    # parser.add_argument(
    #     '--epochs',
    #     type=int,
    #     default=8,
    #     help='number of epochs'
    # )
    # parser.add_argument(
    #     '--batch_size',
    #     type=int,
    #     default=50,
    #     help='batch size'
    # )
    parser.add_argument(
        '--n_GC_layers',
        type=int,
        default=2,
        help='number of GC layers'
    )
    parser.add_argument(
        '--n_GC_units',
        type=int,
        default=128,
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
    evaluate(args)

