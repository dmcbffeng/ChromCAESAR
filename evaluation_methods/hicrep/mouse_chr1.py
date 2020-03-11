import numpy as np
import os
import argparse
from scipy.interpolate import interp2d
from scipy.stats import zscore
from scipy.signal import convolve2d, convolve
from model import model_fn


MOUSE_CHR_SIZES = {'chr1': (3000000, 195300000), 'chr2': (3100000, 182000000), 'chr3': (3000000, 159900000),
                   'chr4': (3100000, 156300000), 'chr5': (3000000, 151700000), 'chr6': (3100000, 149500000),
                   'chr7': (3000000, 145300000), 'chr8': (3000000, 129300000), 'chr9': (3000000, 124400000),
                   'chr10': (3100000, 130500000), 'chr11': (3100000, 121900000), 'chr12': (3000000, 120000000),
                   'chr13': (3000000, 120300000), 'chr14': (3000000, 124800000), 'chr15': (3100000, 103900000),
                   'chr16': (3100000, 98100000), 'chr17': (3000000, 94800000), 'chr18': (3000000, 90600000),
                   'chr19': (3100000, 61300000), 'chrX': (3100000, 170800000)
                   }

HUMAN_CHR_SIZES = {'chr1': (100000, 248900000), 'chr2': (100000, 242100000), 'chr3': (100000, 198200000),
                   'chr4': (100000, 190100000), 'chr5': (100000, 181400000), 'chr6': (100000, 170700000),
                   'chr7': (100000, 159300000), 'chr8': (100000, 145000000), 'chr9': (100000, 138200000),
                   'chr10': (100000, 133700000), 'chr11': (100000, 135000000), 'chr12': (100000, 133200000),
                   'chr13': (100000, 114300000), 'chr14': (100000, 106800000), 'chr15': (100000, 101900000),
                   'chr16': (100000, 90200000), 'chr17': (100000, 83200000), 'chr18': (100000, 80200000),
                   'chr19': (100000, 58600000), 'chr20': (100000, 64400000), 'chr21': (100000, 46700000),
                   'chr22': (100000, 50800000), 'chrX': (100000, 156000000)
                   }


def load_chrom_sizes(reference_genome):
    """
    Load chromosome sizes for a reference genome
    """
    my_path = os.path.abspath(os.path.dirname(__file__))
    f = open(os.path.join(my_path, reference_genome + '.chrom.sizes'))
    lengths = {}
    for line in f:
        [ch, l] = line.strip().split()
        lengths[ch] = int(l)
    return lengths


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


def load_all_strata(ch, chrom_file, res=1000, n_strata=250, reference='hg38'):
    lengths = load_chrom_sizes(reference)

    strata = [np.zeros((lengths[ch] // res + 1 - i,)) for i in range(n_strata)]  # first N strata
    print('Loading strata for {0} at {1} resolution ...'.format(ch, res))
    count = 0
    for line in open(chrom_file):
        if count % 500000 == 0:
            print(' Line: {0}'.format(count))
        count += 1
        [p1, p2, v] = line.strip().split()
        p1, p2, v = int(p1) // res, int(p2) // res, float(v)
        if p1 > p2:
            p1, p2 = p2, p1
        if p2 - p1 >= n_strata:
            continue
        strata[p2 - p1][p1] += v

    return strata


def load_micro_strata(ch, n_strata, window_size=3, cell_line='hESC'):
    path = '/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/high_res_map_project_training_data/MicroC'
    micro_strata = [np.load(f'{path}/{cell_line}/{ch}/{ch}_200bp_strata_{i}.npy') for i in range(n_strata + window_size)]
    for i in range(n_strata):
        print('Micro strata:', i)
        new_micro_strata = np.zeros((len(micro_strata[i]) - window_size + 1,))
        for j in range(- window_size + 1, window_size):
            stratum_id = abs(i + j)
            conv_window = window_size - abs(j)
            if conv_window > 1:
                _stratum = convolve(micro_strata[stratum_id], np.ones((conv_window, )), mode='valid')
            else:
                _stratum = micro_strata[stratum_id]
            padding = len(_stratum) - len(new_micro_strata)
            # print(f'Strata {i}; Length {len(new_micro_strata)}')

            if padding > 0:
                _stratum = _stratum[padding // 2:-padding // 2]
            new_micro_strata += _stratum
        new_micro_strata /= window_size ** 2
        np.savetxt(f'micro_strata/strata_{i}.txt', np.log(new_micro_strata + 1))


def generate_regions(cell_line, ch, strata, epigenetic_data, inp_window=15):
    if cell_line == 'mESC':
        st_ed_pos = MOUSE_CHR_SIZES
    else:
        st_ed_pos = HUMAN_CHR_SIZES
    st, ed = st_ed_pos[ch]

    for pos in range(st, ed - 249999, 25000):
        epi = epigenetic_data[ch][pos // 200 - (inp_window // 2): pos // 200 + 1250 + (inp_window // 2), :]
        epis = np.array([epi])

        hic = np.zeros((250, 250))
        for i in range(250):
            # for j in range(pos // 1000, pos // 1000 + 250 - i):
            #     ii, jj = i, i + j - pos // 1000
            #     hic[ii, jj] = strata[i][j]
            #     hic[jj, ii] = strata[i][j]
            for j in range(i, 250):
                hic[i, j] = strata[j - i][i + pos // 1000]
                hic[j, i] = strata[j - i][i + pos // 1000]
        f = interp2d(np.arange(250), np.arange(250), hic)
        new_co = np.linspace(-0.4, 249.4, 1250)
        hic = f(new_co, new_co)
        hic = np.log(hic + 1)
        # np.save(f'outputs/pred_{ch}_{pos}_{pos + 250000}.npy', hic)
        hics = np.array([hic])
        yield pos, hics, epis


def cal_normalization_mat():
    m = np.ones((1250, 1250))
    m[:250, :250] += 1
    m[:500, :500] += 1
    m[:750, :750] += 1
    m[:1000, :1000] += 1
    m[250:, 250:] += 1
    m[500:, 500:] += 1
    m[750:, 750:] += 1
    m[1000:, 1000:] += 1
    return m


def args():
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
        default='1,4,7,10,13,17,18',
        help='"All", "all" or comma-separated chromosome ids'
    )
    parser.add_argument(
        '--inp_resolutions',
        type=str,
        default='1000',
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
        default='128',
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
    return args


def load_model(args):
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

    model.load_weights('v16_temp_model_11.h5')
    return model


def prepare_data(ch='chr2', reference='hg38', window_size=1):
    epi_names = ['ATAC_seq', 'CTCF', 'H3K4me1', 'H3K4me3',
                 'H3K9ac', 'H3K27ac', 'H3K27me3', 'H3K36me3']
    epi_data = load_epigenetic_data(['chr2'], epi_names)

    model = load_model(args())

    strata = load_all_strata('chr2', '/nfs/turbo/umms-drjieliu/proj/4dn/data/bulkHiC/H1-hESC/processed/chr2_1kb.txt')

    # norm = cal_normalization_mat()
    window = np.ones((window_size, window_size)) / window_size / window_size

    length = load_chrom_sizes(reference)[ch] // 200 + 1
    new_strata = [np.zeros((length - i,)) for i in range(1000)]

    for pos, hics, epis in generate_regions('hESC', 'chr2', strata, epi_data):
        print(pos)
        output = model.predict([hics, epis]).reshape([1250, 1250])
        output = (output + output.T) / 2
        output = np.exp(output) - 1
        if window_size > 1:
            output = convolve2d(output, window, mode='same')  #/ norm
        # np.save(f'outputs/pred_{ch}_{pos}_{pos + 250000}.npy', output)

        for i in range(1000):
            strata_i = np.diag(output[i:, :1250-i])
            # print(strata_i.shape, pos // 200 + 500 - i, pos // 200 + 750 - i, 500 - i // 2)
            new_strata[i][pos // 200 + 500 - i: pos // 200 + 750 - i] = strata_i[500 - i // 2: 750 - i // 2]

    for i in range(1000):
        print('Saving:', i)
        np.savetxt(f'strata/strata_{i}.txt', new_strata[i])

    # load_micro_strata(ch, 1000, window_size=window_size)


def hicrep(ch='chr2'):
    st, ed = HUMAN_CHR_SIZES[ch]
    cors = np.zeros((1000,))
    for j in range(1000):
        print(j)
        r1 = np.loadtxt(f'strata/strata_{j}.txt')[st // 200 + 1000: ed // 200 - 1000 - j]
        r2 = np.loadtxt(f'micro_strata/strata_{j}.txt')[st // 200 + 1000: ed // 200 - 1000 - j]
        r2 = np.exp(r2) - 1
        r1 = np.rint(r1 / np.mean(r1) * np.mean(r2))
        cor = np.corrcoef(r1, r2)[0, 1]
        cors[j] = cor
    np.savetxt('hicrep_corr_v16_w3_v3int.txt', cors)


# load_micro_strata('chr2', 1000, window_size=3)
# prepare_data()
hicrep()
