from io import BytesIO
import numpy as np
from tensorflow.python.lib.io import file_io
from scipy.stats import zscore


HUMAN_CHR_SIZES = {'chr1': (100000, 248900000), 'chr2': (100000, 242100000), 'chr3': (100000, 198200000),
                   'chr4': (100000, 190100000), 'chr5': (100000, 181400000), 'chr6': (100000, 170700000),
                   'chr7': (100000, 159300000), 'chr8': (100000, 145000000), 'chr9': (100000, 138200000),
                   'chr10': (100000, 133700000), 'chr11': (100000, 135000000), 'chr12': (100000, 133200000),
                   'chr13': (100000, 114300000), 'chr14': (100000, 106800000), 'chr15': (100000, 101900000),
                   'chr16': (100000, 90200000), 'chr17': (100000, 83200000), 'chr18': (100000, 80200000),
                   'chr19': (100000, 58600000), 'chr20': (100000, 64400000), 'chr21': (100000, 46700000),
                   'chr22': (100000, 50800000), 'chrX': (100000, 156000000)
                   }
MOUSE_CHR_SIZES = {'chr1': (3000000, 195300000), 'chr2': (3100000, 182000000), 'chr3': (3000000, 159900000),
                   'chr4': (3100000, 156300000), 'chr5': (3000000, 151700000), 'chr6': (3100000, 149500000),
                   'chr7': (3000000, 145300000), 'chr8': (3000000, 129300000), 'chr9': (3000000, 124400000),
                   'chr10': (3100000, 130500000), 'chr11': (3100000, 121900000), 'chr12': (3000000, 120000000),
                   'chr13': (3000000, 120300000), 'chr14': (3000000, 124800000), 'chr15': (3100000, 103900000),
                   'chr16': (3100000, 98100000), 'chr17': (3000000, 94800000), 'chr18': (3000000, 90600000),
                   'chr19': (3100000, 61300000), 'chrX': (3100000, 170800000)
                   }
chromosome_sizes = {'hESC': HUMAN_CHR_SIZES, 'mESC': MOUSE_CHR_SIZES}


def load_npy(src, bucket='gs://micro_c_training_data/'):
    # Loading numpy in Google Cloud Storage
    f = BytesIO(file_io.read_file_to_string(bucket + src, binary_mode=True))
    arr = np.load(f)
    return arr


def load_epigenetic_data(cell_line, chromosomes, epi_names, verbose=1):
    epigenetic_data = {}
    res = 200

    for ch in chromosomes:
        epigenetic_data[ch] = None
        for i, k in enumerate(epi_names):
            src = 'Epi/{0}/{1}/{2}_{3}bp_{4}.npy'.format(cell_line, ch, ch, res, k)
            s = load_npy(src)
            # print(ch, k, s.shape)
            s = zscore(s)
            if verbose:
                print(ch, k, len(s))
            if i == 0:
                epigenetic_data[ch] = np.zeros((len(s), len(epi_names)))
            epigenetic_data[ch][:, i] = s
    return epigenetic_data


def generate_batches(cell_line, chromosomes, resolutions, epi_names, epigenetic_data, batch_size, inp_window):
    st_ed_pos = chromosome_sizes[cell_line]
    chr_sizes = {ch: st_ed_pos[ch] for ch in chromosomes}

    idx2pos = {}
    pointer = 0
    for ch in chromosomes:
        st, ed = chr_sizes[ch]
        positions = np.arange(st, ed - 249999, 125000)
        indices = np.arange(pointer, pointer + len(positions) * len(resolutions))
        for i, idx in enumerate(indices):
            j, k = i // len(resolutions), i % len(resolutions)
            idx2pos[idx] = (ch, positions[j], resolutions[k])
        pointer += len(indices)

    n_samples = len(idx2pos)
    sample_ids = np.arange(n_samples)
    np.random.shuffle(sample_ids)
    n_batches = n_samples // batch_size
    nBins = 1250
    max_distance = 250000
    print(n_batches)

    for j in range(n_batches):
        print(' Batch', j)

        batch_ids = sample_ids[j * batch_size: (j + 1) * batch_size]

        epi = np.zeros((batch_size, nBins + inp_window - 1, len(epi_names)))
        micros = np.zeros((batch_size, nBins, nBins))
        hics = np.zeros((batch_size, nBins, nBins))
        # path = '/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/high_res_map_project_training_data'

        for i, idx in enumerate(batch_ids):
            ch, pos, res = idx2pos[idx]
            epi[i, :, :] = epigenetic_data[ch][pos // 200 - (inp_window // 2): pos // 200 + 1250 + (inp_window // 2), :]
            micros[i, :, :] = load_npy(
                'MicroC/{0}/processed/{1}/{2}_200bp_{3}_{4}.npy'.format(
                    cell_line, ch, ch, pos, pos + max_distance))
            hics[i, :, :] = load_npy(
                'HiC/{0}/processed/{1}/{2}_{3}bp_{4}_{5}.npy'.format(
                    cell_line, ch, ch, res, pos, pos + max_distance))
            # micros[i, :, :] = np.load(
            #     f'{path}/MicroC/{cell_line}/{ch}/{ch}_200bp_{pos}_{pos + max_distance}.npy')
            # hics[i, :, :] = np.load(f'{path}/HiC/{cell_line}/{ch}/{ch}_{res}bp_{pos}_{pos + max_distance}.npy')

        yield (epi, hics), micros



