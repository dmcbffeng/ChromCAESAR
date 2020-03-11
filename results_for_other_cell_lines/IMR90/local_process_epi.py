import pyBigWig
import numpy as np
import os
from scipy.stats import zscore


# ATAC_seq, CTCF, H3K4me1, H3K4me3, H3K9ac, H3K27ac, H3K27me3, H3K36me3
# H3K4me2, H3K9me3, H3K79me2
# Nanog, Rad21


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


def load_bigwig_for_one_region(path, chromosome, start_pos, end_pos, resolution, output_path=None):
    """
    Load bigwig file and save the signal as a 1-D numpy array

    Args:
        path (str): recommended: {cell_type}_{assay_type}_{reference_genome}.bigwig (e.g., mESC_CTCF_mm10.bigwig)
        start_pos (int):
        end_pos (int):
        resolution (int):
        output_path (str): recommended: {cell_type}_{assay_type}_{reference_genome}_{resolution}bp.npy

    Return:
        vec (numpy.array):
    """
    nBins = (end_pos - start_pos) // resolution
    bw = pyBigWig.open(path)
    vec = bw.stats(chromosome, start_pos, end_pos, exact=True, nBins=nBins)
    for i in range(len(vec)):
        if vec[i] is None:
            vec[i] = 0
    if output_path is not None:
        np.save(output_path, vec)
    return vec


def load_bedGraph_for_one_region(path, chromosome, start_pos, end_pos, resolution, output_path=None, score_column=5):
    # score_column: which column is score
    assert start_pos % resolution == 0
    assert end_pos % resolution == 0
    nBins = (end_pos - start_pos) // resolution
    epi_signal = np.zeros((nBins, ))

    f = open(path)
    for line in f:
        # print(line)
        lst = line.strip().split()
        ch, p1, p2, v = lst[0], int(lst[1]), int(lst[2]), float(lst[score_column - 1])
        if ch != chromosome or p1 < start_pos or p2 >= end_pos:
            continue
        pp1, pp2 = p1 // resolution, int(np.ceil(p2 / resolution))
        for i in range(pp1, pp2):
            value = (min(p2, (i + 1) * resolution) - max(p1, i * resolution)) / resolution * v
            # print(pp1, pp2, i, value)
            epi_signal[i - start_pos // resolution] += value
    if output_path is not None:
        np.save(output_path, epi_signal)
    return epi_signal


# def load_bigWig_for_entire_genome(path, name, rg, resolution=200, epi_path=''):
#     """
#         Load bigwig file and save the signal as a 1-D numpy array
#
#         Args:
#             path (str): recommended: {cell_type}_{assay_type}_{reference_genome}.bigwig (e.g., mESC_CTCF_mm10.bigwig)
#             name (str): the name for the epigenetic mark
#             rg (str): reference genome
#             resolution (int): resolution
#             epi_path (str): the folders for storing data of all chromosomes
#
#         No return value
#         """
#     bw = pyBigWig.open(path)
#     chromosome_sizes = load_chrom_sizes(rg)
#     del chromosome_sizes['chrY']
#
#     for ch, end_pos in chromosome_sizes.items():
#         nBins = end_pos // resolution
#         end_pos = nBins * resolution  # remove the 'tail'
#
#         vec = bw.stats(ch, 0, end_pos, exact=True, nBins=nBins)
#         for i in range(len(vec)):
#             if vec[i] is None:
#                 vec[i] = 0
#         np.save('{0}/{1}/{2}_{3}bp_{4}.npy'.format(epi_path, ch, ch, resolution, name), vec)
#
#
# def load_bedGraph_for_entire_genome(path, name, rg, resolution=200, epi_path=''):
#     chromosome_sizes = load_chrom_sizes(rg)
#     del chromosome_sizes['chrY']
#     epi_signal = {ch: np.zeros((length // resolution, )) for ch, length in chromosome_sizes.items()}
#
#     f = open(path)
#     for line in f:
#         # print(line)
#         lst = line.strip().split()
#         ch, p1, p2, v = lst[0], int(lst[1]), int(lst[2]), float(lst[3])
#         if ch not in chromosome_sizes:
#             continue
#         pp1, pp2 = p1 // resolution, int(np.ceil(p2 / resolution))
#         for i in range(pp1, pp2):
#             value = (min(p2, (i + 1) * resolution) - max(p1, i * resolution)) / resolution * v
#             # print(pp1, pp2, i, value)
#             epi_signal[ch][i] += value
#
#     for ch in chromosome_sizes:
#         np.save(f'{epi_path}/{ch}/{ch}_{resolution}bp_{name}.npy', epi_signal[ch])
#
#
# def load_epigenetic_data(chromosomes, epi_names, cell_type='hESC', verbose=1):
#     """
#     Load epigenetic data from processed file (.npy files)
#
#     Args:
#         epi_names (list): the folder for storing data from different chromosomes
#         verbose (int):
#
#     Return:
#          epigenetic_data (dict): {chromosome: numpy.array (shape: n_bins * n_channels)}
#     """
#
#     epigenetic_data = {}
#     res = 200
#
#     for ch in chromosomes:
#         epigenetic_data[ch] = None
#         for i, k in enumerate(epi_names):
#             path = '/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/high_res_map_project_training_data/Epi'
#             path = f'{path}/{cell_type}/{ch}/{ch}_{res}bp_{k}.npy'
#             s = np.load(path)
#             # print(ch, k, s.shape)
#             s = zscore(s)
#             if verbose:
#                 print(ch, k, len(s))
#             if i == 0:
#                 epigenetic_data[ch] = np.zeros((len(s), len(epi_names)))
#             epigenetic_data[ch][:, i] = s
#             # epigenetic_data[ch] = epigenetic_data[ch].T
#     return epigenetic_data


if __name__ == "__main__":
    epigenetic_features = ['ATAC_seq', 'CTCF', 'H3K4me1', 'H3K4me3',
                           'H3K9ac', 'H3K27ac', 'H3K27me3', 'H3K36me3']
    path = '/nfs/turbo/umms-drjieliu/proj/4dn/data/Epigenomic_Data/human_tissues/PA'
    for epi in epigenetic_features:
        print(epi)
        if epi == 'ATAC_seq':
            load_bigwig_for_one_region('{0}/pancreas_DNase_seq_hg38.bigWig'.format(path, epi), 'chr1', 0, 248956000, resolution=200,
                                         output_path='epi/IMR90_chr1_{0}_200bp.npy'.format(epi))
        else:
            load_bigwig_for_one_region('{0}/pancreas_{1}_hg38.bigWig'.format(path, epi), 'chr1', 0, 248956000, resolution=200,
                                       output_path='epi/IMR90_chr1_{0}_200bp.npy'.format(epi))

