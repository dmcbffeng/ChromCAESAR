import pyBigWig
import numpy as np
import os


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
    if output_path is not None:
        np.save(output_path, vec)
    return vec


def load_bigWig_for_entire_genome(path, name, rg, resolution=200, epi_path=''):
    """
        Load bigwig file and save the signal as a 1-D numpy array

        Args:
            path (str): recommended: {cell_type}_{assay_type}_{reference_genome}.bigwig (e.g., mESC_CTCF_mm10.bigwig)
            name (str): the name for the epigenetic mark
            rg (str): reference genome
            resolution (int): resolution
            epi_path (str): the folders for storing data of all chromosomes

        No return value
        """
    bw = pyBigWig.open(path)
    chromosome_sizes = load_chrom_sizes(rg)
    del chromosome_sizes['chrY']

    for ch, end_pos in chromosome_sizes.items():
        nBins = end_pos // resolution
        end_pos = nBins * resolution  # remove the 'tail'

        vec = bw.stats(ch, 0, end_pos, exact=True, nBins=nBins)
        for i in range(len(vec)):
            if vec[i] is None:
                vec[i] = 0
        np.save(f'{epi_path}/{ch}/{ch}_{resolution}bp_{name}.npy', vec)


def load_bedGraph_for_entire_genome(path, name, rg, resolution=200, epi_path=''):
    chromosome_sizes = load_chrom_sizes(rg)
    del chromosome_sizes['chrY']
    epi_signal = {ch: np.zeros((length // resolution, )) for ch, length in chromosome_sizes.items()}

    f = open(path)
    for line in f:
        # print(line)
        lst = line.strip().split()
        ch, p1, p2, v = lst[0], int(lst[1]), int(lst[2]), float(lst[3])
        if ch not in chromosome_sizes:
            continue
        pp1, pp2 = p1 // resolution, int(np.ceil(p2 / resolution))
        for i in range(pp1, pp2):
            value = (min(p2, (i + 1) * resolution) - max(p1, i * resolution)) / resolution * v
            # print(pp1, pp2, i, value)
            epi_signal[ch][i] += value

    for ch in chromosome_sizes:
        np.save(f'{epi_path}/{ch}/{ch}_{resolution}bp_{name}.npy', epi_signal[ch])


if __name__ == '__main__':
    signals = ['ATAC_seq', 'CTCF', 'H3K4me1', 'H3K4me3', 'H3K9ac', 'H3K27ac', 'H3K27me3', 'H3K36me3',
               'H3K4me2', 'H3K9me3', 'H3K79me2', 'Nanog', 'Rad21']

    # hESC
    path = '/nfs/turbo/umms-drjieliu/proj/4dn/data/Epigenomic_Data/hESC'
    for name in signals:
        print('human', name)
        if name == 'ATAC_seq':
            pass
            # load_bedGraph_for_entire_genome(f'{path}/hESC_{name}_hg38.bedGraph',
            #                                 name, 'hg38', 200,
            #                                 '/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/high_res_map_project_training_data/Epi/hESC')
        else:
            load_bigWig_for_entire_genome(f'{path}/hESC_{name}_hg38.bigWig',
                                          name, 'hg38', 200,
                                          '/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/high_res_map_project_training_data/Epi/hESC')

    # mESC
    path = '/nfs/turbo/umms-drjieliu/proj/4dn/data/Epigenomic_Data/mESC'
    for name in signals:
        print('mouse', name)
        if name in ['ATAC_seq', 'H3K4me2', 'H3K79me2', 'Rad21']:
            pass
            # load_bedGraph_for_entire_genome(f'{path}/mESC_{name}_mm10.bedGraph',
            #                                 name, 'mm10', 200,
            #                                 '/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/high_res_map_project_training_data/Epi/mESC')
        else:
            load_bigWig_for_entire_genome(f'{path}/mESC_{name}_mm10.bigwig',
                                          name, 'mm10', 200,
                                          '/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/high_res_map_project_training_data/Epi/mESC')


