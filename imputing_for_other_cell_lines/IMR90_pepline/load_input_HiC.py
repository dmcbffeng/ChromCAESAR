import numpy as np


def strata_from_a_processed_file(ch, chrom_file, end_pos,
                                 norm, resolution, n_strata):
    """
    Process the contact pair files into numpy arrays
    The files should follow this format:
        position 1 - position 2 - contacts

    Args:
        ch (str): chromosome name
        chrom_file (str): path. E.g. data/chr1_1kb.txt
        norm (bool): whether to do normalization (调整每条斜线的所有数据平均值使其等于与训练数据hESC的对应斜线有相同的平均值)
        end_pos (int):
        resolution (int): resolutions
        n_strata (int): 计算多少条斜线

    Return:
        A list of numpy.array (size = n_strata)
    """
    assert end_pos % resolution == 0
    length = end_pos // resolution
    strata = [np.zeros((length - i,)) for i in range(n_strata)]  # first N strata
    print('Loading strata for {0} at {1} resolution ...'.format(ch, res))
    count = 0
    for line in open(chrom_file):
        if count % 500000 == 0:
            print(' Line: {0}'.format(count))
        count += 1
        [p1, p2, v] = line.strip().split()
        p1, p2, v = int(p1) // resolution, int(p2) // resolution, float(v)
        if p1 > p2:
            p1, p2 = p2, p1
        if p2 - p1 >= n_strata or p2 >= end_pos:
            continue
        strata[p2 - p1][p1] += v
    if norm:
        exp = np.loadtxt(f'hESC_exp_all_{resolution}bp.txt')
        for i in range(len(strata)):
            strata[i] = strata[i] / exp[i]
    return strata



