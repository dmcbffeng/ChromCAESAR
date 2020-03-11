import numpy as np
from scipy.interpolate import interp2d


def contact_maps_from_a_processed_file(ch, chrom_file, reference,
                                       oe_norm, chrom_eff_sizes, resolutions, type='mESC'):
    """
    Process the contact pair files into numpy arrays
    The files should follow this format:
        position 1 - position 2 - contacts

    Args:
        ch (str): chromosome name
        chrom_file (str): path. E.g. data/chr1_1kb.txt
        reference (str): reference genome
        oe_norm (bool): whether to do OE normalization (each stratum divides by its average value)
        output_dir (str): output path
        chrom_eff_sizes (dict): effective (used) region for all chromosomes {chromosome: (strat_pos, end_pos)}
        resolutions (list): resolutions

    No return value
    """
    lengths = {'chr1': 248956422}
    max_distance = 250000

    for res in resolutions:
        n_strata = max_distance // res
        # e.g., max = 200 kb, res = 200 bp ==> 1000 bins, 1000 strata
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

        exps = [np.mean(sta) for sta in strata]
        np.savetxt('IMR90_HiC_exp_1000bp_hg19.txt', np.array(exps))

        if oe_norm:
            exp = np.loadtxt('hESC_HiC_exp_{0}bp.txt'.format(res))
            for i in range(len(strata)):
                strata[i] = strata[i] / exp[i]

        st, ed = chrom_eff_sizes[ch]
        for pos in range(st, ed - max_distance + 1, 25000):
            print('{0} Pos: {1}'.format(ch, pos))
            nBins = max_distance // res
            m = np.zeros((nBins, nBins))
            for i in range(nBins):
                for j in range(i, nBins):
                    m[i, j] += strata[j - i][i + pos // res]
                    m[j, i] += strata[j - i][i + pos // res]
            if res != 200:
                fold_change = res // 200
                fc_ = 1 / fold_change
                f = interp2d(np.arange(nBins), np.arange(nBins), m)
                new_co = np.linspace(-0.5 + fc_ / 2, nBins - 0.5 - fc_ / 2, nBins * fold_change)
                m = f(new_co, new_co)
            fname = '/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/high_res_map_project_training_data/HiC/{0}/{1}/{2}_{3}bp_{4}_{5}.npy'.format(type, ch, ch, res, pos, pos + max_distance)
            np.save(fname, np.log(m + 1))
            # upload_data(ch, pos, type, res)


if __name__ == "__main__":
    contact_maps_from_a_processed_file('chr1', 'IMR_chr1_1000bp_hg19.txt', 'hg19', True,
                                       {'chr1': (100000, 248900000)}, [1000], 'IMR90')


