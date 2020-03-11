import numpy as np

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


def cal_exp(chrom_files, chrom_sizes, resolution, n_strata, type='mESC'):
    lengths, sum_ = np.zeros((n_strata, )), np.zeros((n_strata, ))
    for ch in chrom_files:
        # print(ch)
        f = open(chrom_files[ch])
        st, ed = chrom_sizes[ch]
        l = (ed - st) // resolution
        lengths += np.arange(l, l - n_strata, -1)
        count = 0
        for line in f:
            if count % 400000 == 0:
                print(ch, count)
            count += 1
            [p1, p2, v] = line.strip().split()
            p1, p2, v = int(p1) // resolution, int(p2) // resolution, float(v)
            if p1 > p2:
                p1, p2 = p2, p1
            if p2 - p1 >= n_strata:
                continue
            sum_[p2 - p1] += v
    # 200 bp
    exp_ = sum_ / lengths
    np.savetxt(f'expectations/{type}_exp_all_200bp.txt', exp_)

    # 1kb
    sum2, lengths2 =\
        np.array([np.sum(sum_[i * 5: (i + 1) * 5]) for i in range(400)]), \
        np.array([np.sum(lengths[i * 5: (i + 1) * 5]) for i in range(400)])
    exp2 = sum2 / lengths2
    np.savetxt(f'expectations/{type}_exp_all_1000bp.txt', exp2)

    # 2kb
    sum2, lengths2 = \
        np.array([np.sum(sum_[i * 10: (i + 1) * 10]) for i in range(200)]), \
        np.array([np.sum(lengths[i * 10: (i + 1) * 10]) for i in range(200)])
    exp2 = sum2 / lengths2
    np.savetxt(f'expectations/{type}_exp_all_2000bp.txt', exp2)

    # 5kb
    sum2, lengths2 = \
        np.array([np.sum(sum_[i * 25: (i + 1) * 25]) for i in range(80)]), \
        np.array([np.sum(lengths[i * 25: (i + 1) * 25]) for i in range(80)])
    exp2 = sum2 / lengths2
    np.savetxt(f'expectations/{type}_exp_all_5000bp.txt', exp2)

    return exp_


mouse_paths = {ch: f'/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/mESC/processed/hic_contacts/{ch}_200bp.txt'
               for ch in MOUSE_CHR_SIZES}
human_paths = {ch: f'/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/human/processed/hic_contacts/{ch}_200bp.txt'
               for ch in MOUSE_CHR_SIZES}

cal_exp(mouse_paths, MOUSE_CHR_SIZES, 200, 2000, 'mESC')

cal_exp(human_paths, HUMAN_CHR_SIZES, 200, 2000, 'hESC')

