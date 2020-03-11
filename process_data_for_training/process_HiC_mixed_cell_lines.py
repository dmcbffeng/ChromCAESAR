import numpy as np
import os
from scipy.interpolate import interp2d


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


def count_contacts(name, chrom_file, columns, ch):
    f = open('total_reads.txt', 'a')
    resolution = 1000
    print('Counting: ', name)
    count = 0
    reads = 0
    for line in open(chrom_file):
        if count % 500000 == 0:
            print(' Counting: Line: {0}'.format(count))
        count += 1
        lst = line.strip().split()
        if len(columns) == 3:
            p1, p2, v = lst[columns[0]], lst[columns[1]], lst[columns[2]]
        elif len(columns) == 4:
            c1, c2 = lst[columns[0]], lst[columns[2]]
            if c1 != ch or c2 != ch:
                continue
            p1, p2, v = lst[columns[1]], lst[columns[3]], 1.0
        else:
            raise ValueError

        p1, p2, v = int(p1) // resolution, int(p2) // resolution, float(v)
        if p1 > p2:
            p1, p2 = p2, p1
        if p2 - p1 >= 250:
            continue
        reads += v
    f.write(f'{ch} {name} {reads}\n')
    f.close()
    #     n_reads.append(reads)
    # np.savetxt('n_reads.txt', np.array(n_reads))
    # n_reads = [elm / np.average(n_reads) * len(n_reads) for elm in n_reads]
    # np.savetxt('n_reads_2.txt', np.array(n_reads))
    return reads


def contact_maps_from_processed_files(ch, chrom_files, reference, n_reads,
                                      oe_norm, chrom_eff_sizes, resolution, type='mix'):
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
    n_reads = [elm / np.average(n_reads) * len(n_reads) for elm in n_reads]
    lengths = load_chrom_sizes(reference)
    max_distance = 250000
    st, ed = chrom_eff_sizes[ch]

    n_strata = max_distance // resolution
    # e.g., max = 200 kb, res = 200 bp ==> 1000 bins, 1000 strata
    strata = [np.zeros((lengths[ch] // resolution + 1 - i,)) for i in range(n_strata)]  # first N strata
    print('Loading strata for {0} at {1} resolution ...'.format(ch, resolution))
    count = 0
    for chrom_file, n_read in zip(chrom_files, n_reads):
        print(chrom_file)
        for line in open(chrom_file):
            if count % 500000 == 0:
                print(' Line: {0}'.format(count))
            count += 1
            if chrom_file.endswith('summary.txt'):
                lst = line.strip().split()
                c1, c2 = lst[1], lst[4]
                if c1 != ch or c2 != ch:
                    continue
                p1, p2, v = lst[2], lst[5], 1.0
            else:
                [p1, p2, v] = line.strip().split()
            p1, p2, v = int(p1) // resolution, int(p2) // resolution, float(v)
            if p1 > p2:
                p1, p2 = p2, p1
            if p2 - p1 >= n_strata or p1 < st // resolution or p2 >= ed // resolution:
                continue
            strata[p2 - p1][p1] += v / n_read
    if oe_norm:
        exp = np.loadtxt('expectations/{0}_exp_all_{1}bp.txt'.format(type, res))
        for i in range(len(strata)):
            strata[i] = strata[i] / exp[i]

    for pos in range(st, ed - max_distance + 1, 125000):
        print('{0} Pos: {1}'.format(ch, pos))
        nBins = max_distance // resolution
        m = np.zeros((nBins, nBins))
        for i in range(nBins):
            for j in range(i, nBins):
                m[i, j] += strata[j - i][i + pos // resolution]
                m[j, i] += strata[j - i][i + pos // resolution]
        if resolution != 200:
            fold_change = resolution // 200
            fc_ = 1 / fold_change
            f = interp2d(np.arange(nBins), np.arange(nBins), m)
            new_co = np.linspace(-0.5 + fc_ / 2, nBins - 0.5 - fc_ / 2, nBins * fold_change)
            m = f(new_co, new_co)
        fname = '/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/high_res_map_project_training_data/HiC/{0}/{1}/{2}_{3}bp_{4}_{5}.npy'.format(
            type, ch, ch, resolution, pos, pos + max_distance)
        np.save(fname, np.log(m + 1))
        # upload_data(ch, pos, type, res)


def process_hics():
    chrom_lengths = load_chrom_sizes('hg38')
    del chrom_lengths['chrY']

    for ch in chrom_lengths:
        st, ed = HUMAN_CHR_SIZES[ch]
        reads = []
        # IMR90
        path1 = '/nfs/turbo/umms-drjieliu/proj/4dn/data/bulkHiC/human_tissues/IMR90/HiC.IMR90.rep1.nodup.summary.txt'
        reads.append(count_contacts('IMR90', path1, [1, 2, 4, 5], ch))
        # H1-hESC
        path2 = f'/nfs/turbo/umms-drjieliu/proj/4dn/data/bulkHiC/H1-hESC/processed/{ch}_1kb.txt'
        reads.append(count_contacts('hESC', path2, [0, 1, 2], ch))
        # K562
        path3 = '/nfs/turbo/umms-drjieliu/proj/4dn/data/bulkHiC/K562/4DNFI8Y9SRP2.hic'
        os.system(
            f' java -jar ~/JuiceTools/juicer_tools_1.14.08.jar dump observed NONE {path3} {ch} {ch} BP 1000 K562.txt'
        )
        path3 = 'K562.txt'
        reads.append(count_contacts('K562', path3, [0, 1, 2], ch))
        # HFF
        path4 = '/nfs/turbo/umms-drjieliu/proj/4dn/data/bulkHiC/Dekker_HFF/raw/HFF_HiC.hic'
        os.system(
            f' java -jar ~/JuiceTools/juicer_tools_1.14.08.jar dump observed NONE {path4} {ch} {ch} BP 1000 HFF.txt'
        )
        path4 = 'HFF.txt'
        reads.append(count_contacts('HFF', path4, [0, 1, 2], ch))
        # GM12878
        path5 = '/nfs/turbo/umms-drjieliu/proj/4dn/data/bulkHiC/GM12878/GM12878.hic'
        os.system(
            f' java -jar ~/JuiceTools/juicer_tools_1.14.08.jar dump observed NONE {path5} {ch} {ch} BP 1000 GM12878.txt'
        )
        path5 = 'GM12878.txt'
        reads.append(count_contacts('GM12878', path5, [0, 1, 2], ch))

        chrom_files = [path1, path2, path3, path4, path5]
        contact_maps_from_processed_files(ch, chrom_files, 'hg38', reads, False, HUMAN_CHR_SIZES, 1000, 'mix')


if __name__ == '__main__':
    # K562: /nfs/turbo/umms-drjieliu/proj/4dn/data/bulkHiC/K562/4DNFI8Y9SRP2.hic
    # IMR90: /nfs/turbo/umms-drjieliu/proj/4dn/data/bulkHiC/human_tissues/IMR90/HiC.IMR90.rep1.nodup.summary.txt
    # H1-hESC: /nfs/turbo/umms-drjieliu/proj/4dn/data/bulkHiC/H1-hESC/processed/chr1_1kb.txt
    # HFF: /nfs/turbo/umms-drjieliu/proj/4dn/data/bulkHiC/Dekker_HFF/raw/HFF_HiC.hic
    # GM12878: /nfs/turbo/umms-drjieliu/proj/4dn/data/bulkHiC/GM12878/GM12878.hic
    process_hics()
