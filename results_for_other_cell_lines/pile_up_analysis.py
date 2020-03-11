import numpy as np
import pandas as pd

mouse_start_end = {'chr1': (3000000, 195300000), 'chr2': (3100000, 182000000), 'chr3': (3000000, 159900000),
                   'chr4': (3100000, 156300000), 'chr5': (3000000, 151700000), 'chr6': (3100000, 149500000),
                   'chr7': (3000000, 145300000), 'chr8': (3000000, 129300000), 'chr9': (3000000, 124400000),
                   'chr10': (3100000, 130500000), 'chr11': (3100000, 121900000), 'chr12': (3000000, 120000000),
                   'chr13': (3000000, 120300000), 'chr14': (3000000, 124800000), 'chr15': (3100000, 103900000),
                   'chr16': (3100000, 98100000), 'chr17': (3000000, 94800000), 'chr18': (3000000, 90600000),
                   'chr19': (3100000, 61300000), 'chrX': (3100000, 170800000)}
human_start_end = {'chr1': (100000, 248900000), 'chr2': (100000, 242100000), 'chr3': (100000, 198200000),
                   'chr4': (100000, 190100000), 'chr5': (100000, 181400000), 'chr6': (100000, 170700000),
                   'chr7': (100000, 159300000), 'chr8': (100000, 145000000), 'chr9': (100000, 138200000),
                   'chr10': (100000, 133700000), 'chr11': (100000, 135000000), 'chr12': (100000, 133200000),
                   'chr13': (100000, 114300000), 'chr14': (100000, 106800000), 'chr15': (100000, 101900000),
                   'chr16': (100000, 90200000), 'chr17': (100000, 83200000), 'chr18': (100000, 80200000),
                   'chr19': (100000, 58600000), 'chr20': (100000, 64400000), 'chr21': (100000, 46700000),
                   'chr22': (100000, 50800000), 'chrX': (100000, 156000000)}


def find_pile_up_region(chromosome, pos1, pos2, padding=10):
    """Find a 250-kb region which covers the chosen position best
    For example, for contacts between chr1:153500000-153501000, chr1:153500000-153501000,
    region chr1:153400000-153600000 is the best.

    Return:
         A list [chromosome, region_start_position] + [new coordinates in this sub-region]
    """
    human_start = human_start_end[chromosome][0]
    resolution = 200
    p11, p22 = min(pos1, pos2), max(pos1, pos2)
    center = (p11 + p22) / 2
    closest_center = int(round((center - human_start) / 125000) * 125000 + human_start)
    start_pos = closest_center - 125000
    p11, p22 = (p11 - start_pos) // resolution, (p22 - start_pos) // resolution
    return [chromosome, start_pos, p11 - padding, p11 + padding + 1, p22 - padding, p22 + padding + 1]


def load_positions(file='Pancreas.signifpairs.txt', chromosome='chr7', distance_range=(2000, 40000)):
    eqtls = pd.read_csv(file, sep='\t')
    eqtls = eqtls[eqtls.variant_id.str.startswith(f'{chromosome}_')]
    eqtls.tss_distance = np.abs(eqtls.tss_distance)
    eqtls = eqtls[eqtls.tss_distance < distance_range[1]]
    eqtls = eqtls[eqtls.tss_distance > distance_range[0]]

    variants = list(eqtls.variant_id)
    variants = np.array([int(elm.split('_')[1]) for elm in variants])
    tss_distance = np.array(eqtls.tss_distance)
    tss = variants - tss_distance
    return variants, tss


def pile_up_analysis(eqtl_file='Pancreas.signifpairs.txt',
                     pred_path='/home/fanfeng/Documents/high_res_map_prediction/pancreas/outputs',
                     chromosome='chr7', distance_range=(2000, 40000), padding=10):
    variants, tss = load_positions(file=eqtl_file, chromosome=chromosome, distance_range=distance_range)
    piled_up = np.zeros((padding * 2 + 1, padding * 2 + 1))
    i, length = 0, len(tss)
    st, ed = human_start_end[chromosome]
    for v, t in zip(variants, tss):
        if i % 50 == 0:
            print(f' {i} / {length}')
        i += 1
        if min(v, t) <= st + 250000 or max(v, t) >= ed - 250000:
            continue
        lst = find_pile_up_region(chromosome, v, t, padding)
        [ch, pos, a00, a10, a01, a11] = lst
        # print(v, t, lst)
        path = '/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/high_res_map_project_training_data/HiC/hESC/'
        x = np.load(f'{path}/{ch}/{ch}_1000bp_{pos}_{pos+250000}.npy')
        # x = np.load(f'{pred_path}/pred_{ch}_{pos}.npy')
        # x = x + x.T
        x = x[a00: a10, a01: a11]
        piled_up += x
    np.save('piled_up_0_40_HiC.npy', piled_up)


pile_up_analysis()
