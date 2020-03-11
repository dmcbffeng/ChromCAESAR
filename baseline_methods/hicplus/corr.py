import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def visualize_corr(ch, pos, n_th_strata):
    micro = np.load(f'/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/mESC/temp/{ch}/map_200_{pos}_{pos + 200000}_OE.npy')
    hic = np.load(f'/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/mESC/temp/{ch}/hic_200_{pos}_{pos + 200000}_OE.npy')
    pred = np.load(f'/nfs/turbo/umms-drjieliu/res_hicplus/{ch}/pred_{pos}_{pos + 200000}.npy')[0, :, :]

    m = np.exp(np.diag(micro[n_th_strata:, :len(micro) - n_th_strata])) - 1
    h = np.exp(np.diag(hic[n_th_strata:, :len(micro) - n_th_strata])) - 1
    p = np.exp(np.diag(pred[n_th_strata:, :len(micro) - n_th_strata])) - 1

    x = np.arange(len(micro) - n_th_strata)

    plt.figure(figsize=(10, 6))
    plt.subplot(311)
    plt.plot(x, h)
    plt.subplot(312)
    plt.plot(x, m)
    plt.subplot(313)
    plt.plot(x, p)
    # plt.savefig('corr_test.png')
    plt.show()

    print(np.corrcoef(m, h)[0, 1])
    print(np.corrcoef(m, p)[0, 1])


def strata_all(chroms, n_th, window=5):
    start_end_pos = {'chr1': (3000000, 195300000), 'chr2': (3100000, 182000000), 'chr3': (3000000, 159900000),
                     'chr4': (3100000, 156300000), 'chr5': (3000000, 151700000), 'chr6': (3100000, 149500000),
                     'chr7': (3000000, 145300000), 'chr8': (3000000, 129300000), 'chr9': (3000000, 124400000),
                     'chr10': (3100000, 130500000), 'chr11': (3100000, 121900000), 'chr12': (3000000, 120000000),
                     'chr13': (3000000, 120300000), 'chr14': (3000000, 124800000), 'chr15': (3100000, 103900000),
                     'chr16': (3100000, 98100000), 'chr17': (3000000, 94800000), 'chr18': (3000000, 90600000),
                     'chr19': (3100000, 61300000), 'chrX': (3100000, 170800000)
                     }
    mh, mp = [], []
    for ch in chroms:
        st, ed = start_end_pos[ch]
        strata_pred = np.zeros(((ed - st) // 200,))
        count = np.zeros(((ed - st) // 200,))
        for pos in range(st, ed-499999, 100000):
            print(ch, pos)
            pred = np.load(f'/nfs/turbo/umms-drjieliu/res_hicplus/{ch}/pred_{pos}_{pos + 200000}.npy')[0, :, :]
            pred = convolve2d(pred, np.ones((window, window)) / (window ** 2), mode='same')

            pred = np.exp(np.diag(pred[n_th:, :len(pred) - n_th])) - 1
            # print(hic.shape)
            # print(strata_hic[(pos - st) // 200: (pos - st) // 200 + 1000 - n_th].shape)

            strata_pred[(pos - st) // 200: (pos - st) // 200 + 1000 - n_th] += pred
            count[(pos - st) // 200: (pos - st) // 200 + 1000 - n_th] += 1

        strata_pred = np.delete(strata_pred, np.argwhere(count == 0))
        count = np.delete(count, np.argwhere(count == 0))

        strata_micro = np.loadtxt(f'/home/fanfeng/Documents/microC/src/GraphAE/v1.12_200bp/corr/{ch}_{n_th}_micro.txt')
        strata_hic = np.loadtxt(f'/home/fanfeng/Documents/microC/src/GraphAE/v1.12_200bp/corr/{ch}_{n_th}_hic.txt')
        strata_pred = strata_pred / count
        np.savetxt(f'{ch}_{n_th}_pred.txt', strata_pred)

        mh.append(np.corrcoef(strata_micro, strata_hic)[0, 1])
        mp.append(np.corrcoef(strata_micro, strata_pred)[0, 1])

    return chroms, mh, mp


def strata_all_v2(ch, first_strata=0, last_strata=50):
    start_end_pos = {'chr1': (3000000, 195300000), 'chr2': (3100000, 182000000), 'chr3': (3000000, 159900000),
                     'chr4': (3100000, 156300000), 'chr5': (3000000, 151700000), 'chr6': (3100000, 149500000),
                     'chr7': (3000000, 145300000), 'chr8': (3000000, 129300000), 'chr9': (3000000, 124400000),
                     'chr10': (3100000, 130500000), 'chr11': (3100000, 121900000), 'chr12': (3000000, 120000000),
                     'chr13': (3000000, 120300000), 'chr14': (3000000, 124800000), 'chr15': (3100000, 103900000),
                     'chr16': (3100000, 98100000), 'chr17': (3000000, 94800000), 'chr18': (3000000, 90600000),
                     'chr19': (3100000, 61300000), 'chrX': (3100000, 170800000)
                     }
    st, ed = start_end_pos[ch]
    strata_pred = [np.zeros(((ed - st) // 200 - i,)) for i in range(first_strata, last_strata)]
    count = [np.ones(((ed - st) // 200 - i,)) * 1e-9 for i in range(first_strata, last_strata)]
    for pos in range(st, ed-499999, 100000):
        print(ch, pos)
        pred = np.load(f'/nfs/turbo/umms-drjieliu/usr/temp_Fan/res_hicplus/{ch}/pred_{pos}_{pos + 200000}.npy')[0, :, :]
        # pred = np.load(f'/nfs/turbo/umms-drjieliu/usr/temp_Fan/res1.12/{ch}/pred_{pos}_{pos + 200000}.npy')[0, :, :]
        # print(pred.shape)
        # print(strata_hic[(pos - st) // 200: (pos - st) // 200 + 1000 - n_th].shape)
        for i in range(first_strata, last_strata):
            stra = np.diag(pred[i:, :1000-i])
            # print(stra.shape)
            # print(strata_pred[i - first_strata].shape)
            strata_pred[i - first_strata][(pos - st) // 200: (pos - st) // 200 + 1000 - i] += stra
            count[i - first_strata][(pos - st) // 200: (pos - st) // 200 + 1000 - i] += 1

    res = []
    for i in range(first_strata, last_strata):
        pred = strata_pred[i - first_strata] / count[i - first_strata]
        micro = np.load(f'/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/mESC/processed/{ch}/{ch}_200_strata_{i+1}.npy')
        micro = micro[st // 200: ed // 200 - i]
        res.append(np.corrcoef(pred, micro)[0, 1])

    return res


def strata_hic_micro(ch, first_strata=0, last_strata=50):
    start_end_pos = {'chr1': (3000000, 195300000), 'chr2': (3100000, 182000000), 'chr3': (3000000, 159900000),
                     'chr4': (3100000, 156300000), 'chr5': (3000000, 151700000), 'chr6': (3100000, 149500000),
                     'chr7': (3000000, 145300000), 'chr8': (3000000, 129300000), 'chr9': (3000000, 124400000),
                     'chr10': (3100000, 130500000), 'chr11': (3100000, 121900000), 'chr12': (3000000, 120000000),
                     'chr13': (3000000, 120300000), 'chr14': (3000000, 124800000), 'chr15': (3100000, 103900000),
                     'chr16': (3100000, 98100000), 'chr17': (3000000, 94800000), 'chr18': (3000000, 90600000),
                     'chr19': (3100000, 61300000), 'chrX': (3100000, 170800000)
                     }
    st, ed = start_end_pos[ch]
    res = []
    for i in range(first_strata, last_strata):
        print(i)
        micro = np.load(f'/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/mESC/processed/{ch}/{ch}_200_strata_{i + 1}.npy')
        hic = np.load(f'/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/mESC/processed/{ch}/{ch}_200_strata_{i + 1}.npy')
        f = interp1d(np.arange(len(hic)), hic, fill_value='extrapolate')
        hic_new = f(np.arange(-0.4, len(hic) - 0.4, 0.2))
        res.append(np.corrcoef(hic_new[st // 200: ed // 200 - i], micro[st // 200: ed // 200 - i])[0, 1])
    return res


f = open('hicmicro_log_chr3.txt', 'w')
f.write('chr3\n')
res = strata_hic_micro('chr3', 0, 50)
for i in range(0, 50):
    f.write(f'Strata {i + 1}: {res[i]}\n')
f.close()
