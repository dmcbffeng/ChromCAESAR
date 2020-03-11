import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.signal import convolve2d


def back_to_HiC(mat):
    l = len(mat)
    assert l % 5 == 0
    new_mat = np.zeros((l // 5, l // 5))

    for i in range(l // 5):
        for j in range(l // 5):
            new_mat[i, j] = np.mean(mat[5*i: 5*(i+1), 5*j: 5*(j+1)])
    return new_mat


def remove_normalization(mat):
    exps = np.loadtxt('hESC_exp_all_200bp.txt')
    # mat = (mat + mat.T) / 2
    for i in range(len(mat)):
        for j in range(len(mat)):
            mat[i, j] = np.log(mat[i, j] * np.power(exps[abs(i - j)], 0.25) + 1)
            # mat[i, j] = np.log((np.exp(mat[i, j]) - 1) * exps[abs(i - j)] + 1)
    return mat


def vis_micro(ch, pos, res):
    path = '/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/high_res_map_project_training_data'
    pred = np.load(f'output/pred_{ch}_{res}bp_{pos}.npy')[300:600, 300:600]
    pred = convolve2d(pred, np.ones((2, 2)) / 4, mode='same')
    pred = remove_normalization(pred)
    pred = (pred + pred.T) / 2
    hic = np.load(f'{path}/HiC/IMR90/{ch}/{ch}_1000bp_{pos}_{pos + 250000}.npy')[300:600, 300:600]
    hic = back_to_HiC(hic)

    plt.figure(figsize=(18, 9))
    plt.subplot(121)
    sns.heatmap(pred / 1.5, vmax=1, vmin=0, cmap='Reds', square=True)
    plt.subplot(122)
    sns.heatmap(hic / 8, vmax=1, vmin=0, cmap='Reds', square=True)
    plt.savefig(f'IMR90_pred_{ch}_{res}bp_{pos}_300_600.png')
    plt.show()


vis_micro('chr1', 153500000, 1000)

# 96-100, 96-152 (153500000)
# 155840	chr1_153634941_C_T_b38	ENSG00000160679.12	830
# 155841	chr1_153645297_C_G_b38	ENSG00000160679.12	11186

# 45-49, 45-91, 45-259 (145850000)
# 134202	chr1_145919802_C_G_b38	ENSG00000211451.12	789
# 134203	chr1_145928206_C_T_b38	ENSG00000211451.12	9193
# 134204	chr1_145961815_G_A_b38	ENSG00000211451.12	42802

