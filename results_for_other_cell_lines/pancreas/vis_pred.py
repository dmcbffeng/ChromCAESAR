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


def visualize_HiC_triangle(HiC, output, fig_size=(12, 6.5),
                           vmin=0, vmax=None, cmap='Reds', colorbar=True,
                           colorbar_orientation='vertical',
                           x_ticks=None, fontsize=24):
    """
        Visualize matched HiC and epigenetic signals in one figure
        Args:
            HiC (numpy.array): Hi-C contact map, only upper triangle is used.
            output (str): the output path. Must in a proper format (e.g., 'png', 'pdf', 'svg', ...).
            fig_size (tuple): (width, height). Default: (12, 8)
            vmin (float): min value of the colormap. Default: 0
            vmax (float): max value of the colormap. Will use the max value in Hi-C data if not specified.
            cmap (str or plt.cm): which colormap to use. Default: 'Reds'
            colorbar (bool): whether to add colorbar for the heatmap. Default: True
            colorbar_orientation (str): "horizontal" or "vertical". Default: "vertical"
            x_ticks (list): a list of strings. Will be added at the bottom. THE FIRST TICK WILL BE AT THE START OF THE SIGNAL, THE LAST TICK WILL BE AT THE END.
            fontsize (int): font size. Default: 24

        No return. Save a figure only.
        """
    # font = {'size': fontsize}
    # rc('font', **font)
    # rcParams['font.sans-serif'] = "Arial"
    # rcParams['font.family'] = "sans-serif"

    N = len(HiC)
    coordinate = np.array([[[(x + y) / 2, y - x] for y in range(N + 1)] for x in range(N + 1)])
    X, Y = coordinate[:, :, 0], coordinate[:, :, 1]
    vmax = vmax if vmax is not None else np.max(HiC)

    fig, ax = plt.subplots(figsize=fig_size)
    im = plt.pcolormesh(X, Y, HiC, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.yticks([], [])
    plt.ylim([0, N])
    plt.xlim([0, N])

    if colorbar:
        if colorbar_orientation == 'horizontal':
            _left, _width, _bottom, _height = 0.12, 0.25, 0.75, 0.03
        elif colorbar_orientation == 'vertical':
            _left, _width, _bottom, _height = 0.9, 0.02, 0.3, 0.5
        else:
            raise ValueError('Wrong orientation!')
        cbar = plt.colorbar(im, cax=fig.add_axes([_left, _bottom, _width, _height]),
                            orientation=colorbar_orientation)
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.outline.set_visible(False)

    # plt.savefig(output)
    plt.show()


def low_res(mat, fold=5):
    l = len(mat)
    new_mat = np.zeros((l // fold, l // fold))
    for i in range(l // fold):
        for j in range(l // fold):
            new_mat[i, j] = np.sum(mat[i * fold: (i + 1) * fold, j * fold: (j + 1) * fold]) / fold ** 2
    return new_mat


def vis_micro(ch, pos, res):
    # path = '/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/high_res_map_project_training_data'
    pred = np.load(f'outputs/pred_{ch}_{pos}.npy')
    pred = convolve2d(pred, np.ones((2, 2)) / 4, mode='same')
    # pred = remove_normalization(pred)
    pred = (pred + pred.T) / 2
    # pred = low_res(pred, 10)
    # hic = np.load(f'{path}/HiC/hESC/{ch}/{ch}_1000bp_{pos}_{pos + 250000}.npy')
    # hic = back_to_HiC(hic)
    cut = (550, 650)
    # cut = None
    pred = pred[cut[0]:cut[1], cut[0]:cut[1]] if cut else pred
    name = f'vis/pred_{ch}_{res}bp_{pos}_{cut[0]}_{cut[1]}.png' if cut else f'vis/pred_{ch}_{res}bp_{pos}.png'
    # name = f'vis/pred_{ch}_{res}bp_{pos}_lr.png'
    visualize_HiC_triangle(pred, name, vmax=1.2, colorbar=False)


vis_micro('chr1', 161600000, 1000)

# 11475000 850,950  34-49 40-49
# 7020	chr1_11594047_A_G_b38	ENSG00000132879.13	-60810
# 7021	chr1_11650934_G_A_b38	ENSG00000132879.13	-3923
# 7022	chr1_11651804_G_A_b38	ENSG00000132879.13	-3053

# ENSG00000162746.14	FCRLB	chr1	161721563	161728143	+	7507
# 161600000 550,650 26-58 58-91 58-95
# 66319	chr1_161715191_A_T_b38	ENSG00000162746.14	-6372
# 66320	chr1_161728187_G_C_b38	ENSG00000162746.14	6624
# 66321	chr1_161728897_AT_A_b38	ENSG00000162746.14	7334
# 66322	chr1_161729022_G_A_b38	ENSG00000162746.14	7459

