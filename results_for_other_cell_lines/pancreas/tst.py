import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def oe_norm(mat):
    exps = [np.mean(np.diag(mat[i:, :len(mat)-i])) for i in range(len(mat))]
    for i in range(len(mat)):
        for j in range(len(mat)):
            mat[i, j] = mat[i, j] / exps[abs(i - j)]
    return mat


x = np.load('piled_up_0_40_HiC.npy')
# print(x)

plt.figure()
sns.heatmap(oe_norm(x), vmax=np.max(x), vmin=(np.min(x)), cmap='Reds')
plt.savefig('pile_up_naive_norm_0_40_HiC.png')
plt.show()

