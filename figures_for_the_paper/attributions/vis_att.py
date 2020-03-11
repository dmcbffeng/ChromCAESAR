import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


yticks = ['ATAC-seq', 'CTCF', 'H3K4me1', 'H3K4me3',
          'H3K9ac', 'H3K27ac', 'H3K27me3', 'H3K36me3']
xticks = np.linspace(0, 600, 5)
xticklabels = ['23.025 Mb', '', '23.125 Mb', '', '23.225 Mb']

m = np.load('att_chr7_22975000.npy')
plt.figure(figsize=(10, 6))
g = sns.heatmap(m.T[:, 475:1075], cmap='coolwarm', yticklabels=yticks, xticklabels=xticklabels)
g.set_xticks(xticks)
g.set_yticklabels(g.get_ymajorticklabels(), fontsize=24, rotation=0)
g.set_xticklabels(g.get_xmajorticklabels(), fontsize=24, rotation=0)
plt.tight_layout()
plt.savefig('pancreas_chr7_22975000.png')
plt.show()

