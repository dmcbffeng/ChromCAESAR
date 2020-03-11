import numpy as np
import matplotlib.pyplot as plt


def load_data(resource):
    f = open(f'hic{resource.lower()}_log_chr3_v2.txt')
    next(f)
    v = []
    for line in f:
        v.append(float(line.strip().split()[-1]))
    return v


fig, ax = plt.subplots(figsize=(12, 9))
plt.plot(np.arange(1, 201), load_data('gcn'), label='GCN Prediction')
plt.plot(np.arange(1, 201), load_data('plus'), label='HiCPlus Prediction')
plt.plot(np.arange(1, 201), load_data('reg'), label='HiC-Reg Prediction')
plt.plot(np.arange(1, 201), load_data('micro'), label='Original HiC')
plt.xlim([1, 200])
plt.ylim([-0.05, 0.8])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(fontsize=24)
plt.xlabel('Contact Distance (Kb)', fontsize=32)
plt.ylabel('Correlation', fontsize=32)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.tight_layout()
plt.savefig('results_mouse.png')
plt.show()


