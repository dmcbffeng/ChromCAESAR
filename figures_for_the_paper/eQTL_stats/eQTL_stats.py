import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def hist_eQTL(file='Pancreas.signifpairs.txt'):
    x = pd.read_csv(file, sep='\t')
    distances = np.abs(np.array(x.tss_distance))

    plt.figure(figsize=(12, 9))
    plt.hist(np.log10(distances), bins=100, range=(2, 6.5), color='black')
    plt.xlim([2, 6.5])
    plt.savefig('eQTL_dist_log.png')
    plt.show()


hist_eQTL()
