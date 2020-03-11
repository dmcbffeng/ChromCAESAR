import numpy as np


corrs = []
for i in range(1000):
    s1 = np.loadtxt(f'HFF_strata/chr2_strata_{i}.txt')
    s2 = np.loadtxt(f'../hicrep/micro_strata/strata_{i}.txt')
    corrs.append(np.corrcoef(s1, s2)[0, 1])

np.savetxt('corrs.txt', np.array(corrs))

