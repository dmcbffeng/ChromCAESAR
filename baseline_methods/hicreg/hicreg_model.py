from sklearn.ensemble import RandomForestRegressor
import numpy as np
from scipy.stats import zscore
import pickle
import warnings
import random
warnings.filterwarnings('ignore')


def load_all_functional_data(chromosomes, epi_sets):
    """
    Load all required functional data
    :param dict: (dict {str: float}) functional data names and normalization factors (Recommended above)
    :return: dict {chromosome: numpy.array (500 * num_of_functional_datasets)}
    """
    functional_data = {}

    for chrom in chromosomes:
        functional_data[chrom] = None
        for i, k in enumerate(epi_sets):
            s = np.loadtxt(f'/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/mESC/processed/{chrom}/{chrom}_200_{k}.txt')
            s = zscore(s)
            print(chrom, k, len(s))
            if i == 0:
                functional_data[chrom] = s
            else:
                functional_data[chrom] = np.vstack((functional_data[chrom], s))
        functional_data[chrom] = functional_data[chrom].T
        print(functional_data[chrom].shape)  # 大致形状： 1,000,000 * 11
    return functional_data


def load_data(chromosomes, functional_datasets, batch_size=100000):
    all_func = load_all_functional_data(chromosomes, functional_datasets)

    # 为了方便，每一次训练一条strata(一条斜线， 训练一条斜线，大约900，000条数据)
    for ch in chromosomes:
        for i in range(1000):  # 一共有1000条strata
            strata = np.load(
                f'/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/mESC/processed/{ch}/{ch}_200_strata_{i+1}.npy'
            )
            count = 0
            while count + batch_size <= len(strata):
                if random.random() > 0.5:
                    count += batch_size
                    continue
                print(f'Chromosome {ch} - Strata {i + 1} - Process {count / len(strata) * 100} %')
                # Contact between node i and node j
                output_ = strata[count: count + batch_size]
                input_ = np.zeros((batch_size, 3 * len(functional_datasets) + 1))  # 34维输入

                # 前11维：node i的epigenetic data
                input_[:, :len(functional_datasets)] = \
                    all_func[ch][count: count + batch_size, :]
                # 中间11维：node j的epigenetic data
                input_[:, len(functional_datasets):2 * len(functional_datasets)] = \
                    all_func[ch][count + i: count + batch_size + i, :]
                # 后11维：node i和j之间所有信号的平均(为了方便代码 包含i和j)
                # 理论上说包含i j不影响结果 其实是一种线性组合
                for j in range(batch_size):
                    input_[j, 2 * len(functional_datasets):3 * len(functional_datasets)] = \
                        np.mean(all_func[ch][count + j: count + j + i + 1], axis=0)
                # 最后1维 i和j之间的距离
                input_[:, -1] = np.ones((batch_size, )) * i

                count += batch_size
                yield input_, output_


def make_predictions(model, chromosomes, functional_datasets, batch_size=100000):
    all_func = load_all_functional_data(chromosomes, functional_datasets)

    # 为了方便，每一次训练一条strata(一条斜线， 训练一条斜线，大约900，000条数据)
    for ch in chromosomes:
        for i in range(50):  # 一共有1000条strata
            strata = np.load(
                f'/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/mESC/processed/{ch}/{ch}_200_strata_{i + 1}.npy'
            )
            count = 0
            output_strata = np.zeros(strata.shape)
            while count + batch_size <= len(strata):
                print(f'Chromosome {ch} - Strata {i+1} - Process {count / len(strata) * 100} %')
                # Contact between node i and node j
                input_ = np.zeros((batch_size, 3 * len(functional_datasets) + 1))  # 34维输入

                # 前11维：node i的epigenetic data
                input_[:, :len(functional_datasets)] = \
                    all_func[ch][count: count + batch_size, :]
                # 中间11维：node j的epigenetic data
                input_[:, len(functional_datasets):2 * len(functional_datasets)] = \
                    all_func[ch][count + i: count + batch_size + i, :]
                # 后11维：node i和j之间所有信号的平均(为了方便代码 包含i和j)
                # 理论上说包含i j不影响结果 其实是一种线性组合
                for j in range(batch_size):
                    input_[j, 2 * len(functional_datasets):3 * len(functional_datasets)] = \
                        np.mean(all_func[ch][count + j: count + j + i + 1, :], axis=0)
                # 最后1维 i和j之间的距离
                input_[:, -1] = np.ones((batch_size,)) * i

                output_ = model.predict(input_)
                output_strata[count: count + batch_size] = output_
                count += batch_size
            np.save(f'res/{ch}_strata_{i+1}_prediction.npy', output_strata)


def HiCReg_model(chromosomes, functional_datasets):
    # 根据文章， 他们只用了20个tree，按理说咱们分辨率高应该多用一些
    # 但是没时间去试太多了，反正是baseline方法
    model = RandomForestRegressor(n_estimators=20, warm_start=True)

    for input_, output_ in load_data(chromosomes, functional_datasets):
        model.fit(input_, output_)

    return model


if __name__ == '__main__':
    training_chromsomes = ['chr1', 'chr2']
    functional_datasets = [
        'ATAC_seq', 'H3K4me1', 'H3K4me2', 'H3K4me3',
        'H3K9ac', 'H3K9me2', 'H3K9me3', 'H3K27ac',
        'H3K27me3', 'H3K36me3', 'H3K79me2'
    ]
    # Train on chr1 & chr2
    # model = HiCReg_model(training_chromsomes, functional_datasets)
    # pickle.dump(model, open('model_v1.sav', 'wb'))

    model = pickle.load(open('model_v1.sav', 'rb'))

    # Test on chr3
    testing_chromosomes = ['chr1', 'chr2', 'chr3']
    make_predictions(model, testing_chromosomes, functional_datasets)

