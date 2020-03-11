from keras.layers import Input, Conv2D, Reshape, Permute, Add
from keras.models import Model
import numpy as np
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def batch_generator_200bp(chosen_chromosomes, batch_size):
    start_end_pos = {'chr1': (3000000, 195300000), 'chr2': (3100000, 182000000), 'chr3': (3000000, 159900000),
                     'chr4': (3100000, 156300000), 'chr5': (3000000, 151700000), 'chr6': (3100000, 149500000),
                     'chr7': (3000000, 145300000), 'chr8': (3000000, 129300000), 'chr9': (3000000, 124400000),
                     'chr10': (3100000, 130500000), 'chr11': (3100000, 121900000), 'chr12': (3000000, 120000000),
                     'chr13': (3000000, 120300000), 'chr14': (3000000, 124800000), 'chr15': (3100000, 103900000),
                     'chr16': (3100000, 98100000), 'chr17': (3000000, 94800000), 'chr18': (3000000, 90600000),
                     'chr19': (3100000, 61300000), 'chrX': (3100000, 170800000)
                     }
    idx2pos = {}
    pointer = 0
    for chrom in chosen_chromosomes:
        st, ed = start_end_pos[chrom]
        positions = np.arange(st, ed-499999, 100000)
        indices = np.arange(pointer, pointer + len(positions))
        for idx, pos in zip(indices, positions):
            idx2pos[idx] = (chrom, pos)
        pointer += len(positions)
    n_samples = pointer

    sample_ids = np.arange(n_samples)
    np.random.shuffle(sample_ids)
    n_batches = n_samples // batch_size
    current_batch = 0

    while True:
        batch_ids = sample_ids[current_batch * batch_size: (current_batch + 1) * batch_size]
        micros = np.zeros((batch_size, 1000, 1000))
        hics = np.zeros((batch_size, 1000, 1000))

        for i, idx in enumerate(batch_ids):
            # print(i, idx)
            ch, pos = idx2pos[idx]
            micros[i, :, :] = np.load(f'/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/mESC/temp/{ch}/map_200_{pos}_{pos+200000}_OE.npy')
            hics[i, :, :] = np.load(f'/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/mESC/temp/{ch}/hic_200_{pos}_{pos+200000}.npy')

        yield hics, micros

        current_batch += 1
        if current_batch == n_batches:
            current_batch = 0
            np.random.shuffle(sample_ids)


def build_model():
    hic0 = Input(shape=(1000, 1000))
    hic = Reshape([1000, 1000, 1])(hic0)

    hidden_1 = Conv2D(8, (9, 9), padding='same', activation='relu')(hic)
    hidden_2 = Conv2D(8, (1, 1), padding='same', activation='relu')(hidden_1)
    hidden_3 = Conv2D(1, (5, 5), padding='same', activation='relu')(hidden_2)

    res = Reshape([1000, 1000])(hidden_3)
    res_T = Permute([2, 1])(res)
    pred = Add()([res, res_T])

    m = Model(inputs=hic0, outputs=pred)
    m.compile(optimizer='sgd', loss='mse')
    m.summary()
    return m


def predict_all(model, chosen_chromosomes):
    start_end_pos = {'chr1': (3000000, 195300000), 'chr2': (3100000, 182000000), 'chr3': (3000000, 159900000),
                     'chr4': (3100000, 156300000), 'chr5': (3000000, 151700000), 'chr6': (3100000, 149500000),
                     'chr7': (3000000, 145300000), 'chr8': (3000000, 129300000), 'chr9': (3000000, 124400000),
                     'chr10': (3100000, 130500000), 'chr11': (3100000, 121900000), 'chr12': (3000000, 120000000),
                     'chr13': (3000000, 120300000), 'chr14': (3000000, 124800000), 'chr15': (3100000, 103900000),
                     'chr16': (3100000, 98100000), 'chr17': (3000000, 94800000), 'chr18': (3000000, 90600000),
                     'chr19': (3100000, 61300000), 'chrX': (3100000, 170800000)
                     }
    for chrom in chosen_chromosomes:
        st, ed = start_end_pos[chrom]
        cnt = 0
        for pos in range(st, ed - 499999, 100000):
            hic = np.zeros((1, 1000, 1000))
            hic[0, :, :] = np.load(f'/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/mESC/temp/{chrom}/hic_200_{pos}_{pos+200000}.npy')
            output = model.predict(hic)
            cnt += 1
            np.save(f'/nfs/turbo/umms-drjieliu/res_hicplus/{chrom}/pred_{pos}_{pos+200000}.npy', output)


if __name__ == '__main__':
    model = build_model()
    chosen_chromosomes = ['chr1', 'chr2']
    start_end_pos = {'chr1': (3000000, 195300000), 'chr2': (3100000, 182000000), 'chr3': (3000000, 159900000),
                     'chr4': (3100000, 156300000), 'chr5': (3000000, 151700000), 'chr6': (3100000, 149500000),
                     'chr7': (3000000, 145300000), 'chr8': (3000000, 129300000), 'chr9': (3000000, 124400000),
                     'chr10': (3100000, 130500000), 'chr11': (3100000, 121900000), 'chr12': (3000000, 120000000),
                     'chr13': (3000000, 120300000), 'chr14': (3000000, 124800000), 'chr15': (3100000, 103900000),
                     'chr16': (3100000, 98100000), 'chr17': (3000000, 94800000), 'chr18': (3000000, 90600000),
                     'chr19': (3100000, 61300000), 'chrX': (3100000, 170800000)
                     }
    chromosome_lengths = [(start_end_pos[chrom][1] - start_end_pos[chrom][0] - 400000) // 100000 for chrom in
                          chosen_chromosomes]

    print(chosen_chromosomes)
    print(chromosome_lengths)
    n_samples = sum(chromosome_lengths)
    batch_size = 200
    n_per_epoch = n_samples // batch_size
    print(n_samples, n_per_epoch)

    model.fit_generator(
        batch_generator_200bp(chosen_chromosomes, batch_size),
        epochs=800,
        steps_per_epoch=n_per_epoch
    )

    # _json = model.to_json()
    # with open("model.json", "w") as json_file:
    #     json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_hicplus.h5")
    print("Saved model to disk")

    # loaded_model_json = open('v1/model.json', 'r').read()
    # loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    # model.load_weights("model_v1.01.h5")
    # print("Loaded model from disk")

    predict_all(model, ['chr1', 'chr2', 'chr3'])
