"""
Predict the contact maps for IMR-90 chromosome 1
"""
from local_process_epi import load_bigwig_for_one_region, load_bedGraph_for_one_region
from load_input_HiC import strata_from_a_processed_file
from model import build_model, set_args


def IMR90_pepline():
    # Step 1: process bigWig files and save as numpy array
    epigenetic_features = ['ATAC_seq', 'CTCF', 'H3K4me1', 'H3K4me3',
                           'H3K9ac', 'H3K27ac', 'H3K27me3', 'H3K36me3']
    path = '/nfs/turbo/umms-drjieliu/proj/4dn/data/Epigenomic_Data/IMR90'
    for epi in epigenetic_features:
        if epi == 'ATAC_seq':
            load_bedGraph_for_one_region(f'{path}/IMR90_{epi}_hg19.bedGraph', 'chr1', 0, 248956000, resolution=200,
                                         output_path=f'IMR90_chr1_{epi}_200bp.npy')
        else:
            load_bigwig_for_one_region(f'{path}/IMR90_{epi}_hg19.bigWig', 'chr1', 0, 248956000, resolution=200,
                                       output_path=f'IMR90_chr1_{epi}_200bp.npy')

    # Step 2: Load Hi-C data (pre-processed by JuiceTools.jar) into memory (about 500 Mb memory)
    # Hi-C data are in 1 kb resolution
    # Store as diagonal lines, in which the indices indicate its distance with the exact diagonal
    strata = strata_from_a_processed_file('chr1', 'IMR_chr1_100bp.txt', 248956000,
                                          norm=True, resolution=1000, n_strata=250)
    # Load first 250 diagonal lines (i.e. max_distance = 250kb)

    # Step 3: Build model
    args = set_args()  # Use default values
    model = build_model(args)

    # Step 4: predict
    # 沿着对角线不断yield有overlap的小方块区域，一个个区域扔进模型去跑predict
    # 

