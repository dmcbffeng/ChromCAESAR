Code:
    process_data
        - local_process_epi.py: process epigenetic data (bigwig or bedGraph format)
    run_model
        - layers.py: graph convolution layers (our model is based on graph convolution networks)
        - model.py: build the keras model
        - predict.py: generate predictions (when you are running, just use the default args)

Required Packages:
    'tensorflow-gpu==1.13.1'
    'keras==2.2.4'
    'numpy==1.16.0'
    'scipy==1.1.0'



AIM 1: Use the model to predict high-resolution contact maps for 10 human tissues and cell lines:
    Adrenal Gland (AD)
    Aorta (AO)
    Bladder (BL)
    *Fibroblast (IMR90)
    Hippocampus (HC)
    Left Ventricle (LV)
    Lung (LG)
    Ovary (OV)
    Pancreas (PA)
    Right Ventricle (RV)
    Small Bowel (SB)
    Spleen (SX)
    *GM12878


1 Download and process 1-D epigenetic datasets
    Required:
        ATAC_seq, CTCF, H3K4me1, H3K4me3,
        H3K9ac, H3K27ac, H3K27me3, H3K36me3
    Download from ENCODE (https://www.encodeproject.org/matrix/?type=Experiment)
    [find tissue - find epigenetic dataset - "file details" - "Bing Ren from UCSD"]
    "AD_CTCF_hg38.bigWig"
        i. reference genome: GRCh38 (hg38)
        ii. format: bigwig
        iii. replicate: the most
        iv. fold change over control
        v. record the link of download in a .txt file!
        (MUST DOUBLE CHECK)
    process the data with local_process_epi.py
        AD_chr1_CTCF_200bp.npy

2 Download and process Hi-C data
    (See Google drive "human_tissues.docx")
    The downloaded Hi-C might be in different format
    * Start from GM12878 and IMR90
    .hic format: https://github.com/aidenlab/juicer/wiki/Data-Extraction
    >> java -jar JuiceTools/juicer_tools_1.11.04_jcuda.0.8.jar dump observed NONE Downloads/IMR90.hic chr1 chr1 BP 1000 IMR90_chr1_1000bp.txt
        [position 1] [pos 2] [contacts]
        47000	47000	1.0  (47001-48000) - (47001-48000) - 1 contact detected
        51000	52000	1.0
        52000	52000	4.0
    "/nfs/turbo/umms-drjieliu/proj/4dn/data/bulkHiC"


AIM 2: Analyze eQTL data with the high-resolution contact maps


