import os
import sys
import gzip
sys.path.insert(0, '../')
print(sys.path)
from hic_enhancer import pairs_to_processed_files


file = '/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/human/raw/H1-hESC_raw.pairs.gz'


# chroms = [f'chr{i}' for i in list(range(1, 20)) + ['X']]
chroms = ['chr20', 'chr21', 'chr22']
paths = {c: f'/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/human/processed/hic_contacts/{c}_200bp.txt' for c in chroms}
pairs_to_processed_files(file, paths, zip=True)


