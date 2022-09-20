import sys
sys.path.append('..')

from se3_transformer.data_loading.qm9 import OGBDataset

dataset = OGBDataset(data_dir='/sharefs/baaihealth/yuancheng/unimap/pcqm4m-v2-train.sdf')

def getitem(i,j):
    try:
        mol = dataset[j]
        return None, None
    except:
        return i, j


import multiprocessing
pool = multiprocessing.Pool(16)
import rdkit

error_list = []
index_list = []
from tqdm import tqdm
for i, j in tqdm(pool.imap(getitem, enumerate(range(3378606)), chunksize=100),total=100):
    if i != None:
        index_list.append(i)
        error_list.append(j)