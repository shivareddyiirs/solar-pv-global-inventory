import os,glob, pickle
import numpy as np
from random import shuffle
def make_records(directory):
    """
    Makes a Pickle of a list of record dicts storing data and meta information
    """
    npz_files = glob.glob(os.path.join(directory, '*.npz'))
    meta_files = glob.glob(os.path.join(directory,'*.geojson'))

    records = []
    for npz in npz_files:
        ii = npz.split('/')[-1].split('.')[0]
        meta = [m for m in meta_files if m.split('/')[-1].split('.')[0]==ii][0]
        records.append({'data':npz, 'meta':meta})
    print("Total records are "+str(len(records)))
    shuffle(records)
    pickle.dump(records, open(os.path.join(directory,'records.pickle'),'wb'))
    print("testing the sample records")
    trn_records=pickle.load(open(os.path.join(directory,'records.pickle'),'rb'))
    val_indexes = [1,2]
    val_records = [rr for ii,rr in enumerate(trn_records) if ii in val_indexes]
    for record in val_records:
        data=record["data"]
        meta=record["meta"]
        test=np.load(data)
        print(type(test['data']))
   
if __name__ == "__main__":
    make_records("C:\\hpc\\data\\training\\S2_unet2")
