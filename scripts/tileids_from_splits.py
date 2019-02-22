"""
Generate tileids (input to german-based model) from 
data splits (input to our model)
"""
import os
import pickle

country = 'ghana'
setsize = 'full'
splits = ['train', 'val', 'test']
tile_dir = '/home/roserustowicz/MTLCC-all/africa/tileids/ghana'

for split in splits:
    cur_file = '_'.join([country, setsize, split])
    print('cur file: ', cur_file)   
    with open(os.path.join(tile_dir, cur_file), "rb") as f:
        inlist = list(pickle.load(f))
        print(inlist)
        outfname = os.path.join(tile_dir, cur_file + '.tileids')
        with open(outfname, 'w') as outf:
            for item in inlist:
                for append_i in range(9):
                    cur_item = item + str(append_i)
                    outf.write("%s\n" % cur_item)


