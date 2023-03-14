
conda env 

1. Download and extract qm9_crude.msgpack from https://dataverse.harvard.edu/file.xhtml?fileId=4327190&version=4.0
2. Place in data/qm9/raw
3. Run `python data/qm9/preprocess.py`

1. Download and extract drugs_crude.msgpack from https://dataverse.harvard.edu/file.xhtml?fileId=4360331&version=4.0
2. Place in data/geom/raw
3. Run `python data/geom/preprocess.py`

Alternatively, create a symlink:
```
ln -s /path/to/qm9_crude.msgpack data/qm9/raw/qm9_crude.msgpack
ln -s /path/to/drugs_crude.msgpack data/geom/raw/drugs_crude.msgpack
```

The preprocessing caches the entire dataset in a .npy file, which is much faster to load than .msgpack files.
