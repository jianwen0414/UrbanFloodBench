"""
FloodDataset — Universal Lazy Loader.

Responsible for:
    * Discovering Model / Event folders under RAW_DATA_PATH.
    * Lazy-loading dynamic CSV files per event on __getitem__.
    * Caching static files (e.g. 1d_nodes_static.csv) in memory
      after first read to minimise RAM & I/O overhead.
"""
