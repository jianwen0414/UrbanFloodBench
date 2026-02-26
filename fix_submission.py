import pandas as pd
import numpy as np
import time

def fix_csv(input_path, output_path):
    print(f"Loading {input_path}...")
    start_time = time.time()
    
    # Optimize memory usage by specifyingdtypes
    dtypes = {
        'row_id': np.int32,
        'model_id': np.int8,
        'event_id': np.int8,
        'node_type': np.int8,
        'node_id': np.int32,
        'water_level': np.float32
    }
    df = pd.read_csv(input_path, dtype=dtypes)
    print(f"Loaded {len(df)} rows in {time.time() - start_time:.2f} seconds.")
    
    print("Sorting values internally...")
    start_time = time.time()
    # Use stable sort (mergesort) to maintain the existing chronological order
    # while interleaving 1D and 2D predictions properly per event.
    df.sort_values(by=['model_id', 'event_id', 'node_type'], kind='mergesort', inplace=True)
    print(f"Sorted in {time.time() - start_time:.2f} seconds.")
    
    print("Resetting row_id...")
    start_time = time.time()
    df['row_id'] = np.arange(len(df), dtype=np.int32)
    print(f"Row IDs reset in {time.time() - start_time:.2f} seconds.")
    
    print(f"Saving to {output_path}...")
    start_time = time.time()
    df.to_csv(output_path, index=False)
    print(f"Saved in {time.time() - start_time:.2f} seconds.")
    print("Finished successfully!")

if __name__ == '__main__':
    fix_csv('submission_full_with_1d.csv', 'submission_full_with_1d_fixed.csv')
