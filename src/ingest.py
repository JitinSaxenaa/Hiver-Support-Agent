import os
import sys
import glob
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def get_raw_data_path():
    possible_paths = [
        'data/raw/twcs/twcs.csv',
        'data/raw/twcs.csv',
        '../data/raw/twcs/twcs.csv',
        '../data/raw/twcs.csv'
    ]
    for p in possible_paths:
        if os.path.isfile(p):
            return p
    return None

def download_dataset_if_needed():
    raw_path = get_raw_data_path()
    if raw_path is not None:
        return raw_path
    
    print('Raw dataset not found locally. Checking Kaggle credentials...')
    kaggle_user = os.getenv('KAGGLE_USERNAME')
    kaggle_key = os.getenv('KAGGLE_KEY')
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    
    if not ((kaggle_user and kaggle_key) or kaggle_json.exists()):
        raise RuntimeError(
            'Cannot download dataset: No Kaggle credentials found!\n'
            'Please either:\n'
            ' 1. Set KAGGLE_USERNAME and KAGGLE_KEY in your .env file, or\n'
            ' 2. Place ~/.kaggle/kaggle.json on your machine, or\n'
            ' 3. Manually download thoughtvector/customer-support-on-twitter and extract twcs.csv into data/raw/twcs/twcs.csv'
        )
    
    print('Downloading Kaggle dataset thoughtvector/customer-support-on-twitter...')
    os.makedirs('data/raw', exist_ok=True)
    import subprocess
    cmd = [sys.executable, '-m', 'kaggle', 'datasets', 'download', 'thoughtvector/customer-support-on-twitter', '-p', 'data/raw', '--unzip']
    subprocess.run(cmd, check=True)
    
    raw_path = get_raw_data_path()
    if raw_path is None:
        raise FileNotFoundError('Failed to locate twcs.csv after Kaggle download.')
    return raw_path

def ingest_brand_data(brand='AppleSupport', subsample_size=25000, seed=42, force=False):
    os.makedirs('data/processed', exist_ok=True)
    cached_path = f'data/processed/{brand.lower()}_raw_subsample.csv'
    
    if os.path.isfile(cached_path) and not force:
        print(f'Loading cached brand dataset from {cached_path}...')
        return pd.read_csv(cached_path)
    
    raw_path = download_dataset_if_needed()
    print(f'Ingesting data for brand [{brand}] from {raw_path}...')
    t0 = time.time()
    
    chunks = []
    chunk_size = 250000
    total_scanned = 0
    
    # Ingest rows relevant to the brand
    for chunk in pd.read_csv(raw_path, chunksize=chunk_size, low_memory=False):
        total_scanned += len(chunk)
        mask = (chunk['author_id'] == brand) | (chunk['text'].str.contains(f'@{brand}', case=False, na=False))
        matched = chunk[mask]
        if not matched.empty:
            chunks.append(matched)
            
    df_brand = pd.concat(chunks, ignore_index=True)
    print(f'Scanned {total_scanned:,} total rows; found {len(df_brand):,} rows for @{brand} in {time.time()-t0:.2f}s.')
    
    # Save raw brand slice for threading
    full_brand_path = f'data/processed/{brand.lower()}_full_raw.csv'
    df_brand.to_csv(full_brand_path, index=False)
    print(f'Saved full brand raw slice ({len(df_brand):,} rows) to {full_brand_path}')
    return df_brand

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ingest customer support data for a specific brand')
    parser.add_argument('--brand', type=str, default=os.getenv('BRAND_NAME', 'AppleSupport'))
    parser.add_argument('--subsample', type=int, default=int(os.getenv('SUBSAMPLE_SIZE', 25000)))
    parser.add_argument('--seed', type=int, default=int(os.getenv('RANDOM_SEED', 42)))
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    
    ingest_brand_data(brand=args.brand, subsample_size=args.subsample, seed=args.seed, force=args.force)
