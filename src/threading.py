import os
import re
import time
import argparse
import pandas as pd
import numpy as np
from src.ingest import ingest_brand_data

def clean_tweet_text(text):
    if not isinstance(text, str):
        return ''
    # Replace multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def reconstruct_threads(df_raw, brand='AppleSupport', subsample_size=25000, seed=42):
    print(f'Reconstructing conversation threads for brand [{brand}]...')
    t0 = time.time()
    
    # Fast index lookup
    df_raw = df_raw.drop_duplicates(subset=['tweet_id'])
    tweet_dict = df_raw.set_index('tweet_id').to_dict('index')
    
    pairs = []
    brand_tweets = df_raw[df_raw['author_id'] == brand]
    
    for _, brand_row in brand_tweets.iterrows():
        parent_id = brand_row['in_response_to_tweet_id']
        if pd.isna(parent_id):
            continue
        parent_id = int(parent_id)
        if parent_id in tweet_dict:
            parent_row = tweet_dict[parent_id]
            # Must be an inbound customer inquiry
            if parent_row.get('inbound') == True:
                inbound_txt = clean_tweet_text(parent_row.get('text', ''))
                brand_txt = clean_tweet_text(brand_row.get('text', ''))
                
                # Exclude trivial/empty messages or non-ASCII spam
                if len(inbound_txt) >= 15 and len(brand_txt) >= 10:
                    pairs.append({
                        'thread_id': f'{parent_id}_{brand_row.name}',
                        'inbound_tweet_id': parent_id,
                        'inbound_author_id': parent_row.get('author_id', ''),
                        'inbound_text': inbound_txt,
                        'inbound_created_at': parent_row.get('created_at', ''),
                        'brand_tweet_id': int(brand_row['tweet_id']),
                        'brand_author_id': brand,
                        'brand_reply_text': brand_txt,
                        'brand_created_at': brand_row.get('created_at', '')
                    })
                    
    df_pairs = pd.DataFrame(pairs)
    print(f'Reconstructed {len(df_pairs):,} valid customer->{brand} conversation pairs in {time.time()-t0:.2f}s.')
    
    # Remove duplicate inbound queries
    df_pairs = df_pairs.drop_duplicates(subset=['inbound_tweet_id'])
    
    # Parse dates to create month stratification
    df_pairs['inbound_datetime'] = pd.to_datetime(df_pairs['inbound_created_at'], errors='coerce')
    df_pairs['month_key'] = df_pairs['inbound_datetime'].dt.to_period('M').astype(str)
    
    # Subsampling with time stratification
    if len(df_pairs) > subsample_size:
        print(f'Subsampling to {subsample_size:,} pairs using time stratification...')
        # Stratified sample by month
        sampled_dfs = []
        month_groups = df_pairs.groupby('month_key')
        for month, group in month_groups:
            frac = len(group) / len(df_pairs)
            n_month = int(np.round(frac * subsample_size))
            if n_month > 0:
                sampled_dfs.append(group.sample(n=min(n_month, len(group)), random_state=seed))
        df_subsample = pd.concat(sampled_dfs, ignore_index=True)
        # Adjust if rounding caused slight mismatch
        if len(df_subsample) > subsample_size:
            df_subsample = df_subsample.sample(n=subsample_size, random_state=seed)
        elif len(df_subsample) < subsample_size:
            remaining = df_pairs[~df_pairs['inbound_tweet_id'].isin(df_subsample['inbound_tweet_id'])]
            extra = remaining.sample(n=subsample_size - len(df_subsample), random_state=seed)
            df_subsample = pd.concat([df_subsample, extra], ignore_index=True)
    else:
        df_subsample = df_pairs
        
    df_subsample = df_subsample.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    out_path = 'data/processed/grounding_corpus.csv'
    os.makedirs('data/processed', exist_ok=True)
    df_subsample.to_csv(out_path, index=False)
    print(f'Successfully saved {len(df_subsample):,} grounding pairs to {out_path}.')
    return df_subsample

def load_grounding_corpus(brand='AppleSupport', subsample_size=25000, seed=42, force=False):
    out_path = 'data/processed/grounding_corpus.csv'
    if os.path.isfile(out_path) and not force:
        print(f'Loading existing grounding corpus from {out_path}...')
        return pd.read_csv(out_path)
    
    df_raw = ingest_brand_data(brand=brand, force=force)
    return reconstruct_threads(df_raw, brand=brand, subsample_size=subsample_size, seed=seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--brand', type=str, default='AppleSupport')
    parser.add_argument('--subsample', type=int, default=25000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    
    load_grounding_corpus(brand=args.brand, subsample_size=args.subsample, seed=args.seed, force=args.force)
