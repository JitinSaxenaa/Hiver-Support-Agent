import pandas as pd
import numpy as np
import re
import time

csv_path = 'data/raw/twcs/twcs.csv'
candidates = ['AppleSupport', 'AmazonHelp', 'SpotifyCares', 'Delta', 'Uber_Support']

print('Loading dataset for candidate analysis...')
start_time = time.time()

# Load full dataset into dataframe with essential columns
df = pd.read_csv(csv_path, dtype={
    'tweet_id': 'int64',
    'author_id': 'str',
    'inbound': 'bool',
    'created_at': 'str',
    'text': 'str',
    'response_tweet_id': 'str',
    'in_response_to_tweet_id': 'float64'
})

print(f'Loaded {len(df):,} rows in {time.time() - start_time:.2f}s')

# Create index of tweet_id to row for fast lookup
tweet_id_map = dict(zip(df['tweet_id'], df.index))

def is_english_approx(text):
    if not isinstance(text, str):
        return False
    # Check ASCII printable character ratio
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return (ascii_chars / max(len(text), 1)) > 0.85

results = {}

for brand in candidates:
    print(f'\n--- Analyzing {brand} ---')
    # Brand outbound tweets
    brand_tweets = df[(df['author_id'] == brand) & (~df['inbound'])]
    brand_outbound_count = len(brand_tweets)
    
    # Inbound tweets replying to or mentioning brand
    # Parent tweets that the brand replied to
    parent_ids = brand_tweets['in_response_to_tweet_id'].dropna().astype('int64')
    valid_parent_ids = [pid for pid in parent_ids if pid in tweet_id_map]
    
    parent_indices = [tweet_id_map[pid] for pid in valid_parent_ids]
    parent_df = df.iloc[parent_indices]
    
    # Check English ratio of inbound parents
    english_parents = sum(parent_df['text'].apply(is_english_approx))
    english_ratio = english_parents / max(len(parent_df), 1)
    
    # Estimate total inbound volume directed to this brand
    # In twcs dataset, inbound tweets mentioning @brand
    # We can check tweets that received a reply from brand
    # Thread depth: how many tweets are in the conversation
    # For sample of 500 brand tweets, trace thread backwards
    sample_parents = parent_df.head(500)
    depths = []
    for idx, row in sample_parents.iterrows():
        depth = 1
        curr_in_reply = row['in_response_to_tweet_id']
        while pd.notna(curr_in_reply) and int(curr_in_reply) in tweet_id_map and depth < 10:
            depth += 1
            curr_row = df.iloc[tweet_id_map[int(curr_in_reply)]]
            curr_in_reply = curr_row['in_response_to_tweet_id']
        depths.append(depth)
    avg_depth = np.mean(depths) if depths else 1.0
    
    results[brand] = {
        'brand_outbound_replies': brand_outbound_count,
        'matched_inbound_parents': len(valid_parent_ids),
        'english_ratio': round(english_ratio * 100, 2),
        'avg_thread_depth': round(float(avg_depth), 2)
    }
    print(f'  Outbound replies: {brand_outbound_count:,}')
    print(f'  Matched inbound inquiries: {len(valid_parent_ids):,}')
    print(f'  English %: {english_ratio * 100:.1f}%')
    print(f'  Avg thread depth: {avg_depth:.2f}')

import json
with open('data/processed/candidate_brands_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\nCandidate analysis saved to data/processed/candidate_brands_metrics.json')
