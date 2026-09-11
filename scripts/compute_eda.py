import pandas as pd
import numpy as np
from collections import Counter
import json
import time

print("Starting EDA on twcs.csv...")
start_time = time.time()

# Let's count authors where inbound == False (these are official brand handles)
brand_counts = Counter()
total_rows = 0

chunk_size = 250000
csv_path = 'data/raw/twcs/twcs.csv'

for chunk in pd.read_csv(csv_path, chunksize=chunk_size, usecols=['tweet_id', 'author_id', 'inbound', 'created_at', 'text', 'response_tweet_id', 'in_response_to_tweet_id']):
    total_rows += len(chunk)
    brand_mask = (chunk['inbound'] == False)
    brand_counts.update(chunk.loc[brand_mask, 'author_id'].value_counts().to_dict())
    print(f"Processed {total_rows} rows so far...")

print(f"\nTotal rows in dataset: {total_rows}")
print("\nTop 15 brands by outbound reply volume:")
for brand, count in brand_counts.most_common(15):
    print(f"  {brand}: {count:,} tweets")

# Let's inspect the top 8 brands in detail
top_brands = [b for b, _ in brand_counts.most_common(8)]
print(f"\nAnalyzing top brands: {top_brands}")

# Save brand counts summary to JSON
with open('data/processed/eda_brand_summary.json', 'w') as f:
    json.dump({
        'total_rows': total_rows,
        'top_brands': dict(brand_counts.most_common(20))
    }, f, indent=2)

print(f"EDA pass 1 completed in {time.time() - start_time:.2f} seconds.")
