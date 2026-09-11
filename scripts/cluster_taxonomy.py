import os
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import sys
import pandas as pd
import numpy as np
import time
import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

print('Loading grounding corpus sample for clustering...')
df = pd.read_csv('data/processed/grounding_corpus.csv')
sample_df = df.sample(n=1500, random_state=42).reset_index(drop=True)

print('Loading SentenceTransformer all-MiniLM-L6-v2...')
t0 = time.time()
model = SentenceTransformer('all-MiniLM-L6-v2')
print(f'Model loaded in {time.time()-t0:.2f}s')

texts = sample_df['inbound_text'].tolist()
print('Encoding texts...')
embeddings = model.encode(texts, show_progress_bar=False, batch_size=64)

k = 8
print(f'Fitting KMeans with k={k}...')
kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
labels = kmeans.fit_predict(embeddings)
sample_df['cluster'] = labels

# Find exemplars closest to each centroid
cluster_results = {}
for i in range(k):
    cluster_indices = np.where(labels == i)[0]
    cluster_embeddings = embeddings[cluster_indices]
    centroid = kmeans.cluster_centers_[i]
    distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
    closest_indices = cluster_indices[np.argsort(distances)[:5]]
    exemplars = sample_df.iloc[closest_indices]['inbound_text'].tolist()
    cluster_results[f'Cluster_{i}'] = {
        'count': int(len(cluster_indices)),
        'exemplars': exemplars
    }
    safe_exs = [ex.encode('ascii', 'replace').decode('ascii') for ex in exemplars[:3]]
    print(f'\n--- Cluster {i} ({len(cluster_indices)} samples) ---')
    for ex in safe_exs:
        print(f'  * {ex}')

with open('data/processed/cluster_exemplars.json', 'w', encoding='utf-8') as f:
    json.dump(cluster_results, f, indent=2, ensure_ascii=False)
print('\nClustering analysis saved to data/processed/cluster_exemplars.json')
