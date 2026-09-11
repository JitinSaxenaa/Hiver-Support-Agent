import os
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import time
import argparse
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Set
from sentence_transformers import SentenceTransformer

INDEX_CORPUS_PATH = 'data/processed/retrieval_index_corpus.csv'
EMBEDDINGS_CACHE_PATH = 'models/retrieval_embeddings.npy'
METADATA_CACHE_PATH = 'models/retrieval_metadata.csv'

_model: Optional[SentenceTransformer] = None
_index_embeddings: Optional[np.ndarray] = None
_index_df: Optional[pd.DataFrame] = None

def sanitize_reply_for_display(text: str) -> str:
    cleaned = re.sub(r'^(?:@\w+\s*)+', '', str(text)).strip()
    cleaned = re.sub(r'https?://\S+', '', cleaned).strip()
    return re.sub(r'\s{2,}', ' ', cleaned)

def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def build_or_load_index(corpus_path: str = INDEX_CORPUS_PATH, force_rebuild: bool = False, max_docs: int = 15000):
    global _index_embeddings, _index_df
    
    if not force_rebuild and os.path.isfile(EMBEDDINGS_CACHE_PATH) and os.path.isfile(METADATA_CACHE_PATH):
        _index_embeddings = np.load(EMBEDDINGS_CACHE_PATH)
        _index_df = pd.read_csv(METADATA_CACHE_PATH)
        return _index_embeddings, _index_df

    print(f'Building semantic retrieval index from {corpus_path}...')
    t0 = time.time()
    
    if not os.path.isfile(corpus_path):
        from src.threading import load_grounding_corpus
        load_grounding_corpus()
        
    df = pd.read_csv(corpus_path)
    if len(df) > max_docs:
        print(f'Subsampling index corpus to {max_docs:,} for optimal latency budget...')
        df = df.iloc[:max_docs].reset_index(drop=True)

    model = get_embedding_model()
    texts = df['inbound_text'].astype(str).tolist()
    
    print(f'Encoding {len(texts):,} inbound customer inquiries with all-MiniLM-L6-v2...')
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=128, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    os.makedirs('models', exist_ok=True)
    np.save(EMBEDDINGS_CACHE_PATH, embeddings)
    df.to_csv(METADATA_CACHE_PATH, index=False)

    _index_embeddings = embeddings
    _index_df = df
    print(f'Retrieval index built and cached successfully in {time.time()-t0:.2f}s.')
    return _index_embeddings, _index_df

def retrieve_similar_threads(
    query_text: str,
    top_k: int = 3,
    exclude_tweet_ids: Optional[Set[int]] = None,
    exclude_exact_text: bool = True,
    min_similarity: float = 0.20
) -> List[Dict[str, Any]]:
    global _index_embeddings, _index_df
    if _index_embeddings is None or _index_df is None:
        build_or_load_index()

    model = get_embedding_model()
    query_vec = model.encode([query_text], show_progress_bar=False, normalize_embeddings=True)[0]
    query_vec = np.asarray(query_vec, dtype=np.float32)

    # Cosine similarity via dot product (both are L2 normalized)
    scores = np.dot(_index_embeddings, query_vec)

    # Top candidates indices
    sorted_indices = np.argsort(-scores)

    results = []
    exclude_set = set(exclude_tweet_ids) if exclude_tweet_ids else set()
    cleaned_query = query_text.strip().lower()

    for idx in sorted_indices:
        if len(results) >= top_k:
            break
        row = _index_df.iloc[idx]
        tid = int(row['inbound_tweet_id'])

        # LEAVE-ONE-OUT LEAKAGE GUARDS:
        # 1. Exclude tweet id if in excluded set
        if tid in exclude_set:
            continue
        # 2. Exclude near-duplicate or exact text matches
        row_text = str(row['inbound_text']).strip().lower()
        if exclude_exact_text and (row_text == cleaned_query or (len(cleaned_query) > 20 and cleaned_query in row_text)):
            continue

        score = float(scores[idx])
        if score < min_similarity and len(results) > 0:
            break

        results.append({
            'inbound_tweet_id': tid,
            'inbound_text': str(row['inbound_text']),
            'brand_reply_text': sanitize_reply_for_display(row['brand_reply_text']),
            'similarity_score': round(score, 4)
        })

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild', action='store_true')
    args = parser.parse_args()

    build_or_load_index(force_rebuild=args.rebuild)
    sample_query = '@AppleSupport my battery on iOS 11 is draining super fast on iPhone 8'
    safe_q = sample_query.encode('ascii', 'replace').decode('ascii')
    print(f'Testing retrieval for query: {safe_q}')
    matches = retrieve_similar_threads(sample_query, top_k=3)
    for i, m in enumerate(matches, 1):
        score = m['similarity_score']
        in_t = m['inbound_text'].encode('ascii', 'replace').decode('ascii')
        br_t = m['brand_reply_text'].encode('ascii', 'replace').decode('ascii')
        print(f'Match {i} (Similarity: {score}):')
        print(f'  Inbound: {in_t}')
        print(f'  Brand Reply: {br_t}\n')
