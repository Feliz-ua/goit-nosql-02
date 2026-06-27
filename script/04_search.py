from pathlib import Path
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

DATA_FILE = BASE_DIR / "data" / "arxiv_subset.parquet"
EMB_FILE = BASE_DIR / "embeddings" / "embeddings.npy"
ENV_FILE = BASE_DIR / ".env"


def load_resources():
    load_dotenv(ENV_FILE)

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("Не знайдено PINECONE_API_KEY у .env")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)
    model = SentenceTransformer(MODEL_NAME)
    df = pd.read_parquet(DATA_FILE)
    embeddings = np.load(EMB_FILE)

    return index, model, df, embeddings


def encode_query(model, query: str) -> np.ndarray:
    emb = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return emb.astype(np.float32)


def print_pinecone_results(title: str, results):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    matches = results.get("matches", [])
    if not matches:
        print("Нічого не знайдено.")
        return

    for i, match in enumerate(matches, start=1):
        meta = match.get("metadata", {})
        abstract = meta.get("abstract", "")
        snippet = abstract[:200].replace("\n", " ")

        print(f"{i}. {meta.get('title', 'н/д')}")
        print(f"   оцінка: {match.get('score', 'н/д')}")
        print(f"   категорія: {meta.get('category', 'н/д')}")
        print(f"   рік: {meta.get('year', 'н/д')}")
        print(f"   анотація: {snippet}...")
        print()


def semantic_search(index, query_vector, top_k=TOP_K, search_filter=None):
    results = index.query(
        vector=query_vector.tolist(),
        top_k=top_k,
        include_metadata=True,
        filter=search_filter
    )
    return results


def top_k_indices_desc(scores: np.ndarray, k: int):
    idx = np.argsort(scores)[::-1][:k]
    return idx


def top_k_indices_asc(scores: np.ndarray, k: int):
    idx = np.argsort(scores)[:k]
    return idx


def print_local_results(header: str, indices, scores, df):
    print("\n" + "-" * 100)
    print(header)
    print("-" * 100)

    for rank, idx in enumerate(indices, start=1):
        row = df.iloc[idx]
        abstract = str(row["abstract"])[:200].replace("\n", " ")
        print(f"{rank}. {row['title']}")
        print(f"   оцінка: {scores[idx]:.6f}")
        print(f"   категорія: {row['category']}")
        print(f"   рік: {row['year']}")
        print(f"   анотація: {abstract}...")
        print()


def compare_local_metrics(query_vector, embeddings, df, top_k=TOP_K):
    cosine_scores = embeddings @ query_vector
    dot_scores = embeddings @ query_vector
    l2_scores = np.linalg.norm(embeddings - query_vector, axis=1)

    cosine_idx = top_k_indices_desc(cosine_scores, top_k)
    dot_idx = top_k_indices_desc(dot_scores, top_k)
    l2_idx = top_k_indices_asc(l2_scores, top_k)

    print_local_results("Локальний пошук: Cosine similarity", cosine_idx, cosine_scores, df)
    print_local_results("Локальний пошук: Dot product", dot_idx, dot_scores, df)
    print_local_results("Локальний пошук: L2 distance", l2_idx, l2_scores, df)

    print("\nПорівняння індексів:")
    print("Cosine top-5:", cosine_idx.tolist())
    print("Dot top-5   :", dot_idx.tolist())
    print("L2 top-5    :", l2_idx.tolist())


def main():
    index, model, df, embeddings = load_resources()

    query_1 = "teaching machines to recognize objects in pictures"
    qvec_1 = encode_query(model, query_1)

    results_1 = semantic_search(index, qvec_1, top_k=TOP_K)
    print_pinecone_results(
        f"Чистий семантичний пошук для запиту: {query_1}",
        results_1
    )

    query_2 = "reinforcement learning"
    qvec_2 = encode_query(model, query_2)

    current_year = pd.Timestamp.now().year
    filter_a = {
        "$and": [
            {"category": {"$eq": "cs.LG"}},
            {"year": {"$gte": current_year - 5}}
        ]
    }

    results_a = semantic_search(index, qvec_2, top_k=TOP_K, search_filter=filter_a)
    print_pinecone_results(
        f"Фільтр A: reinforcement learning, category=cs.LG, year>={current_year - 5}",
        results_a
    )

    filter_b = {"year": {"$lt": 2015}}
    results_b = semantic_search(index, qvec_2, top_k=TOP_K, search_filter=filter_b)
    print_pinecone_results(
        "Фільтр B: reinforcement learning, year < 2015, будь-яка категорія",
        results_b
    )

    compare_local_metrics(qvec_1, embeddings, df, top_k=TOP_K)

#   print("\nПояснення:")
#   print("- Фільтр A обмежує пошук сучасними статтями з cs.LG, тому результати більш вузькі й тематично однорідні.")
#   print("- Фільтр B дозволяє старіші роботи з будь-яких категорій, тому видача може бути історично ширшою та менш однорідною.")
#   print("- Для нормалізованих ембеддингів cosine similarity і dot product зазвичай дають однаковий рейтинг.")
#   print("- L2 distance може дати той самий або дуже схожий рейтинг для нормалізованих векторів, але інтерпретується як відстань, а не подібність.")


if __name__ == "__main__":
    main()