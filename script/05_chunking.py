import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "arxiv_subset.parquet"
ENV_FILE = BASE_DIR / ".env"

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768

INDEX_FIXED = "arxiv-chunks-fixed"
INDEX_SEMANTIC = "arxiv-chunks-semantic"

TOP_K = 5
UPSERT_BATCH_SIZE = 100
EMBED_BATCH_SIZE = 64

CHUNK_MAX_WORDS = 120
OVERLAP_WORDS = 20
TOP_LONGEST = 30


load_dotenv(ENV_FILE)

api_key = os.environ.get("PINECONE_API_KEY")
if not api_key:
    raise ValueError("Не знайдено PINECONE_API_KEY у .env")

pc = Pinecone(api_key=api_key)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet(DATA_FILE)


def ensure_index(index_name: str):
    existing = list(pc.list_indexes().names())
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Створено індекс: {index_name}")
    else:
        print(f"Індекс уже існує: {index_name}")

    return pc.Index(index_name)


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def split_sentences(text: str):
    text = normalize_spaces(text)
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def fixed_size_chunking(text: str, chunk_size=CHUNK_MAX_WORDS, overlap=OVERLAP_WORDS):
    words = re.findall(r"\S+", normalize_spaces(text))
    if not words:
        return []

    chunks = []
    step = max(1, chunk_size - overlap)

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if not chunk_words:
            continue
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break

    return chunks


def semantic_chunking(text: str, max_words=CHUNK_MAX_WORDS):
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        sent_words = sentence.split()
        sent_len = len(sent_words)

        if sent_len > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk).strip())
                current_chunk = []
                current_len = 0

            for i in range(0, sent_len, max_words):
                part = " ".join(sent_words[i:i + max_words]).strip()
                if part:
                    chunks.append(part)
            continue

        if current_len + sent_len <= max_words:
            current_chunk.append(sentence)
            current_len += sent_len
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk).strip())
            current_chunk = [sentence]
            current_len = sent_len

    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return chunks


def build_chunk_records(df_subset: pd.DataFrame, strategy_name: str):
    all_records = []

    for _, row in df_subset.iterrows():
        abstract = normalize_spaces(row["abstract"])
        title = normalize_spaces(row["title"])

        if strategy_name == "fixed":
            chunks = fixed_size_chunking(abstract)
        elif strategy_name == "semantic":
            chunks = semantic_chunking(abstract)
        else:
            raise ValueError("Невідома стратегія chunking")

        for chunk_num, chunk_text in enumerate(chunks):
            all_records.append({
                "vector_id": f"{strategy_name}_{row['id']}_{chunk_num}",
                "text_for_embedding": f"{title} [SEP] {chunk_text}",
                "metadata": {
                    "arxiv_id": str(row["id"]),
                    "title": title[:500],
                    "chunk_text": chunk_text[:1000],
                    "chunk_num": int(chunk_num),
                    "year": int(row["year"]),
                    "category": str(row["category"]),
                }
            })

    return all_records


def embed_texts(texts):
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return embeddings.astype(np.float32)


def upsert_chunk_records(index, index_name: str, records, batch_size=UPSERT_BATCH_SIZE):
    if not records:
        print(f"Немає записів для завантаження в {index_name}")
        return

    texts = [r["text_for_embedding"] for r in records]
    embeddings = embed_texts(texts)

    for start in tqdm(
        range(0, len(records), batch_size),
        desc=f"Upsert -> {index_name}"
    ):
        end = min(start + batch_size, len(records))
        batch = []

        for i in range(start, end):
            batch.append({
                "id": records[i]["vector_id"],
                "values": embeddings[i].tolist(),
                "metadata": records[i]["metadata"]
            })

        index.upsert(vectors=batch)

    print(f"Завантажено чанків в {index_name}: {len(records)}")


def search_chunks(index, query: str, top_k=TOP_K):
    query_vec = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)

    results = index.query(
        vector=query_vec.tolist(),
        top_k=top_k,
        include_metadata=True
    )
    return results


def print_search_results(header: str, results):
    print("\n" + "=" * 100)
    print(header)
    print("=" * 100)

    matches = results.get("matches", [])
    if not matches:
        print("Нічого не знайдено.")
        return

    for i, match in enumerate(matches, start=1):
        meta = match.get("metadata", {})
        snippet = str(meta.get("chunk_text", ""))[:220].replace("\n", " ")
        print(f"{i}. {meta.get('title', 'N/A')}")
        print(f"   score: {match.get('score', 'N/A')}")
        print(f"   arxiv_id: {meta.get('arxiv_id', 'N/A')}")
        print(f"   chunk_num: {meta.get('chunk_num', 'N/A')}")
        print(f"   year: {meta.get('year', 'N/A')}")
        print(f"   category: {meta.get('category', 'N/A')}")
        print(f"   chunk: {snippet}...")
        print()


def main():
    df_local = df.copy()
    df_local["abstract_len_words"] = df_local["abstract"].astype(str).apply(
        lambda x: len(normalize_spaces(x).split())
    )
    longest_df = df_local.sort_values("abstract_len_words", ascending=False).head(TOP_LONGEST)

    print(f"Вибрано {len(longest_df)} найдовших статей.")
    print(
        longest_df[["id", "title", "abstract_len_words"]]
        .head(10)
        .to_string(index=False)
    )

    fixed_records = build_chunk_records(longest_df, "fixed")
    semantic_records = build_chunk_records(longest_df, "semantic")

    print(f"\nFixed-size chunks: {len(fixed_records)}")
    print(f"Semantic chunks:   {len(semantic_records)}")

    fixed_index = ensure_index(INDEX_FIXED)
    semantic_index = ensure_index(INDEX_SEMANTIC)

    upsert_chunk_records(fixed_index, INDEX_FIXED, fixed_records)
    upsert_chunk_records(semantic_index, INDEX_SEMANTIC, semantic_records)

    test_queries = [
        "deep learning for image recognition",
        "reinforcement learning with policy optimization",
        "graph neural networks for molecular property prediction"
    ]

    for query in test_queries:
        print(f"\n\nТестовий запит: {query}")

        fixed_results = search_chunks(fixed_index, query)
        semantic_results = search_chunks(semantic_index, query)

        print_search_results("Результати для fixed-size chunking", fixed_results)
        print_search_results("Результати для semantic chunking", semantic_results)


if __name__ == "__main__":
    main()