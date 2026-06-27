from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "arxiv_subset.parquet"
OUTPUT_DIR = BASE_DIR / "embeddings"
OUTPUT_FILE = OUTPUT_DIR / "embeddings.npy"
MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 64

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Не знайдено файл датасету: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_FILE)

    texts = (
        df["title"].fillna("").astype(str).str.strip()
        + " [SEP] "
        + df["abstract"].fillna("").astype(str).str.strip()
    ).tolist()

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print(f"\nЗагальна кількість оброблених текстів: {len(texts)}")
    print(f"Розмірність ембеддингів: {embeddings.shape[1]}")
    print(f"Норма першого ембеддингу: {np.linalg.norm(embeddings[0]):.6f}")

    np.save(OUTPUT_FILE, embeddings)
    print(f"Ембеддинги збережено у: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()