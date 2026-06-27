from pathlib import Path
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

INDEX_NAME = "arxiv-papers"
DIMENSION = 768
METRIC = "cosine"
CLOUD = "aws"
REGION = "us-east-1"

def get_pinecone_index():
    load_dotenv(ENV_FILE)

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("У файлі .env не знайдено PINECONE_API_KEY")

    pc = Pinecone(api_key=api_key)

    existing_indexes = list(pc.list_indexes().names())
    print("Наявні індекси:", existing_indexes)

    if INDEX_NAME not in existing_indexes:
        print(f"Індекс '{INDEX_NAME}' не знайдено. Створюємо...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(
                cloud=CLOUD,
                region=REGION
            )
        )
        print(f"Індекс '{INDEX_NAME}' створено.")
    else:
        print(f"Індекс '{INDEX_NAME}' вже існує.")

    index = pc.Index(INDEX_NAME)
    print(f"Підключення до індексу '{INDEX_NAME}' успішне.")
    return index

if __name__ == "__main__":
    index = get_pinecone_index()