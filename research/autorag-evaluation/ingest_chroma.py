import asyncio
import pandas as pd
from autorag.vectordb import load_vectordb_from_yaml


async def main():
    # Load corpus
    df = pd.read_parquet("data/aiops_corpus.parquet")

    print(f"Corpus documents: {len(df)}")

    # Load Chroma
    db = load_vectordb_from_yaml(
        "resources/chroma/vectordb.yaml",
        "local",
        "."
    )

    ids = df["doc_id"].astype(str).tolist()
    texts = df["contents"].fillna("").astype(str).tolist()

    # Add documents and generate BGE embeddings
    await db.add(ids=ids, texts=texts)

    print("Documents indexed successfully.")
    print("Chroma document count:", db.collection.count())


if __name__ == "__main__":
    asyncio.run(main())
