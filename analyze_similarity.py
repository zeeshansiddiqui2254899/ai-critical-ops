import os
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def batched(iterable: List[str], batch_size: int) -> List[List[str]]:
    return [iterable[i : i + batch_size] for i in range(0, len(iterable), batch_size)]


def main() -> None:
    load_dotenv()

    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
    similarity_threshold_str = os.getenv("SIMILARITY_THRESHOLD", "0.85")
    try:
        similarity_threshold = float(similarity_threshold_str)
    except ValueError:
        similarity_threshold = 0.85

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    input_path = "critical_ops_issues.csv"
    if not os.path.exists(input_path):
        raise SystemExit(
            f"Input CSV not found: {input_path}. Run fetch_jira_data.py first."
        )

    df = pd.read_csv(input_path)
    if "summary" not in df.columns or "description" not in df.columns:
        raise SystemExit("CSV missing required columns: summary, description")

    df["description"] = df["description"].fillna("")
    df["combined"] = df["summary"].fillna("") + "\n" + df["description"]

    texts: List[str] = df["combined"].astype(str).tolist()

    # Embed in batches to reduce API round-trips
    embeddings: List[List[float]] = []
    batch_size = 64
    for chunk in tqdm(batched(texts, batch_size), total=max(1, (len(texts) + batch_size - 1) // batch_size), desc="Embedding"):
        response = client.embeddings.create(model=embed_model, input=chunk)
        for item in response.data:
            embeddings.append(item.embedding)

    if len(embeddings) != len(texts):
        raise RuntimeError("Mismatch between number of texts and embeddings")

    emb_matrix = np.array(embeddings, dtype=np.float32)
    sim_matrix = cosine_similarity(emb_matrix)

    clusters: List[List[int]] = []
    visited = set()
    n = len(sim_matrix)
    for i in range(n):
        if i in visited:
            continue
        group = [j for j in range(n) if sim_matrix[i, j] >= similarity_threshold]
        clusters.append(group)
        visited.update(group)

    print(f"Found {len(clusters)} duplicate clusters (threshold={similarity_threshold}).")

    index_to_cluster = {}
    for cluster_id, members in enumerate(clusters):
        for idx in members:
            index_to_cluster[idx] = cluster_id

    df["cluster_id"] = df.index.map(index_to_cluster)

    output_path = "critical_ops_clusters.csv"
    df.to_csv(output_path, index=False)
    print(f"Clusters saved to {output_path}")


if __name__ == "__main__":
    main()



