import os
import json
from typing import List, Dict, Tuple
import numpy as np
import google.generativeai as genai
from openai import OpenAI as OpenAIClient

# Simple local knowledge base stored as JSONL with embeddings
KB_PATH = os.getenv("KB_INDEX_PATH", "kb_index.jsonl")


def _load_kb() -> List[Dict]:
    if not os.path.exists(KB_PATH):
        return []
    items: List[Dict] = []
    with open(KB_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def _save_kb(items: List[Dict]) -> None:
    tmp_path = KB_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    os.replace(tmp_path, KB_PATH)


def _ensure_embedding(text: str, embed_model: str) -> List[float]:
    provider = os.getenv("AI_PROVIDER", "gemini").lower()
    try:
        if provider == "openai":
            client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
            resp = client.embeddings.create(model=embed_model, input=[text or ""])
            return (resp.data[0].embedding if resp and resp.data else []) or []
        # default gemini
        res = genai.embed_content(model=embed_model, content=text or "")
        emb = res.get("embedding") or (res.get("data", {}) or {}).get("embedding")
        if isinstance(emb, dict) and "values" in emb:
            emb = emb.get("values")
        return emb or []
    except Exception:
        return []


def upsert_kb(items_to_add: List[Tuple[str, str]], embed_model: str) -> None:
    """
    items_to_add: list of (id, text)
    """
    existing = _load_kb()
    id_to_idx = {obj.get("id"): i for i, obj in enumerate(existing)}
    changed = False
    for _id, _text in items_to_add:
        if not _id:
            continue
        record = {"id": _id, "text": _text or ""}
        idx = id_to_idx.get(_id)
        if idx is None:
            record["embedding"] = _ensure_embedding(record["text"], embed_model)
            existing.append(record)
            changed = True
        else:
            if (existing[idx].get("text") or "") != record["text"]:
                existing[idx]["text"] = record["text"]
                existing[idx]["embedding"] = _ensure_embedding(record["text"], embed_model)
                changed = True
    if changed:
        _save_kb(existing)


def retrieve_context_for_text(text: str, top_k: int, max_chars: int, embed_model: str) -> str:
    """
    Returns a concatenated context string from top_k similar KB entries.
    """
    kb = _load_kb()
    if not kb:
        return ""
    query_emb = _ensure_embedding(text or "", embed_model)
    if not query_emb:
        return ""
    emb_mat = np.array([it.get("embedding") or [] for it in kb], dtype=float)
    if emb_mat.size == 0 or emb_mat.shape[1] == 0:
        return ""
    q = np.array(query_emb, dtype=float)
    if q.shape[0] != emb_mat.shape[1]:
        # dimension mismatch; skip retrieval
        return ""
    # cosine similarity
    qn = q / (np.linalg.norm(q) + 1e-8)
    norms = np.linalg.norm(emb_mat, axis=1, keepdims=True) + 1e-8
    emb_norm = emb_mat / norms
    sim = emb_norm.dot(qn)
    idxs = np.argsort(-sim)[: max(1, top_k)]
    pieces: List[str] = []
    for i in idxs:
        rec = kb[int(i)]
        pieces.append(f"{rec.get('id')}: {rec.get('text','')}")
    context = "\n---\n".join(pieces)
    if len(context) > max_chars:
        context = context[:max_chars]
    return context


