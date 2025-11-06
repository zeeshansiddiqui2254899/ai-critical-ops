import os
import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials
import gspread
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import re


def batched(lst: List[str], size: int) -> List[List[str]]:
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text or "")
    t = text.lower()
    t = re.sub(r"[\w\.-]+@[\w\.-]+", " ", t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"\b[A-Z]{2,}-\d+\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def classify_feature_error(client: OpenAI, model: str, text: str, feature_labels: List[str] = None, error_labels: List[str] = None) -> Dict[str, str]:
    constraint = ""
    if feature_labels:
        constraint += f"\nFeature choices: {', '.join(feature_labels)}"
    if error_labels:
        constraint += f"\nError type choices: {', '.join(error_labels)}"
    prompt = (
        "You are labeling a Jira incident. Read the text and output JSON with keys 'feature' and 'error_type'. "
        "Feature should be the module or integration touched (e.g., Finance, GoodLeap, LightReach, Proposals, Contracts). "
        "Error type should be the failure class (e.g., Auth, Mapping, Validation, Timeout, Missing-Config, API-Error, Performance). "
        "Keep each label <= 3 words. If unsure, make your best inference (do not return Unknown)." + constraint + "\n\nText:\n" + text
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = resp.choices[0].message.content.strip()
    import json as _json, re as _re
    m = _re.search(r"\{[\s\S]*\}", content)
    if m:
        try:
            obj = _json.loads(m.group(0))
            feature = str(obj.get("feature") or "").strip() or "General"
            error_type = str(obj.get("error_type") or "").strip() or "General"
            return {"feature": feature, "error_type": error_type}
        except Exception:
            pass
    return {"feature": "General", "error_type": "General"}


def main() -> None:
    load_dotenv()

    base = os.getenv("JIRA_BASE_URL") or os.getenv("JIRA_HOST")
    email = os.getenv("JIRA_EMAIL")
    token = os.getenv("JIRA_API_TOKEN")
    sheet_id = os.getenv("SHEET_ID")
    project = os.getenv("JIRA_PROJECT", "CO")
    days = int(os.getenv("JIRA_LOOKBACK_DAYS", "60"))

    if not (base and email and token and sheet_id):
        raise SystemExit("Missing JIRA/Sheet envs")

    # Find board for project
    r = requests.get(
        base.rstrip("/") + "/rest/agile/1.0/board",
        params={"projectKeyOrId": project, "maxResults": 50},
        headers={"Accept": "application/json"},
        auth=(email, token),
    )
    r.raise_for_status()
    values = (r.json() or {}).get("values", [])
    board_id = (values or [{}])[0].get("id")
    if board_id is None:
        raise SystemExit("No board found for project")

    # Pull issues from last N days
    feature_field_id = "customfield_10015"
    error_field_id = "customfield_10016"
    issues: List[Dict[str, Any]] = []
    start_at = 0
    remaining = 500
    while remaining > 0:
        batch = min(50, remaining)
        resp = requests.get(
            base.rstrip("/") + f"/rest/agile/1.0/board/{board_id}/issue",
            params={
                "jql": f"created >= -{days}d ORDER BY created DESC",
                "startAt": start_at,
                "maxResults": batch,
            "fields": f"summary,description,created,updated,status,{feature_field_id},{error_field_id}",
            },
            headers={"Accept": "application/json"},
            auth=(email, token),
        )
        resp.raise_for_status()
        data = resp.json() or {}
        arr = data.get("issues", [])
        if not arr:
            break
        for it in arr:
            f = it.get("fields", {}) or {}
            issues.append(
                {
                    "id": it.get("key", ""),
                    "summary": f.get("summary") or "",
                    "description": f.get("description") or "",
                    "feature": f.get(feature_field_id, "Unknown"),
                    "error_type": f.get(error_field_id, "Unknown"),
                    "created": f.get("created", ""),
                    "updated": f.get("updated", ""),
                    "status": (f.get("status", {}) or {}).get("name", "Unknown") if isinstance(f.get("status"), dict) else (f.get("status") or "Unknown"),
                }
            )
        got = len(arr)
        start_at += got
        remaining -= got
        if got < batch:
            break

    if not issues:
        print("No issues via board.")
        return

    # Coerce select fields to strings
    def coerce_field(v):
        if isinstance(v, list):
            vals = []
            for item in v:
                if isinstance(item, dict):
                    vals.append(item.get("value") or item.get("name") or str(item))
                else:
                    vals.append(str(item))
            return ", ".join([s for s in vals if s]) or "Unknown"
        if isinstance(v, dict):
            return v.get("value") or v.get("name") or str(v)
        if v is None:
            return "Unknown"
        return str(v)

    df = pd.DataFrame(issues)
    if "feature" in df.columns:
        df["feature"] = df["feature"].apply(coerce_field)
    if "error_type" in df.columns:
        df["error_type"] = df["error_type"].apply(coerce_field)

    # Embeddings and clustering
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    classify_model = os.getenv("OPENAI_CLASSIFY_MODEL", chat_model)
    feature_vocab = [s.strip() for s in os.getenv("FEATURE_LABELS", "").split(",") if s.strip()]
    error_vocab = [s.strip() for s in os.getenv("ERROR_TYPE_LABELS", "").split(",") if s.strip()]

    # AI classification override (from text)
    ai_features: List[str] = []
    ai_errors: List[str] = []
    for _, row in df.iterrows():
        text_full = (row.get("summary") or "") + "\n" + (row.get("description") or "")
        labels = classify_feature_error(client, classify_model, text_full, feature_vocab or None, error_vocab or None)
        ai_features.append(labels["feature"])
        ai_errors.append(labels["error_type"])
    df["feature"] = ai_features
    df["error_type"] = ai_errors

    df["description"] = df["description"].fillna("")
    df["combined"] = df["summary"].fillna("") + "\n" + df["description"]
    df["normalized"] = df["combined"].apply(normalize_text)

    texts = df["normalized"].astype(str).tolist()
    embeddings: List[List[float]] = []
    for chunk in batched(texts, 64):
        resp = client.embeddings.create(model=embed_model, input=chunk)
        for item in resp.data:
            embeddings.append(item.embedding)
    emb = np.array(embeddings, dtype=np.float32)
    sim = cosine_similarity(emb)

    threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.78"))
    clusters: List[List[int]] = []
    visited = set()
    for i in range(len(sim)):
        if i in visited:
            continue
        group = [j for j in range(len(sim)) if sim[i, j] >= threshold]
        clusters.append(group)
        visited.update(group)
    index_to_cluster = {}
    for cid, members in enumerate(clusters):
        for m in members:
            index_to_cluster[m] = cid
    df["cluster_id"] = df.index.map(index_to_cluster)

    rows: List[Dict[str, Any]] = []
    now = datetime.datetime.now().strftime("%Y-%m-%d")
    for cid, group in df.groupby("cluster_id"):
        text = "\n".join(group["combined"].astype(str).tolist())
        prompt = (
            "Write a clear, non-superlative explanation for a product manager. "
            "Use simple language (no hype). Provide 3–5 short sentences (<= 120 words) that cover: "
            "1) What failed and where, 2) symptoms and scope, 3) likely root cause, 4) one next action.\n\n" + text
        )
        resp = client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        summary = resp.choices[0].message.content.strip()
        # Build concise crux (5–10 words)
        try:
            crux_prompt = (
                "From this summary, extract a concise 5–10 word crux capturing the main failure and cause. "
                "Return only the phrase.\n\n" + summary
            )
            crux_resp = client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": crux_prompt}],
                temperature=0,
            )
            crux = crux_resp.choices[0].message.content.strip()
        except Exception:
            crux = "Concise crux not available"
        # Plain text list of all example ids (no hyperlink)
        example_link = ", ".join([str(x) for x in group["id"].astype(str).tolist()])

        rows.append(
            {
                "cluster_id": cid,
                "recurring_summary": summary,
                "Crux": crux,
                "feature": (group["feature"].mode(dropna=True).astype(str).iloc[0] if "feature" in group and not group["feature"].isna().all() else "Unknown"),
                "error_type": (group["error_type"].mode(dropna=True).astype(str).iloc[0] if "error_type" in group and not group["error_type"].isna().all() else "Unknown"),
                "total_tickets": int(len(group)),
                "example_ids": example_link,
                "status": (group["status"].mode(dropna=True).astype(str).iloc[0] if "status" in group and not group["status"].isna().all() else "Unknown"),
            }
        )

    out_df = pd.DataFrame(rows)

    creds = Credentials.from_service_account_file(
        "service_account.json",
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    gc = gspread.authorize(creds)
    sheet = gc.open_by_key(sheet_id).sheet1
    sheet.clear()
    sheet.append_row(out_df.columns.tolist())
    if len(out_df) > 0:
        sheet.append_rows(out_df.values.tolist(), value_input_option='USER_ENTERED')
    print(f"✅ Wrote {len(out_df)} clusters to Google Sheet")


if __name__ == "__main__":
    main()


