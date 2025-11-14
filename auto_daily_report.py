import os
import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from jira import JIRA
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials
import gspread
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import re
AI_PROVIDER = (os.getenv("AI_PROVIDER") or "").strip().lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")
_HAS_GENAI = False
if AI_PROVIDER == "gemini" or GEMINI_API_KEY:
    try:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=GEMINI_API_KEY)
        _HAS_GENAI = True
    except Exception:
        _HAS_GENAI = False
def _pick_gemini_chat_model(default_name: str) -> str:
    chat_name = default_name
    try:
        models = list(getattr(genai, "list_models", lambda: [])())
        names = [getattr(m, "name", "") for m in models]
        # Prefer 2.5 family if available
        preferred = [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "models/gemini-2.5-pro",
            "models/gemini-2.5-flash",
        ]
        for cand in preferred:
            if cand in names:
                return cand
        # Fallback to any 2.5
        for n in names:
            if "gemini-2.5" in n:
                return n
    except Exception:
        pass
    return chat_name

def get_env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def discover_project_key(base_url: str, email: str, token: str, name: str) -> Optional[str]:
    try:
        resp = requests.get(
            base_url.rstrip("/") + "/rest/api/3/project/search",
            params={"query": name, "maxResults": 50},
            headers={"Accept": "application/json"},
            auth=(email, token),
        )
        if resp.status_code != 200:
            return None
        values = (resp.json() or {}).get("values", [])
        for v in values:
            if (v.get("name") or "").strip().lower() == name.strip().lower():
                return v.get("key")
        if values:
            return values[0].get("key")
    except Exception:
        return None
    return None


def fetch_jira_issues_last_days(
    base_url: str,
    email: str,
    token: str,
    project_input: str,
    days: int,
    max_results: int = 200,
) -> List[Dict[str, Any]]:
    # Resolve to key if possible; accept either key or name and query both
    resolved_key = discover_project_key(base_url, email, token, project_input) or project_input
    jql_project_clause = f'project in ("{project_input}", "{resolved_key}")'
    jql = f'{jql_project_clause} AND created >= -{days}d ORDER BY created DESC'

    records: List[Dict[str, Any]] = []
    # Use fixed custom field IDs
    feature_field_id = "customfield_10015"
    error_field_id = "customfield_10016"

    # 1) Try Jira Python library (REST v3)
    try:
        options = {"server": base_url, "rest_api_version": "3"}
        jira_client = JIRA(options=options, basic_auth=(email, token))
        issues = jira_client.search_issues(jql, maxResults=max_results)
        for issue in issues:
            fields = issue.fields
            records.append(
                {
                    "id": issue.key,
                    "summary": getattr(fields, "summary", "") or "",
                    "description": getattr(fields, "description", "") or "",
                    "feature": getattr(fields, feature_field_id, "Unknown"),
                    "error_type": getattr(fields, error_field_id, "Unknown"),
                    "created": getattr(fields, "created", ""),
                    "updated": getattr(fields, "updated", ""),
                }
            )
    except Exception:
        pass

    if records:
        return records

    # 2) Try new POST /rest/api/3/search/jql with JSON body
    url = base_url.rstrip("/") + "/rest/api/3/search/jql"
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    auth = (email, token)

    start_at = 0
    page_size = 100
    remaining = max_results

    while remaining > 0:
        batch_size = min(page_size, remaining)
        body = {
            "jql": jql,
            "startAt": start_at,
            "maxResults": batch_size,
            "fields": [
                "summary",
                "description",
                "created",
                "updated",
                feature_field_id,
                error_field_id,
            ],
        }
        resp = requests.post(url, json=body, headers=headers, auth=auth)
        if resp.status_code != 200:
            break
        data = resp.json() or {}
        issues = data.get("issues", [])
        if not issues:
            break
        for issue in issues:
            key = issue.get("key", "")
            fields = issue.get("fields", {}) or {}
            records.append(
                {
                    "id": key,
                    "summary": fields.get("summary") or "",
                    "description": fields.get("description") or "",
                    "feature": fields.get(feature_field_id, "Unknown"),
                    "error_type": fields.get(error_field_id, "Unknown"),
                    "status": (fields.get("status", {}) or {}).get("name", "Unknown") if isinstance(fields.get("status"), dict) else (fields.get("status") or "Unknown"),
                    "created": fields.get("created", ""),
                    "updated": fields.get("updated", ""),
                }
            )
        fetched = len(issues)
        start_at += fetched
        remaining -= fetched
        if fetched < batch_size:
            break

    if records:
        return records

    # 3) Final fallback: old GET /search (if tenant still supports it)
    url_get = base_url.rstrip("/") + "/rest/api/3/search"
    headers = {"Accept": "application/json"}
    auth = (email, token)

    start_at = 0
    remaining = max_results
    while remaining > 0:
        batch_size = min(100, remaining)
        params = {
            "jql": jql,
            "startAt": start_at,
            "maxResults": batch_size,
            "fields": f"summary,description,status,created,updated,{feature_field_id},{error_field_id}",
        }
        resp = requests.get(url_get, params=params, headers=headers, auth=auth)
        if resp.status_code != 200:
            break
        data = resp.json()
        issues = data.get("issues", [])
        if not issues:
            break
        for issue in issues:
            key = issue.get("key", "")
            fields = issue.get("fields", {}) or {}
            records.append(
                {
                    "id": key,
                    "summary": fields.get("summary") or "",
                    "description": fields.get("description") or "",
                    "feature": fields.get(feature_field_id, "Unknown"),
                    "error_type": fields.get(error_field_id, "Unknown"),
                    "status": (fields.get("status", {}) or {}).get("name", "Unknown") if isinstance(fields.get("status"), dict) else (fields.get("status") or "Unknown"),
                    "created": fields.get("created", ""),
                    "updated": fields.get("updated", ""),
                }
            )
        fetched = len(issues)
        start_at += fetched
        remaining -= fetched
        if fetched < batch_size:
            break

    # 4) Agile board fallback using project key
    try:
        boards_url = base_url.rstrip("/") + "/rest/agile/1.0/board"
        r = requests.get(
            boards_url,
            params={"projectKeyOrId": resolved_key, "maxResults": 50},
            headers={"Accept": "application/json"},
            auth=(email, token),
        )
        if r.status_code == 200:
            values = (r.json() or {}).get("values", [])
            board_id = None
            for b in values:
                if "critical" in (b.get("name") or "").lower():
                    board_id = b.get("id")
                    break
            if board_id is None and values:
                board_id = values[0].get("id")
            if board_id is not None:
                start_at = 0
                remaining = max_results
                while remaining > 0:
                    batch_size = min(50, remaining)
                    issues_url = base_url.rstrip("/") + f"/rest/agile/1.0/board/{board_id}/issue"
                    params = {
                        "jql": f"created >= -{days}d ORDER BY created DESC",
                        "startAt": start_at,
                        "maxResults": batch_size,
                        "fields": f"summary,description,status,created,updated,{feature_field_id},{error_field_id}",
                    }
                    resp = requests.get(issues_url, params=params, headers={"Accept": "application/json"}, auth=(email, token))
                    if resp.status_code != 200:
                        break
                    data = resp.json() or {}
                    issues = data.get("issues", [])
                    if not issues:
                        break
                    for issue in issues:
                        key = issue.get("key", "")
                        fields = issue.get("fields", {}) or {}
                        records.append(
                            {
                                "id": key,
                                "summary": fields.get("summary") or "",
                                "description": fields.get("description") or "",
                                "feature": fields.get(feature_field_id, "Unknown"),
                                "error_type": fields.get(error_field_id, "Unknown"),
                                "status": (fields.get("status", {}) or {}).get("name", "Unknown") if isinstance(fields.get("status"), dict) else (fields.get("status") or "Unknown"),
                                "created": fields.get("created", ""),
                                "updated": fields.get("updated", ""),
                            }
                        )
                    fetched = len(issues)
                    start_at += fetched
                    remaining -= fetched
                    if fetched < batch_size:
                        break
    except Exception:
        pass

    return records


def batched(lst: List[str], size: int) -> List[List[str]]:
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text or "")
    t = text.lower()
    # remove emails and urls
    t = re.sub(r"[\w\.-]+@[\w\.-]+", " ", t)
    t = re.sub(r"https?://\S+", " ", t)
    # replace ticket keys like ABC-123
    t = re.sub(r"\b[A-Z]{2,}-\d+\b", " ", t)
    # collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def coerce_text(value: Any) -> str:
    """Coerce Jira field (which may be str/dict/PropertyHolder) to plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    # PropertyHolder from jira library can stringify, but try to extract body if present
    try:
        if hasattr(value, "body"):
            body = getattr(value, "body")
            return body if isinstance(body, str) else str(body or "")
        if isinstance(value, dict):
            # Attempt to flatten Atlassian Document Format by collecting text fields
            def walk(node):
                parts = []
                if isinstance(node, dict):
                    txt = node.get("text")
                    if isinstance(txt, str):
                        parts.append(txt)
                    for k in ("content", "paragraphs", "items", "children"):
                        if k in node:
                            parts.append(walk(node[k]))
                elif isinstance(node, list):
                    for item in node:
                        parts.append(walk(item))
                return " ".join([p for p in parts if p])
            flattened = walk(value)
            if flattened:
                return flattened
        return str(value)
    except Exception:
        return str(value)


def classify_feature_error(client: OpenAI, model: str, text: str, feature_labels: List[str] = None, error_labels: List[str] = None) -> Dict[str, str]:
    # Truncate to control token/cost
    if text and len(text) > 4000:
        text = text[:4000]
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
    try:
        if AI_PROVIDER == "gemini" and _HAS_GENAI:
            model_obj = genai.GenerativeModel(_pick_gemini_chat_model(GEMINI_CHAT_MODEL))
            r = model_obj.generate_content(prompt)
            content = (getattr(r, "text", "") or "").strip()
        else:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=120,
                response_format={"type": "json_object"},
            )
            content = (resp.choices[0].message.content or "").strip()
    except Exception:
        return {"feature": "General", "error_type": "General"}
    # naive parse: try to extract JSON
    import json, re as _re
    m = _re.search(r"\{[\s\S]*\}", content)
    if m:
        try:
            obj = json.loads(m.group(0))
            feature = str(obj.get("feature") or "").strip() or "General"
            error_type = str(obj.get("error_type") or "").strip() or "General"
            return {"feature": feature, "error_type": error_type}
        except Exception:
            pass
    # fallback: return short extracted labels from free text
    return {"feature": "General", "error_type": "General"}

def _sanitize_text_for_crux(s: str) -> str:
    """Remove noise and imperative phrasing; return up to ~10 informative tokens."""
    if not isinstance(s, str):
        s = str(s or "")
    t = s.lower()
    # strip hex-like ids, ticket ids, urls, emails
    t = re.sub(r"\b0x[0-9a-f]+\b", " ", t)
    t = re.sub(r"\b[a-z]{2,}\-\d+\b", " ", t)
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"[\w\.-]+@[\w\.-]+", " ", t)
    stop = {
        "you","can","please","should","could","would","may","might","we","our","let's",
        "ensure","make","fix","address","issue","issues","problem","problems","error","errors",
        "need","needs","needed","note","notes","thanks","thank","info","information"
    }
    toks = [w for w in re.findall(r"[a-z0-9\-]+", t) if w and w not in stop]
    if not toks:
        return "concise crux not available"
    return " ".join(toks[:10])


def main() -> None:
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    jira_base_url = os.getenv("JIRA_BASE_URL") or os.getenv("JIRA_HOST")
    jira_email = os.getenv("JIRA_EMAIL")
    jira_api_token = os.getenv("JIRA_API_TOKEN")
    sheet_id = os.getenv("SHEET_ID")

    has_ai_key = bool(openai_api_key)
    if not (has_ai_key and jira_base_url and jira_email and jira_api_token and sheet_id):
        raise SystemExit("Missing required env: OPENAI_API_KEY, JIRA_BASE_URL/JIRA_HOST, JIRA_EMAIL, JIRA_API_TOKEN, SHEET_ID")

    # Accept either key or name from JIRA_PROJECT; fallback to legacy var
    project_input = os.getenv("JIRA_PROJECT") or os.getenv("CRITICAL_OPS_BOARD_NAME", "Critical Ops")
    days = get_env_int("JIRA_LOOKBACK_DAYS", 1)

    base_url = os.getenv("OPENAI_BASE_URL")
    client = None
    if openai_api_key:
        client = OpenAI(api_key=openai_api_key, base_url=base_url) if base_url else OpenAI(api_key=openai_api_key)
    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    classify_model = os.getenv("OPENAI_CLASSIFY_MODEL", chat_model)
    # Optional fixed vocabularies (comma-separated)
    feature_vocab = [s.strip() for s in os.getenv("FEATURE_LABELS", "").split(",") if s.strip()]
    error_vocab = [s.strip() for s in os.getenv("ERROR_TYPE_LABELS", "").split(",") if s.strip()]

    # Google Sheets auth (service account file must exist)
    creds = Credentials.from_service_account_file(
        "service_account.json",
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    try:
        import json as _json
        with open("service_account.json", "r") as _f:
            _sa = _json.load(_f) or {}
            _email = _sa.get("client_email", "")
            print(f"Using Google service account: {_email}")
    except Exception:
        pass
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)
    try:
        print(f"Using OpenAI models: chat={chat_model}, embed={embed_model}")
    except Exception:
        pass
    # Find or create the "last 15 days" worksheet (case/space-insensitive)
    def _normalize_title(t: str) -> str:
        return re.sub(r"\s+", "", (t or "").strip().lower())
    target_title = "last 15 days"
    worksheet = None
    try:
        for w in sh.worksheets():
            if _normalize_title(w.title) == _normalize_title(target_title):
                worksheet = w
                break
    except Exception:
        worksheet = None
    if worksheet is None:
        worksheet = sh.add_worksheet(title=target_title, rows="1000", cols="20")
    sheet = worksheet

    # Step 1: Fetch Jira
    issues = fetch_jira_issues_last_days(
        jira_base_url, jira_email, jira_api_token, project_input, days, max_results=200
    )

    if not issues:
        # Final inline fallback via Agile API
        resolved_key = discover_project_key(jira_base_url, jira_email, jira_api_token, project_input) or project_input
        try:
            boards_url = jira_base_url.rstrip("/") + "/rest/agile/1.0/board"
            r = requests.get(
                boards_url,
                params={"projectKeyOrId": resolved_key, "maxResults": 50},
                headers={"Accept": "application/json"},
                auth=(jira_email, jira_api_token),
            )
            if r.status_code == 200:
                values = (r.json() or {}).get("values", [])
                board_id = (values or [{}])[0].get("id")
                if board_id is not None:
                    issues_url = jira_base_url.rstrip("/") + f"/rest/agile/1.0/board/{board_id}/issue"
                    agg = []
                    start_at = 0
                    remaining = 200
                    while remaining > 0:
                        batch = min(50, remaining)
                        resp = requests.get(
                            issues_url,
                            params={
                                "jql": f"updated >= -{days}d ORDER BY updated DESC",
                                "startAt": start_at,
                                "maxResults": batch,
                                "fields": "summary,description,created,updated,customfield_10015,customfield_10016",
                            },
                            headers={"Accept": "application/json"},
                            auth=(jira_email, jira_api_token),
                        )
                        if resp.status_code != 200:
                            break
                        data = resp.json() or {}
                        arr = data.get("issues", [])
                        if not arr:
                            break
                        for it in arr:
                            f = it.get("fields", {}) or {}
                            agg.append(
                                {
                                    "id": it.get("key", ""),
                                    "summary": f.get("summary") or "",
                                    "description": f.get("description") or "",
                                    "feature": f.get("customfield_10015", "Unknown"),
                                    "error_type": f.get("customfield_10016", "Unknown"),
                                    "created": f.get("created", ""),
                                    "updated": f.get("updated", ""),
                                }
                            )
                        got = len(arr)
                        start_at += got
                        remaining -= got
                        if got < batch:
                            break
                    issues = agg
        except Exception:
            pass

    if not issues:
        print("No new issues found.")
        return

    df = pd.DataFrame(issues)
    # Ensure text fields are strings (Jira may return complex objects)
    if "summary" in df.columns:
        df["summary"] = df["summary"].apply(coerce_text).fillna("")
    else:
        df["summary"] = ""
    if "description" in df.columns:
        df["description"] = df["description"].apply(coerce_text).fillna("")
    else:
        df["description"] = ""
    df["combined"] = df["summary"] + "\n" + df["description"]
    df["normalized"] = df["combined"].apply(normalize_text)

    # Step 2: AI classification (per-issue)
    ai_features: List[str] = []
    ai_errors: List[str] = []
    for _, row in df.iterrows():
        text_full = (row.get("summary") or "") + "\n" + (row.get("description") or "")
        labels = classify_feature_error(client, classify_model, text_full, feature_vocab or None, error_vocab or None)
        ai_features.append(labels["feature"])
        ai_errors.append(labels["error_type"])
    df["feature"] = ai_features
    df["error_type"] = ai_errors

    # Step 3: Embeddings (batched)
    texts = df["normalized"].astype(str).tolist()
    embeddings: List[List[float]] = []
    for chunk in batched(texts, 64):
        try:
            if AI_PROVIDER == "gemini" and _HAS_GENAI:
                for _t in chunk:
                    try:
                        er = genai.embed_content(model=GEMINI_EMBED_MODEL, content=_t)
                        vec = None
                        if isinstance(er, dict):
                            vec = (er.get("embedding") or {}).get("values")
                        else:
                            emb = getattr(er, "embedding", None)
                            vec = getattr(emb, "values", None) if emb is not None else None
                        embeddings.append(vec or [0.0])
                    except Exception:
                        embeddings.append([0.0])
            else:
                if client is not None:
                    resp = client.embeddings.create(model=embed_model, input=chunk)
                    for item in resp.data:
                        embeddings.append(item.embedding)
                else:
                    raise RuntimeError("No remote embedding provider available")
        except Exception:
            # Local TF-IDF fallback (better than all-zero vectors)
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                tfidf = TfidfVectorizer(max_features=1024)
                tfidf_matrix = tfidf.fit_transform(texts)
                embeddings = tfidf_matrix.astype("float32").toarray().tolist()
                break  # computed all at once
            except Exception:
                embeddings.extend([[0.0] for _ in chunk])

    emb_matrix = np.array(embeddings, dtype=np.float32)
    # If embeddings are constant/zero, avoid collapsing everything into one cluster
    if emb_matrix.size == 0 or np.allclose(emb_matrix, emb_matrix[0]):
        sim = np.eye(len(texts), dtype=np.float32)
    else:
        sim = cosine_similarity(emb_matrix)

    # Step 4: Clustering by threshold
    threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.78"))
    clusters: List[List[int]] = []
    visited = set()
    n = len(sim)
    for i in range(n):
        if i in visited:
            continue
        group = [j for j in range(n) if sim[i, j] >= threshold]
        clusters.append(group)
        visited.update(group)
    index_to_cluster: Dict[int, int] = {}
    for cid, members in enumerate(clusters):
        for m in members:
            index_to_cluster[m] = cid
    df["cluster_id"] = df.index.map(index_to_cluster)

    # Step 5: Summaries
    rows: List[Dict[str, Any]] = []
    now = datetime.datetime.now().strftime("%Y-%m-%d")
    # Export per-ticket clusters for downstream summarization
    try:
        export_cols = ["cluster_id"]
        for col in ["id", "summary", "description", "feature", "error_type", "created", "updated"]:
            if col in df.columns:
                export_cols.append(col)
        if "combined" in df.columns:
            export_cols.append("combined")
        else:
            df["combined"] = (df.get("summary", "")).astype(str) + "\n" + (df.get("description", "")).astype(str)
            export_cols.append("combined")
        df[export_cols].to_csv("critical_ops_clusters.csv", index=False)
        print(f"Saved critical_ops_clusters.csv with {len(df)} rows.")
    except Exception as _e:
        print(f"Warning: failed to export critical_ops_clusters.csv: {_e}")
    for cid, group in df.groupby("cluster_id"):
        text = "\n".join(group["combined"].astype(str).tolist())
        # Truncate context to control token/cost
        if len(text) > 8000:
            text = text[:8000]
        prompt = (
            "You are a senior incident analyst. Write a clear, neutral, third‑person summary (no direct address). "
            "Do NOT say 'you', 'we', or 'can'. Avoid filler/marketing. "
            "Provide 5–7 concise sentences (120–180 words) covering: "
            "1) what failed and where, 2) symptoms and scope (who/how many), "
            "3) likely root cause, 4) one recommended next action.\n\n"
            f"Context:\n{text}"
        )
        try:
            if AI_PROVIDER == "gemini" and _HAS_GENAI:
                model_obj = genai.GenerativeModel(_pick_gemini_chat_model(GEMINI_CHAT_MODEL))
                gen_cfg = {"temperature": 0.2, "max_output_tokens": 300}
                r = model_obj.generate_content(prompt, generation_config=gen_cfg)
                summary = (getattr(r, "text", "") or "").strip()
            else:
                resp = client.chat.completions.create(
                    model=chat_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=300,
                )
                summary = resp.choices[0].message.content.strip()
        except Exception:
            summary = ""
        if not summary:
            # Deterministic backup if LLM unavailable
            sample = " ".join(group["summary"].astype(str).tolist()[:3])
            summary = f"Grouped similar incidents. Representative notes: {sample[:240]}..."

        # Concise 5–10 word crux for the cluster
        try:
            crux_prompt = (
                "From the summary below, output ONLY a 5–10 word noun phrase capturing the core failure/root cause. "
                "No punctuation except hyphens. No verbs like 'fix', 'use', 'can'. No second‑person words.\n\n"
                f"{summary}"
            )
            if AI_PROVIDER == "gemini" and _HAS_GENAI:
                model_obj = genai.GenerativeModel(_pick_gemini_chat_model(GEMINI_CHAT_MODEL))
                gen_cfg = {"temperature": 0.0, "max_output_tokens": 32}
                rr = model_obj.generate_content(crux_prompt, generation_config=gen_cfg)
                crux = (getattr(rr, "text", "") or "").strip()
            else:
                crux_resp = client.chat.completions.create(
                    model=chat_model,
                    messages=[{"role": "user", "content": crux_prompt}],
                    temperature=0,
                    max_tokens=32,
                )
                crux = crux_resp.choices[0].message.content.strip()
        except Exception:
            crux = ""
        crux = _sanitize_text_for_crux(crux)
        if crux == "concise crux not available" or not crux.strip():
            # Simple TF-IDF keyphrase fallback
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                sample_texts = group["combined"].astype(str).tolist()
                vec = TfidfVectorizer(max_features=8, ngram_range=(1, 2), stop_words="english")
                X = vec.fit_transform(sample_texts)
                terms = vec.get_feature_names_out()
                crux = _sanitize_text_for_crux(" ".join(list(terms)[:6]))
            except Exception:
                crux = "Concise crux not available"

        # Plain text list of all example ids (no hyperlink)
        example_link = ", ".join([str(x) for x in group["id"].astype(str).tolist()])

        # Earliest create date in the cluster (YYYY-MM-DD)
        create_date_val = ""
        try:
            if "created" in group.columns:
                created_ts = pd.to_datetime(group["created"], errors="coerce", utc=True)
                created_ts = created_ts.dropna()
                if not created_ts.empty:
                    create_date_val = created_ts.min().strftime("%Y-%m-%d")
        except Exception:
            create_date_val = ""

        rows.append(
            {
                "cluster_id": cid,
                "recurring_summary": summary,
                "crux": crux,
                "feature": (group["feature"].mode()[0] if not group["feature"].isna().all() else "Unknown"),
                "error_type": (group["error_type"].mode()[0] if not group["error_type"].isna().all() else "Unknown"),
                "total_tickets": int(len(group)),
                "example_ids": example_link,
                "status": (group["status"].mode()[0] if "status" in group and not group["status"].isna().all() else "Unknown"),
                "create_date": create_date_val,
            }
        )

    out_df = pd.DataFrame(rows)

    # Step 5: Write to Google Sheet (replace)
    sheet.clear()
    if len(out_df) == 0:
        sheet.append_row(["cluster_id", "recurring_summary", "crux", "feature", "error_type", "total_tickets", "example_ids", "status", "create_date"])
    else:
        # Ensure column order places crux beside recurring_summary
        desired_cols = [
            "cluster_id",
            "recurring_summary",
            "crux",
            "feature",
            "error_type",
            "total_tickets",
            "example_ids",
            "status",
            "create_date",
        ]
        # Reindex if all expected columns exist; otherwise fall back to current order
        if all(c in out_df.columns for c in desired_cols):
            out_df = out_df[desired_cols]
        sheet.append_row(out_df.columns.tolist())
        sheet.append_rows(out_df.values.tolist())
    print(f"✅ Google Sheet updated successfully at {datetime.datetime.now()} with {len(out_df)} clusters.")


if __name__ == "__main__":
    main()


