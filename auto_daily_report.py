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
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import Tuple
import re as _rx
from kb_store import upsert_kb, retrieve_context_for_text


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


def _make_gemini_model(model_name: str):
    try:
        from google.generativeai.types import HarmCategory, HarmBlockThreshold, SafetySetting
        safety_settings = [
            SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUAL_CONTENT, threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.BLOCK_NONE),
        ]
    except Exception:
        # Fallback to dict-based safety config if types not available
        safety_settings = [
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUAL_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
    generation_config = {"temperature": 0.2, "max_output_tokens": 256}
    return genai.GenerativeModel(model_name, generation_config=generation_config, safety_settings=safety_settings)


def _safe_generate_text(model_name: str, prompt: str) -> Tuple[bool, str]:
    """Call Gemini and return (ok, text). Retries with simplified prompt if blocked/empty."""
    def _extract_text(resp_obj) -> str:
        txt = (getattr(resp_obj, "text", None) or "").strip()
        if txt:
            return txt
        try:
            cands = getattr(resp_obj, "candidates", []) or []
            for cand in cands:
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", None)
                if parts:
                    collected = []
                    for p in parts:
                        # p can be a dict or an object with .text
                        t = getattr(p, "text", None)
                        if not t and isinstance(p, dict):
                            t = p.get("text")
                        if t:
                            collected.append(t)
                    if collected:
                        return "\n".join(collected).strip()
        except Exception:
            pass
        return ""

    # Guard prompt length to avoid model refusing on very long inputs
    if isinstance(prompt, str) and len(prompt) > 20000:
        prompt = prompt[-20000:]
    try:
        mdl = _make_gemini_model(model_name)
        resp = mdl.generate_content(prompt)
        txt = _extract_text(resp)
        if txt:
            return True, txt
        # Retry with simplified prompt if empty
        retry_prompt = "Provide a neutral, purely technical summary without opinions.\n\n" + prompt
        resp2 = mdl.generate_content(retry_prompt)
        txt2 = _extract_text(resp2)
        return (True, txt2) if txt2 else (False, "")
    except Exception:
        return False, ""


def classify_feature_error(model_name: str, text: str, feature_labels: List[str] = None, error_labels: List[str] = None) -> Dict[str, str]:
    # Limit text size to keep prompts reliable
    if isinstance(text, str) and len(text) > 15000:
        text = text[-15000:]
    # Default vocab if not provided
    default_features = [
        "Finance", "GoodLeap", "LightReach", "Proposals", "Contracts", "Reports",
        "Integrations", "Authentication", "UI", "Data Pipeline"
    ]
    default_errors = [
        "Validation", "Mapping", "Timeout", "Missing-Config", "API-Error",
        "Auth", "Performance", "Data-Quality"
    ]
    feature_labels = feature_labels or default_features
    error_labels = error_labels or default_errors
    constraint = ""
    if feature_labels:
        constraint += f"\nFeature choices: {', '.join(feature_labels)}"
    if error_labels:
        constraint += f"\nError type choices: {', '.join(error_labels)}"
    prompt = (
        "You are labeling a Jira incident. Read the text and output JSON with keys 'feature' and 'error_type'. "
        "Feature should be the most relevant product module/integration. "
        "Error type should be the failure class. "
        "Choose the best labels from the choices if possible, or infer a specific 1–3 word label (do NOT return General/Unknown).\n"
        + f"Feature choices: {', '.join(feature_labels)}\n"
        + f"Error type choices: {', '.join(error_labels)}\n\n"
        "Return strictly JSON like {\"feature\":\"...\",\"error_type\":\"...\"}.\n\nText:\n" + text
    )
    ok, content = _safe_generate_text(model_name, prompt)
    # naive parse: try to extract JSON
    import json, re as _re
    m = _re.search(r"\{[\s\S]*\}", content)
    if m:
        try:
            obj = json.loads(m.group(0))
            feature = str(obj.get("feature") or "").strip() or "General"
            error_type = str(obj.get("error_type") or "").strip() or "General"
            if feature.lower() != "general" or error_type.lower() != "general":
                return {"feature": feature, "error_type": error_type}
        except Exception:
            pass
    # Heuristic fallback from text
    t = (text or "").lower()
    patterns_feature = [
        ("GoodLeap", r"\bgood\s*leap\b|\bgoodleap\b"),
        ("LightReach", r"\blightreach\b"),
        ("Proposals", r"\bproposal|quot(e|ing)\b"),
        ("Contracts", r"\bcontract(s)?\b"),
        ("Finance", r"\bloan|credit|finance|funding\b"),
        ("Reports", r"\breport\b"),
        ("Integrations", r"\bwebhook|callback|integration|api\b"),
        ("Authentication", r"\bauth(entication)?|oauth|token|login\b"),
        ("UI", r"\bbutton|dropdown|screen|ui\b"),
        ("Data Pipeline", r"\bpipeline|etl|ingest(ion)?\b"),
    ]
    patterns_error = [
        ("Validation", r"\bvalidation|invalid|required|format|regex|parse\b"),
        ("Mapping", r"\bmapping|map(ped|ping)?|transform\b"),
        ("Timeout", r"\btime[ -]?out|timed out\b"),
        ("Missing-Config", r"\bconfig(uration)? missing|missing config|env var|secret\b"),
        ("API-Error", r"\bhttp\s*(4|5)\d{2}|bad gateway|gateway|service unavailable|rate limit|429\b"),
        ("Auth", r"\bauth(entication)?|unauthorized|forbidden|expired token|login\b"),
        ("Performance", r"\blatenc(y|ies)|slow|sluggish\b"),
        ("Data-Quality", r"\bdata (mismatch|inconsisten|stale|quality)\b"),
    ]
    feat = "General"
    err = "General"
    for label, pat in patterns_feature:
        if _rx.search(pat, t):
            feat = label
            break
    for label, pat in patterns_error:
        if _rx.search(pat, t):
            err = label
            break
    return {"feature": feat, "error_type": err}


def main() -> None:
    load_dotenv()

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    jira_base_url = os.getenv("JIRA_BASE_URL") or os.getenv("JIRA_HOST")
    jira_email = os.getenv("JIRA_EMAIL")
    jira_api_token = os.getenv("JIRA_API_TOKEN")
    sheet_id = os.getenv("SHEET_ID")

    if not (gemini_api_key and jira_base_url and jira_email and jira_api_token and sheet_id):
        raise SystemExit("Missing required env: GEMINI_API_KEY, JIRA_BASE_URL/JIRA_HOST, JIRA_EMAIL, JIRA_API_TOKEN, SHEET_ID")

    # Accept either key or name from JIRA_PROJECT; fallback to legacy var
    project_input = os.getenv("JIRA_PROJECT") or os.getenv("CRITICAL_OPS_BOARD_NAME", "Critical Ops")
    days = get_env_int("JIRA_LOOKBACK_DAYS", 1)

    # Configure Gemini
    genai.configure(api_key=gemini_api_key)
    embed_model = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")
    chat_model = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
    classify_model = os.getenv("GEMINI_CLASSIFY_MODEL", chat_model)
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
    sheet = gc.open_by_key(sheet_id).sheet1

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
                                "jql": f"created >= -{days}d ORDER BY created DESC",
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

    # Upsert to local KB for context-aware prompts
    try:
        use_kb = os.getenv("USE_KB_CONTEXT", "1") == "1"
        if use_kb:
            items = [(str(r["id"]), str(r["combined"])) for _, r in df.iterrows()]
            upsert_kb(items, embed_model)
    except Exception:
        pass

    # Step 2: AI classification (per-issue)
    ai_features: List[str] = []
    ai_errors: List[str] = []
    for _, row in df.iterrows():
        text_full = (row.get("summary") or "") + "\n" + (row.get("description") or "")
        kb_ctx = ""
        if use_kb:
            try:
                kb_ctx = retrieve_context_for_text(text_full, top_k=3, max_chars=3000, embed_model=embed_model)
            except Exception:
                kb_ctx = ""
        if kb_ctx:
            text_for_classify = text_full + "\n\nSimilar tickets context:\n" + kb_ctx
        else:
            text_for_classify = text_full
        labels = classify_feature_error(classify_model, text_for_classify, feature_vocab or None, error_vocab or None)
        ai_features.append(labels["feature"])
        ai_errors.append(labels["error_type"])
    df["feature"] = ai_features
    df["error_type"] = ai_errors

    # Step 3: Embeddings (batched)
    texts = df["normalized"].astype(str).tolist()
    embeddings: List[List[float]] = []
    for chunk in batched(texts, 32):
        for t in chunk:
            try:
                res = genai.embed_content(model=embed_model, content=t)
                emb = res.get("embedding") or (res.get("data", {}) or {}).get("embedding")
                # Some versions return {'embedding': {'values': [...]}}
                if isinstance(emb, dict) and "values" in emb:
                    emb = emb.get("values")
                if emb is None:
                    emb = []
                embeddings.append(emb)
            except Exception:
                embeddings.append([])

    emb_matrix = np.array(embeddings, dtype=np.float32)
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
    for cid, group in df.groupby("cluster_id"):
        text = "\n".join(group["combined"].astype(str).tolist())
        kb_ctx = ""
        if use_kb:
            try:
                kb_ctx = retrieve_context_for_text(text, top_k=5, max_chars=3500, embed_model=embed_model)
            except Exception:
                kb_ctx = ""
        prompt = (
            "Write a clear, non-superlative explanation for a product manager. "
            "Use simple language (no hype). Provide 3–5 short sentences (<= 120 words) that cover: "
            "1) What failed and where, 2) symptoms and scope, 3) likely root cause, 4) one next action.\n\n"
            + ("Relevant similar tickets:\n" + kb_ctx + "\n\n" if kb_ctx else "")
            + text
        )
        ok, summary = _safe_generate_text(chat_model, prompt)
        if not ok or not summary:
            # Deterministic fallback summary from data
            sample_summary = (str(group["summary"].iloc[0]) if "summary" in group.columns and len(group) > 0 else "").strip()
            sample_desc = (str(group["description"].iloc[0]) if "description" in group.columns and len(group) > 0 else "").strip()
            feature_hint = (str(group["feature"].mode()[0]) if "feature" in group and not group["feature"].isna().all() else "General")
            error_hint = (str(group["error_type"].mode()[0]) if "error_type" in group and not group["error_type"].isna().all() else "General")
            tickets_count = int(len(group))
            first_sentence = (sample_summary or sample_desc).split(".")[0][:180]
            summary = (
                f"{first_sentence}."
                f" Observed in {tickets_count} ticket(s); scope appears limited to the reported cases."
                f" Likely area: {feature_hint}; failure type looks like {error_hint}."
                " Root cause not yet confirmed based on ticket text."
                " Next action: reproduce, check recent changes, and review logs for this path."
            )

        # Concise 5–10 word crux for the cluster
        try:
            crux_prompt = (
                "From this summary, extract a concise 5–10 word crux capturing the main failure and cause. "
                "Return only the phrase.\n\n" + summary
            )
            ok2, crux = _safe_generate_text(chat_model, crux_prompt)
            if not ok2 or not crux:
                # Deterministic crux fallback from hints
                crux = f"{feature_hint} {error_hint} issue in reported flow"
        except Exception:
            crux = f"{feature_hint} {error_hint} issue in reported flow"

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


