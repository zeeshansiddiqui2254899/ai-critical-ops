import os
import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials
import gspread
import google.generativeai as genai
from openai import OpenAI as OpenAIClient
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import Tuple
import re as _rx
from kb_store import upsert_kb, retrieve_context_for_text


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
        safety_settings = [
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUAL_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
    generation_config = {"temperature": 0.2, "max_output_tokens": 256}
    return genai.GenerativeModel(model_name, generation_config=generation_config, safety_settings=safety_settings)


def _safe_generate_text(model_name: str, prompt: str) -> Tuple[bool, str]:
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
    if isinstance(prompt, str) and len(prompt) > 20000:
        prompt = prompt[-20000:]
    try:
        mdl = _make_gemini_model(model_name)
        resp = mdl.generate_content(prompt)
        txt = _extract_text(resp)
        if txt:
            return True, txt
        resp2 = mdl.generate_content("Provide a neutral, purely technical response.\n\n" + prompt)
        txt2 = _extract_text(resp2)
        return (True, txt2) if txt2 else (False, "")
    except Exception:
        return False, ""


def classify_feature_error(model_name: str, text: str, feature_labels: List[str] = None, error_labels: List[str] = None) -> Dict[str, str]:
    constraint = ""
    default_features = ["Finance","GoodLeap","LightReach","Proposals","Contracts","Reports","Integrations","Authentication","UI","Data Pipeline"]
    default_errors = ["Validation","Mapping","Timeout","Missing-Config","API-Error","Auth","Performance","Data-Quality"]
    feature_labels = feature_labels or default_features
    error_labels = error_labels or default_errors
    constraint = f"\nFeature choices: {', '.join(feature_labels)}\nError type choices: {', '.join(error_labels)}"
    prompt = (
        "You are labeling a Jira incident. Read the text and output JSON with keys 'feature' and 'error_type'. "
        "Feature should be the most relevant product module/integration. Error type should be the failure class. "
        "Choose from the choices when possible, or infer a specific 1–3 word label. Do NOT return General/Unknown. "
        "Return strictly JSON like {\"feature\":\"...\",\"error_type\":\"...\"}."
        + constraint + "\n\nText:\n" + text
    )
    ok, content = _safe_generate_text(model_name, prompt)
    import json as _json, re as _re
    m = _re.search(r"\{[\s\S]*\}", content)
    if m:
        try:
            obj = _json.loads(m.group(0))
            feature = str(obj.get("feature") or "").strip() or "General"
            error_type = str(obj.get("error_type") or "").strip() or "General"
            if feature.lower() != "general" or error_type.lower() != "general":
                return {"feature": feature, "error_type": error_type}
        except Exception:
            pass
    # Heuristic fallback
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
    # Configure provider
    provider = os.getenv("AI_PROVIDER", "gemini").lower()
    if provider == "openai":
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise SystemExit("AI_PROVIDER=openai but OPENAI_API_KEY missing")
        openai_client = OpenAIClient(api_key=openai_key)
        embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        classify_model = os.getenv("OPENAI_CLASSIFY_MODEL", chat_model)
    else:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        embed_model = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")
        chat_model = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
        classify_model = os.getenv("GEMINI_CLASSIFY_MODEL", chat_model)
    feature_vocab = [s.strip() for s in os.getenv("FEATURE_LABELS", "").split(",") if s.strip()]
    error_vocab = [s.strip() for s in os.getenv("ERROR_TYPE_LABELS", "").split(",") if s.strip()]

    # AI classification override (from text)
    ai_features: List[str] = []
    ai_errors: List[str] = []
    for _, row in df.iterrows():
        text_full = (row.get("summary") or "") + "\n" + (row.get("description") or "")
        labels = classify_feature_error(classify_model, text_full, feature_vocab or None, error_vocab or None)
        ai_features.append(labels["feature"])
        ai_errors.append(labels["error_type"])
    df["feature"] = ai_features
    df["error_type"] = ai_errors

    df["description"] = df["description"].fillna("")
    df["combined"] = df["summary"].fillna("") + "\n" + df["description"]
    df["normalized"] = df["combined"].apply(normalize_text)

    texts = df["normalized"].astype(str).tolist()

    # Upsert to local KB
    try:
        use_kb = os.getenv("USE_KB_CONTEXT", "1") == "1"
        if use_kb:
            items = [(str(r["id"]), str(r["combined"])) for _, r in df.assign(combined=df["summary"].fillna("") + "\n" + df["description"].fillna("")).iterrows()]
            upsert_kb(items, embed_model)
    except Exception:
        use_kb = False
    embeddings: List[List[float]] = []
    if provider == "openai":
        for chunk in batched(texts, 64):
            try:
                resp = openai_client.embeddings.create(model=embed_model, input=chunk)
                for item in resp.data:
                    emb = getattr(item, "embedding", None) or []
                    if not emb:
                        emb = [0.0] * (3072 if "text-embedding-3-large" in (embed_model or "").lower() else 1536)
                    embeddings.append(emb)
            except Exception:
                dim = 3072 if "text-embedding-3-large" in (embed_model or "").lower() else 1536
                for _ in chunk:
                    embeddings.append([0.0] * dim)
    else:
        for chunk in batched(texts, 32):
            for t in chunk:
                try:
                    res = genai.embed_content(model=embed_model, content=t)
                    emb = res.get("embedding") or (res.get("data", {}) or {}).get("embedding")
                    if isinstance(emb, dict) and "values" in emb:
                        emb = emb.get("values")
                    embeddings.append(emb or [])
                except Exception:
                    embeddings.append([])
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
            sample_summary = (str(group["summary"].iloc[0]) if "summary" in group.columns and len(group) > 0 else "").strip()
            sample_desc = (str(group["description"].iloc[0]) if "description" in group.columns and len(group) > 0 else "").strip()
            feature_hint = (str(group["feature"].mode(dropna=True).astype(str).iloc[0]) if "feature" in group and not group["feature"].isna().all() else "General")
            error_hint = (str(group["error_type"].mode(dropna=True).astype(str).iloc[0]) if "error_type" in group and not group["error_type"].isna().all() else "General")
            tickets_count = int(len(group))
            first_sentence = (sample_summary or sample_desc).split(".")[0][:180]
            summary = (
                f"{first_sentence}."
                f" Observed in {tickets_count} ticket(s); scope appears limited."
                f" Likely area: {feature_hint}; failure type looks like {error_hint}."
                " Root cause not yet confirmed. Next action: reproduce and check logs."
            )
        # Build concise crux (5–10 words)
        try:
            crux_prompt = (
                "From this summary, extract a concise 5–10 word crux capturing the main failure and cause. "
                "Return only the phrase.\n\n" + summary
            )
            ok2, crux = _safe_generate_text(chat_model, crux_prompt)
            crux = crux if ok2 and crux else f"{feature_hint} {error_hint} issue in reported flow"
        except Exception:
            crux = f"{feature_hint} {error_hint} issue in reported flow"
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


