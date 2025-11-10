import os
import datetime
import pandas as pd
import numpy as np
from jira import JIRA
import requests
import google.generativeai as genai
from openai import OpenAI as OpenAIClient
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv
from kb_store import upsert_kb, retrieve_context_for_text

# -----------------------------
# 1️⃣  Load environment variables
# -----------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
JIRA_BASE_URL = os.getenv("JIRA_BASE_URL") or os.getenv("JIRA_HOST")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
SHEET_ID = os.getenv("SHEET_ID")

# -----------------------------
# 2️⃣  Connect to APIs
# -----------------------------
jira = JIRA(server=JIRA_BASE_URL, basic_auth=(JIRA_EMAIL, JIRA_API_TOKEN))
provider = os.getenv("AI_PROVIDER", "gemini").lower()
if provider == "openai":
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise SystemExit("AI_PROVIDER=openai but OPENAI_API_KEY missing")
    openai_client = OpenAIClient(api_key=openai_key)
    CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
else:
    if not GEMINI_API_KEY:
        raise SystemExit("Missing GEMINI_API_KEY environment variable")
    genai.configure(api_key=GEMINI_API_KEY)
    CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
    EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")

creds = Credentials.from_service_account_file(
    "service_account.json",
    scopes=["https://www.googleapis.com/auth/spreadsheets"]
)
gc = gspread.authorize(creds)
sh = gc.open_by_key(SHEET_ID)

# Log the service account email for troubleshooting permissions
try:
    import json as _json
    with open("service_account.json", "r") as _f:
        _sa = _json.load(_f) or {}
        _email = _sa.get("client_email", "")
        print(f"Using Google service account: {_email}")
except Exception:
    pass

# -----------------------------
# Helper: resolve field id by display name (case-insensitive)
# -----------------------------
def resolve_field_id_by_name(jira_client: JIRA, display_name: str) -> str:
    try:
        fields = jira_client.fields()  # list of dicts
        target = display_name.strip().lower()
        for f in fields:
            name = (f.get("name") or "").strip().lower()
            if name == target:
                return f.get("id")
    except Exception:
        return ""
    return ""

# Coerce Jira option field values to simple strings
def coerce_option(v):
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

# Normalize and classify helpers
import re as _re
import re as _rx

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text or "")
    t = text.lower()
    t = _re.sub(r"[\w\.-]+@[\w\.-]+", " ", t)
    t = _re.sub(r"https?://\S+", " ", t)
    t = _re.sub(r"\b[A-Z]{2,}-\d+\b", " ", t)
    t = _re.sub(r"\s+", " ", t).strip()
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


def _safe_generate_text(model_name: str, prompt: str) -> str:
    provider = os.getenv("AI_PROVIDER", "gemini").lower()
    if provider == "openai":
        try:
            client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            txt = (resp.choices[0].message.content or "").strip()
            if txt:
                return txt
            resp2 = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Provide a neutral, purely technical response.\n\n" + prompt}],
                temperature=0.1,
            )
            return (resp2.choices[0].message.content or "").strip()
        except Exception:
            return ""
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
            return txt
        # retry with neutral instruction
        resp2 = mdl.generate_content("Provide a neutral, purely technical response.\n\n" + prompt)
        return _extract_text(resp2)
    except Exception:
        return ""


def classify_feature_error(text: str, feature_labels=None, error_labels=None) -> dict:
    # default choices
    feature_labels = feature_labels or ["Finance","GoodLeap","LightReach","Proposals","Contracts","Reports","Integrations","Authentication","UI","Data Pipeline"]
    error_labels = error_labels or ["Validation","Mapping","Timeout","Missing-Config","API-Error","Auth","Performance","Data-Quality"]
    constraint = f"\nFeature choices: {', '.join(feature_labels)}\nError type choices: {', '.join(error_labels)}"
    prompt = (
        "You are labeling a Jira incident. Read the text and output JSON with keys 'feature' and 'error_type'. "
        "Feature is the most relevant module/integration. Error type is the failure class. "
        "Choose from the choices when possible, or infer a specific 1–3 word label. Do NOT return General/Unknown. "
        "Return strictly JSON like {\"feature\":\"...\",\"error_type\":\"...\"}."
        + constraint + "\n\nText:\n" + text
    )
    try:
        content = _safe_generate_text(CHAT_MODEL, prompt)
        import json as _json
        m = _re.search(r"\{[\s\S]*\}", content)
        if m:
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

# -----------------------------
# 3️⃣  Fetch Jira Issues (last 15 days) via Agile board (reliable)
# -----------------------------
completion_field_name = "Change completion date"
completion_field_id = resolve_field_id_by_name(jira, completion_field_name)

project_key = os.getenv("JIRA_PROJECT", "CO")
boards_url = (JIRA_BASE_URL or JIRA_HOST).rstrip("/") + "/rest/agile/1.0/board"
resp_b = requests.get(boards_url, params={"projectKeyOrId": project_key, "maxResults": 50}, auth=(JIRA_EMAIL, JIRA_API_TOKEN), headers={"Accept": "application/json"})
resp_b.raise_for_status()
values = (resp_b.json() or {}).get("values", [])
board_id = (values or [{}])[0].get("id")
if board_id is None:
    raise SystemExit("No board found for project")

issues_url = (JIRA_BASE_URL or JIRA_HOST).rstrip("/") + f"/rest/agile/1.0/board/{board_id}/issue"
fields_list = ["summary", "description", "priority", "status", "customfield_10015", "customfield_10016"]
if completion_field_id:
    fields_list.append(completion_field_id)

data = []
start_at = 0
remaining = 500
# Determine how far back to fetch based on months history (default 5 months)
months_history = int(os.getenv("MONTHS_HISTORY", "5"))
days_back = max(90, months_history * 31)
while remaining > 0:
    batch = min(50, remaining)
    r = requests.get(
        issues_url,
        params={
            "jql": f"statusCategory=Done AND updated >= -{days_back}d ORDER BY updated DESC",
            "startAt": start_at,
            "maxResults": batch,
            "fields": ",".join(fields_list),
        },
        auth=(JIRA_EMAIL, JIRA_API_TOKEN),
        headers={"Accept": "application/json"},
    )
    r.raise_for_status()
    payload = r.json() or {}
    arr = payload.get("issues", [])
    if not arr:
        break
    for it in arr:
        f = it.get("fields", {}) or {}
        completion_date = f.get(completion_field_id) if completion_field_id else None
        completion_date = completion_date or f.get("updated")
        data.append({
            "id": it.get("key", ""),
            "summary": f.get("summary") or "",
            "description": f.get("description") or "",
            "feature": coerce_option(f.get("customfield_10015", "Unknown")),
            "error_type": coerce_option(f.get("customfield_10016", "Unknown")),
            "priority": (f.get("priority", {}) or {}).get("name", "Unknown") if isinstance(f.get("priority"), dict) else (f.get("priority") or "Unknown"),
            "completion_date": completion_date,
            "status": (f.get("status", {}) or {}).get("name", "Unknown") if isinstance(f.get("status"), dict) else (f.get("status") or "Unknown"),
        })
    got = len(arr)
    start_at += got
    remaining -= got
    if got < batch:
        break

if not data:
    print("No Jira issues found in last 90 days.")
    raise SystemExit(0)

df_all = pd.DataFrame(data)
df_all["completion_date"] = pd.to_datetime(df_all["completion_date"], errors="coerce", utc=True)

# Upsert full-month data into KB to build global context
try:
    genai  # ensure configured
    items_all = [(str(r["id"]), f"{r.get('summary','')}\n{r.get('description','')}") for _, r in df_all.iterrows()]
    upsert_kb(items_all, EMBED_MODEL)
except Exception:
    pass

def month_bounds(dt_utc: pd.Timestamp):
    start = pd.Timestamp(dt_utc.year, dt_utc.month, 1, tz="UTC")
    if dt_utc.month == 12:
        end = pd.Timestamp(dt_utc.year + 1, 1, 1, tz="UTC")
    else:
        end = pd.Timestamp(dt_utc.year, dt_utc.month + 1, 1, tz="UTC")
    return start, end

# Build last N months tabs (configurable via MONTHS_HISTORY)
def write_month(tab_name: str, df_month: pd.DataFrame):
    if df_month.empty:
        try:
            worksheet = sh.worksheet(tab_name)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = sh.add_worksheet(title=tab_name, rows="1000", cols="20")
        worksheet.clear()
        worksheet.append_row(["Cluster ID","Recurring Summary (AI Root Cause)","Crux","Feature","Error Type","Total Tickets","Example Ticket IDs","Status","Resolution Suggestion","Create Date"])
        return

    df_month = df_month.copy()
    # Compose and normalize text for classification
    df_month["combined"] = (
        df_month["summary"].fillna("") + " " +
        df_month["description"].fillna("") + " " +
        df_month["feature"].fillna("") + " " +
        df_month["error_type"].fillna("")
    )

    # AI feature/error_type classification per ticket (context-aware)
    ai_feats = []
    ai_errs = []
    for _, row in df_month.iterrows():
        text_full = (row.get("summary") or "") + "\n" + (row.get("description") or "")
        kb_ctx = ""
        try:
            kb_ctx = retrieve_context_for_text(text_full, top_k=3, max_chars=3000, embed_model=EMBED_MODEL)
        except Exception:
            kb_ctx = ""
        text_for_classify = text_full + ("\n\nSimilar tickets context:\n" + kb_ctx if kb_ctx else "")
        labels = classify_feature_error(text_for_classify)
        ai_feats.append(labels["feature"]) 
        ai_errs.append(labels["error_type"]) 
    df_month["feature"] = ai_feats
    df_month["error_type"] = ai_errs

    df_month["normalized"] = df_month["combined"].apply(normalize_text)

    def get_embedding(text: str):
        provider = os.getenv("AI_PROVIDER", "gemini").lower()
        try:
            if provider == "openai":
                resp = openai_client.embeddings.create(model=EMBED_MODEL, input=[text])
                return (resp.data[0].embedding if resp and resp.data else []) or []
            res = genai.embed_content(model=EMBED_MODEL, content=text)
            emb = res.get("embedding") or (res.get("data", {}) or {}).get("embedding")
            if isinstance(emb, dict) and "values" in emb:
                emb = emb.get("values")
            return emb or []
        except Exception:
            return []

    if len(df_month) < 2:
        df_month["embedding"] = [np.zeros((1,)) for _ in range(len(df_month))]
        df_month["cluster_id"] = 0
    else:
        df_month["embedding"] = df_month["normalized"].apply(get_embedding)
        emb = np.vstack(df_month["embedding"])
        sim = cosine_similarity(emb)
        dist = 1 - sim
        model = AgglomerativeClustering(metric='precomputed', linkage='average', distance_threshold=0.4, n_clusters=None)
        labels = model.fit_predict(dist)
        df_month["cluster_id"] = labels

    summaries = []
    for cid, group in df_month.groupby("cluster_id"):
        text = "\n".join(group["combined"].tolist())
        kb_ctx = ""
        try:
            kb_ctx = retrieve_context_for_text(text, top_k=5, max_chars=3500, embed_model=EMBED_MODEL)
        except Exception:
            kb_ctx = ""
        prompt = f"""Write a clear, non-superlative explanation for a product manager.
        Use simple language (no hype). Provide 3–5 short sentences (<= 120 words) that cover:
        1) What failed and where (system/vendor/module),
        2) Observable symptoms and scope (who/how many were affected),
        3) Likely root cause, and
        4) One recommended next action or prevention step.{("\\nRelevant similar tickets:\\n" + kb_ctx) if kb_ctx else ""}
        ---
        {text}
        ---
        """
        try:
            root_cause = _safe_generate_text(CHAT_MODEL, prompt)
            if not root_cause:
                # Deterministic fallback built from data
                sample_summary = (str(group["summary"].iloc[0]) if "summary" in group.columns and len(group) > 0 else "").strip()
                sample_desc = (str(group["description"].iloc[0]) if "description" in group.columns and len(group) > 0 else "").strip()
                feature_hint = (str(group["feature"].mode()[0]) if "feature" in group and not group["feature"].isna().all() else "General")
                error_hint = (str(group["error_type"].mode()[0]) if "error_type" in group and not group["error_type"].isna().all() else "General")
                tickets_count = int(len(group))
                first_sentence = (sample_summary or sample_desc).split(".")[0][:180]
                root_cause = (
                    f"{first_sentence}."
                    f" Observed in {tickets_count} ticket(s); scope limited to reported cases."
                    f" Likely area: {feature_hint}; failure type suggests {error_hint}."
                    " Root cause not confirmed from ticket text."
                    " Next action: reproduce, diff recent changes, and review integration logs."
                )
        except Exception as e:
            root_cause = (
                f"Observed pattern without AI summary due to error. "
                f"Likely area {feature_hint} with {error_hint} symptoms. ({e})"
            )

        # concise crux 5–10 words
        try:
            crux_prompt = (
                "From this summary, extract a concise 5–10 word crux capturing the main failure and cause. "
                "Return only the phrase.\n\n" + root_cause
            )
            crux = _safe_generate_text(CHAT_MODEL, crux_prompt)
            if not crux:
                crux = f"{feature_hint} {error_hint} issue in reported flow"
        except Exception:
            crux = f"{feature_hint} {error_hint} issue in reported flow"

        create_date = pd.to_datetime(group["completion_date"], utc=True).min().strftime("%Y-%m-%d")
        example_link = ", "; example_link = ", ".join([str(x) for x in group["id"].tolist()[:5]])

        summaries.append({
            "Cluster ID": cid,
            "Recurring Summary (AI Root Cause)": root_cause,
            "Crux": crux,
            "Feature": group["feature"].mode()[0] if not group["feature"].isna().all() else "Unknown",
            "Error Type": group["error_type"].mode()[0] if not group["error_type"].isna().all() else "Unknown",
            "Total Tickets": len(group),
            "Example Ticket IDs": example_link,
            "Status": group["status"].mode()[0] if "status" in group and not group["status"].isna().all() else "Unknown",
            "Resolution Suggestion": "See summary",
            "Create Date": create_date,
        })

    df_summary = pd.DataFrame(summaries)
    try:
        worksheet = sh.worksheet(tab_name)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = sh.add_worksheet(title=tab_name, rows="1000", cols="20")
    worksheet.clear()
    worksheet.append_row(df_summary.columns.tolist())
    if not df_summary.empty:
        worksheet.append_rows(df_summary.values.tolist())

# Determine current and previous two months and write tabs
now_utc = pd.Timestamp.now(tz="UTC")
months = []
cur = now_utc
for _ in range(months_history):
    months.append(cur)
    y = cur.year if cur.month > 1 else cur.year - 1
    m = cur.month - 1 if cur.month > 1 else 12
    cur = pd.Timestamp(y, m, 1, tz="UTC")

for mdt in months:
    start = pd.Timestamp(mdt.year, mdt.month, 1, tz="UTC")
    end = pd.Timestamp(mdt.year + 1, 1, 1, tz="UTC") if mdt.month == 12 else pd.Timestamp(mdt.year, mdt.month + 1, 1, tz="UTC")
    tab = mdt.strftime("%b-%Y")
    df_month = df_all[(df_all["completion_date"] >= start) & (df_all["completion_date"] < end)]
    write_month(tab, df_month)

print(f"✅ Google Sheet updated with last {months_history} months tabs.")

# Auto-refresh Summary tab
try:
    from update_google_sheet import update_summary_tab
    update_summary_tab()
except Exception:
    # Non-fatal if the helper is unavailable
    pass


