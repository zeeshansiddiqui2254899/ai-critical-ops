import os
import datetime
import pandas as pd
import numpy as np
from jira import JIRA
import requests
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

# -----------------------------
# 1️⃣  Load environment variables
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
JIRA_BASE_URL = os.getenv("JIRA_BASE_URL") or os.getenv("JIRA_HOST")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
SHEET_ID = os.getenv("SHEET_ID")

# -----------------------------
# 2️⃣  Connect to APIs
# -----------------------------
client = OpenAI(api_key=OPENAI_API_KEY)
jira = JIRA(server=JIRA_BASE_URL, basic_auth=(JIRA_EMAIL, JIRA_API_TOKEN))
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

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

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text or "")
    t = text.lower()
    t = _re.sub(r"[\w\.-]+@[\w\.-]+", " ", t)
    t = _re.sub(r"https?://\S+", " ", t)
    t = _re.sub(r"\b[A-Z]{2,}-\d+\b", " ", t)
    t = _re.sub(r"\s+", " ", t).strip()
    return t

def classify_feature_error(text: str, feature_labels=None, error_labels=None) -> dict:
    constraint = ""
    if feature_labels:
        constraint += f"\nFeature choices: {', '.join(feature_labels)}"
    if error_labels:
        constraint += f"\nError type choices: {', '.join(error_labels)}"
    prompt = (
        "You are labeling a Jira incident. Read the text and output JSON with keys 'feature' and 'error_type'. "
        "Feature is the module/integration (e.g., Finance, GoodLeap, LightReach, Proposals, Contracts). "
        "Error type is the failure class (e.g., Auth, Mapping, Validation, Timeout, Missing-Config, API-Error, Performance). "
        "Keep each label <= 3 words. If unsure, make your best inference (do not return Unknown)." + constraint + "\n\nText:\n" + text
    )
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = resp.choices[0].message.content.strip()
        import json as _json
        m = _re.search(r"\{[\s\S]*\}", content)
        if m:
            obj = _json.loads(m.group(0))
            feature = str(obj.get("feature") or "").strip() or "General"
            error_type = str(obj.get("error_type") or "").strip() or "General"
            return {"feature": feature, "error_type": error_type}
    except Exception:
        pass
    return {"feature": "General", "error_type": "General"}

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

    # AI feature/error_type classification per ticket (override if Unknown)
    ai_feats = []
    ai_errs = []
    for _, row in df_month.iterrows():
        text_full = (row.get("summary") or "") + "\n" + (row.get("description") or "")
        labels = classify_feature_error(text_full)
        ai_feats.append(labels["feature"]) 
        ai_errs.append(labels["error_type"]) 
    df_month["feature"] = ai_feats
    df_month["error_type"] = ai_errs

    df_month["normalized"] = df_month["combined"].apply(normalize_text)

    def get_embedding(text: str):
        res = client.embeddings.create(input=text, model="text-embedding-3-small")
        return res.data[0].embedding

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
        prompt = f"""Write a clear, non-superlative explanation for a product manager.
        Use simple language (no hype). Provide 3–5 short sentences (<= 120 words) that cover:
        1) What failed and where (system/vendor/module),
        2) Observable symptoms and scope (who/how many were affected),
        3) Likely root cause, and
        4) One recommended next action or prevention step.
        ---
        {text}
        ---
        """
        try:
            resp = client.chat.completions.create(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.2)
            root_cause = resp.choices[0].message.content.strip()
        except Exception as e:
            root_cause = f"Summary unavailable: {e}"

        # concise crux 5–10 words
        try:
            crux_prompt = (
                "From this summary, extract a concise 5–10 word crux capturing the main failure and cause. "
                "Return only the phrase.\n\n" + root_cause
            )
            crux_resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": crux_prompt}],
                temperature=0,
            )
            crux = crux_resp.choices[0].message.content.strip()
        except Exception:
            crux = "Concise crux not available"

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


