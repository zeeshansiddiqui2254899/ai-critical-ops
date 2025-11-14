import os
import sys
import pandas as pd
from typing import List, Dict
from openai import OpenAI
from google.oauth2.service_account import Credentials
import gspread
from datetime import datetime


def load_clusters_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    if "cluster_id" not in df.columns:
        raise ValueError("Input CSV must contain 'cluster_id' column.")
    # Build 'combined' if not present
    if "combined" not in df.columns:
        summary_col = df["summary"] if "summary" in df.columns else ""
        desc_col = df["description"] if "description" in df.columns else ""
        if isinstance(summary_col, str):
            # If column missing, above assignment yields a scalar string; coerce to empty list
            df["combined"] = ""
        else:
            df["combined"] = summary_col.fillna("").astype(str) + "\n" + desc_col.fillna("").astype(str)
    return df


def summarize_clusters(
    df: pd.DataFrame,
    client: OpenAI,
    model: str,
    temperature: float = 0.2,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    # Coerce dates
    def _parse_ts(s: str):
        try:
            return pd.to_datetime(s, errors="coerce", utc=True)
        except Exception:
            return pd.NaT
    if "created" in df.columns:
        df["created_ts"] = df["created"].apply(_parse_ts)
    else:
        df["created_ts"] = pd.NaT
    if "updated" in df.columns:
        df["updated_ts"] = df["updated"].apply(_parse_ts)
    else:
        df["updated_ts"] = pd.NaT

    for cluster_id, group in df.groupby("cluster_id"):
        texts = group["combined"].astype(str).tolist()
        joined_text = "\n".join(texts)
        if len(joined_text) > 8000:
            joined_text = joined_text[:8000]

        ticket_ids = [str(x) for x in group.get("id", pd.Series(dtype=str)).astype(str).tolist()]
        features = sorted(set([str(x) for x in group.get("feature", pd.Series(dtype=str)).astype(str).tolist() if str(x) and str(x) != "nan"]))
        error_types = sorted(set([str(x) for x in group.get("error_type", pd.Series(dtype=str)).astype(str).tolist() if str(x) and str(x) != "nan"]))

        first_seen = pd.NaT
        last_seen = pd.NaT
        try:
            cts = group["created_ts"]
            uts = group["updated_ts"]
            series = pd.concat([cts, uts]).dropna()
            if not series.empty:
                first_seen = series.min()
                last_seen = series.max()
        except Exception:
            pass
        first_seen_str = first_seen.strftime("%Y-%m-%d") if pd.notna(first_seen) else ""
        last_seen_str = last_seen.strftime("%Y-%m-%d") if pd.notna(last_seen) else ""

        count = int(len(group))
        # Simple severity heuristic
        severity = "Low"
        if count >= 5:
            severity = "High"
        elif count >= 3:
            severity = "Medium"

        # Frequency as spread days + count (string)
        try:
            spread_days = 0
            if pd.notna(first_seen) and pd.notna(last_seen):
                spread_days = max(0, int((last_seen - first_seen).total_seconds() // 86400))
            frequency = f"{spread_days}d + {count}"
        except Exception:
            frequency = str(count)

        # LLM: produce PRD-aligned fields in JSON
        prompt = (
            "You are generating a pattern summary for a cluster of related Jira issues.\n"
            "Return a single JSON object with keys EXACTLY: "
            "pattern_summary, detailed_summary, root_cause, recommendation.\n"
            "Requirements:\n"
            "- pattern_summary: one concise sentence capturing the repeated issue (no hype).\n"
            "- detailed_summary: 3–4 short sentences explaining what breaks, where, and context.\n"
            "- root_cause: likely cause in one sentence.\n"
            "- recommendation: one clear engineering action.\n\n"
            f"Cluster context (examples):\n{joined_text}\n"
        )
        pattern_summary = ""
        detailed_summary = ""
        root_cause = ""
        recommendation = ""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=320,
                response_format={"type": "json_object"},
            )
            content = (resp.choices[0].message.content or "").strip()
            import json, re as _re
            m = _re.search(r"\{[\s\S]*\}", content)
            if m:
                obj = json.loads(m.group(0))
                pattern_summary = (obj.get("pattern_summary") or "").strip()
                detailed_summary = (obj.get("detailed_summary") or "").strip()
                root_cause = (obj.get("root_cause") or "").strip()
                recommendation = (obj.get("recommendation") or "").strip()
        except Exception:
            # Fallbacks
            sample = " ".join(texts[:1])
            pattern_summary = pattern_summary or f"Repeated issue pattern: {sample[:120]}..."
            detailed_summary = detailed_summary or "Related issues show similar failures across multiple tickets over the tracked period."
            root_cause = root_cause or "Cause not confidently inferred."
            recommendation = recommendation or "Investigate logs and add validation and error handling in the affected module."

        rows.append({
            "Cluster ID": cluster_id,
            "Pattern Summary": pattern_summary,
            "Detailed Summary": detailed_summary,
            "Root Cause (AI Inferred)": root_cause,
            "Recommendation (AI Inferred)": recommendation,
            "Ticket Count": count,
            "Tickets": ", ".join(ticket_ids),
            "Features Impacted": ", ".join(features) if features else "",
            "Error Types": ", ".join(error_types) if error_types else "",
            "First Seen": first_seen_str,
            "Last Seen": last_seen_str,
            "Frequency": frequency,
            "Severity Indicator (Auto)": severity,
        })

    return pd.DataFrame(rows)


def main() -> None:
    input_csv = os.getenv("CLUSTERS_INPUT", "critical_ops_clusters.csv")
    output_csv = os.getenv("PATTERNS_OUTPUT", "critical_ops_patterns.csv")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise SystemExit("Missing OPENAI_API_KEY")
    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    sheet_id = os.getenv("SHEET_ID")

    df = load_clusters_csv(input_csv)

    client = OpenAI(api_key=openai_api_key)
    result = summarize_clusters(df, client, chat_model, temperature=0.2)
    result.to_csv(output_csv, index=False)
    print(f"Saved pattern summaries to {output_csv} (rows={len(result)})")

    # Optionally write to Google Sheet ("Patterns" tab) with PRD columns
    if sheet_id:
        try:
            creds = Credentials.from_service_account_file(
                "service_account.json",
                scopes=["https://www.googleapis.com/auth/spreadsheets"],
            )
            gc = gspread.authorize(creds)
            sh = gc.open_by_key(sheet_id)
            title = "Patterns"
            try:
                ws = sh.worksheet(title)
            except gspread.exceptions.WorksheetNotFound:
                ws = sh.add_worksheet(title=title, rows="1000", cols="10")
            ws.clear()
            headers = [
                "Cluster ID",
                "Pattern Summary",
                "Detailed Summary",
                "Root Cause (AI Inferred)",
                "Recommendation (AI Inferred)",
                "Ticket Count",
                "Tickets",
                "Features Impacted",
                "Error Types",
                "First Seen",
                "Last Seen",
                "Frequency",
                "Severity Indicator (Auto)",
            ]
            ws.append_row(headers)
            if not result.empty:
                ws.append_rows(result[headers].astype(str).values.tolist())
            print(f"Updated Google Sheet tab '{title}' with {len(result)} rows.")
        except Exception as e:
            print(f"Warning: failed to update Google Sheet Patterns tab: {e}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise

import os
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    load_dotenv()

    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    input_path = "critical_ops_clusters.csv"
    if not os.path.exists(input_path):
        raise SystemExit(
            f"Input CSV not found: {input_path}. Run analyze_similarity.py first."
        )

    df = pd.read_csv(input_path)
    if "cluster_id" not in df.columns or "combined" not in df.columns:
        raise SystemExit("CSV missing required columns: cluster_id, combined")

    summaries: List[Dict[str, object]] = []

    for cluster_id, group in df.groupby("cluster_id", dropna=False):
        texts = group["combined"].dropna().astype(str).tolist()
        if not texts:
            continue

        joined_text = "\n\n".join(texts)
        prompt = (
            "You are analyzing related incident tickets. Summarize the recurring problem/root cause "
            "succinctly in one sentence. Mention any key systems or vendors if obvious.\n\n"
            f"Tickets:\n{joined_text}\n\nReturn exactly one concise sentence."
        )

        response = client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        summary = response.choices[0].message.content.strip()
        summaries.append({
            "cluster_id": cluster_id,
            "summary": summary,
            "count": len(group),
        })

    pattern_df = pd.DataFrame(summaries)
    output_path = "critical_ops_patterns.csv"
    pattern_df.to_csv(output_path, index=False)
    print(f"Saved cluster summaries to {output_path}")


if __name__ == "__main__":
    main()



