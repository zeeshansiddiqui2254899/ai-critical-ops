import os
import sys
import pandas as pd
from typing import List, Dict
from openai import OpenAI
from google.oauth2.service_account import Credentials
import gspread


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
    summaries: List[Dict[str, object]] = []
    for cluster_id, group in df.groupby("cluster_id"):
        # Join context. Keep reasonable cap to control token size.
        texts = group["combined"].astype(str).tolist()
        joined_text = "\n".join(texts)
        if len(joined_text) > 8000:
            joined_text = joined_text[:8000]
        prompt = (
            "Summarize the following related issue reports and infer the likely recurring problem or root cause.\n"
            "Return ONE concise sentence, neutral and specific, no second-person, no hype:\n\n"
            f"{joined_text}\n"
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=120,
            )
            summary = (resp.choices[0].message.content or "").strip()
        except Exception:
            # Deterministic fallback if API fails
            sample = " ".join(group["combined"].astype(str).tolist()[:1])
            summary = f"Grouped similar incidents; representative theme: {sample[:140]}..."

        summaries.append({"cluster_id": cluster_id, "summary": summary, "count": int(len(group))})

    return pd.DataFrame(summaries)


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

    # Optionally write to Google Sheet ("Patterns" tab)
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
            headers = ["cluster_id", "summary", "count"]
            ws.append_row(headers)
            if not result.empty:
                ws.append_rows(result[headers].values.tolist())
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



