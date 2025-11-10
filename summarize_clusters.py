import os
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai


def main() -> None:
    load_dotenv()

    chat_model = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Missing GEMINI_API_KEY")
    genai.configure(api_key=api_key)

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

        try:
            model = genai.GenerativeModel(chat_model)
            response = model.generate_content(prompt)
            summary = (response.text or "").strip()
        except Exception:
            summary = "Summary unavailable"
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



