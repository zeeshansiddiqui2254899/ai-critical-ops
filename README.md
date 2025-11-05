# AI-Driven Critical Ops Analyzer

End-to-end pipeline to analyze Jira Critical Ops issues using OpenAI for embeddings, duplicate detection, clustering, and summaries.

## Setup

1. Create and activate a virtual environment, then install dependencies:
```bash
pip install -r requirements.txt
```
2. Copy `env.example` to `.env` and fill the values:
```bash
cp env.example .env
```

## Run

```bash
python fetch_jira_data.py
python analyze_similarity.py
python summarize_clusters.py
```

Outputs:
- critical_ops_issues.csv
- critical_ops_clusters.csv
- critical_ops_patterns.csv

Configure models via `OPENAI_EMBED_MODEL` and `OPENAI_CHAT_MODEL` in `.env`.

