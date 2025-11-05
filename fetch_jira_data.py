import os
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
import requests
from jira import JIRA


def get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def main() -> None:
    load_dotenv()

    jira_base_url = os.getenv("JIRA_BASE_URL") or os.getenv("JIRA_HOST")
    jira_email = os.getenv("JIRA_EMAIL")
    jira_api_token = os.getenv("JIRA_API_TOKEN")

    if not jira_base_url or not jira_email or not jira_api_token:
        raise SystemExit(
            "Missing Jira configuration. Ensure JIRA_BASE_URL (or JIRA_HOST), JIRA_EMAIL, and JIRA_API_TOKEN are set in .env"
        )

    days = get_env_int("JIRA_QUERY_DAYS", 60)
    max_results = get_env_int("MAX_RESULTS", 500)

    board_name = os.getenv("CRITICAL_OPS_BOARD_NAME", "Critical Ops") or "Critical Ops"

    # Discover project key for the given board/project name
    project_key = None
    try:
        proj_url = jira_base_url.rstrip("/") + "/rest/api/3/project/search"
        resp = requests.get(
            proj_url,
            params={"query": board_name, "maxResults": 50},
            headers={"Accept": "application/json"},
            auth=(jira_email, jira_api_token),
        )
        if resp.status_code == 200:
            values = (resp.json() or {}).get("values", [])
            # Prefer exact name match, else first contains
            for v in values:
                if (v.get("name") or "").strip().lower() == board_name.strip().lower():
                    project_key = v.get("key")
                    break
            if not project_key and values:
                project_key = values[0].get("key")
    except Exception:
        pass

    jql_project = project_key or board_name
    jql = f'project = "{jql_project}" AND created >= -{days}d ORDER BY created DESC'

    # Jira Cloud REST API v3 with pagination
    records: List[Dict[str, Any]] = []

    # Try using jira library with REST API v3
    try:
        options = {"server": jira_base_url, "rest_api_version": "3"}
        jira_client = JIRA(options=options, basic_auth=(jira_email, jira_api_token))
        issues = jira_client.search_issues(jql, maxResults=max_results)
        for issue in issues:
            fields = issue.fields
            records.append(
                {
                    "id": issue.key,
                    "summary": getattr(fields, "summary", "") or "",
                    "description": getattr(fields, "description", "") or "",
                    "feature": getattr(fields, "customfield_10015", "Unknown"),
                    "error_type": getattr(fields, "customfield_10016", "Unknown"),
                    "created": getattr(fields, "created", ""),
                    "updated": getattr(fields, "updated", ""),
                }
            )
    except Exception:
        pass

    if not records:
        # Try direct REST with POST /rest/api/3/search/jql (new API)
        url = jira_base_url.rstrip("/") + "/rest/api/3/search/jql"
        headers = {"Accept": "application/json"}
        auth = (jira_email, jira_api_token)

        start_at = 0
        page_size = 100
        remaining = max_results

        while remaining > 0:
            batch_size = min(page_size, remaining)
            payload = {
                "queries": [
                    {
                        "query": jql,
                        "startAt": start_at,
                        "maxResults": batch_size,
                    }
                ],
                "fields": [
                    "summary",
                    "description",
                    "created",
                    "updated",
                    "customfield_10015",
                    "customfield_10016",
                ],
            }
            resp = requests.post(url, json=payload, headers=headers, auth=auth)
            if resp.status_code == 200:
                data = resp.json()
                try:
                    issues = data["queries"][0]["issues"]
                except Exception:
                    issues = []
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
                            "feature": fields.get("customfield_10015", "Unknown"),
                            "error_type": fields.get("customfield_10016", "Unknown"),
                            "created": fields.get("created", ""),
                            "updated": fields.get("updated", ""),
                        }
                    )
                fetched = len(issues)
                start_at += fetched
                remaining -= fetched
                if fetched < batch_size:
                    break
            else:
                # Fallback to Agile API
                break

    if not records:
        # Agile API fallback: find board by name, then fetch issues
        headers = {"Accept": "application/json"}
        auth = (jira_email, jira_api_token)
        boards_url = jira_base_url.rstrip("/") + "/rest/agile/1.0/board"
        resp = requests.get(boards_url, params={"name": board_name}, headers=headers, auth=auth)
        if resp.status_code != 200:
            raise SystemExit(f"Jira board lookup failed: {resp.status_code} {resp.text[:200]}")
        values = (resp.json() or {}).get("values", [])
        board_id = None
        for b in values:
            if b.get("name") == board_name:
                board_id = b.get("id")
                break
        if board_id is None and values:
            board_id = values[0].get("id")
        if board_id is None:
            raise SystemExit("No matching Jira board found for name '{board_name}'.")

        start_at = 0
        remaining = max_results
        while remaining > 0:
            batch_size = min(50, remaining)
            issues_url = jira_base_url.rstrip("/") + f"/rest/agile/1.0/board/{board_id}/issue"
            params = {
                "jql": f"created >= -{days}d ORDER BY created DESC",
                "startAt": start_at,
                "maxResults": batch_size,
                "fields": "summary,description,created,updated,customfield_10015,customfield_10016",
            }
            resp = requests.get(issues_url, params=params, headers=headers, auth=auth)
            if resp.status_code != 200:
                raise SystemExit(f"Jira agile issues failed: {resp.status_code} {resp.text[:200]}")
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
                        "feature": fields.get("customfield_10015", "Unknown"),
                        "error_type": fields.get("customfield_10016", "Unknown"),
                        "created": fields.get("created", ""),
                        "updated": fields.get("updated", ""),
                    }
                )
            fetched = len(issues)
            start_at += fetched
            remaining -= fetched
            if fetched < batch_size:
                break

    df = pd.DataFrame.from_records(records)
    output_path = "critical_ops_issues.csv"
    df.to_csv(output_path, index=False)
    print(f"Exported {len(df)} issues to {output_path}")


if __name__ == "__main__":
    main()


