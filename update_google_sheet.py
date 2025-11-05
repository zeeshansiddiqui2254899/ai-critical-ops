import os
from datetime import datetime
import pandas as pd
import gspread
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv


def get_sheet():
    load_dotenv()
    sheet_id = os.getenv("SHEET_ID")
    creds = Credentials.from_service_account_file(
        "service_account.json",
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    gc = gspread.authorize(creds)
    return gc.open_by_key(sheet_id)


def update_monthly_tab(df: pd.DataFrame):
    sh = get_sheet()
    month_tab = datetime.now().strftime("%b-%Y")
    try:
        ws = sh.worksheet(month_tab)
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=month_tab, rows="2000", cols="26")
    ws.clear()
    set_with_dataframe(ws, df)
    print(f"✅ Updated monthly tab: {month_tab}")


def update_summary_tab():
    sh = get_sheet()
    tabs = [w.title for w in sh.worksheets() if w.title not in ["Summary", "Config", "Sheet1"]]

    summary_rows = []
    for tab in tabs:
        try:
            ws = sh.worksheet(tab)
        except gspread.exceptions.WorksheetNotFound:
            continue
        data = ws.get_all_records()
        if not data:
            continue
        df = pd.DataFrame(data)
        total_clusters = len(df)
        total_tickets = df.get("Total Tickets", pd.Series(dtype=float)).sum() if "Total Tickets" in df else 0
        top_feature = (
            df["Feature"].value_counts().idxmax() if "Feature" in df and not df["Feature"].empty else "N/A"
        )
        top_root_cause = (
            df["Recurring Summary (AI Root Cause)"].value_counts().idxmax()
            if "Recurring Summary (AI Root Cause)" in df and not df["Recurring Summary (AI Root Cause)"].empty
            else "N/A"
        )
        avg_age = round(df.get("Avg Age (Days)", pd.Series(dtype=float)).mean() or 0, 1)

        summary_rows.append(
            {
                "Month": tab,
                "Total Clusters": total_clusters,
                "Total Tickets": total_tickets,
                "Top Feature": top_feature,
                "Top Root Cause": top_root_cause,
                "Avg Ticket Age": avg_age,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("Month")
    try:
        sws = sh.worksheet("Summary")
    except gspread.exceptions.WorksheetNotFound:
        sws = sh.add_worksheet(title="Summary", rows="200", cols="20")
    sws.clear()
    set_with_dataframe(sws, summary_df)
    print("📊 Summary tab updated successfully!")


if __name__ == "__main__":
    # Example: re-hydrate from a CSV result if needed
    if os.path.exists("critical_ops_patterns.csv"):
        df = pd.read_csv("critical_ops_patterns.csv")
        update_monthly_tab(df)
    update_summary_tab()


