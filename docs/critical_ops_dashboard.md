# 📊 Critical Ops Dashboard — Google Sheets Architecture

*(AI Agent + Google Sheets Output)*

---

## 🧭 Overview

The purpose of this dashboard is to transform the **AI-generated cluster and issue data** from Jira into a **decision-oriented view of product health**.

It converts raw ticket data into insights like:

- “Which modules are breaking most often?”
- “What types of errors are recurring?”
- “Which patterns are systemic vs isolated?”
- “Are we improving over time?”

---

## 🧩 1. SUMMARY DASHBOARD

### 🎯 Objective

To give leadership a single-glance overview of product health and recurring problem patterns across all features.

### 📊 Structure

| Column | Description | Formula / Source |
| --- | --- | --- |
| **Month** | The month of reporting (e.g., `Nov-2025`) | Derived from tab name |
| **Total Clusters** | Total unique clusters created by the AI agent | `=COUNTUNIQUE(ClusterID)` |
| **Total Tickets** | Total number of issues under those clusters | `=SUM(MonthlyTab!TotalTickets)` |
| **Recurring Clusters (%)** | % clusters with >1 ticket | `=COUNTIF(MonthlyTab!TotalTickets,">1")/COUNTUNIQUE(MonthlyTab!ClusterID)` |
| **Top 3 Features** | Most frequent features | `=TEXTJOIN(", ",TRUE,INDEX(SORT(UNIQUE(MonthlyTab!Feature),COUNTIF(MonthlyTab!Feature,UNIQUE(MonthlyTab!Feature)),FALSE),SEQUENCE(3)))` |
| **Most Common Error Type** | Mode of Error Type | `=MODE(SPLIT(TEXTJOIN(",",TRUE,MonthlyTab!ErrorType),","))` |
| **New Root Causes** | Count of root causes not seen in prior months | Compare vs previous month tab using `MATCH()` or script |
| **Stability Trend (Chart)** | Total Tickets by month | Line chart bound to this table |

### 🖼️ Visualization Ideas

- Line chart: Total tickets by month
- Donut chart: Error type distribution
- Table with color-coded stability indicators

---

## 🧱 2. FEATURE INSIGHTS

### 🎯 Objective

Pinpoint which modules or features are driving the majority of recurring issues.

### 📊 Structure

| Column | Description | Formula / Source |
| --- | --- | --- |
| **Feature** | Product module (e.g., Finance, Proposals, Reports) | From AI output |
| **Total Tickets** | Total tickets for the feature | `=COUNTIF(MonthlyTab!Feature,A2)` |
| **Recurring Clusters** | Clusters with `TotalTickets > 1` | `=COUNTIFS(MonthlyTab!Feature,A2,MonthlyTab!TotalTickets,">1")` |
| **Recurrence %** | Share of recurring clusters | `=IFERROR( RecurringClusters/TotalTickets ,0)` |
| **Common Root Cause** | Top recurring AI summary | `=INDEX(MonthlyTab!"Recurring Summary (AI Root Cause)",MATCH(A2,MonthlyTab!Feature,0))` |
| **Error Type Mix** | % split of error types | Pivot table on `Feature × ErrorType` |
| **Stability Rating** | Traffic-light indicator | `=IFS( Recurrence%>0.4,"Poor", Recurrence%>0.2,"Moderate", TRUE,"Stable")` |

### 🖼️ Visualization Ideas

- Stacked bar: Error Type Mix per Feature
- Conditional format: Stability Rating (Red/Yellow/Green)

---

## ⚙️ 3. ERROR TYPE INSIGHTS

### 🎯 Objective

Understand the **nature** of recurring issues — functional, validation, configuration, UI/UX, etc. — to separate technical debt from usability or training gaps.

### 📊 Structure

| Column | Description | Formula / Source |
| --- | --- | --- |
| **Error Type** | e.g., Functional, Config, Validation, UI/UX | From AI |
| **Total Clusters** | Count of clusters per Error Type | `=COUNTIF(MonthlyTab!ErrorType,A2)` |
| **% of Total** | Distribution percentage | `=TotalClusters / SUM(TotalClusters)` |
| **Top Features** | Top 2–3 features by count | `=TEXTJOIN(", ",TRUE,INDEX(SORT(UNIQUE(FILTER(MonthlyTab!Feature,MonthlyTab!ErrorType=A2)),COUNTIF(MonthlyTab!Feature,FILTER(MonthlyTab!Feature,MonthlyTab!ErrorType=A2)),FALSE),SEQUENCE(3)))` |
| **Example Root Cause** | Representative AI summary | `=INDEX(MonthlyTab!"Recurring Summary (AI Root Cause)",MATCH(A2,MonthlyTab!ErrorType,0))` |

### 🖼️ Visualization Ideas

- Pie chart: Error Type % share
- Bar chart: Error Type vs Cluster Count
- Line chart: Error Type trend across months

---

## 🔁 4. RECURRENCE TRACKER

### 🎯 Objective

Track which issue clusters reappear across months to detect **chronic problems** and **systemic regressions**.

### 📊 Structure

| Column | Description | Formula / Source |
| --- | --- | --- |
| **Cluster ID** | Unique cluster ID from AI | From AI |
| **Feature** | Feature for the cluster | From AI |
| **Recurring Summary** | Root cause / pattern summary | From AI |
| **Ticket Count** | Tickets per cluster | From AI |
| **First Seen** | First month cluster appeared | Compare across month tabs |
| **Last Seen** | Most recent month appeared | Compare across month tabs |
| **Frequency** | Number of months appeared | Count across month tabs |
| **Status** | Categorization by frequency | `=IFS(Frequency>2,"Chronic",Frequency=2,"Reappeared",TRUE,"Fixed")` |

### 🖼️ Visualization Ideas

- Heat map: Feature × Frequency
- Bubble chart: Recurring clusters by ticket count

---

## 📅 5. MONTHLY SNAPSHOT (Generated by Agent)

### 🎯 Objective

Keep a clean record of all AI outputs for that month to enable later pattern analysis.

### 📊 Structure (auto-generated)

| Column | Description |
| --- | --- |
| **Cluster ID** | Unique AI cluster identifier |
| **Feature** | Jira feature field |
| **Error Type** | Categorized issue type |
| **Recurring Summary (AI Root Cause)** | AI-generated root cause summary |
| **Crux** | 5–10 word distilled essence of the issue |
| **Total Tickets** | Count of issues grouped into the cluster |
| **Completion Date** | From Jira “Change completion date” |
| **Description** | Normalized concatenation of summary + description |
| **Cluster Month** | Auto-set from current month |

> The AI agent will auto-generate this tab every day at **12 PM** (if automated scheduling is enabled) and push updated data to the relevant month (e.g., `Nov-2025`).

---

## 📈 6. SUMMARY CHARTS (Optional Dashboard View)

### 🎯 Objective

Convert spreadsheet metrics into visual, story-driven insights for leadership.

### 📌 Suggested Visuals

| Chart | Purpose |
| --- | --- |
| **Line Chart** | Total clusters per month → product stability trend |
| **Donut Chart** | Error Type distribution → where the system fails most |
| **Bar Chart** | Feature vs Recurrence % → identify risky modules |
| **Heatmap** | Cluster frequency vs Feature → spotlight chronic modules |
| **Table** | Crux + Root cause summaries per cluster |

---

## 🧠 Insights to Track (for PM Review Meetings)

| Key Question | Metric / Visualization | Decision Driver |
| --- | --- | --- |
| Which module breaks most often? | Feature Insights tab | Prioritize stability work |
| Are we fixing or reintroducing bugs? | Recurrence Tracker | Evaluate QA/regression coverage |
| What types of issues dominate? | Error Type Insights | Allocate engineering vs UX vs config effort |
| New vs repeat problems? | Summary + Recurrence Trend | Distinguish novel vs legacy issues |
| Is stability improving MoM? | Summary trend | Executive KPI for reliability |

---

## 🚀 Implementation Notes

- AI Agent pushes data → Monthly Tab (`Nov-2025`, `Dec-2025`…)
- Script auto-updates Summary tab every run
- PM & QA teams review **Recurrence Tracker** monthly
- Charts & pivots update automatically
- Product briefings can reference Summary + Feature tabs directly


