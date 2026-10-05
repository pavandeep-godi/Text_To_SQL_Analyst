# Data Analyst Agent

A small, beginner-friendly proof of concept that turns business questions into analysis of synthetic sales and procurement data. It shows a chart, a short explanation, the result table, and the SQL used.

> This is a demo, not a financial reporting system. The included records are generated examples and do not represent real customers, suppliers, or transactions.

> **A note about the AI:** This demo uses a general-purpose language model; it is not a custom-trained machine-learning chatbot. It can understand many phrasings, but ambiguous or unsupported requests may need clarification. Check the question interpretation and results, and include a metric, group, and year when helpful.

## What it can answer

- Sales summaries, monthly trends, top products, regions, and customer segments
- Procurement spend by item category
- Supplier spend, quality, and delivery performance
- Annual sales-versus-procurement comparisons
- Questions about 2023, 2024, 2025, or 2026

Example questions:

- “Show monthly sales trends in 2025”
- “Which products generated the most revenue?”
- “Compare sales and procurement spend by year”
- “Which suppliers have the best on-time delivery?”
- “Which regions had the highest sales in 2025?”
- “Compare sales by customer segment”
- “Compare suppliers by procurement spend and quality rating”

## Get started

You need Python 3.10 or newer.

1. Open a terminal in this project folder.
2. Create and activate a virtual environment.
3. Install the libraries listed in `requirements.txt`.
4. Copy `.env.example` to `.env` and add your Groq API key to `GROQ_API_KEY`.
5. Run `python data_generator.py` to create the sample CSV files. The generator creates 2,000 sales rows and 2,000 procurement rows across 2023–2026.
6. Run `streamlit run app.py`.
7. Open the local web address printed in the terminal, type a question, and select **Generate analysis**.

Select **About this demo** on the main page whenever you want to open the short introduction, dataset explanation, and description of how the analysis works. It stays collapsed until clicked. Answers include a plain-language takeaway and a chart or KPI summary, along with the result table.

The app has no left sidebar or display-unit input. Monetary amounts are automatically abbreviated in USD millions for readability; the downloadable CSV and raw query results preserve exact amounts. Charts stretch across the available page width, with horizontal category bars for easier label reading.

For annual comparisons, **Sales** means net sales after discounts and **Procurement spend** means purchase-order totals. The analysis also shows gross sales, gross margin, procurement as a percentage of sales, and year-over-year movement. Full years are compared with full years; 2026 year-to-date growth is compared with the same dates in 2025. Procurement-to-sales is not a profit or margin measure.

The demo's datasets are already included in `data/`. Run the generator again whenever you want to replace them with the same reproducible sample data.

## How it works

1. **Sample data:** `data_generator.py` creates realistic-looking sales and procurement CSV data. Each dataset has more than 20 columns covering dates, products or suppliers, amounts, statuses, delivery, and business dimensions.
2. **Understand the question:** Groq can work with complete sentences, shorthand, colloquial phrasing, common abbreviations, and minor typos. It rewrites the request as a clear question without changing its intent, shows that interpretation in the app, and selects a supported analysis and optional year filter. Requests outside the available analyses are flagged rather than silently treated as a different question.
3. **Generate the insight:** After the local query returns aggregated results, Groq writes a short explanation grounded in those results.
4. **Run safely:** DuckDB runs a reviewed, parameterized SQL template against the local CSV files. Arbitrary user- or model-written SQL is not executed.
5. **Validate with AI:** A separate Groq review checks whether the narrative answers the question and whether its claims are supported by the computed result. If it finds an error, Groq receives the exact review notes, corrects the insight from the query output, and validates it again (up to two correction attempts). Any issue that remains is shown as a warning. Basic Python checks also flag empty results and negative sales/spend values.
6. **Visualize:** Charts, KPI cards, the AI-written takeaway, validation status, and result table are displayed in Streamlit.

## Required Groq setup

Groq is required to analyze questions. Copy `.env.example` to `.env`, enter your key as the value of `GROQ_API_KEY`, and restart Streamlit. You can optionally change `GROQ_MODEL`; the configured default is `qwen/qwen3.8-27b` (confirm model availability in your Groq account). Keep `.env` private and never commit or share it. You must have access to the Groq API; usage limits and terms depend on your account.

Each analysis calls Groq to classify the question, write an insight from the query output, and independently validate that insight against the question and computed result. If validation finds a problem, AI automatically revises the insight using the review notes and query results, then validates it again (up to two correction attempts). The question and a small, aggregated result (up to 12 rows) are sent to Groq for these steps. Do not enter confidential or personally identifiable information. The full CSV data and SQL execution stay local; generated SQL is never executed. If the insight still fails after the retries, the app shows a warning and recommends human review.

## Data fields

**Sales** include order and line IDs, order date/year/month, customer ID/name/segment/industry/region/country, product and SKU details, quantity, unit price, discount, gross/net revenue, gross margin, currency, channel, sales representative, payment, shipping, fulfillment, warehouse, returns, and contract type.

**Procurement** include purchase order and requisition IDs, PO and delivery dates, fiscal year, supplier details and tier, item details, ordered/received quantities, unit and line costs, currency, payment terms/status, buyer, invoice status, quality rating, lead time, late days, on-time status, cost center, ship-to region, contract flag, tax, and freight.

All monetary amounts use USD. The dataset is synthetic, and values are randomized with a fixed seed so it can be regenerated consistently.

## Project files

- `app.py` — Streamlit user interface
- `insights_agent.py` — question classification, safe query templates, validation, and narrative insights
- `data_generator.py` — reproducible data generator
- `data/sales.csv` and `data/procurement.csv` — generated sample datasets
- `tests/test_agent.py` — automated checks
- `requirements.txt` — Python dependencies

## Run tests

From the project folder, run `python -m unittest discover -s tests`.
