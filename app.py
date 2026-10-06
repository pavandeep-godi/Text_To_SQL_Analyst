"""Streamlit interface for the Data Analyst Agent demo."""

from __future__ import annotations

from typing import Any

import streamlit as st
import pandas as pd
import plotly.express as px

from insights_agent import (
    DEFAULT_DISPLAY_UNIT,
    DISPLAY_UNITS,
    analyze,
    format_money,
    has_groq_api_key,
)

st.set_page_config(
    page_title="Data Analyst Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)
PLOTLY_CONFIG = {
    "displayModeBar": True,
    "modeBarButtons": [["toImage"]],
    "displaylogo": False,
}
CHART_TEMPLATE = "plotly_white"
CHART_TEXT_COLOR = "#1c302a"
CHART_GRID_COLOR = "rgba(127,127,127,.24)"
CHART_PLOT_BACKGROUND = "rgba(0,0,0,0)"
CHART_COLORS = ["#123f73", "#3e83c4", "#72aade", "#b9d9f2"]
st.markdown(
    """
    <style>
        :root {--ink:#1c302a;--muted:#687b72;--forest:#173c32;--forest-2:#245849;--mint:#bcebd1;--coral:#e8875d;}
        [data-testid="stHeader"] {background:transparent;}
        .block-container {padding:clamp(4rem,4vw,4.5rem) clamp(.9rem,4vw,2.2rem) 3rem;max-width:1500px;}
        [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], button[data-testid="collapsedControl"] {display:none !important;}
        .hero {min-height:168px;padding:1.65rem 1.9rem;border:1px solid #28594a;border-radius:23px;background:linear-gradient(112deg,#122f29 0%,#1d5142 66%,#286754 100%);color:#fff;margin:.5rem 0 1.15rem;box-shadow:0 16px 36px #173c321c;position:relative;overflow:hidden;}
        .hero:after {content:"";position:absolute;width:250px;height:250px;border-radius:50%;right:-55px;top:-120px;background:radial-gradient(circle,#bcebd155 0%,#bcebd100 70%);pointer-events:none;}
        .hero-kicker {color:#bcebd1;text-transform:uppercase;letter-spacing:.17em;font-size:.72rem;font-weight:750;margin:0 0 .6rem;}
        .hero h1 {color:#fff;letter-spacing:-.045em;margin:0 0 .45rem;font-size:clamp(1.8rem,4vw,2.55rem);line-height:1.1;}
        .hero p {color:#e0eee6;margin:0;font-size:1.02rem;max-width:690px;}
        [data-testid="stMetric"] {background:rgba(127,127,127,.07);border:1px solid rgba(127,127,127,.3);padding:16px 18px;border-radius:17px;box-shadow:0 7px 20px rgba(23,60,50,.06);}
        [data-testid="stMetricLabel"] p {font-weight:650;opacity:.82;}
        [data-testid="stMetricValue"] {letter-spacing:-.03em;}
        [data-testid="stTextInput"] input {border:1px solid rgba(127,127,127,.45);border-radius:12px;background:transparent !important;color:inherit !important;min-height:3.1rem;}
        [data-testid="stTextInput"] input:focus {border-color:#2d795e;box-shadow:0 0 0 .15rem #2d795e2b;}
        button[kind="primary"] {background:#1d624c;border:1px solid #1d624c;border-radius:11px;color:#fff;font-weight:680;min-height:2.75rem;}
        button[kind="primary"]:hover {background:#164d3c;border-color:#164d3c;color:#fff;}
        [data-testid="stExpander"] {background:transparent;border:1px solid rgba(127,127,127,.3);border-radius:14px;}
        [data-testid="stAlert"] {border-radius:13px;border:1px solid rgba(127,127,127,.3);}
        [data-testid="stPlotlyChart"], .stPlotlyChart {width:100% !important;}
        [data-testid="stPlotlyChart"] .main-svg .bg {fill:Canvas !important;fill-opacity:1 !important;}
        [data-testid="stPlotlyChart"] .main-svg text {fill:CanvasText !important;}
        [data-testid="stPlotlyChart"] .main-svg .gridlayer path {stroke:color-mix(in srgb, CanvasText 18%, Canvas) !important;}
        h1,h2,h3 {letter-spacing:-.025em;}
        .small-note {color:inherit;opacity:.78;font-size:.9rem;}
        .section-note {padding:.7rem .9rem;border-left:3px solid #2d795e;background:rgba(45,121,94,.14);border-radius:0 10px 10px 0;color:inherit;font-size:.91rem;}
        @media (max-width: 640px) {
            .block-container {padding:4rem .85rem 2rem;}
            .hero {min-height:0;padding:1.25rem 1.15rem;border-radius:18px;}
            .hero p {font-size:.96rem;}
            [data-testid="stMetric"] {padding:12px;border-radius:14px;}
        }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.expander("About this demo", expanded=False):
    about_scope, about_data, about_method = st.columns(3, gap="large")
    with about_scope:
        st.markdown("#### About the company")
        st.write("This fictional company is a B2B technology provider that sells analytics software, cloud products, hardware, and professional services to business and public-sector customers.")
        st.write("It also purchases components, cloud and logistics services, packaging, and other supplies to fulfil customer orders; all records are synthetic and for demonstration only.")
    with about_data:
        st.markdown("#### What the datasets cover")
        st.write("Sales includes customers, products, revenue, discounts, payments, shipping, and returns. Procurement includes suppliers, purchase orders, spend, delivery, quality, and invoices.")
    with about_method:
        st.markdown("#### How answers are produced")
        st.write("Groq interprets each question, writes a result-based insight, and independently validates it. If validation finds an issue, AI automatically corrects the insight and validates it again, up to two times. DuckDB runs reviewed SQL locally; the model cannot generate executable SQL.")
    st.caption(
        "This demo uses a general-purpose AI model, not a custom-trained chatbot. "
        "It can understand many ways of asking, though ambiguous or unsupported requests may need rephrasing. "
        "Review the interpretation and results; name a metric, group, and year when helpful. "
        "Synthetic data only—do not submit confidential information."
    )

st.markdown(
    '<div class="hero"><div class="hero-kicker">Fieldnote · Business intelligence</div>'
    '<h1>Make the numbers make sense.</h1>'
    '<p>Ask a business question in plain language. Get a grounded answer, a useful visual, and a transparent validation check.</p></div>',
    unsafe_allow_html=True,
)

display_unit = DEFAULT_DISPLAY_UNIT

st.markdown("### What can I ask about?")
st.caption("Explore two sample datasets. Choose a starter question, or type a full question, shorthand, or rough phrase—the AI will restate it clearly before analyzing.")


def use_starter_question(prompt: str) -> None:
    st.session_state["question_input"] = prompt


sales_topic, procurement_topic = st.columns(2, gap="medium")
with sales_topic:
    with st.container(border=True):
        st.markdown("#### 🛍️ Sales data")
        st.write("Orders, customers, products, regions, sales channels, discounts, net sales, gross margin, payments, shipping, and returns.")
        sales_examples_left, sales_examples_right = st.columns(2)
        with sales_examples_left:
            st.button(
                "Monthly sales trends",
                key="starter_sales_trend",
                width="stretch",
                on_click=use_starter_question,
                args=("Show monthly sales trends in 2025",),
            )
        with sales_examples_right:
            st.button(
                "Top products",
                key="starter_top_products",
                width="stretch",
                on_click=use_starter_question,
                args=("Which products generated the most revenue?",),
            )
        sales_examples_left, sales_examples_right = st.columns(2)
        with sales_examples_left:
            st.button(
                "Sales by region",
                key="starter_sales_region",
                width="stretch",
                on_click=use_starter_question,
                args=("Which regions had the highest sales in 2025?",),
            )
        with sales_examples_right:
            st.button(
                "Sales by segment",
                key="starter_sales_segment",
                width="stretch",
                on_click=use_starter_question,
                args=("Compare sales by customer segment",),
            )

with procurement_topic:
    with st.container(border=True):
        st.markdown("#### 📦 Procurement data")
        st.write("Purchase orders, suppliers, item categories, spend, quantities, delivery dates, lead times, quality, invoice status, and payment terms.")
        procurement_examples_left, procurement_examples_right = st.columns(2)
        with procurement_examples_left:
            st.button(
                "Supplier delivery",
                key="starter_supplier_delivery",
                width="stretch",
                on_click=use_starter_question,
                args=("Which suppliers have the best on-time delivery?",),
            )
        with procurement_examples_right:
            st.button(
                "Spend by category",
                key="starter_procurement_spend",
                width="stretch",
                on_click=use_starter_question,
                args=("Show procurement spend by category in 2025",),
            )
        procurement_examples_left, procurement_examples_right = st.columns(2)
        with procurement_examples_left:
            st.button(
                "Supplier quality & spend",
                key="starter_supplier_scorecard",
                width="stretch",
                on_click=use_starter_question,
                args=("Compare suppliers by procurement spend and quality rating",),
            )
        with procurement_examples_right:
            st.button(
                "Annual sales vs spend",
                key="starter_annual_comparison",
                width="stretch",
                on_click=use_starter_question,
                args=("Compare annual sales and procurement spend by year",),
            )

st.markdown("#### Your business question")

question = st.text_input(
    "Your business question",
    label_visibility="collapsed",
    key="question_input",
    placeholder="e.g. ‘sales 2025 month trend’ or ‘which place sold most last year?’",
    help="Use shorthand or casual wording, compare supported years by group, or ask why sales, spend, margin, discounts, quality, or delivery metrics may be high or low.",
)
st.caption(
    "AI note: this demo uses a general-purpose model rather than a custom-trained chatbot. "
    "It handles many phrasings, but ambiguous or unsupported requests may need clarification; "
    "check the interpreted question and include a metric, group, or year when you can."
)
st.caption("Monetary amounts are shown in compact USD millions. Exact values are available in the results table and CSV download.")
api_key_configured = has_groq_api_key()
if not api_key_configured:
    st.warning("Groq API key required. Add GROQ_API_KEY to a local .env file using .env.example as a guide, then restart the app.")
run = st.button("Generate analysis", type="primary", disabled=not question.strip() or not api_key_configured)

if run:
    with st.spinner("Analyzing the sample data..."):
        try:
            result = analyze(question, display_unit=display_unit)
        except Exception as exc:
            st.error(f"Analysis could not be completed: {exc}")
            st.stop()

    st.subheader(result.title)
    # Escape currency markers so Streamlit does not mistake dollar amounts for LaTeX.
    st.markdown(result.narrative.replace("$", r"\$"))
    if result.interpreted_question.strip().casefold().rstrip(".!?") != result.question.strip().casefold().rstrip(".!?"):
        st.caption(f"Understood as: {result.interpreted_question}")

    st.caption(f"Groq interpreted this as `{result.intent}` · Rows returned: {len(result.frame):,}")
    if result.ai_validation_passed:
        if result.ai_correction_attempts:
            attempt_label = "attempt" if result.ai_correction_attempts == 1 else "attempts"
            st.success(
                f"AI validation passed after Groq corrected the insight ({result.ai_correction_attempts} revision {attempt_label})."
            )
        else:
            st.success("AI validation passed: the reviewer found the insight supported by the query result.")
    else:
        st.warning(
            f"AI validation still flagged the insight after {result.ai_correction_attempts} automatic revision attempt(s): "
            + ("; ".join(result.ai_validation_issues) or "human review is recommended")
        )

    if result.intent == "annual_comparison" and not result.frame.empty:
        if result.comparison_years:
            earlier_year, later_year = result.comparison_years
            st.caption(
                f"Comparing only {earlier_year} and {later_year}. Sales means net sales after discounts; "
                "procurement spend is purchase-order spend. Their ratio is not profit or a margin."
            )
            comparison_rows = result.frame.set_index("fiscal_year")
            earlier = comparison_rows.loc[earlier_year]
            later = comparison_rows.loc[later_year]

            if later_year == 2026:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(
                    "Sales · 2026 YTD", format_money(later["sales_ytd"], display_unit),
                    f"{later['sales_yoy_change']} vs matched 2025 period",
                )
                c2.metric(
                    "Procurement spend · 2026 YTD", format_money(later["spend_ytd"], display_unit),
                    f"{later['spend_yoy_change']} vs matched 2025 period",
                )
                c3.metric("Sales change", later["sales_yoy_change"])
                c4.metric("Spend change", later["spend_yoy_change"])
            else:

                def comparison_delta(current: float, previous: float) -> str:
                    difference = current - previous
                    percent = difference / abs(previous) * 100 if previous else 0.0
                    sign = "+" if difference >= 0 else "−"
                    return f"{sign}{format_money(abs(difference), display_unit)} · {percent:+.1f}%"

                c1, c2, c3, c4 = st.columns(4)
                c1.metric(
                    f"Sales · {later_year}", format_money(later["sales_revenue"], display_unit),
                    comparison_delta(later["sales_revenue"], earlier["sales_revenue"]),
                )
                c2.metric(
                    f"Procurement spend · {later_year}", format_money(later["procurement_spend"], display_unit),
                    comparison_delta(later["procurement_spend"], earlier["procurement_spend"]),
                )
                c3.metric(
                    f"Gross margin · {later_year}", format_money(later["gross_margin"], display_unit),
                    comparison_delta(later["gross_margin"], earlier["gross_margin"]),
                )
                ratio_delta = later["procurement_to_sales_pct"] - earlier["procurement_to_sales_pct"]
                c4.metric(
                    f"Procurement / sales · {later_year}", f"{later['procurement_to_sales_pct']:.1f}%",
                    f"{ratio_delta:+.1f} percentage points",
                )
        else:
            st.caption(
                "Sales here means net sales after discounts. Procurement spend is purchase-order spend; "
                "comparing these is not the same as comparing profit and margin. 2026 is year-to-date; "
                "its year-over-year change uses the matching period in 2025."
            )
            complete_year = result.frame.loc[result.frame["fiscal_year"] < 2026].iloc[-1]
            c1, c2, c3, c4 = st.columns(4)
            full_year = int(complete_year["fiscal_year"])
            c1.metric(f"{full_year} sales", format_money(complete_year["sales_revenue"], display_unit))
            c2.metric(f"{full_year} procurement spend", format_money(complete_year["procurement_spend"], display_unit))
            c3.metric(f"{full_year} gross sales", format_money(complete_year["gross_sales"], display_unit))
            c4.metric(f"Procurement / sales ({full_year})", f"{complete_year['procurement_to_sales_pct']:.1f}%")
    elif result.intent == "sales_summary" and not result.frame.empty:
        row = result.frame.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Net revenue", format_money(row["net_revenue"], display_unit))
        c2.metric("Orders", f"{int(row['orders']):,}")
        c3.metric("Gross margin", format_money(row["gross_margin"], display_unit))
        c4.metric("Average discount", f"{row['avg_discount_pct'] * 100:.1f}%")
        divisor, suffix = DISPLAY_UNITS[display_unit]
        summary_chart = pd.DataFrame({
            "Measure": ["Net revenue", "Gross margin"],
            "Amount": [row["net_revenue"] / divisor, row["gross_margin"] / divisor],
            "Value label": [format_money(row["net_revenue"], display_unit), format_money(row["gross_margin"], display_unit)],
        })
        summary_figure = px.bar(
            summary_chart,
            x="Measure",
            y="Amount",
            color="Measure",
            text="Value label",
            title="Revenue and gross margin",
            labels={"Amount": f"USD ({suffix})"},
            color_discrete_sequence=CHART_COLORS,
            template=CHART_TEMPLATE,
        )
        summary_figure.update_traces(textposition="outside", cliponaxis=False, textfont=dict(color=CHART_TEXT_COLOR, size=13))
        st.plotly_chart(summary_figure, width="stretch", config=PLOTLY_CONFIG)
    if result.intent != "sales_summary" and not result.frame.empty and result.chart_type != "metric":
        divisor, suffix = DISPLAY_UNITS[display_unit]
        unit_axis_label = f"USD ({suffix})"
        chart_specs: list[tuple[str, Any, bool]] = []

        def add_horizontal_metric_chart(
            frame: pd.DataFrame,
            metric: str,
            title: str,
            axis_label: str,
            *,
            monetary: bool = False,
            suffix_label: str = "",
        ) -> None:
            chart_data = frame.copy()
            chart_data["_plot_value"] = chart_data[metric] / divisor if monetary else chart_data[metric]
            if monetary:
                labels = chart_data[metric].map(lambda value: format_money(value, display_unit))
            else:
                labels = chart_data[metric].map(lambda value: f"{value:,.1f}{suffix_label}")
            category = frame.columns[0]
            chart = px.bar(
                chart_data,
                x="_plot_value",
                y=category,
                orientation="h",
                color="_plot_value",
                color_continuous_scale=[CHART_COLORS[-1], CHART_COLORS[0]],
                text=labels,
                title=title,
                labels={"_plot_value": axis_label, category: category.replace("_", " ").title()},
                template=CHART_TEMPLATE,
            )
            chart.update_coloraxes(showscale=False)
            chart.update_traces(textposition="outside", cliponaxis=False, textfont=dict(color=CHART_TEXT_COLOR, size=12))
            chart_specs.append((title, chart, True))

        if result.intent == "driver_analysis":
            driver_chart_frame = result.frame.copy()
            driver_chart_frame.insert(
                0,
                "driver_label",
                driver_chart_frame["driver_type"] + " · " + driver_chart_frame["driver"],
            )
            monetary_metric = result.driver_metric in {"procurement_spend", "sales_revenue", "gross_margin"}
            if result.driver_metric == "supplier_quality":
                axis_label, suffix_label = "Average quality rating (out of 5)", ""
            elif result.driver_metric == "delivery_performance":
                axis_label, suffix_label = "Late deliveries", "%"
            elif result.driver_metric == "discount_pct":
                axis_label, suffix_label = "Average discount", "%"
            else:
                axis_label, suffix_label = unit_axis_label, ""
            add_horizontal_metric_chart(
                driver_chart_frame,
                "metric_value",
                result.title,
                axis_label,
                monetary=monetary_metric,
                suffix_label=suffix_label,
            )
        elif result.intent == "grouped_year_comparison":
            chart_data = result.frame.copy()
            chart_data["fiscal_year"] = chart_data["fiscal_year"].astype(str)
            chart_data = chart_data.sort_values("metric_value", ascending=True)
            monetary_metric = result.comparison_metric in {"sales_revenue", "gross_margin", "procurement_spend"}
            chart_labels = (
                chart_data["metric_value"].map(lambda value: format_money(value, display_unit))
                if monetary_metric
                else chart_data["metric_value"].map(lambda value: f"{value:.1f}%")
            )
            chart = px.bar(
                chart_data,
                x="metric_value",
                y="group_name",
                color="fiscal_year",
                orientation="h",
                barmode="group",
                text=chart_labels,
                title=result.title,
                labels={"metric_value": "USD (M)" if monetary_metric else "Rate (%)", "group_name": result.comparison_group.title()},
                color_discrete_sequence=CHART_COLORS[:2],
                template=CHART_TEMPLATE,
                hover_data=["contribution_pct"] if "contribution_pct" in chart_data else None,
            )
            chart.update_traces(textposition="outside", cliponaxis=False, textfont=dict(size=11))
            chart_specs.append((result.title, chart, True))
        elif result.intent == "annual_comparison":
            chart_data = result.frame.melt(
                id_vars="fiscal_year",
                value_vars=["sales_revenue", "procurement_spend"],
                var_name="Measure",
                value_name="Amount",
            )
            chart_data["Amount"] /= divisor
            chart_data["fiscal_year"] = chart_data["fiscal_year"].astype(str)
            chart_data["Measure"] = chart_data["Measure"].map(
                {"sales_revenue": "Sales", "procurement_spend": "Procurement spend"}
            )
            chart = px.bar(
                chart_data,
                x="fiscal_year",
                y="Amount",
                color="Measure",
                title="Sales and procurement spend by year",
                labels={"Amount": unit_axis_label, "fiscal_year": "Fiscal year"},
                color_discrete_map={"Sales": CHART_COLORS[0], "Procurement spend": CHART_COLORS[1]},
                template=CHART_TEMPLATE,
                barmode="group",
            )
            chart.update_traces(texttemplate="$%{y:.2f}M", textposition="outside", cliponaxis=False, textfont=dict(size=11))
            chart_specs.append(("Annual sales and procurement comparison", chart, False))
        elif result.intent == "sales_trend":
            chart_data = result.frame.copy()
            value_columns = [column for column in ("net_revenue", "gross_margin") if column in chart_data]
            chart_data[value_columns] /= divisor
            chart = px.line(
                chart_data,
                x="period",
                y=value_columns,
                markers=True,
                labels={"period": "Month", "net_revenue": f"Sales ({suffix})", "gross_margin": f"Gross margin ({suffix})"},
                color_discrete_sequence=CHART_COLORS,
                template=CHART_TEMPLATE,
            )
            for trace in chart.data:
                last_y = float(trace.y[-1])
                trace.text = [None] * (len(trace.y) - 1) + [f"${last_y:.2f}M"]
                trace.mode = "lines+markers+text"
                trace.textposition = "top right"
                trace.cliponaxis = False
            chart_specs.append(("Monthly sales trend · Endpoint values labeled", chart, False))
        elif result.intent == "supplier_performance":
            add_horizontal_metric_chart(result.frame, "procurement_spend", "Procurement spend by supplier", unit_axis_label, monetary=True)
            add_horizontal_metric_chart(result.frame, "on_time_pct", "On-time delivery by supplier", "On-time deliveries", suffix_label="%")
            add_horizontal_metric_chart(result.frame, "avg_quality_rating", "Average supplier quality rating", "Quality rating (out of 5)")
        elif result.intent == "delivery_performance":
            add_horizontal_metric_chart(result.frame, "on_time_pct", "On-time delivery by supplier", "On-time deliveries", suffix_label="%")
            add_horizontal_metric_chart(result.frame, "avg_lead_time_days", "Average lead time by supplier", "Days")
        else:
            metric_by_intent = {
                "top_products": ("net_revenue", "Net sales by product", True),
                "regional_sales": ("net_revenue", "Net sales by region", True),
                "segment_sales": ("net_revenue", "Net sales by customer segment", True),
                "procurement_spend": ("procurement_spend", "Procurement spend by category", True),
            }
            metric, title, monetary = metric_by_intent.get(
                result.intent,
                ("net_revenue", result.title, "net_revenue" in result.frame.columns),
            )
            add_horizontal_metric_chart(
                result.frame,
                metric,
                title,
                unit_axis_label if monetary else metric.replace("_", " ").title(),
                monetary=monetary,
                suffix_label="%" if metric.endswith("_pct") else "",
            )

        st.subheader("The visual")
        chart_explanations = {
            "annual_comparison": "Paired bars show Sales and Procurement spend side by side. Bar labels are USD millions; 2026 is year-to-date.",
            "grouped_year_comparison": "Bars compare the requested group across the selected years.",
            "driver_analysis": "These are measured contributors or patterns in the sample data, not proof of a single cause.",
            "sales_trend": "The line shows the month-to-month pattern. The latest month is labeled; hover over points for exact values.",
            "supplier_performance": "Separate scorecards keep spend, delivery reliability, and quality on their own scales for a fair comparison.",
            "delivery_performance": "Each supplier is compared on on-time delivery and average lead time; lower lead time is better.",
        }
        if result.intent == "annual_comparison" and result.comparison_years:
            chart_explanations["annual_comparison"] = (
                f"Paired bars compare Sales and Procurement spend for {result.comparison_years[0]} "
                f"and {result.comparison_years[1]} only. Labels are USD millions."
            )
        if result.intent == "grouped_year_comparison" and result.comparison_years:
            matched_period_note = (
                " 2026 uses year-to-date records through 2026-10-05, matched to the same calendar period in the other selected year."
                if 2026 in result.comparison_years
                else ""
            )
            chart_explanations["grouped_year_comparison"] = (
                f"Each pair compares the same {result.comparison_group} across {result.comparison_years[0]} and "
                f"{result.comparison_years[1]}. Differences describe the data, not proven causes.{matched_period_note}"
            )
        if result.intent == "driver_analysis":
            chart_explanations["driver_analysis"] = (
                "The breakdown highlights groups with the largest measured contribution or rate. "
                "These associations can suggest where to investigate, but do not establish causation."
            )
        st.markdown(
            f'<div class="section-note">{chart_explanations.get(result.intent, "Bars are ranked from strongest to weakest; each bar is labeled with its value.")}</div>',
            unsafe_allow_html=True,
        )
        for chart_title, chart, reverse_categories in chart_specs:
            chart.update_layout(
                autosize=True,
                height=max(430, 58 * len(result.frame)) if reverse_categories else 470,
                margin=dict(l=28, r=90, t=58, b=65),
                showlegend=result.intent in ("annual_comparison", "sales_trend"),
                legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor=CHART_PLOT_BACKGROUND,
                font=dict(color=CHART_TEXT_COLOR, family="Inter, sans-serif"),
                xaxis=dict(automargin=True, gridcolor=CHART_GRID_COLOR),
                yaxis=dict(
                    automargin=True,
                    gridcolor=CHART_GRID_COLOR,
                    autorange="reversed" if reverse_categories else True,
                    tickformat=".1f",
                ),
            )
            st.plotly_chart(chart, width="stretch", config=PLOTLY_CONFIG)

        if result.intent == "annual_comparison":
            st.subheader("Annual breakdown")
            display_frame = result.frame[
                [
                    "fiscal_year", "sales_revenue", "gross_sales", "gross_margin",
                    "procurement_spend", "sales_ytd", "spend_ytd", "sales_yoy_change", "spend_yoy_change",
                    "procurement_to_sales_pct",
                ]
            ].copy()
            for column in ("sales_revenue", "gross_sales", "gross_margin", "procurement_spend", "sales_ytd", "spend_ytd"):
                display_frame[column] = display_frame[column].map(
                    lambda amount: format_money(amount, display_unit) if pd.notna(amount) else "—"
                )
            display_frame["procurement_to_sales_pct"] = display_frame["procurement_to_sales_pct"].map(lambda value: f"{value:.1f}%")
            display_frame = display_frame.rename(columns={
                "fiscal_year": "Year",
                "sales_revenue": "Sales (net, after discounts)",
                "gross_sales": "Gross sales (before discounts)",
                "gross_margin": "Gross margin",
                "procurement_spend": "Procurement spend",
                "sales_ytd": "Sales (matched YTD window)",
                "spend_ytd": "Spend (matched YTD window)",
                "sales_yoy_change": "Sales change vs prior year",
                "spend_yoy_change": "Spend change vs prior year",
                "procurement_to_sales_pct": "Procurement as % of sales",
            })
            st.dataframe(display_frame, width="stretch", hide_index=True)
        elif result.intent == "driver_analysis":
            st.subheader("Evidence behind the explanation")
            display_frame = result.frame.copy().rename(columns={
                "driver_type": "Breakdown",
                "driver": "Group",
                "metric_value": "Observed metric",
                "contribution_pct": "Share of total",
                "observations": "Records",
                "related_spend": "Procurement spend",
                "avg_late_days": "Average late days",
                "avg_lead_time_days": "Average lead time (days)",
            })
            if result.driver_metric in {"procurement_spend", "sales_revenue", "gross_margin"}:
                display_frame["Observed metric"] = display_frame["Observed metric"].map(
                    lambda value: format_money(value, display_unit)
                )
            elif result.driver_metric in {"discount_pct", "delivery_performance"}:
                display_frame["Observed metric"] = display_frame["Observed metric"].map(lambda value: f"{value:.1f}%")
            if "Share of total" in display_frame:
                display_frame["Share of total"] = display_frame["Share of total"].map(
                    lambda value: f"{value:.1f}%" if pd.notna(value) else "—"
                )
            if "Procurement spend" in display_frame:
                display_frame["Procurement spend"] = display_frame["Procurement spend"].map(
                    lambda value: format_money(value, display_unit)
                )
            st.dataframe(display_frame, width="stretch", hide_index=True)
        elif result.intent == "grouped_year_comparison":
            st.subheader("Group-by-group comparison")
            display_frame = result.frame.copy()
            monetary_metric = result.comparison_metric in {"sales_revenue", "gross_margin", "procurement_spend"}
            display_frame["metric_value"] = display_frame["metric_value"].map(
                lambda value: format_money(value, display_unit) if monetary_metric else f"{value:.1f}%"
            )
            if "contribution_pct" in display_frame:
                display_frame["contribution_pct"] = display_frame["contribution_pct"].map(
                    lambda value: f"{value:.1f}%" if pd.notna(value) else "—"
                )
            display_frame = display_frame.rename(columns={
                "fiscal_year": "Year",
                "group_name": result.comparison_group.title(),
                "metric_value": result.comparison_metric.replace("_", " ").title(),
                "observations": "Records",
                "contribution_pct": "Share of year total",
                "avg_late_days": "Average late days",
                "avg_lead_time_days": "Average lead time (days)",
            })
            st.dataframe(display_frame, width="stretch", hide_index=True)

    if result.warnings:
        with st.expander("Data validation notes"):
            for warning in result.warnings:
                st.warning(warning)

    with st.expander("View result data"):
        st.dataframe(result.frame, width="stretch", hide_index=True)
        st.download_button(
            "Download results as CSV",
            data=result.frame.to_csv(index=False).encode("utf-8"),
            file_name="analysis_results.csv",
            mime="text/csv",
        )
    with st.expander("How the analysis was produced"):
        st.write("Groq interpreted the question, generated the narrative insight, and ran an independent AI review against the computed result. DuckDB executed a reviewed SQL template; the application does not execute SQL written by the user or an LLM.")
        st.code(result.sql, language="sql")
else:
    st.info("Enter a question above and select **Generate analysis** to begin.")

st.divider()
st.markdown('<p class="small-note">Proof of concept · Synthetic data only · Not intended for financial reporting</p>', unsafe_allow_html=True)
