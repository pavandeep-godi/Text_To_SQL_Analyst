from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from data_generator import generate_procurement, generate_sales
from insights_agent import _validate_with_ai, analyze, classify_question, format_money


class DataGenerationTests(unittest.TestCase):
    def test_generated_datasets_have_business_columns_and_four_years(self) -> None:
        import random

        sales = generate_sales(random.Random(4), count=40)
        procurement = generate_procurement(random.Random(5), count=40)
        self.assertGreaterEqual(len(sales[0]), 20)
        self.assertGreaterEqual(len(procurement[0]), 20)
        self.assertEqual({2023, 2024, 2025, 2026}, {row["fiscal_year"] for row in sales} | {row["fiscal_year"] for row in procurement})


class AgentTests(unittest.TestCase):
    def test_compact_usd_display_units(self) -> None:
        self.assertEqual("$5.46M", format_money(5_463_403.71, "Millions (M)"))
        self.assertEqual("$54.6 lakh", format_money(5_463_403.71, "Lakhs"))
        self.assertEqual("$0.5 crore", format_money(5_463_403.71, "Crores"))

    def test_groq_key_is_required(self) -> None:
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "Groq is required"):
                classify_question("Show procurement spend in 2025")

    def test_annual_comparison_returns_four_years(self) -> None:
        with (
            patch("insights_agent._llm_classify", return_value=("annual_comparison", None)),
            patch("insights_agent._generate_ai_narrative", return_value="AI-generated comparison insight."),
            patch("insights_agent._validate_with_ai", return_value=(True, [])),
        ):
            result = analyze("Compare sales and procurement spend by year")
        self.assertEqual("annual_comparison", result.intent)
        self.assertEqual("bar", result.chart_type)
        self.assertEqual([2023, 2024, 2025, 2026], result.frame["fiscal_year"].tolist())
        self.assertEqual([], result.warnings)
        self.assertIn("read_csv_auto", result.sql)
        self.assertIn("sales_revenue", result.frame.columns)
        self.assertIn("sales_ytd", result.frame.columns)
        self.assertIn("spend_ytd", result.frame.columns)
        self.assertIn("sales_yoy_change", result.frame.columns)
        self.assertIn("spend_yoy_change", result.frame.columns)
        self.assertIn("procurement_to_sales_pct", result.frame.columns)
        self.assertLess(result.frame["procurement_to_sales_pct"].max(), 200)
        self.assertIn("2025-10-05", result.sql)
        self.assertIn("2026-10-05", result.sql)
        prior_ytd_sales = result.frame.loc[result.frame["fiscal_year"] == 2025, "sales_ytd"].iloc[0]
        current_ytd_sales = result.frame.loc[result.frame["fiscal_year"] == 2026, "sales_ytd"].iloc[0]
        ytd_sales_change = float(
            result.frame.loc[result.frame["fiscal_year"] == 2026, "sales_yoy_change"].iloc[0].rstrip("%")
        )
        self.assertAlmostEqual((current_ytd_sales / prior_ytd_sales - 1) * 100, ytd_sales_change, places=1)
        self.assertIn("AI-generated", result.narrative)
        self.assertTrue(result.ai_validation_passed)

    def test_sales_summary_honors_year(self) -> None:
        with (
            patch("insights_agent._llm_classify", return_value=("sales_summary", 2024)),
            patch("insights_agent._generate_ai_narrative", return_value="AI-generated sales insight."),
            patch("insights_agent._validate_with_ai", return_value=(False, ["Revenue claim not supported."])),
        ):
            result = analyze("Give me a sales summary for 2024")
        self.assertEqual("sales_summary", result.intent)
        self.assertEqual(2024, result.year)
        self.assertGreater(result.frame.iloc[0]["net_revenue"], 0)
        self.assertFalse(result.ai_validation_passed)
        self.assertIn("AI validation flagged", result.warnings[-1])

    def test_ai_reviewer_parses_groq_validation_verdict(self) -> None:
        message = SimpleNamespace(content='{"passed": false, "issues": ["Unsupported comparison."]}')
        response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_kwargs: response)
            )
        )
        with patch("insights_agent._groq_client", return_value=client):
            passed, issues = _validate_with_ai(
                "Compare revenue", "annual_comparison", None,
                pd.DataFrame([{"fiscal_year": 2025, "net_revenue": 100}]), "Revenue increased."
            )
        self.assertFalse(passed)
        self.assertEqual(["Unsupported comparison."], issues)

    def test_rejects_empty_question(self) -> None:
        with self.assertRaises(ValueError):
            analyze("   ")


if __name__ == "__main__":
    unittest.main()
