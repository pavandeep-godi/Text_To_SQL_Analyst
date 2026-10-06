from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from data_generator import generate_procurement, generate_sales
from insights_agent import _comparison_years, _driver_metric_for, _grouped_comparison_spec, _llm_classify, _normalize_classification, _validate_with_ai, analyze, classify_question, format_money


class DataGenerationTests(unittest.TestCase):
    def test_generated_datasets_have_business_columns_and_four_years(self) -> None:
        import random

        sales = generate_sales(random.Random(4), count=40)
        procurement = generate_procurement(random.Random(5), count=40)
        self.assertGreaterEqual(len(sales[0]), 20)
        self.assertGreaterEqual(len(procurement[0]), 20)
        self.assertEqual({2023, 2024, 2025, 2026}, {row["fiscal_year"] for row in sales} | {row["fiscal_year"] for row in procurement})


class AgentTests(unittest.TestCase):
    def test_normalizes_llm_string_years_and_common_intent_aliases(self) -> None:
        self.assertEqual(
            ("sales_trend", 2025, "Show the monthly sales trend for 2025."),
            _normalize_classification(
                {"intent": "sales_trend", "year": "2025", "interpreted_question": "Show the monthly sales trend for 2025."},
                "sales 2025 month trend",
            ),
        )
        self.assertEqual(
            ("regional_sales", 2025, "Which region had the highest sales in 2025?"),
            _normalize_classification(
                {"intent": "sales-by-region", "year": 2025, "interpreted_question": "Which region had the highest sales in 2025?"},
                "which place sold most last year?",
            ),
        )
        self.assertEqual(
            ("delivery_performance", None, "Which suppliers have the best delivery?"),
            _normalize_classification(
                {"intent": "supplier_delivery", "year": "all years", "interpreted_question": "Which suppliers have the best delivery?"},
                "vendors delivery best",
            ),
        )

    def test_flexible_classifier_handles_shorthand_and_relative_year(self) -> None:
        message = SimpleNamespace(
            content='{"interpreted_question":"Which regions had the highest sales in 2025?",'
            '"intent":"sales-by-region","year":"2025"}'
        )
        response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
        create = unittest.mock.Mock(return_value=response)
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        with patch("insights_agent._groq_client", return_value=client):
            result = _llm_classify("REV by region last yr pls")

        self.assertEqual(
            ("regional_sales", 2025, "Which regions had the highest sales in 2025?"),
            result,
        )
        request = create.call_args.kwargs
        self.assertEqual({"type": "json_object"}, request["response_format"])
        self.assertIn("minor spelling mistakes", request["messages"][0]["content"])
        self.assertIn("unsupported", request["messages"][0]["content"])
        self.assertIn("2024 vs 2025 sales and procurement breakdown", request["messages"][0]["content"])

    def test_extracts_only_explicitly_compared_supported_years(self) -> None:
        self.assertEqual((2024, 2025), _comparison_years("2024 vs 2025 breakdown"))
        self.assertEqual((2025, 2026), _comparison_years("Compare 2025 and 2026 sales"))
        self.assertEqual((2025, 2026), _comparison_years("Compare 2026 vs 2025 sales"))
        self.assertEqual((2023, 2025), _comparison_years("sales data from 2023 to 2025"))
        self.assertIsNone(_comparison_years("Show sales in 2025"))
        self.assertIsNone(_comparison_years("Compare sales in 2022 and 2025"))

    def test_resolves_grouped_year_comparison_measure_and_dimension(self) -> None:
        self.assertEqual(
            ("sales_revenue", "segment"),
            _grouped_comparison_spec("2023 vs 2025 sales by segment", "Compare sales by customer segment in 2023 and 2025."),
        )
        self.assertEqual(
            ("procurement_spend", "category"),
            _grouped_comparison_spec("procurement spend by category 2024 vs 2025", "Compare procurement spend by item category."),
        )

    def test_why_question_maps_to_supported_metric(self) -> None:
        self.assertEqual(
            "procurement_spend",
            _driver_metric_for("Why is procurement spend high?", "Which suppliers contribute most to procurement spend?"),
        )
        self.assertEqual(
            "delivery_performance",
            _driver_metric_for("What is causing late deliveries?", "Which suppliers have the highest late-delivery rates?"),
        )

    def test_two_year_analysis_filters_to_requested_years_and_reports_deltas(self) -> None:
        captured: dict[str, pd.DataFrame] = {}

        def generate(_question: str, _interpreted: str, _intent: str, _year: int | None, frame: pd.DataFrame, _unit: str) -> str:
            captured["frame"] = frame.copy()
            return "AI-generated two-year comparison insight."

        with (
            patch(
                "insights_agent._llm_classify",
                return_value=("annual_comparison", None, "Compare sales and procurement in 2024 and 2025."),
            ),
            patch("insights_agent._generate_ai_narrative", side_effect=generate),
            patch("insights_agent._validate_with_ai", return_value=(True, [])),
        ):
            result = analyze("2024 vs 2025 breakdown")

        self.assertEqual((2024, 2025), result.comparison_years)
        self.assertEqual("Sales vs procurement: 2024 vs 2025", result.title)
        self.assertEqual([2024, 2025], result.frame["fiscal_year"].tolist())
        self.assertEqual([2024, 2025], captured["frame"]["fiscal_year"].tolist())
        self.assertTrue(result.ai_validation_passed)

    def test_two_year_sales_by_segment_returns_each_segment_for_both_years(self) -> None:
        with (
            patch(
                "insights_agent._llm_classify",
                return_value=("grouped_year_comparison", None, "Compare sales by customer segment in 2023 and 2025."),
            ),
            patch("insights_agent._generate_ai_narrative", return_value="AI-generated segment comparison.") as generate,
            patch("insights_agent._validate_with_ai", return_value=(True, [])),
        ):
            result = analyze("2023 vs 2025 sales by segment")

        self.assertEqual("grouped_year_comparison", result.intent)
        self.assertEqual((2023, 2025), result.comparison_years)
        self.assertEqual("segment", result.comparison_group)
        self.assertEqual("sales_revenue", result.comparison_metric)
        self.assertEqual({2023, 2025}, set(result.frame["fiscal_year"]))
        self.assertGreaterEqual(result.frame["group_name"].nunique(), 4)
        self.assertEqual(2 * result.frame["group_name"].nunique(), len(result.frame))
        self.assertIn("contribution_pct", result.frame.columns)
        self.assertEqual("AI-generated segment comparison.", result.narrative)
        self.assertTrue(result.ai_validation_passed)
        self.assertIn("customer_segment", result.sql)

    def test_grouped_comparison_with_2026_uses_matching_year_to_date_window(self) -> None:
        with (
            patch(
                "insights_agent._llm_classify",
                return_value=("grouped_year_comparison", None, "Compare sales by segment for 2025 and 2026."),
            ),
            patch("insights_agent._generate_ai_narrative", return_value="Matched-period comparison."),
            patch("insights_agent._validate_with_ai", return_value=(True, [])),
        ):
            result = analyze("2025 vs 2026 sales by segment")

        self.assertEqual((2025, 2026), result.comparison_years)
        self.assertIn("STRFTIME(CAST(order_date AS DATE)", result.sql)
        self.assertIn("'10-05'", result.sql)
        self.assertEqual({2025, 2026}, set(result.frame["fiscal_year"]))

    def test_why_procurement_spend_returns_category_and_supplier_evidence(self) -> None:
        with (
            patch(
                "insights_agent._llm_classify",
                return_value=("driver_analysis", None, "Which suppliers and categories contribute most to procurement spend?"),
            ),
            patch("insights_agent._generate_ai_narrative", return_value="The largest observed contributors are supported by the breakdown."),
            patch("insights_agent._validate_with_ai", return_value=(True, [])),
        ):
            result = analyze("Why is procurement spend high?")

        self.assertEqual("driver_analysis", result.intent)
        self.assertEqual("procurement_spend", result.driver_metric)
        self.assertEqual({"Item category", "Supplier"}, set(result.frame["driver_type"]))
        self.assertIn("contribution_pct", result.frame.columns)
        self.assertIn("observations", result.frame.columns)
        self.assertGreaterEqual(len(result.frame), 6)
        self.assertLessEqual(len(result.frame), 10)
        self.assertTrue(result.ai_validation_passed)

    def test_driver_queries_cover_supported_metrics(self) -> None:
        questions_and_metrics = [
            ("Why are sales revenue low?", "sales_revenue"),
            ("Why is gross margin low?", "gross_margin"),
            ("Why are discounts high?", "discount_pct"),
            ("Why is supplier quality low?", "supplier_quality"),
            ("What is behind late deliveries?", "delivery_performance"),
        ]
        for question, expected_metric in questions_and_metrics:
            with self.subTest(metric=expected_metric):
                with (
                    patch("insights_agent._llm_classify", return_value=("driver_analysis", None, question)),
                    patch("insights_agent._generate_ai_narrative", return_value="Measured driver summary."),
                    patch("insights_agent._validate_with_ai", return_value=(True, [])),
                ):
                    result = analyze(question)
                self.assertEqual(expected_metric, result.driver_metric)
                self.assertFalse(result.frame.empty)
                self.assertIn("metric_value", result.frame.columns)

    def test_unsupported_or_unrankable_requests_are_not_misclassified(self) -> None:
        with self.assertRaisesRegex(ValueError, "couldn't confidently match"):
            _normalize_classification(
                {"intent": "unsupported", "year": None},
                "What was our profit?",
            )
        self.assertEqual(
            ("annual_comparison", None, "Compare yearly sales and spend."),
            _normalize_classification(
                {
                    "intent": "sales-vs-procurement",
                    "year": None,
                    "interpreted_question": "Compare yearly sales and spend.",
                },
                "sales spend each year",
            ),
        )

    def test_rejects_invalid_llm_year(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid year"):
            _normalize_classification({"intent": "sales_trend", "year": "2025 please"}, "sales trend")

    def test_analysis_retains_original_and_grammatically_interpreted_question(self) -> None:
        interpreted = "Show the monthly sales trend for 2025."
        with (
            patch("insights_agent._llm_classify", return_value=("sales_trend", 2025, interpreted)),
            patch("insights_agent._generate_ai_narrative", return_value="AI-generated monthly sales insight.") as generate,
            patch("insights_agent._validate_with_ai", return_value=(True, [])) as validate,
        ):
            result = analyze("sales 2025 month trend")
        self.assertEqual("sales 2025 month trend", result.question)
        self.assertEqual(interpreted, result.interpreted_question)
        self.assertEqual(interpreted, generate.call_args.args[1])
        self.assertEqual(interpreted, validate.call_args.args[-1])

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
            patch("insights_agent._llm_classify", return_value=("annual_comparison", None, "Compare annual sales and procurement spend by year.")),
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
            patch("insights_agent._llm_classify", return_value=("sales_summary", 2024, "Give me a sales summary for 2024.")),
            patch("insights_agent._generate_ai_narrative", return_value="AI-generated sales insight."),
            patch("insights_agent._correct_ai_narrative", return_value="Corrected sales insight."),
            patch(
                "insights_agent._validate_with_ai",
                side_effect=[
                    (False, ["Revenue claim not supported."]),
                    (False, ["Corrected claim still needs review."]),
                    (False, ["Validation still fails."]),
                ],
            ),
        ):
            result = analyze("Give me a sales summary for 2024")
        self.assertEqual("sales_summary", result.intent)
        self.assertEqual(2024, result.year)
        self.assertGreater(result.frame.iloc[0]["net_revenue"], 0)
        self.assertFalse(result.ai_validation_passed)
        self.assertEqual(2, result.ai_correction_attempts)
        self.assertIn("after 2 automatic correction", result.warnings[-1])

    def test_failed_supplier_insight_is_corrected_and_revalidated(self) -> None:
        with (
            patch("insights_agent._llm_classify", return_value=("supplier_performance", None, "Compare suppliers by on-time delivery.")),
            patch("insights_agent._generate_ai_narrative", return_value="Incorrect supplier ranking."),
            patch(
                "insights_agent._correct_ai_narrative",
                return_value="Nimbus Cloud Supply ranks second by on-time delivery at 43.23%.",
            ) as correction,
            patch(
                "insights_agent._validate_with_ai",
                side_effect=[
                    (False, ["Supplier ranking does not match on-time percentages."]),
                    (True, []),
                ],
            ) as validation,
        ):
            result = analyze("Which suppliers have the best on-time delivery?")

        self.assertTrue(result.ai_validation_passed)
        self.assertEqual(1, result.ai_correction_attempts)
        self.assertEqual("Nimbus Cloud Supply ranks second by on-time delivery at 43.23%.", result.narrative)
        correction.assert_called_once()
        self.assertEqual(2, validation.call_count)

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
