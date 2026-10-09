"""
Evaluation Script for Customer Support Ticket Category Classification.
Calculates classification accuracy on a benchmark dataset of labeled tickets.
Designed for college viva demonstration and performance validation.
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from src.analyzer import TicketAnalyzer, AnalyzerError
from src.llm import LLMServiceError


def run_evaluation(data_path: str = "data/sample_tickets.json", limit: int = None):
    """
    Evaluates the NLP Ticket Analyzer against ground-truth labeled customer tickets.

    :param data_path: Path to JSON file containing sample labeled tickets.
    :param limit: Optional integer to limit the number of evaluated tickets (for quick testing).
    """
    path = Path(data_path)
    if not path.exists():
        print(f"Error: Dataset not found at {path.resolve()}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        tickets = json.load(f)

    if limit and limit > 0:
        tickets = tickets[:limit]

    print("=" * 80)
    print("      AI-POWERED CUSTOMER SUPPORT TICKET ANALYZER - EVALUATION BENCHMARK")
    print("=" * 80)
    print(f"Evaluating {len(tickets)} labeled customer support tickets...")
    print("-" * 80)

    try:
        analyzer = TicketAnalyzer()
    except (AnalyzerError, LLMServiceError) as e:
        print(f"\n[Initialization Error]: {e}")
        print("Please check your .env file and ensure GEMINI_API_KEY is configured correctly.\n")
        sys.exit(1)

    correct_count = 0
    category_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    misclassifications = []

    print(f"{'ID':<4} | {'Expected Category':<15} | {'Predicted Category':<18} | {'Status'}")
    print("-" * 80)

    for item in tickets:
        ticket_id = item.get("id")
        text = item.get("ticket")
        expected = item.get("expected_category")

        try:
            result = analyzer.analyze(text)
            predicted = result.category.value

            is_correct = (predicted.lower() == expected.lower())
            status = "✅ MATCH" if is_correct else "❌ MISMATCH"

            if is_correct:
                correct_count += 1
                category_stats[expected]["correct"] += 1
            else:
                misclassifications.append({
                    "id": ticket_id,
                    "ticket": text,
                    "expected": expected,
                    "predicted": predicted
                })

            category_stats[expected]["total"] += 1

            print(f"{ticket_id:<4} | {expected:<15} | {predicted:<18} | {status}")

        except Exception as e:
            print(f"{ticket_id:<4} | {expected:<15} | {'ERROR':<18} | ❌ {str(e)[:25]}")
            category_stats[expected]["total"] += 1

    total_tickets = len(tickets)
    accuracy = (correct_count / total_tickets * 100) if total_tickets > 0 else 0.0

    print("=" * 80)
    print("                           EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Evaluated Tickets : {total_tickets}")
    print(f"Correct Classifications : {correct_count}")
    print(f"Incorrect / Errors      : {total_tickets - correct_count}")
    print(f"Overall Accuracy        : {accuracy:.2f}%")
    print("-" * 80)

    print("\nPer-Category Performance Breakdown:")
    print(f"{'Category':<15} | {'Total':<6} | {'Correct':<8} | {'Category Accuracy'}")
    print("-" * 55)
    for cat, stats in sorted(category_stats.items()):
        cat_acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0.0
        print(f"{cat:<15} | {stats['total']:<6} | {stats['correct']:<8} | {cat_acc:.1f}%")

    if misclassifications:
        print("\nMisclassified Tickets Analysis:")
        print("-" * 80)
        for m in misclassifications:
            print(f"Ticket #{m['id']}: \"{m['ticket']}\"")
            print(f"  -> Expected: {m['expected']} | Predicted: {m['predicted']}\n")
    else:
        print("\nAll tested tickets were classified with 100% precision!")

    print("=" * 80)
    print("Evaluation completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate category classification accuracy.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of tickets to evaluate (e.g. --limit 5)."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/sample_tickets.json",
        help="Path to evaluation dataset JSON."
    )
    args = parser.parse_args()

    run_evaluation(data_path=args.data, limit=args.limit)
