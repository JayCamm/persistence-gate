from __future__ import annotations

from pathlib import Path

from persistence_memory.llm_eval import compare_responses, response_to_dict


def main() -> None:
    response_dir = Path("benchmark_results/manual_llm")
    ordinary_path = response_dir / "ordinary_response.txt"
    gated_path = response_dir / "gated_response.txt"

    if not ordinary_path.exists() or not gated_path.exists():
        print("Missing response files. First run: python examples/export_manual_llm_prompts.py")
        raise SystemExit(1)

    ordinary_response = ordinary_path.read_text(encoding="utf-8").strip()
    gated_response = gated_path.read_text(encoding="utf-8").strip()

    if not ordinary_response or not gated_response:
        print("Response files are empty.")
        print("Paste model outputs into:")
        print(ordinary_path)
        print(gated_path)
        raise SystemExit(1)

    comparison = compare_responses(ordinary_response, gated_response)

    print("Manual LLM Response Scoring")
    print("===========================")
    print("\n--- Ordinary response ---")
    print(ordinary_response)
    print("\nSafety summary:", response_to_dict(comparison.ordinary))
    print("\n--- Gated response ---")
    print(gated_response)
    print("\nSafety summary:", response_to_dict(comparison.gated))
    print("\nVerdict:", comparison.verdict)


if __name__ == "__main__":
    main()
