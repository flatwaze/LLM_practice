from pathlib import Path
import requests
import json

from src.llm_client import LLMClient


LLM_MODEL = "google/gemma-4-31b-it"


def main():
    #reading questions
    q_path = Path(__file__).resolve().parent / "data" / "rag_eval_questions.json"
    with q_path.open(encoding='utf-8') as file:
        questions = json.load(file)

    try:
        llm_client = LLMClient(model=LLM_MODEL, api_key_env="OPENROUTER_API_KEY")
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    results = []
    results_path = Path(__file__).resolve().parent / "experiments" / "baseline_evaluation.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with llm_client.session:
        for q in questions:
            print(f"\n{q['query_id']}: {q['query']}")
            messages = [
                {
                    "role": "system",
                    "content": "Answer the question accurately and directly in at most 120 words. Avoid introductions and repetition. Include code only if needed. If you do not know the answer, say so."
                },
                {
                    "role": "user",
                    "content": q["query"]
                }
            ]

            try:
                answer = llm_client.generate(messages)
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                print(f"Error: {exc}")
                results.append({
                    **q,
                    "model": LLM_MODEL,
                    "temperature": 0,
                    "max_tokens": llm_client.max_tokens,
                    "baseline": None,
                    "error": str(exc)
                })
            else:
                print(f"\nAnswer:\n{answer}")
                results.append({
                    **q,
                    "model": LLM_MODEL,
                    "temperature": 0,
                    "max_tokens": llm_client.max_tokens,
                    "baseline": {"answer": answer}
                })

            with results_path.open("w", encoding="utf-8") as file:
                json.dump(results, file, ensure_ascii=False, indent=2)

            print('-----------------')


if __name__ == "__main__":
    main()
