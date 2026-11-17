
from llm_sdk import Small_LLM_Model


def main() -> None:
    model = Small_LLM_Model()

    prompt = "The capital of France is"

    input_ids = model.encode(prompt)[0].tolist()

    generated_tokens: list[int] = []

    for _ in range(10):
        logits = model.get_logits_from_input_ids(input_ids)

        best_token_id = max(
            range(len(logits)),
            key=lambda token_id: logits[token_id],
        )

        input_ids.append(best_token_id)
        generated_tokens.append(best_token_id)

    result = model.decode(generated_tokens)

    print("Generated text:", result)


if __name__ == "__main__":
    main()
