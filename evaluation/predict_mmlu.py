import argparse

from pathlib import Path
from typing import Any, cast

import torch

from datasets import concatenate_datasets, load_dataset
from model_utils import DEFAULT_MODELS, get_additional_config, get_system_message, model_to_filename
from prompt import BasePromptBuilder, PromptBuilderFactory
from vllm import LLM, SamplingParams


DATASET_NAME = "saiteki-kai/mmlu-redux-dialects"
TEST_SPLIT = "test"
OUTPUT_DIR = Path("output/mmlu_predictions/by_model")

CHOICES = ["A", "B", "C", "D"]
SUPPORTED_LANGUAGES = ["English", "Italian", "Friulian", "Venetian", "Lombard", "Sicilian", "Ligurian"]

SAMPLING_PARAMS = SamplingParams(temperature=0.0, max_tokens=1, logprobs=2000, seed=42)
TOP_K_TOKENS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MMLU prediction runner with different prompt templates.")
    parser.add_argument("--model", choices=DEFAULT_MODELS, required=True)
    parser.add_argument("--config", required=True, help="Config file for prompt template")
    return parser.parse_args()


def _format_choices(answers: list[str]) -> str:
    if len(answers) != len(CHOICES):
        msg = f"Expected {len(CHOICES)} answer choices, got {len(answers)}."
        raise ValueError(msg)

    return "\n".join(f"{choice}. {answer}" for choice, answer in zip(CHOICES, answers, strict=True))


def _get_choice_variant_token_ids(tokenizer: Any, choice: str) -> list[int]:
    variant_token_ids: list[int] = []

    for variant in [choice, f" {choice}"]:
        token_ids = tokenizer(variant, add_special_tokens=False).input_ids

        if len(token_ids) != 1:
            continue

        token_id = token_ids[0]

        if token_id not in variant_token_ids:
            variant_token_ids.append(token_id)

    if not variant_token_ids:
        msg = f"Could not find any single-token variants for choice {choice!r}."
        raise ValueError(msg)

    return variant_token_ids


def _aggregate_choice_logprob(step_logprobs: dict[int, Any], token_ids: list[int]) -> float:
    variant_logprobs = [
        token_info.logprob for token_id in token_ids if (token_info := step_logprobs.get(token_id)) is not None
    ]

    if not variant_logprobs:
        return float("-inf")

    return cast(float, torch.logsumexp(torch.as_tensor(variant_logprobs), dim=0).item())


def _extract_top_tokens(step_logprobs: dict[int, Any], tokenizer: Any) -> tuple[list[str], list[float]]:
    sorted_logprobs = sorted(step_logprobs.items(), key=lambda item: item[1].logprob, reverse=True)
    top_k = sorted_logprobs[:TOP_K_TOKENS]

    current_tokens: list[str] = []
    current_token_logprobs: list[float] = []
    for token_id, token_info in top_k:
        decoded_token = getattr(token_info, "decoded_token", None)
        if decoded_token is None:
            decoded_token = tokenizer.decode([token_id])

        current_tokens.append(decoded_token)
        current_token_logprobs.append(token_info.logprob)

    return current_tokens, current_token_logprobs


def _build_predictions(outputs: Any, token_id_map: dict[str, list[int]], tokenizer: Any) -> dict[str, Any]:
    probs = []
    predictions = []
    top_tokens = []
    top_token_logprobs = []

    for output in outputs:
        if not output.outputs:
            raise ValueError("Model output is missing generated candidates.")

        first_output = output.outputs[0]
        logprobs = first_output.logprobs

        if logprobs is None:
            raise ValueError("Logprobs are not available. Set sampling logprobs > 0.")

        if not logprobs:
            raise ValueError("Model output is missing step logprobs.")

        step_logprobs = logprobs[0]
        choice_logprobs = [_aggregate_choice_logprob(step_logprobs, token_id_map[choice]) for choice in CHOICES]
        current_tokens, current_token_logprobs = _extract_top_tokens(step_logprobs, tokenizer)

        probs.append(choice_logprobs)
        predictions.append(first_output.text)
        top_tokens.append(current_tokens)
        top_token_logprobs.append(current_token_logprobs)

    probs_tensor = torch.softmax(torch.as_tensor(probs), dim=-1)

    return {
        "probs": probs_tensor.tolist(),
        "prediction": predictions,
        "top_tokens": top_tokens,
        "top_token_logprobs": top_token_logprobs,
    }


def row_to_prompt(prompt_builder: BasePromptBuilder, row: dict[str, Any]) -> dict[str, str | list[dict[str, str]]]:
    payload = prompt_builder.build(
        subject=row["subset"],
        question=row["question"],
        choices=_format_choices(row["choices"]),
    )

    return {"messages": payload} if isinstance(payload, list) else {"prompt": payload}


def main() -> None:
    args = parse_args()

    model_id = args.model
    config_path = Path(args.config)

    test_dataset = load_dataset(DATASET_NAME, split=TEST_SPLIT)

    llm = LLM(model_id, max_logprobs=2000, language_model_only=True, additional_config=get_additional_config(model_id))
    token_id_map = {choice: _get_choice_variant_token_ids(llm.get_tokenizer(), choice) for choice in CHOICES}
    sys_msg = get_system_message(model_id)

    def predict_batch(batch: dict[str, Any]) -> dict[str, Any]:
        outputs = llm.chat(
            batch["messages"],
            sampling_params=SAMPLING_PARAMS,
            chat_template_kwargs={"enable_thinking": False},
            add_generation_prompt=False,
            continue_final_message=True,
            use_tqdm=False,
        )

        return _build_predictions(outputs, token_id_map, llm.get_tokenizer())

    mapped_datasets = []

    for language in SUPPORTED_LANGUAGES:
        prompt_builder = PromptBuilderFactory.from_yaml(config_path, language, system_message=sys_msg)

        dataset_lang = test_dataset.filter(lambda e, language=language: e["lang"] == language)
        dataset_lang = dataset_lang.map(lambda row, prompt_builder=prompt_builder: row_to_prompt(prompt_builder, row))
        dataset_lang = dataset_lang.map(predict_batch, batched=True, batch_size=1)

        mapped_datasets.append(dataset_lang)

    dataset = concatenate_datasets(mapped_datasets)
    dataset = dataset.add_column("model", [model_id] * len(dataset))

    output_file = OUTPUT_DIR / f"{model_to_filename(model_id)}.parquet"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(output_file))


if __name__ == "__main__":
    main()
