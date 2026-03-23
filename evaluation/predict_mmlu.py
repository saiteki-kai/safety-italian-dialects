import argparse

from collections.abc import Callable
from functools import partial
from pathlib import Path

import torch

from datasets import load_dataset
from mmlu_utils import (
    get_answer_prefill,
    get_answer_prefix,
    get_json_instruction,
    get_one_letter_instruction,
    get_subject_system_message,
)
from model_utils import DEFAULT_MODELS, get_llm_additional_config, get_system_message, model_to_filename
from vllm import LLM, SamplingParams
from vllm.tokenizers import get_tokenizer


CHOICES = ["A", "B", "C", "D"]

PROMPT_MODES = ["json", "fewshot"]
TOP_K_TOKENS = 20


def _get_choice_variant_token_ids(tokenizer, choice):
    variant_token_ids = []

    for variant in [choice, f" {choice}"]:
        token_ids = tokenizer(variant, add_special_tokens=False).input_ids

        if token_ids[-1] not in variant_token_ids:
            variant_token_ids.append(token_ids[-1])

    if not variant_token_ids:
        msg = f"Could not find any single-token variants for choice {choice!r}."
        raise ValueError(msg)

    return variant_token_ids


def _aggregate_choice_logprob(step_logprobs, token_ids):
    variant_logprobs = [
        token_info.logprob for token_id in token_ids if (token_info := step_logprobs.get(token_id)) is not None
    ]

    if not variant_logprobs:
        return float("-inf")

    return torch.logsumexp(torch.as_tensor(variant_logprobs), dim=0).item()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-mode", choices=PROMPT_MODES, default="json")
    parser.add_argument("--model", choices=DEFAULT_MODELS, required=True)
    parser.add_argument("--output-dir", default="output/mmlu_predictions/by_model")
    parser.add_argument("--output-file", default=None)
    return parser.parse_args()


def _format_choices(answers):
    return "\n".join(f"{choice}. {answer}" for choice, answer in zip(CHOICES, answers, strict=True))


def _format_question_choices_prompt(question, answers):
    return f"{question}\n{_format_choices(answers)}"


def _get_fixed_fewshot_examples(fewshot_dataset, lang, subject):
    return fewshot_dataset.filter(lambda e: e["lang"] == lang and e["subset"] == subject)


def _create_json_messages(sys_msg, question, answers, subject, lang="English"):
    assistant_prefill = get_answer_prefill(lang)
    user_prompt = _format_question_choices_prompt(question, answers)

    messages = [{"role": "system", "content": sys_msg}] if sys_msg else []
    messages.append(
        {
            "role": "system",
            "content": get_subject_system_message(subject, lang) + "\n" + get_json_instruction(lang),
        }
    )
    messages.append({"role": "user", "content": user_prompt})
    messages.append({"role": "assistant", "content": assistant_prefill})

    return messages


def _create_fewshot_messages(  # noqa: PLR0913
    sys_msg,
    question,
    answers,
    subject,
    lang="English",
    fewshot_dataset=None,
):
    messages = [{"role": "system", "content": sys_msg}] if sys_msg else []
    messages.append(
        {
            "role": "system",
            "content": get_subject_system_message(subject, lang) + "\n" + get_one_letter_instruction(lang),
        }
    )
    answer_prefix = get_answer_prefix(lang)

    shots = _get_fixed_fewshot_examples(fewshot_dataset, lang, subject)

    for shot in shots:
        shot_user_prompt = _format_question_choices_prompt(shot["question"], shot["choices"])
        shot_answer = CHOICES[shot["answer"]]

        messages.extend(
            [
                {"role": "user", "content": shot_user_prompt},
                {"role": "assistant", "content": f"{answer_prefix}{shot_answer}"},
            ]
        )

    user_prompt = _format_question_choices_prompt(question, answers)
    messages.extend(
        [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": answer_prefix},
        ]
    )
    return messages


def make_predict_batch(  # noqa: PLR0913
    llm: LLM,
    model_id: str,
    token_id_map: dict,
    sampling_params: SamplingParams,
    message_builder: Callable[[str | None, str, list[str], str, str], list[dict[str, str]]],
    tokenizer,
):
    def predict_batch(batch):
        sys_msg = get_system_message(model_id)

        messages = [
            message_builder(sys_msg, question, answers, subject, lang)
            for question, answers, subject, lang in zip(
                batch["question"],
                batch["choices"],
                batch["subset"],
                batch["lang"],
                strict=True,
            )
        ]

        outputs = llm.chat(
            messages,
            sampling_params=sampling_params,
            chat_template_kwargs={"enable_thinking": False},
            add_generation_prompt=False,
            continue_final_message=True,
            use_tqdm=False,
        )

        probs = []
        predictions = []
        top_tokens = []
        top_token_logprobs = []
        for output in outputs:
            logprobs = output.outputs[0].logprobs

            if logprobs is None:
                msg = (
                    "Logprobs are not available. Please make sure to set `logprobs` "
                    "in `sampling_params` to a value greater than 0."
                )
                raise ValueError(msg)

            step_logprobs = logprobs[0]
            choice_logprobs = [_aggregate_choice_logprob(step_logprobs, token_id_map[choice]) for choice in CHOICES]
            sorted_logprobs = sorted(step_logprobs.items(), key=lambda item: item[1].logprob, reverse=True)
            top_k = sorted_logprobs[:TOP_K_TOKENS]

            current_tokens = []
            current_token_logprobs = []
            for token_id, token_info in top_k:
                decoded_token = getattr(token_info, "decoded_token", None)
                if decoded_token is None:
                    decoded_token = tokenizer.decode([token_id])

                current_tokens.append(decoded_token)
                current_token_logprobs.append(token_info.logprob)

            probs.append(choice_logprobs)
            predictions.append(output.outputs[0].text)
            top_tokens.append(current_tokens)
            top_token_logprobs.append(current_token_logprobs)

        probs = torch.as_tensor(probs)
        probs = torch.softmax(probs, dim=-1)

        return {
            "probs": probs.tolist(),
            "prediction": predictions,
            "top_tokens": top_tokens,
            "top_token_logprobs": top_token_logprobs,
        }

    return predict_batch


def main():
    args = parse_args()
    model_id = args.model

    original_dataset = load_dataset("saiteki-kai/mmlu-redux-dialects", split="test")
    fewshot_dataset = load_dataset("saiteki-kai/mmlu-redux-dialects", split="validation")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=2000,
        # structured_outputs=StructuredOutputsParams(choice=CHOICES),
        seed=42,
    )

    llm = LLM(
        model_id,
        max_logprobs=2000,
        language_model_only=True,
        additional_config=get_llm_additional_config(model_id),
    )

    tokenizer = get_tokenizer(model_id)
    token_id_map = {choice: _get_choice_variant_token_ids(tokenizer, choice) for choice in CHOICES}

    message_builder = (
        partial(_create_fewshot_messages, fewshot_dataset=fewshot_dataset)
        if args.prompt_mode == "fewshot"
        else _create_json_messages
    )
    predict_batch = make_predict_batch(
        llm,
        model_id,
        token_id_map,
        sampling_params,
        message_builder=message_builder,
        tokenizer=tokenizer,
    )

    dataset = original_dataset.map(predict_batch, batched=True, batch_size=1)
    dataset = dataset.add_column("model", [model_id] * len(dataset))

    output_file = args.output_file
    if output_file is None:
        output_file = str(Path(args.output_dir) / f"{model_to_filename(model_id)}.parquet")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_file)


if __name__ == "__main__":
    main()
