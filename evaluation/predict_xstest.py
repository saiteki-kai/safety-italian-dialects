import argparse

from pathlib import Path

from datasets import load_dataset
from src.eval_utils.model_utils import get_additional_config, get_system_message, model_to_filename
from vllm import LLM, SamplingParams


INPUT_DATASET = "output/xstest_translation"
INPUT_SPLIT = "test"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-dir", default="output/responses/by_model")
    parser.add_argument("--output-file", default=None)
    return parser.parse_args()


def prepare_messages(model_id, text):
    messages = []

    system_message = get_system_message(model_id)

    if system_message:
        messages.append({"role": "system", "content": system_message})

    messages.append({"role": "user", "content": text})

    return messages


reccommended_generation_configs = {
    "Fastweb/FastwebMIIA-7B": SamplingParams(temperature=0.1, top_p=0.9, repetition_penalty=1.1),
    "sapienzanlp/Minerva-7B-instruct-v1.0": SamplingParams(temperature=0.4, repetition_penalty=1.1),
    "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA": SamplingParams(temperature=0.6, top_p=0.9),
    "utter-project/EuroLLM-9B-Instruct-2512": SamplingParams(temperature=1.0),
    "utter-project/EuroLLM-22B-Instruct-2512": SamplingParams(temperature=1.0),
    "swiss-ai/Apertus-8B-Instruct-2509": SamplingParams(temperature=0.8, top_p=0.9),
    "swiss-ai/Apertus-70B-Instruct-2509": SamplingParams(temperature=0.8, top_p=0.9),
    "Qwen/Qwen3-8B": SamplingParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5),
    "Qwen/Qwen3.5-9B": SamplingParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5),
    "Qwen/Qwen3.5-27B": SamplingParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5),
    "Almawave/Velvet-14B": SamplingParams(temperature=1.0),
}


def get_sampling_params(model_id):
    gen_config = reccommended_generation_configs[model_id]

    return SamplingParams(
        temperature=gen_config.temperature if gen_config.temperature is not None else 1.0,
        top_p=gen_config.top_p if gen_config.top_p is not None else 1.0,
        top_k=gen_config.top_k if gen_config.top_k is not None else 0,
        min_p=gen_config.min_p if gen_config.min_p is not None else 0.0,
        repetition_penalty=gen_config.repetition_penalty if gen_config.repetition_penalty is not None else 1.0,
        presence_penalty=gen_config.presence_penalty if gen_config.presence_penalty is not None else 0.0,
        max_tokens=2048,
        seed=0,
        n=5,
    )


def main():
    args = parse_args()
    model_id = args.model_name_or_path

    original_dataset = load_dataset(INPUT_DATASET, split=INPUT_SPLIT)

    llm = LLM(
        model_id,
        language_model_only=True,
        additional_config=get_additional_config(model_id),
    )
    sampling_params = get_sampling_params(model_id)

    def generate_batch(batch: dict):
        messages = [prepare_messages(model_id, text) for text in batch["prompt"]]
        outputs = llm.chat(messages, sampling_params, chat_template_kwargs={"enable_thinking": False})

        return {
            "input_tokens": [len(output.prompt_token_ids or []) for output in outputs],
            "output_tokens": [[len(seq.token_ids) for seq in output.outputs] for output in outputs],
            "response": [[seq.text for seq in output.outputs] for output in outputs],
        }

    dataset = original_dataset.map(generate_batch, batched=True)
    dataset = dataset.add_column("model", [model_id] * len(dataset))

    output_file = args.output_file
    if output_file is None:
        output_file = str(Path(args.output_dir) / f"{model_to_filename(model_id)}.parquet")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_file)


if __name__ == "__main__":
    main()
