import argparse

from functools import partial
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from translation_utils import (
    DIALECTS,
    ENGLISH_LABEL,
    GOOGLE_TRANSLATE_CODES,
    DatasetTranslationRunConfig,
    TranslationRetryConfig,
    build_checkpoint_path,
    existing_dialect_checkpoints,
    load_checkpoint_state,
    require_complete_checkpoint,
    save_combined_dataset,
    translate_dataset_with_checkpoint,
    translate_with_retry,
)


CHECKPOINT_FIELDS = ("questions", "choices", "answers", "subsets")
DATASET_ID = "cais/mmlu"
DATASET_CONFIG = "all"
SPLIT = "dev"
OUTPUT_DIR = Path("output/mmlu_translation")
CHECKPOINT_DIR = OUTPUT_DIR / "dev_checkpoints"
FINAL_OUTPUT_PATH = OUTPUT_DIR / "dev.parquet"

SAVE_EVERY_N_ROWS = 20
COOLDOWN_EVERY_N_ROWS = 0
COOLDOWN_SEC = 0.0

RETRY_CONFIG = TranslationRetryConfig(
    max_retries=3,
    retry_sleep_sec=0.5,
    backoff_base_sec=3.0,
    backoff_max_sec=30.0,
    request_pacing_sec=0.5,
    from_language="en",
    failure_prefix="[FAILED]",
)


def transform_row(row: dict[str, Any], target: str) -> dict[str, Any]:
    return {
        "questions": translate_with_retry(row["question"], target, config=RETRY_CONFIG),
        "choices": [translate_with_retry(choice, target, config=RETRY_CONFIG) for choice in row["choices"]],
        "answers": row["answer"],
        "subsets": row["subject"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate MMLU dev split and save one checkpoint per dialect.")
    parser.add_argument("--dialect", choices=DIALECTS)
    args = parser.parse_args()

    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split=SPLIT)
    dialects = [args.dialect] if args.dialect else DIALECTS

    for dialect in dialects:
        target = GOOGLE_TRANSLATE_CODES[dialect]
        checkpoint_path = build_checkpoint_path(CHECKPOINT_DIR, dialect)
        state = load_checkpoint_state(
            checkpoint_path,
            metadata={"lang": dialect, "target": target},
            sequence_fields=CHECKPOINT_FIELDS,
        )

        translate_dataset_with_checkpoint(
            dataset,
            state=state,
            checkpoint_path=checkpoint_path,
            transform_row=partial(transform_row, target=target),
            run_config=DatasetTranslationRunConfig(
                progress_label=f"Dialect: {dialect}",
                sequence_fields=CHECKPOINT_FIELDS,
                save_every_n_rows=SAVE_EVERY_N_ROWS,
                cooldown_every_n_rows=COOLDOWN_EVERY_N_ROWS,
                cooldown_sec=COOLDOWN_SEC,
            ),
        )
        print(f"Completed checkpoint for {dialect}: {checkpoint_path.as_posix()}")

    expected_rows = len(dataset)
    english_dataset = dataset.rename_column("subject", "subset")
    english_dataset = english_dataset.add_column("lang", [ENGLISH_LABEL] * expected_rows)

    translated_datasets = []
    output_features = english_dataset.features
    for dialect, checkpoint_path in existing_dialect_checkpoints(CHECKPOINT_DIR, DIALECTS):
        target = GOOGLE_TRANSLATE_CODES[dialect]
        state = load_checkpoint_state(
            checkpoint_path,
            metadata={"lang": dialect, "target": target},
            sequence_fields=CHECKPOINT_FIELDS,
        )
        translated_rows = require_complete_checkpoint(
            state,
            sequence_fields=CHECKPOINT_FIELDS,
            expected_rows=expected_rows,
            label=dialect,
            path=checkpoint_path,
        )

        translated = Dataset.from_dict(
            {
                "question": state["questions"],
                "choices": state["choices"],
                "answer": state["answers"],
                "subset": state["subsets"],
                "lang": [dialect] * translated_rows,
            },
            features=output_features,
        )
        translated_datasets.append(translated)

    save_combined_dataset(FINAL_OUTPUT_PATH, english_dataset, translated_datasets)


if __name__ == "__main__":
    main()
