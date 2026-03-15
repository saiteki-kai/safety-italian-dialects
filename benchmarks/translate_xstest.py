import argparse

from functools import partial
from pathlib import Path
from typing import Any

from datasets import load_dataset
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


CHECKPOINT_FIELDS = ("prompts",)
DATASET_ID = "walledai/XSTest"
SPLIT = "test"
OUTPUT_DIR = Path("output/xstest_translation")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FINAL_OUTPUT_PATH = OUTPUT_DIR / "test.parquet"

SAVE_EVERY_N_ROWS = 20
COOLDOWN_EVERY_N_ROWS = 100
COOLDOWN_SEC = 1.0

RETRY_CONFIG = TranslationRetryConfig(
    max_retries=3,
    retry_sleep_sec=0.5,
    backoff_base_sec=3.0,
    backoff_max_sec=30.0,
    request_pacing_sec=0.0,
    from_language="en",
    failure_prefix="[FAILED]",
)


def transform_row(row: dict[str, Any], target: str) -> dict[str, Any]:
    return {
        "prompts": translate_with_retry(row["prompt"], target, config=RETRY_CONFIG),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate XSTest prompts and save one checkpoint per dialect.")
    parser.add_argument("--dialect", choices=DIALECTS)
    args = parser.parse_args()

    original_dataset = load_dataset(DATASET_ID, split=SPLIT)
    dialects = [args.dialect] if args.dialect else DIALECTS

    for dialect in dialects:
        target = GOOGLE_TRANSLATE_CODES[dialect]
        checkpoint_path = build_checkpoint_path(CHECKPOINT_DIR, dialect)
        state = load_checkpoint_state(
            checkpoint_path,
            metadata={"dialect": dialect},
            sequence_fields=CHECKPOINT_FIELDS,
        )

        translate_dataset_with_checkpoint(
            original_dataset,
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

    translated_datasets = []
    expected_rows = len(original_dataset)
    for dialect, checkpoint_path in existing_dialect_checkpoints(CHECKPOINT_DIR, DIALECTS):
        state = load_checkpoint_state(
            checkpoint_path,
            metadata={"dialect": dialect},
            sequence_fields=CHECKPOINT_FIELDS,
        )
        translated_rows = require_complete_checkpoint(
            state,
            sequence_fields=CHECKPOINT_FIELDS,
            expected_rows=expected_rows,
            label=dialect,
            path=checkpoint_path,
        )

        translated = original_dataset.remove_columns(["prompt"])
        translated = translated.add_column("prompt", state["prompts"])
        translated = translated.add_column("lang", [dialect] * translated_rows)
        translated_datasets.append(translated)

    original_dataset = original_dataset.add_column("lang", [ENGLISH_LABEL] * expected_rows)
    save_combined_dataset(FINAL_OUTPUT_PATH, original_dataset, translated_datasets)


if __name__ == "__main__":
    main()
